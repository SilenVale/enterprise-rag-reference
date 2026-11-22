"""一条命令跑完各层演示。输出写明每一层走了哪一格。

嵌入用 BGE-M3，精排用 bge-reranker-v2-m3，加载方式和已经通过的两层相同。
不加载 Qwen2.5-72B。四个存储连本机服务，不用内存替身。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from rag.evaluate.cases import CASES
from rag.evaluate.gate import compare_to_baseline
from rag.evaluate.runner import run_suite
from rag.evaluate.store import EvaluateStore, numbers_to_payload
from rag.evaluate.types import RunConfig
from rag.generate.adapters import LocalExtractiveGenerator
from rag.generate.entailment import EntailmentJudge
from rag.generate.log import GenerateLog
from rag.generate.pipeline import GeneratePipeline
from rag.generate.types import GenerateRequest, ParentRef
from rag.index.catalog import IndexCatalog
from rag.index.embedder import BgeM3Embedder
from rag.index.pipeline import COLLECTION, IndexPipeline
from rag.index.types import Clause, PolicyDocument, SourceRef
from rag.ingest.pipeline import ingest_documents
from rag.ingest.types import BBox, PageModel, SourceDocument, SourceIdentity, TextBlock
from rag.retrieve.params import (
    GENERATION_MAIN_MODEL,
    PATH_TOP_K,
    RRF_K,
    TEXT_INDEX,
    VECTOR_COLLECTION,
)
from rag.retrieve.pipeline import RetrievePipeline
from rag.retrieve.refusal import RefusalLog
from rag.retrieve.rerank import BgeReranker
from rag.retrieve.types import RetrieveRequest, Turn
from rag.storage.elasticsearch_store import ElasticsearchChildTextStore
from rag.storage.hashing import content_hash
from rag.storage.metadata_store import PostgresMetadataStore
from rag.storage.milvus_store import MilvusChildVectorStore
from rag.storage.object_store import MinioBlobStore, open_parent_store
from rag.storage.probe import probe_services
from rag.storage.settings import load_settings
from rag.storage.types import ChildText, ChildVector, ScalarFilter

AS_OF = date(2024, 6, 1)
EXAM = "员工年度体检补贴为每人 640 元，于每年三月发放。"
LODGING = "出差住宿报销上限为每晚 320 元，返回后十日内提交票据。"
PARAPHRASE = "每年三月向员工发放体检补贴，标准是每人 640 元。"
CONTRADICTION = "员工年度体检补贴已经停发，以后不再发放。"


class DemoFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class DemoReport:
    lines: tuple[str, ...]
    passed: bool


class _Draft:
    """把一份草稿送进出库校验。不加载生成主模型。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.model_id = "local-draft"
        self.weights_loaded = False

    def complete(self, prompt) -> str:
        return self.text


class _TracingEmbedder:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.queries: list[str] = []

    def embed_query(self, text: str):
        self.queries.append(text)
        return self.inner.embed_query(text)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _TracingVectors:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple[object, int]] = []

    def search(self, query, *, limit, scalar):
        self.calls.append((scalar, limit))
        return self.inner.search(query, limit=limit, scalar=scalar)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _TracingLexical:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple[str, object, int]] = []

    def search(self, query, *, limit, scalar):
        self.calls.append(("bm25", scalar, limit))
        return self.inner.search(query, limit=limit, scalar=scalar)

    def search_clause(self, clause_number, *, limit, scalar):
        self.calls.append(("clause", scalar, limit))
        return self.inner.search_clause(clause_number, limit=limit, scalar=scalar)

    def list_matching(self, *, limit, scalar):
        self.calls.append(("metadata", scalar, limit))
        return self.inner.list_matching(limit=limit, scalar=scalar)

    def fetch_texts(self, child_ids):
        return self.inner.fetch_texts(child_ids)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main() -> int:
    report = run_demo(echo=True)
    return 0 if report.passed else 1


def run_demo(*, echo: bool = True) -> DemoReport:
    lines: list[str] = []
    try:
        _execute(lines)
        lines.append(_line("端到端", "结果", "通过"))
        passed = True
    except Exception as exc:
        lines.append(_line("端到端", "结果", "未通过", 原因=_one_line(exc)))
        passed = False
    if echo:
        for line in lines:
            print(line, flush=True)
    return DemoReport(tuple(lines), passed)


def _execute(lines: list[str]) -> None:
    settings = load_settings()
    probed = probe_services(settings)
    _expect(probed.backend == "services", "探测没有走本机四个服务")
    metadata = PostgresMetadataStore(settings.metadata_dsn)
    metadata.ensure_schema()
    vectors = MilvusChildVectorStore(settings.milvus_uri, collection=COLLECTION)
    parents = open_parent_store(settings)
    lexical = ElasticsearchChildTextStore(settings.elasticsearch_url, index=TEXT_INDEX)
    _expect_type(metadata, PostgresMetadataStore)
    _expect_type(vectors, MilvusChildVectorStore)
    _expect_type(parents.blobs, MinioBlobStore)
    _expect_type(lexical, ElasticsearchChildTextStore)
    lines.append(
        _line(
            "存储",
            "Milvus。向量检索，HNSW",
            "已接通",
            地址=_host(settings.milvus_uri),
            版本=probed.milvus_version,
            实现=type(vectors).__name__,
            集合=COLLECTION,
        )
    )
    lines.append(
        _line(
            "存储",
            "Elasticsearch。BM25 倒排",
            "已接通",
            地址=_host(settings.elasticsearch_url),
            实现=type(lexical).__name__,
            索引=TEXT_INDEX,
        )
    )
    lines.append(
        _line(
            "存储",
            "对象存储。原始文档，版本快照",
            "已接通",
            地址=_host(settings.object_endpoint),
            实现=type(parents.blobs).__name__,
            桶=settings.object_bucket,
        )
    )
    lines.append(
        _line(
            "存储",
            "元数据库。引用日志，评测记录",
            "已接通",
            地址=_metadata_host(settings.metadata_dsn),
            实现=type(metadata).__name__,
        )
    )

    embedder = BgeM3Embedder()
    embedder.load()
    _expect(embedder.model_id == "BAAI/bge-m3", "嵌入模型不是 BGE-M3")
    _expect("bge-m3" in embedder.resolved_name.lower(), "加载到的不是 BGE-M3")
    lines.append(
        _line(
            "模型",
            "BGE-M3。向量嵌入",
            "已加载",
            模型=embedder.model_id,
            名称=embedder.resolved_name,
            维度=embedder.dimension,
        )
    )

    _show_manual_queue(lines, metadata)
    catalog = IndexCatalog(settings.metadata_dsn)
    catalog.ensure_schema()
    pipeline = IndexPipeline(
        embedder=embedder,
        vectors=vectors,
        parents=parents,
        catalog=catalog,
        as_of=AS_OF,
    )
    _show_index(lines, pipeline, embedder)

    reranker = BgeReranker()
    reranker.load()
    _expect(reranker.ready, "精排模型没有加载")
    _expect(reranker.model_id == "BAAI/bge-reranker-v2-m3", "精排模型不是 bge-reranker-v2-m3")
    lines.append(
        _line(
            "模型",
            "Cross-Encoder 精排。bge-reranker-v2-m3",
            "已加载",
            模型=reranker.model_id,
            名称=reranker.resolved_name,
        )
    )
    refusals = RefusalLog(settings.metadata_dsn)
    refusals.ensure_schema()
    lexical.ensure_ready()
    retrieve_vectors = MilvusChildVectorStore(settings.milvus_uri, collection=VECTOR_COLLECTION)
    retrieve_vectors.ensure_ready(embedder.dimension)
    _expect_type(retrieve_vectors, MilvusChildVectorStore)
    _show_retrieve(lines, embedder, retrieve_vectors, lexical, reranker, refusals)

    judge = EntailmentJudge()
    judge.load()
    _expect(judge.ready, "蕴含模型没有加载")
    _expect(not getattr(judge, "decides_by_string_equality", False), "蕴含不能用字符串全等")
    generate_log = GenerateLog(settings.metadata_dsn)
    generate_log.ensure_schema()
    lines.append(
        _line(
            "模型",
            "Qwen2.5-72B。vLLM，prefix caching",
            "未加载",
            演示档="local-extractive",
            权重="false",
        )
    )
    _show_generate(lines, parents, generate_log, judge)

    _show_gate(lines, settings.metadata_dsn)


def _show_manual_queue(lines: list[str], metadata: PostgresMetadataStore) -> None:
    source_id = _token("src")
    left = "左" * 40
    right = "右" * 40
    page = PageModel(
        width=500,
        height=700,
        text_layer_blocks=(TextBlock(left, BBox(40, 30, 180, 80)),),
        text_tables=(),
        text_layer_ratio=1.0,
        image_blocks=(TextBlock(left, BBox(40, 30, 180, 80)),),
    )
    bad = SourceDocument(
        SourceIdentity(source_id, "half-column", "归档"),
        (page,),
        left + right,
    )
    batch = ingest_documents((bad,), metadata)
    ready = {item.document_ref for item in batch.ready_for_index()}
    rows = metadata.list_parse_failures(source_id)
    _expect(ready == set(), "解析失败的文件进入了索引")
    _expect(len(rows) == 1, "人工队列里没有这条解析失败")
    _expect(rows[0].document_ref == "half-column", "人工队列里的文件对不上")
    _expect(rows[0].status == "queued", "解析失败没有停在人工队列")
    _expect(rows[0].attempt_count == 3, "失败后没有按三次尝试进入队列")
    _expect(rows[0].reason.startswith("字符保有率"), "失败原因不是字符保有率")
    lines.append(
        _line(
            "数据接入",
            "失败处置。重试，然后人工队列",
            "解析失败进入人工队列",
            文件="half-column",
            状态=rows[0].status,
            尝试=rows[0].attempt_count,
            进入索引="否",
            队列="元数据库",
        )
    )


def _show_index(lines: list[str], pipeline: IndexPipeline, embedder: BgeM3Embedder) -> None:
    count = embedder.count_tokens
    document_id = _token("d")
    first_clause = _clause(
        count,
        number="第一条",
        title_path=("差旅制度", "第二章 报销", "第一条 范围"),
        seed=f"{document_id}第一条这一版保持不变，用来复用向量。",
    )
    second_clause = _clause(
        count,
        number="第二条",
        title_path=("差旅制度", "第三章 住宿", "第二条 标准"),
        seed=f"{document_id}第二条切换前仍然对外。",
    )
    current = _policy(
        (first_clause, second_clause),
        (SourceRef("制度库", f"{document_id}-v1", "archive"),),
        document_id=document_id,
        version="v1",
        effective_from=date(2020, 1, 1),
    )
    receipt = pipeline.index_document(current)
    _expect(receipt.activated, "第一版没有生效")
    children = pipeline.catalog.children(document_id, "v1")
    child = next(item for item in children if item.clause_number == "第一条")
    hits = pipeline.lookup(child.embed_text, document_type=current.document_type, limit=5)
    hit = next((item for item in hits if item.child_id == child.child_id), None)
    _expect(hit is not None, "子块没有从向量库命中")
    assert hit is not None
    stored = pipeline.load_parent(document_id, child.parent_id)
    injected = pipeline.injection_for(child.child_id)
    _expect(stored.text == injected.text, "取回的父块和注入文本不一致")
    _expect(first_clause.statement[:12] in stored.text, "对象存储里的父块没有这条条款")
    _expect(not hasattr(hit, "text"), "向量命中带了父块正文")
    lines.append(
        _line(
            "索引工程",
            "父子语义分块。子块检索，父块注入",
            "子块命中后取回父块",
            子块=hit.child_id,
            父块=hit.parent_id,
            取回="对象存储",
            向量命中含正文="否",
        )
    )

    before = len(embedder.document_texts)
    duplicate = _policy(
        (first_clause, second_clause),
        (SourceRef("邮件", f"fwd-{document_id}", "forward"),),
        document_id=_token("d"),
        version="v9",
        effective_from=date(2020, 1, 1),
    )
    reused = pipeline.index_document(duplicate)
    _expect(reused.reused, "相同内容哈希又入库了")
    _expect(reused.document_id == document_id, "复用没有指回已有文档")
    _expect(len(embedder.document_texts) == before, "相同内容哈希又嵌入了")
    _expect(pipeline.catalog.children(duplicate.document_id, duplicate.version) == [], "复用文档又写了子块")
    sources = {
        (item.source_id, item.document_ref)
        for item in pipeline.catalog.list_sources(reused.normalized_hash)
    }
    _expect(("制度库", f"{document_id}-v1") in sources, "原来源没有留下")
    _expect(("邮件", f"fwd-{document_id}") in sources, "后来源没有留下")
    lines.append(
        _line(
            "索引工程",
            "内容寻址去重。按块做差量",
            "同一内容哈希不重复入库",
            复用="是",
            再次嵌入="0",
            来源数=len(sources),
        )
    )

    old_second = next(item for item in children if item.clause_number == "第二条")
    old_parent = next(
        item
        for item in pipeline.catalog.parents(document_id, "v1")
        if item.clause_numbers == ("第二条",)
    )
    old_text = pipeline.load_parent(document_id, old_parent.parent_id).text
    revised_second = _clause(
        count,
        number="第二条",
        title_path=("差旅制度", "第三章 住宿", "第二条 标准"),
        seed=f"{document_id}第二条切换后才对外。",
    )
    revised = _policy(
        (first_clause, revised_second),
        (SourceRef("制度库", f"{document_id}-v2", "archive"),),
        document_id=document_id,
        version="v2",
        effective_from=date(2024, 1, 1),
    )
    staged = pipeline.stage(revised)
    held = pipeline.lookup(old_second.embed_text, document_type=current.document_type, limit=5)
    _expect(pipeline.parents.active_version(document_id) == "v1", "切换前查询指针已经离开旧版")
    _expect(pipeline.load_parent(document_id, old_parent.parent_id).text == old_text, "切换前读到的不是旧版父块")
    _expect({item.version for item in held} == {"v1"}, "切换前检索没有停在旧版")
    _expect(
        all(pipeline.catalog.child(child_id).status == "pending" for child_id in staged.child_ids),
        "新版本在切换前已经生效",
    )
    lines.append(
        _line(
            "索引工程",
            "废止制度打标。保留可追溯",
            "修订后查询在切换前仍读旧版",
            指针="v1",
            检索版本="v1",
            新版状态="pending",
        )
    )

    pipeline.activate(document_id, "v2")
    unchanged = next(item for item in pipeline.catalog.children(document_id, "v1") if item.clause_number == "第一条")
    default_hits = pipeline.lookup(unchanged.embed_text, document_type=current.document_type, limit=5)
    audit_hits = pipeline.lookup(
        unchanged.embed_text,
        document_type=current.document_type,
        limit=5,
        include_obsolete=True,
    )
    _expect(default_hits, "切换后现行版本没有被检索到")
    _expect(all(item.version != "v1" for item in default_hits), "废止版本仍出现在默认检索里")
    _expect(any(item.version == "v2" for item in default_hits), "默认检索没有读到现行版本")
    _expect(any(item.version == "v1" for item in audit_hits), "审计查询带不出废止版本")
    _expect(pipeline.catalog.get_document(document_id, "v1").status == "obsolete", "旧版没有打上废止标")
    lines.append(
        _line(
            "索引工程",
            "废止制度打标。保留可追溯",
            "废止版本默认检索不到",
            默认版本=",".join(sorted({item.version for item in default_hits})),
            废止版本="未出现",
            审计可带出="是",
        )
    )


def _show_retrieve(lines, embedder, vectors, lexical, reranker, refusals) -> None:
    created: list[str] = []
    try:
        _show_permission(lines, embedder, vectors, lexical, reranker, refusals, created)
        _show_rewrite(lines, embedder, vectors, lexical, reranker, refusals, created)
        _show_tiers(lines, embedder, vectors, lexical, reranker, refusals, created)
    finally:
        if created:
            vectors.delete_children(created)
            lexical.delete_children(created)


def _show_permission(lines, embedder, vectors, lexical, reranker, refusals, created) -> None:
    document_type = _token("t")
    token = "qx" + uuid.uuid4().hex[:8]
    restricted_text = f"{token} " * 8 + "受限报销标准每日八百元整"
    public_text = f"食堂每周供应早餐，另附门禁和菜单说明 {token}"
    rows = [
        _retrieve_row(
            restricted_text,
            document_type=document_type,
            permission="restricted",
            child_id=f"a{index:03d}{token[:4]}",
            clause_number=f"第{index + 1}条",
        )
        for index in range(55)
    ]
    public = _retrieve_row(
        public_text,
        document_type=document_type,
        permission="public",
        child_id=f"z{token[:6]}",
        clause_number="附则",
    )
    created.extend(_index_rows(embedder, vectors, lexical, [*rows, public]))
    scope = ScalarFilter(effective_status="effective", document_type=document_type)
    unfiltered = vectors.search(embedder.embed_query(restricted_text), limit=PATH_TOP_K, scalar=scope)
    restricted_ids = {row["child_id"] for row in rows}
    _expect(public["child_id"] not in {hit.child_id for hit in unfiltered}, "公开块挤进了未加权限的前 50")
    _expect(len(unfiltered) == PATH_TOP_K, "未加权限的向量检索没有取满 top 50")
    traced_vectors = _TracingVectors(vectors)
    traced_lexical = _TracingLexical(lexical)
    result = RetrievePipeline(
        embedder=embedder,
        vectors=traced_vectors,
        lexical=traced_lexical,
        reranker=reranker,
        refusals=refusals,
    ).retrieve(RetrieveRequest(restricted_text, role="intern", document_type=document_type, request_id=_token("r")))
    found = _hit_ids(result)
    _expect(public["child_id"] in found, "实习生没有检索到公开制度")
    _expect(found.isdisjoint(restricted_ids), "无权限角色检索到了受限制度")
    _expect(traced_vectors.calls, "向量一路没有把过滤条件推进引擎")
    for scalar, limit in traced_vectors.calls:
        _expect(scalar.permission == "public", "实习生的向量检索没有先滤成公开")
        _expect(scalar.effective_status == "effective", "向量检索没有先滤生效状态")
        _expect(limit == PATH_TOP_K, "向量一路没有按 top 50 取")
    for _kind, scalar, limit in traced_lexical.calls:
        _expect(scalar.permission == "public", "倒排或元数据一路没有先滤成公开")
        _expect(limit == PATH_TOP_K, "倒排一路没有按 top 50 取")
    lines.append(
        _line(
            "检索链路",
            "三路并发召回。向量、BM25 稀疏、元数据过滤",
            "无权限的角色检索不到受限制度",
            角色="intern",
            引擎权限="public",
            受限命中="0",
            后过滤="否",
        )
    )


def _show_rewrite(lines, embedder, vectors, lexical, reranker, refusals, created) -> None:
    document_type = _token("t")
    target = _retrieve_row("差旅住宿每晚上限六百元", document_type=document_type, clause_number="第三条")
    other = _retrieve_row("食堂补贴按月发放", document_type=document_type, clause_number="第八条")
    created.extend(_index_rows(embedder, vectors, lexical, [target, other]))
    tracer = _TracingEmbedder(embedder)
    history = (
        Turn("user", "差旅住宿标准写在哪一条"),
        Turn("assistant", "写在差旅制度第三条"),
    )
    question = "那一条的上限是多少"
    rewritten = "差旅住宿标准第三条的上限是多少"
    result = RetrievePipeline(
        embedder=tracer,
        vectors=vectors,
        lexical=lexical,
        reranker=reranker,
        refusals=refusals,
    ).retrieve(
        RetrieveRequest(question, history=history, document_type=document_type, request_id=_token("r"))
    )
    _expect(result.queries[0] == question, "原问题没有参加召回")
    _expect(rewritten in result.queries, "改写问题没有参加召回")
    _expect(question in tracer.queries, "原问题没有去向量一路")
    _expect(rewritten in tracer.queries, "改写问题没有去向量一路")
    _expect(target["child_id"] in _hit_ids(result), "改写后没有召回对应条款")
    _expect(result.rewrite_model_id != GENERATION_MAIN_MODEL, "改写占用了生成主模型")
    lines.append(
        _line(
            "检索链路",
            "查询改写。指代消解，复合问题拆解",
            "原问题和改写都参与召回",
            原问题=question,
            改写=rewritten,
            改写模型=result.rewrite_model_id,
        )
    )


def _show_tiers(lines, embedder, vectors, lexical, reranker, refusals, created) -> None:
    question = "差旅住宿每晚上限是多少"
    samples = {
        "high": (question, "差旅住宿每晚上限为六百元。"),
        "mid": (question, "本条讨论差旅住宿报销，上限金额未完整给出。"),
        "low": ("量子引力波段的食堂菜谱如何编排", "门禁卡丢失后应当在当日挂失并补办。"),
    }
    outcomes = {}
    for name, (asked, text) in samples.items():
        document_type = _token("t")
        row = _retrieve_row(text, document_type=document_type, clause_number="第三条")
        created.extend(_index_rows(embedder, vectors, lexical, [row]))
        request_id = _token("r")
        result = RetrievePipeline(
            embedder=embedder,
            vectors=vectors,
            lexical=lexical,
            reranker=reranker,
            refusals=refusals,
        ).retrieve(RetrieveRequest(asked, document_type=document_type, request_id=request_id))
        _expect(result.disposition == name, f"{name} 档没有落到这一档，实际是 {result.disposition}")
        _expect(result.fused, f"{name} 档没有经过 RRF")
        _expect(result.passages, f"{name} 档没有精排结果")
        _expect_rrf(result)
        outcomes[name] = result
    low_logged = refusals.list_for(outcomes["low"].request_id)
    _expect(low_logged and low_logged[0][0] == "低分拒答", "低分没有写入拒答日志")
    lines.append(
        _line(
            "检索链路",
            "RRF 融合。免分数归一化",
            "融合用名次",
            k=RRF_K,
            加权余弦与BM25="否",
        )
    )
    lines.append(
        _line(
            "检索链路",
            "Cross-Encoder 精排。bge-reranker-v2-m3",
            "RRF之后精排",
            模型=reranker.model_id,
            高分=f"{outcomes['high'].passages[0].rerank_score:.4f}",
            中分=f"{outcomes['mid'].passages[0].rerank_score:.4f}",
            低分=f"{outcomes['low'].passages[0].rerank_score:.4f}",
        )
    )
    lines.append(
        _line(
            "检索链路",
            "三档分流。高分作答，中分提示，低分拒答",
            "分成高中低三档",
            高=outcomes["high"].route_label,
            中=outcomes["mid"].route_label,
            低=outcomes["low"].route_label,
        )
    )


def _show_generate(lines, parents, generate_log, judge) -> None:
    document_id, exam_id, lodging_id = _ids("g")
    _write_parent(parents, document_id, exam_id, EXAM)
    _write_parent(parents, document_id, lodging_id, LODGING)
    numbered_id = document_id + "ok"
    numbered = GeneratePipeline(
        parents=parents,
        audit=generate_log,
        generator=LocalExtractiveGenerator(),
        entailment=judge,
    ).run(
        GenerateRequest(
            question="体检补贴每人是多少，住宿报销上限是多少？",
            parents=(
                _parent_ref(exam_id, document_id),
                _parent_ref(lodging_id, document_id),
            ),
            request_id=numbered_id,
        )
    )
    _expect(numbered.action == "answer", "带编号的答案没有放行")
    _expect("[1]" in numbered.text and "[2]" in numbered.text, "答案没有带上编号")
    _expect(numbered.model_id == "local-extractive", "编号答案没有走本机档")
    _expect(numbered.weights_loaded is False, "本机档加载了生成权重")
    _expect(GENERATION_MAIN_MODEL not in numbered.model_id, "演示加载了 Qwen2.5-72B")
    _expect({row.outcome for row in generate_log.list_citations(numbered_id)} == {"放行"}, "放行没有写入引用日志")
    lines.append(
        _line(
            "生成防御",
            "编号注入提示词。逐句标注来源",
            "答案带编号",
            编号="[1][2]",
            模型=numbered.model_id,
            权重已加载="否",
            引用日志="元数据库",
        )
    )

    number_id = document_id + "num"
    number = GeneratePipeline(
        parents=parents,
        audit=generate_log,
        generator=_Draft("员工年度体检补贴为每人 1750 元。[1]"),
        entailment=judge,
    ).run(
        GenerateRequest(
            question="体检补贴是多少？",
            parents=(_parent_ref(exam_id, document_id),),
            request_id=number_id,
        )
    )
    _expect(number.action == "refused", "错误数字没有拦住整条答案")
    _expect("1750" not in number.text, "错误数字交了出去")
    _expect("拦截数字" in {row.outcome for row in generate_log.list_citations(number_id)}, "数字拦截没有写入引用日志")
    lines.append(
        _line(
            "生成防御",
            "数值一致性校验。数字必须见于引用原文",
            "错误数字被拦住",
            处置="拦截数字",
            交卷含错误数字="否",
            引用日志="元数据库",
        )
    )

    obsolete_id = document_id + "old"
    obsolete = GeneratePipeline(
        parents=parents,
        audit=generate_log,
        generator=_Draft(f"{EXAM}[1]{LODGING}[2]"),
        entailment=judge,
    ).run(
        GenerateRequest(
            question="体检补贴和住宿报销是多少？",
            parents=(
                _parent_ref(exam_id, document_id, status="obsolete"),
                _parent_ref(lodging_id, document_id),
            ),
            request_id=obsolete_id,
        )
    )
    _expect(obsolete.action == "answer", "废止句被拦截后，仍生效的句子没有留下")
    _expect("640" not in obsolete.text, "废止引用里的金额交了出去")
    _expect("320" in obsolete.text, "仍生效的句子没有留下")
    outcomes = {row.outcome for row in generate_log.list_citations(obsolete_id)}
    _expect("拦截废止" in outcomes, "废止引用没有被拦住")
    lines.append(
        _line(
            "生成防御",
            "版本一致性拦截。全景图写废止版本直接拒答",
            "废止引用被拦住",
            处置="拦截废止",
            废止金额交卷="否",
            现行句="留下",
        )
    )

    entail_id = document_id + "nli"
    entail = GeneratePipeline(
        parents=parents,
        audit=generate_log,
        generator=_Draft(f"{PARAPHRASE}[1]{CONTRADICTION}[1]{LODGING}[2]"),
        entailment=judge,
    ).run(
        GenerateRequest(
            question="体检补贴和住宿报销是多少？",
            parents=(
                _parent_ref(exam_id, document_id),
                _parent_ref(lodging_id, document_id),
            ),
            request_id=entail_id,
        )
    )
    _expect(entail.action == "answer", "删掉不支持的句子后没有留下可读答案")
    _expect(entail.entailment_deletion_executed is True, "不支持的句子没有被删掉")
    _expect("停发" not in entail.text, "得不到支持的句子交了出去")
    _expect("640" in entail.text, "得到支持的句子被删掉了")
    _expect("删句" in {row.outcome for row in generate_log.list_citations(entail_id)}, "删句没有写入引用日志")
    lines.append(
        _line(
            "生成防御",
            "NLI 支持性校验。全景图写不通过删句",
            "蕴含不支持的句子被删掉",
            处置="删句",
            不支持句交卷="否",
            整条降级="否",
            模型=judge.model_id,
        )
    )


def _show_gate(lines, dsn: str) -> None:
    store = EvaluateStore(dsn)
    store.ensure_schema()
    store.replace_cases(CASES)
    loaded = store.load_cases()
    _expect(len(loaded) == len(CASES), "评测集没有从元数据库读回来")
    _expect({case.case_key for case in loaded} == {case.case_key for case in CASES}, "读回的评测集和写入的不一致")
    baseline = run_suite(cases=loaded)
    run_id = uuid.uuid4().hex
    store.save_report(run_id, numbers_to_payload(baseline.numbers), baseline=True)
    current = run_suite(RunConfig(top_k=1), cases=loaded)
    decision = compare_to_baseline(store.load_baseline(), current.numbers)
    _expect(decision.blocked, "评测回退没有阻断")
    _expect(any("Recall@5" in reason for reason in decision.reasons), "阻断原因没有落到 Recall@5")
    lines.append(
        _line(
            "评测门禁",
            "CI 回归门禁。指标回退阻断发布",
            "评测回退会阻断",
            阻断="是",
            变更="topK",
            依据="元数据库基线",
            原因=decision.reasons[0],
        )
    )


def _expect_rrf(result) -> None:
    top = result.fused[0]
    ranks = {}
    for path in result.paths:
        for hit in path.hits:
            if hit.child_id == top.child_id:
                ranks[path.name] = hit.rank
    _expect(ranks, "融合结果对不上三路名次")
    expected = sum(1.0 / (RRF_K + rank) for rank in ranks.values())
    _expect(abs(expected - top.score) < 1e-9, "融合分不是各路名次的倒数和")
    _expect(RRF_K == 60, "RRF 的 k 不是 60")


def _hit_ids(result) -> set[str]:
    found = {hit.child_id for hit in result.passages}
    found.update(hit.child_id for hit in result.fused)
    for path in result.paths:
        found.update(hit.child_id for hit in path.hits)
    return found


def _index_rows(embedder, vectors, lexical, rows: list[dict]) -> list[str]:
    unique = list(dict.fromkeys(row["text"] for row in rows))
    encoded = dict(zip(unique, embedder.embed_documents(unique), strict=True))
    vector_rows = []
    text_rows = []
    for row in rows:
        vector_rows.append(
            ChildVector(
                child_id=row["child_id"],
                vector=encoded[row["text"]],
                parent_id=row["parent_id"],
                document_id=row["document_id"],
                version=row["version"],
                content_hash=row["content_hash"],
                clause_number=row["clause_number"],
                permission=row["permission"],
                effective_status=row["effective_status"],
                document_type=row["document_type"],
            )
        )
        text_rows.append(
            ChildText(
                child_id=row["child_id"],
                text=row["text"],
                parent_id=row["parent_id"],
                document_id=row["document_id"],
                version=row["version"],
                content_hash=row["content_hash"],
                clause_number=row["clause_number"],
                permission=row["permission"],
                effective_status=row["effective_status"],
                document_type=row["document_type"],
            )
        )
    vectors.upsert_children(vector_rows)
    lexical.upsert_children(text_rows)
    return [row["child_id"] for row in rows]


def _retrieve_row(
    text: str,
    *,
    document_type: str,
    clause_number: str = "第一条",
    permission: str = "public",
    effective_status: str = "effective",
    child_id: str | None = None,
) -> dict:
    child_id = child_id or ("c" + uuid.uuid4().hex[:16])
    return {
        "child_id": child_id,
        "text": text,
        "parent_id": "p" + child_id[1:],
        "document_id": "d" + child_id[1:9],
        "version": "v1",
        "content_hash": content_hash(text + child_id),
        "clause_number": clause_number,
        "permission": permission,
        "effective_status": effective_status,
        "document_type": document_type,
    }


def _clause(count_tokens, *, number: str, title_path: tuple[str, ...], seed: str) -> Clause:
    return Clause(number=number, title_path=title_path, statement=_fit(count_tokens, 640, seed))


def _policy(
    clauses: tuple[Clause, ...],
    sources: tuple[SourceRef, ...],
    *,
    document_id: str,
    version: str,
    effective_from: date,
) -> PolicyDocument:
    return PolicyDocument(
        document_id=document_id,
        version=version,
        document_type=document_id,
        permission="public",
        effective_from=effective_from,
        clauses=clauses,
        sources=sources,
    )


def _fit(count_tokens, size: int, seed: str) -> str:
    text = seed.strip()
    while count_tokens(text) < size:
        text += seed
    low = 0
    high = len(text)
    best = text
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue
        if count_tokens(text[:mid]) >= size:
            best = text[:mid]
            high = mid - 1
        else:
            low = mid + 1
    while best and count_tokens(best) > size:
        best = best[:-1]
    while count_tokens(best) < size:
        best += "条"
        if count_tokens(best) > size:
            break
    return best


def _write_parent(store, document_id: str, parent_id: str, text: str) -> None:
    store.write_parent(document_id, "v1", parent_id, text, content_hash(text))


def _parent_ref(parent_id: str, document_id: str, *, status: str = "effective") -> ParentRef:
    end = "2024-12-31" if status == "obsolete" else None
    return ParentRef(
        document_id=document_id,
        parent_id=parent_id,
        version="v1",
        effective_status=status,
        effective_from="2024-01-01",
        effective_to=end,
    )


def _ids(prefix: str) -> tuple[str, str, str]:
    token = uuid.uuid4().hex[:10]
    return f"{prefix}{token}", f"pexam{token}", f"pstay{token}"


def _token(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:8]


def _line(layer: str, cell: str, demo: str, **details: object) -> str:
    parts = [layer, cell, demo]
    if details:
        parts.append(" ".join(f"{key}={value}" for key, value in details.items()))
    return " | ".join(parts)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise DemoFailure(message)


def _expect_type(obj, expected: type) -> None:
    _expect(type(obj) is expected, f"用了 {type(obj).__name__}，不是 {expected.__name__}")
    _expect("memory" not in type(obj).__module__, f"{expected.__name__} 落到了内存模块")


def _host(url: str) -> str:
    host = url
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix) :]
    return host


def _metadata_host(dsn: str) -> str:
    tail = dsn.split("@", 1)[-1]
    return tail.split("?", 1)[0]


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split())
