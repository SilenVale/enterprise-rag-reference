import uuid

import pytest

from rag.retrieve.params import PATH_TOP_K, RRF_K
from rag.retrieve.pipeline import RetrievePipeline
from rag.retrieve.types import RetrieveRequest, Turn
from rag.storage.errors import StorageUnavailable
from rag.storage.types import ScalarFilter

from fixtures import Sleepy, TracingEmbedder, TracingLexical, TracingVectors, index_rows, make_row, new_type

pytestmark = pytest.mark.live

HISTORY = (
    Turn("user", "差旅住宿标准写在哪一条"),
    Turn("assistant", "写在差旅制度第三条"),
)


@pytest.fixture
def corpus(embedder, vectors, lexical):
    created: list[str] = []

    def add(rows: list[dict]) -> list[str]:
        ids = index_rows(embedder, vectors, lexical, rows)
        created.extend(ids)
        return ids

    yield add
    if created:
        vectors.delete_children(created)
        lexical.delete_children(created)


def _pipeline(embedder, vectors, lexical, reranker, refusals, **kwargs):
    return RetrievePipeline(
        embedder=kwargs.get("embedder", embedder),
        vectors=kwargs.get("vectors", vectors),
        lexical=kwargs.get("lexical", lexical),
        reranker=reranker,
        refusals=refusals,
        rewriter=kwargs.get("rewriter"),
    )


def _ids(result) -> set[str]:
    found = {hit.child_id for hit in result.passages}
    found.update(hit.child_id for hit in result.fused)
    for path in result.paths:
        found.update(hit.child_id for hit in path.hits)
    return found


def test_named_models_are_loaded(embedder, reranker):
    assert embedder.model_id == "BAAI/bge-m3"
    assert "bge-m3" in embedder.resolved_name.lower()
    assert embedder.dimension == 1024
    assert reranker.model_id == "BAAI/bge-reranker-v2-m3"
    assert reranker.ready
    assert "bge-reranker-v2-m3" in reranker.resolved_name.lower()


def test_followup_and_original_are_both_searched(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    target = make_row(
        "差旅住宿每晚上限六百元",
        document_type=document_type,
        clause_number="第三条",
    )
    other = make_row("食堂补贴按月发放", document_type=document_type, clause_number="第八条")
    corpus([target, other])
    tracer = TracingEmbedder(embedder)
    result = _pipeline(tracer, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest("那一条的上限是多少", history=HISTORY, document_type=document_type, request_id=new_type())
    )
    assert result.queries[0] == "那一条的上限是多少"
    assert "差旅住宿标准第三条的上限是多少" in result.queries
    assert result.queries[0] in tracer.queries
    assert "差旅住宿标准第三条的上限是多少" in tracer.queries
    assert target["child_id"] in _ids(result)
    assert result.rewrite_failed is False
    assert result.generated_text == ""


def test_compound_subqueries_are_retrieved_then_merged(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    stay = make_row("住宿每晚上限六百元", document_type=document_type, clause_number="第一条")
    days = make_row("报销时限为十五个工作日", document_type=document_type, clause_number="第二条")
    corpus([stay, days])
    result = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest("住宿上限和报销时限分别是什么", document_type=document_type, request_id=new_type())
    )
    assert result.queries[0] == "住宿上限和报销时限分别是什么"
    assert "住宿上限是什么" in result.queries
    assert "报销时限是什么" in result.queries
    assert 2 <= len(result.queries) - 1 <= 3
    assert stay["child_id"] in _ids(result)
    assert days["child_id"] in _ids(result)


def test_rewrite_failure_still_recalls_the_original(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    row = make_row("门禁卡丢失后应当当日挂失", document_type=document_type, clause_number="第五条")
    corpus([row])

    class Broken:
        model_id = "rule-rewrite"

        def rewrite(self, question, history):
            raise RuntimeError("改写模型没有返回")

    tracer = TracingEmbedder(embedder)
    question = "门禁卡丢失后应当当日挂失"
    result = _pipeline(tracer, vectors, lexical, reranker, refusals, rewriter=Broken()).retrieve(
        RetrieveRequest(question, history=HISTORY, document_type=document_type, request_id=new_type())
    )
    assert result.rewrite_failed is True
    assert result.queries == (question,)
    assert tracer.queries == [question]
    assert row["child_id"] in _ids(result)
    assert result.failure == ""


def test_topic_change_does_not_pull_history_into_the_query(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    canteen = make_row("食堂补贴按月发放两百元", document_type=document_type, clause_number="第九条")
    corpus([canteen])
    result = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest("食堂补贴怎么发放", history=HISTORY, document_type=document_type, request_id=new_type())
    )
    assert result.queries == ("食堂补贴怎么发放",)
    assert canteen["child_id"] in _ids(result)


def test_intern_cannot_see_restricted_when_it_would_fill_top_50(
    corpus, embedder, vectors, lexical, reranker, refusals
):
    document_type = new_type()
    token = "qx" + uuid.uuid4().hex[:8]
    restricted_text = f"{token} " * 8 + "受限报销标准每日八百元整"
    public_text = f"食堂每周供应早餐，另附门禁和菜单说明 {token}"
    rows = [
        make_row(
            restricted_text,
            document_type=document_type,
            permission="restricted",
            child_id=f"a{index:03d}{token[:4]}",
            clause_number=f"第{index + 1}条",
        )
        for index in range(55)
    ]
    public = make_row(
        public_text,
        document_type=document_type,
        permission="public",
        child_id=f"z{token[:6]}",
        clause_number="附则",
    )
    corpus([*rows, public])
    scope = ScalarFilter(effective_status="effective", document_type=document_type)
    query_vector = embedder.embed_query(restricted_text)
    unfiltered_vector = vectors.search(query_vector, limit=PATH_TOP_K, scalar=scope)
    unfiltered_lexical = lexical.search(token, limit=PATH_TOP_K, scalar=scope)
    unfiltered_meta = lexical.list_matching(limit=PATH_TOP_K, scalar=scope)
    assert public["child_id"] not in {hit.child_id for hit in unfiltered_vector}
    assert public["child_id"] not in {hit.child_id for hit in unfiltered_lexical}
    assert public["child_id"] not in {hit.child_id for hit in unfiltered_meta}
    assert len(unfiltered_vector) == PATH_TOP_K
    assert len(unfiltered_lexical) == PATH_TOP_K
    assert len(unfiltered_meta) == PATH_TOP_K

    traced_vectors = TracingVectors(vectors)
    traced_lexical = TracingLexical(lexical)
    result = _pipeline(embedder, traced_vectors, traced_lexical, reranker, refusals).retrieve(
        RetrieveRequest(restricted_text, role="intern", document_type=document_type, request_id=new_type())
    )
    assert public["child_id"] in _ids(result)
    restricted_ids = {row["child_id"] for row in rows}
    assert _ids(result).isdisjoint(restricted_ids)
    staff = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest(restricted_text, role="staff", document_type=document_type, request_id=new_type())
    )
    for path in staff.paths:
        assert len(path.hits) == PATH_TOP_K
    assert traced_vectors.calls
    assert traced_lexical.calls
    for scalar, limit in traced_vectors.calls:
        assert scalar.permission == "public"
        assert scalar.effective_status == "effective"
        assert scalar.document_type == document_type
        assert limit == PATH_TOP_K
    for kind, scalar, limit in traced_lexical.calls:
        assert kind in {"bm25", "clause", "metadata"}
        assert scalar.permission == "public"
        assert scalar.effective_status == "effective"
        assert limit == PATH_TOP_K


def test_obsolete_and_pending_stay_out_unless_audit_asks(
    corpus, embedder, vectors, lexical, reranker, refusals
):
    document_type = new_type()
    token = "ob" + uuid.uuid4().hex[:8]
    close = f"{token} 已废止的住宿上限为九百元"
    current_text = f"食堂菜单 {token}"
    obsolete_rows = [
        make_row(
            close,
            document_type=document_type,
            effective_status="obsolete",
            child_id=f"a{index:03d}{token[:4]}",
        )
        for index in range(55)
    ]
    pending = make_row(
        close,
        document_type=document_type,
        effective_status="pending",
        child_id=f"p{token[:6]}",
    )
    current = make_row(
        current_text,
        document_type=document_type,
        effective_status="effective",
        child_id=f"z{token[:6]}",
    )
    corpus([*obsolete_rows, pending, current])
    scope = ScalarFilter(document_type=document_type)
    crowded = vectors.search(embedder.embed_query(close), limit=PATH_TOP_K, scalar=scope)
    assert current["child_id"] not in {hit.child_id for hit in crowded}

    hidden = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest(close, document_type=document_type, request_id=new_type())
    )
    hidden_ids = _ids(hidden)
    assert current["child_id"] in hidden_ids
    assert pending["child_id"] not in hidden_ids
    assert hidden_ids.isdisjoint({row["child_id"] for row in obsolete_rows})

    audit = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest(close, document_type=document_type, include_obsolete=True, request_id=new_type())
    )
    audit_ids = _ids(audit)
    assert current["child_id"] in audit_ids
    assert any(row["child_id"] in audit_ids for row in obsolete_rows)
    assert pending["child_id"] not in audit_ids


def test_document_type_is_applied_inside_the_engine(corpus, embedder, vectors, lexical, reranker, refusals):
    wanted_type = new_type()
    other_type = new_type()
    token = "dt" + uuid.uuid4().hex[:8]
    close = f"{token} " * 6 + "另一类制度的报销标准"
    far = f"门禁说明 {token}"
    others = [
        make_row(close, document_type=other_type, child_id=f"a{index:03d}{token[:4]}")
        for index in range(55)
    ]
    wanted = make_row(far, document_type=wanted_type, child_id=f"z{token[:6]}")
    corpus([*others, wanted])
    crowded = lexical.search(token, limit=PATH_TOP_K, scalar=ScalarFilter(effective_status="effective"))
    # 不限文档类型时，更贴词面的那一类会占满 top 50。
    assert wanted["child_id"] not in {hit.child_id for hit in crowded}
    result = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest(close, document_type=wanted_type, request_id=new_type())
    )
    assert wanted["child_id"] in _ids(result)
    assert _ids(result).isdisjoint({row["child_id"] for row in others})


def test_clause_number_matches_the_whole_token(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    exact = make_row("座席报销规则", document_type=document_type, clause_number="DZ-12.3")
    shorter = make_row("座席报销规则", document_type=document_type, clause_number="DZ-12")
    article = make_row("住宿上限六百元", document_type=document_type, clause_number="第三条")
    longer_article = make_row("住宿上限六百元", document_type=document_type, clause_number="第十三条")
    corpus([exact, shorter, article, longer_article])
    code = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest("DZ-12", document_type=document_type, request_id=new_type())
    )
    bm25 = next(path for path in code.paths if path.name == "bm25")
    assert [hit.child_id for hit in bm25.hits] == [shorter["child_id"]]
    article_result = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest("第三条", document_type=document_type, request_id=new_type())
    )
    article_bm25 = next(path for path in article_result.paths if path.name == "bm25")
    assert article["child_id"] in {hit.child_id for hit in article_bm25.hits}
    assert longer_article["child_id"] not in {hit.child_id for hit in article_bm25.hits}


def test_metadata_path_does_not_follow_wording(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    first = make_row("门禁卡丢失应当挂失", document_type=document_type, child_id="a" + uuid.uuid4().hex[:8])
    second = make_row("报销时限十五个工作日 " * 4, document_type=document_type, child_id="b" + uuid.uuid4().hex[:8])
    corpus([first, second])
    pipe = _pipeline(embedder, vectors, lexical, reranker, refusals)
    about_door = pipe.retrieve(RetrieveRequest("门禁卡丢失", document_type=document_type, request_id=new_type()))
    about_days = pipe.retrieve(RetrieveRequest("报销时限", document_type=document_type, request_id=new_type()))
    meta_door = next(path for path in about_door.paths if path.name == "metadata")
    meta_days = next(path for path in about_days.paths if path.name == "metadata")
    bm25_days = next(path for path in about_days.paths if path.name == "bm25")
    assert [hit.child_id for hit in meta_door.hits] == [hit.child_id for hit in meta_days.hits]
    assert [hit.child_id for hit in meta_days.hits] == [first["child_id"], second["child_id"]]
    assert bm25_days.hits[0].child_id == second["child_id"]
    assert [hit.child_id for hit in bm25_days.hits] != [hit.child_id for hit in meta_days.hits]


def test_rrf_on_live_hits_uses_rank_not_engine_scores(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    row = make_row("差旅住宿每晚上限六百元", document_type=document_type, clause_number="第三条")
    corpus([row])
    result = _pipeline(embedder, vectors, lexical, reranker, refusals).retrieve(
        RetrieveRequest("住宿上限", document_type=document_type, request_id=new_type())
    )
    assert result.failure == ""
    expected: dict[str, float] = {}
    for path in result.paths:
        assert path.name in {"vector", "bm25", "metadata"}
        assert len(path.hits) <= PATH_TOP_K
        for index, hit in enumerate(path.hits, start=1):
            assert hit.rank == index
            expected[hit.child_id] = expected.get(hit.child_id, 0.0) + 1 / (RRF_K + hit.rank)
    assert result.fused
    for item in result.fused:
        assert item.score == pytest.approx(expected[item.child_id])
        raw = []
        for path in result.paths:
            for hit in path.hits:
                if hit.child_id == item.child_id:
                    raw.append(hit.source_score)
        if len(raw) >= 2:
            assert item.score != pytest.approx(sum(raw))
    assert result.latency_ms == pytest.approx(max(path.elapsed_ms for path in result.paths), abs=1)


def test_latency_is_the_slowest_path(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    row = make_row("差旅住宿每晚上限六百元", document_type=document_type)
    corpus([row])
    sleepy_lexical = Sleepy(Sleepy(lexical, 0.35, "search"), 0.35, "list_matching")
    result = _pipeline(embedder, vectors, sleepy_lexical, reranker, refusals).retrieve(
        RetrieveRequest("住宿上限", document_type=document_type, request_id=new_type())
    )
    elapsed = [path.elapsed_ms for path in result.paths]
    slow = max(elapsed)
    assert result.latency_ms == pytest.approx(slow, abs=1)
    assert result.recall_wall_ms < slow + (sum(elapsed) - slow) * 0.5


def test_engine_outage_discards_the_other_paths(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    row = make_row("差旅住宿每晚上限六百元", document_type=document_type, clause_number="第三条")
    corpus([row])

    class Down:
        def search(self, *args, **kwargs):
            raise StorageUnavailable("milvus")

    request_id = new_type()
    result = _pipeline(embedder, Down(), lexical, reranker, refusals).retrieve(
        RetrieveRequest("住宿上限", document_type=document_type, request_id=request_id)
    )
    assert result.disposition == "retrieval_failed"
    assert result.passages == ()
    assert result.paths == ()
    assert result.fused == ()
    assert row["child_id"] not in _ids(result)
    assert refusals.list_for(request_id) == [("检索失败", result.failure)]
    still_there = lexical.search("住宿", limit=5, scalar=ScalarFilter(document_type=document_type, effective_status="effective"))
    assert row["child_id"] in {hit.child_id for hit in still_there}
