"""父块、子块、标题路径、表格和超长降级。"""

from datetime import date

import pytest

from rag.index.chunking import build_chunks
from rag.index.errors import TableHeaderMissing
from rag.index.types import IndexTable, SourceRef
from rag.storage.hashing import content_hash

from factory import make_clause, make_policy, text_of


def _source(source_id: str = "制度库") -> tuple[SourceRef, ...]:
    return (SourceRef(source_id, "ref-1", "archive"),)


def test_parent_chunks_are_clauses_between_512_and_1024_tokens(pipeline, embedder):
    count = embedder.count_tokens
    reimbursement = ("差旅制度", "第二章 报销")
    clauses = (
        make_clause(
            count,
            300,
            number="第一条",
            title_path=reimbursement + ("第一条 范围",),
            seed="第一条范围包括出差产生的住宿和交通。",
        ),
        make_clause(
            count,
            300,
            number="第二条",
            title_path=reimbursement + ("第二条 凭证",),
            seed="第二条凭证包括发票审批单和行程单。",
        ),
        make_clause(
            count,
            640,
            number="第三条",
            title_path=("差旅制度", "第三章 住宿", "第三条 标准"),
            seed="第三条住宿标准按城市级别执行。",
        ),
    )
    policy = make_policy(clauses, _source(), effective_from=date(2020, 1, 1))
    plan = build_chunks(policy, count)
    pipeline.index_document(policy)
    stored = pipeline.catalog.parents(policy.document_id, policy.version)

    assert len(stored) == len(plan.parents) == 2
    for parent in stored:
        assert 512 <= parent.token_count <= 1024
        original = pipeline.parents.read_parent_version(policy.document_id, parent.parent_id, policy.version)
        assert count(original.text) == parent.token_count
    merged = next(parent for parent in stored if parent.clause_numbers == ("第一条", "第二条"))
    alone = next(parent for parent in stored if parent.clause_numbers == ("第三条",))
    assert "第一条" in merged.title_path or "第一条" in pipeline.parents.read_parent_version(
        policy.document_id, merged.parent_id, policy.version
    ).text
    assert alone.injection_mode == "full_parent"
    assert alone.clause_numbers == ("第三条",)


def test_child_chunks_are_between_128_and_256_tokens(pipeline, embedder):
    count = embedder.count_tokens
    policy = make_policy(
        (
            make_clause(
                count,
                700,
                number="第一条",
                title_path=("差旅制度", "第二章 报销", "第一条 范围"),
                seed="子块长度要落在规定区间里。",
            ),
        ),
        _source(),
    )
    pipeline.index_document(policy)
    children = pipeline.catalog.children(policy.document_id, policy.version)

    assert children
    assert all(child.kind == "prose" for child in children)
    assert all(128 <= child.token_count <= 256 for child in children)
    assert all(128 <= count(child.body) <= 256 for child in children)


def test_only_children_are_embedded_and_parents_stay_out_of_the_vector_store(pipeline, embedder):
    count = embedder.count_tokens
    before = len(embedder.document_texts)
    policy = make_policy(
        (
            make_clause(
                count,
                680,
                number="第一条",
                title_path=("差旅制度", "第二章 报销", "第一条 范围"),
                seed="父块全文不能拿去嵌入。",
            ),
        ),
        _source(),
    )
    pipeline.index_document(policy)
    added = embedder.document_texts[before:]
    parents = pipeline.catalog.parents(policy.document_id, policy.version)
    parent_texts = [
        pipeline.parents.read_parent_version(policy.document_id, parent.parent_id, policy.version).text
        for parent in parents
    ]
    children = pipeline.catalog.children(policy.document_id, policy.version)

    assert parent_texts
    assert added
    assert len(added) == len(children)
    for text in added:
        assert text not in parent_texts
        assert all(parent not in text for parent in parent_texts)
    names = set(pipeline.vectors.field_names())
    assert names.isdisjoint({"parent_text", "parent_body", "full_text", "text", "body", "content"})
    assert "vector" in names


def test_child_metadata_carries_parent_id_document_version_and_content_hash(pipeline, embedder):
    count = embedder.count_tokens
    policy = make_policy(
        (
            make_clause(
                count,
                640,
                number="第四条",
                title_path=("差旅制度", "第二章 报销", "第四条 范围"),
                seed="子块元数据要能指回父块。",
            ),
        ),
        _source(),
    )
    pipeline.index_document(policy)
    child = pipeline.catalog.children(policy.document_id, policy.version)[0]
    hits = pipeline.lookup(child.embed_text, document_type=policy.document_type, limit=1)

    assert hits[0].child_id == child.child_id
    assert hits[0].parent_id == child.parent_id
    assert hits[0].document_id == policy.document_id
    assert hits[0].version == policy.version
    assert hits[0].content_hash == child.content_hash == content_hash(child.embed_text)
    assert child.parent_id
    assert child.clause_number == "第四条"


def test_parent_original_is_fetched_from_object_storage_by_id_after_child_hit(pipeline, embedder, settings):
    from rag.storage.object_store import open_parent_store

    count = embedder.count_tokens
    policy = make_policy(
        (
            make_clause(
                count,
                660,
                number="第五条",
                title_path=("差旅制度", "第二章 报销", "第五条 取回"),
                seed="命中子块后按父块编号取回原文。",
            ),
        ),
        _source(),
    )
    pipeline.index_document(policy)
    child = pipeline.catalog.children(policy.document_id, policy.version)[0]
    hit = pipeline.lookup(child.embed_text, document_type=policy.document_type, limit=1)[0]
    reread = open_parent_store(settings).read_parent(hit.document_id, hit.parent_id)

    assert reread.parent_id == hit.parent_id
    assert reread.document_id == policy.document_id
    assert reread.version == policy.version
    assert child.clause_number in reread.text
    assert len(reread.text) > len(child.body)
    assert reread.content_hash == content_hash(reread.text)


def test_embedding_text_is_prefixed_with_title_path(pipeline, embedder):
    count = embedder.count_tokens
    path = ("差旅制度", "第二章 报销", "第一条 住宿")
    before = len(embedder.document_texts)
    policy = make_policy(
        (make_clause(count, 640, number="第一条", title_path=path, seed="嵌入前要拼上标题路径。"),),
        _source(),
    )
    pipeline.index_document(policy)
    added = embedder.document_texts[before:]
    prefix = "差旅制度 / 第二章 报销 / 第一条 住宿\n"

    assert added
    assert all(text.startswith(prefix) for text in added)
    children = pipeline.catalog.children(policy.document_id, policy.version)
    assert all(child.embed_text.startswith(prefix) for child in children)
    assert all(child.title_path == "差旅制度 / 第二章 报销 / 第一条 住宿" for child in children)


def test_table_is_split_by_row_and_each_row_keeps_the_header(pipeline, embedder):
    count = embedder.count_tokens
    header = ("项目", "标准", "说明")
    rows = tuple(
        (
            text_of(count, 150, f"第{index}行住宿报销需要发票。"),
            "按城市级别",
            "附审批单",
        )
        for index in (1, 2)
    )
    table = IndexTable(header, rows)
    policy = make_policy(
        (
            make_clause(
                count,
                520,
                number="第六条",
                title_path=("差旅制度", "第四章 表格", "第六条 标准"),
                seed="表格不能整段切掉表头。",
                table=table,
            ),
        ),
        _source(),
    )
    pipeline.index_document(policy)
    children = pipeline.catalog.children(policy.document_id, policy.version)
    table_children = [child for child in children if child.kind == "table_row"]

    assert table_children
    for child in table_children:
        assert 128 <= child.token_count <= 256
        for label in header:
            assert label in child.body
    assert all(child.kind != "table_row" or "表头：" in child.body for child in children)


def test_table_without_header_is_not_indexed_as_plain_paragraphs(pipeline, embedder):
    count = embedder.count_tokens
    marker = "破表单元格不应入库"
    before = len(embedder.document_texts)
    policy = make_policy(
        (
            make_clause(
                count,
                640,
                number="第七条",
                title_path=("差旅制度", "第四章 表格", "第七条 破表"),
                seed="没有表头就退回表格闸门。",
                table=IndexTable((), ((marker, "没有表头"),)),
            ),
        ),
        _source(),
    )

    with pytest.raises(TableHeaderMissing):
        pipeline.index_document(policy)

    assert marker not in "".join(embedder.document_texts[before:])
    assert pipeline.catalog.children(policy.document_id, policy.version) == []
    assert pipeline.parents.active_version(policy.document_id) is None


def test_overlong_parent_degrades_to_clause_plus_item_explanation(pipeline, embedder):
    count = embedder.count_tokens
    marker = "附加说明不应留在降级父块"
    policy = make_policy(
        (
            make_clause(
                count,
                420,
                number="第八条",
                title_path=("差旅制度", "第五章 超长", "第八条 降级"),
                seed="条款正文仍然要留下来。",
                items=(("（一）", 220, "条项说明写清需要哪些凭证。"),),
                extra_tokens=520,
                extra_seed=marker + "。",
            ),
        ),
        _source(),
    )
    pipeline.index_document(policy)
    parent = pipeline.catalog.parents(policy.document_id, policy.version)[0]
    original = pipeline.parents.read_parent_version(policy.document_id, parent.parent_id, policy.version)

    assert parent.injection_mode == "clause_plus_items"
    assert 512 <= parent.token_count <= 1024
    assert marker not in original.text
    assert "第八条" in original.text
    assert "条项说明" in original.text or "（一）" in original.text
    injection = pipeline.injection_for(pipeline.catalog.children(policy.document_id, policy.version)[0].child_id)
    assert injection.mode == "clause_plus_items"
    assert injection.text == original.text
    assert marker not in injection.text


def test_still_overlong_parent_injects_summary_and_hit_child_full_text(pipeline, embedder):
    count = embedder.count_tokens
    marker = "句末标记只在父块末尾"
    policy = make_policy(
        (
            make_clause(
                count,
                620,
                number="第九条",
                title_path=("差旅制度", "第五章 超长", "第九条 摘要"),
                seed="条款正文长到降级以后仍然放不下。",
                items=(("（一）", 620, marker + "之前还有大段条项说明。"),),
            ),
        ),
        _source(),
    )
    pipeline.index_document(policy)
    parent = pipeline.catalog.parents(policy.document_id, policy.version)[0]
    original = pipeline.parents.read_parent_version(policy.document_id, parent.parent_id, policy.version)
    children = pipeline.catalog.children(policy.document_id, policy.version)
    missed = next(child for child in children if marker not in child.body)
    injection = pipeline.injection_for(missed.child_id)

    assert parent.injection_mode == "summary_plus_child"
    assert parent.token_count > 1024
    assert marker in original.text
    assert parent.summary
    assert count(parent.summary) <= 64
    assert injection.mode == "summary_plus_child"
    assert injection.text.startswith(parent.summary)
    assert missed.body in injection.text
    assert marker not in injection.text
    assert len(injection.text) < len(original.text)
    assert original.text not in "".join(embedder.document_texts)
