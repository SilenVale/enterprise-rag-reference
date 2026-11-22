"""哈希去重、按块增量、废止区间和版本切换。"""

import uuid
from datetime import date

import pytest

from rag.index.pipeline import IndexPipeline
from rag.index.types import SourceRef
from rag.ingest.types import AcceptedDocument, AttemptRecord
from rag.index.boundary import source_from_accepted
from rag.storage.errors import VersionSwitchError
from rag.storage.object_store import MinioBlobStore, VersionedParentStore, split_endpoint
from rag.storage.types import ScalarFilter

from factory import make_clause, make_policy


def _attempt() -> AttemptRecord:
    return AttemptRecord("default", (), (), 1.0, True, "")


def _accepted(source_id: str, document_ref: str, form: str) -> AcceptedDocument:
    return AcceptedDocument(
        source_id=source_id,
        document_ref=document_ref,
        form=form,
        normalized_text="同一份正文",
        tables=(),
        reading_order=(),
        attempts=(_attempt(),),
    )


def _two_clause_policy(count, document_id: str, version: str, *, second_seed: str, effective_from: date):
    first = make_clause(
        count,
        640,
        number="第一条",
        title_path=("差旅制度", "第二章 报销", "第一条 范围"),
        seed=f"{document_id}第一条这一版保持不变，用来复用向量。",
    )
    second = make_clause(
        count,
        640,
        number="第二条",
        title_path=("差旅制度", "第三章 住宿", "第二条 标准"),
        seed=f"{document_id}{second_seed}",
    )
    return make_policy(
        (first, second),
        (SourceRef("制度库", f"{document_id}-{version}", "archive"),),
        document_id=document_id,
        version=version,
        effective_from=effective_from,
    )


def test_same_sha256_normalized_text_is_not_stored_or_embedded_twice(pipeline, embedder):
    count = embedder.count_tokens
    clauses = (
        make_clause(
            count,
            640,
            number="第一条",
            title_path=("差旅制度", "第二章 报销", "第一条 范围"),
            seed="同一份规范化文本只留一份。",
        ),
    )
    mail = source_from_accepted(_accepted("邮件", "fwd-18", "forward"))
    archive = source_from_accepted(_accepted("共享目录", "box-18", "archive"))
    first = make_policy(clauses, (mail,))
    second = make_policy(clauses, (archive,), document_id="d-other", version="v9")
    before = len(embedder.document_texts)

    first_receipt = pipeline.index_document(first)
    embedded = len(embedder.document_texts)
    second_receipt = pipeline.index_document(second)

    assert not first_receipt.reused
    assert second_receipt.reused
    assert second_receipt.document_id == first.document_id
    assert len(embedder.document_texts) == embedded
    assert embedded > before
    sources = {(item.source_id, item.document_ref) for item in pipeline.catalog.list_sources(first_receipt.normalized_hash)}
    assert sources == {("邮件", "fwd-18"), ("共享目录", "box-18")}
    assert pipeline.catalog.children(second.document_id, second.version) == []
    assert len(pipeline.catalog.children(first.document_id, first.version)) == embedded - before


def test_unchanged_chunks_reuse_vectors_without_embedding_again(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "reuse" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条旧标准按城市执行。",
        effective_from=date(2020, 1, 1),
    )
    revised = _two_clause_policy(
        count,
        document_id,
        "v2",
        second_seed="第二条新标准改成按出差天数执行。",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    before = len(embedder.document_texts)
    receipt = pipeline.apply_change(revised)
    added = embedder.document_texts[before:]
    children = pipeline.catalog.children(document_id, "v2")
    reused = [child for child in children if child.vector_reused]
    fresh = [child for child in children if not child.vector_reused]

    assert receipt.reused_hashes
    assert receipt.embedded_hashes
    assert reused and fresh
    assert len(added) == len(fresh)
    assert not any(child.embed_text in added for child in reused)
    old = {child.content_hash: child.vector for child in pipeline.catalog.children(document_id, "v1")}
    for child in reused:
        assert child.vector == old[child.content_hash]


def test_changed_chunks_are_embedded_again(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "change" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条旧的住宿金额写在这里。",
        effective_from=date(2020, 1, 1),
    )
    revised = _two_clause_policy(
        count,
        document_id,
        "v2",
        second_seed="第二条改写后的住宿金额写在这里。",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    before = len(embedder.document_texts)
    receipt = pipeline.apply_change(revised)
    added = embedder.document_texts[before:]
    changed = [child for child in pipeline.catalog.children(document_id, "v2") if not child.vector_reused]

    assert added
    assert {child.embed_text for child in changed} == set(added)
    assert len(added) < len(pipeline.catalog.children(document_id, "v2"))
    assert "change" == pipeline.catalog.events(document_id)[-1].kind
    assert receipt.event_kind == "change"


def test_disappeared_chunks_are_marked_deleted_and_not_removed_before_switch(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "gone" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条会在下一版消失。",
        effective_from=date(2020, 1, 1),
    )
    kept = make_policy(
        (
            make_clause(
                count,
                640,
                number="第一条",
                title_path=("差旅制度", "第二章 报销", "第一条 范围"),
                seed=f"{document_id}第一条这一版保持不变，用来复用向量。",
            ),
        ),
        (SourceRef("制度库", f"{document_id}-v2", "archive"),),
        document_id=document_id,
        version="v2",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    staged = pipeline.stage(kept)

    assert staged.disappeared_ids
    for child_id in staged.disappeared_ids:
        assert pipeline.catalog.child(child_id).status == "effective"
    assert pipeline.vectors.delete_calls == []

    pipeline.activate(document_id, "v2")

    for child_id in staged.disappeared_ids:
        child = pipeline.catalog.child(child_id)
        assert child.status == "deleted"
        hits = pipeline.vectors.search(
            child.vector,
            limit=3,
            scalar=ScalarFilter(effective_status="deleted", document_type=current.document_type),
        )
        assert any(hit.child_id == child_id for hit in hits)
    assert pipeline.vectors.delete_calls == []


def test_old_version_is_obsolete_and_keeps_an_effective_date_range(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "dates" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条旧区间仍然要留着。",
        effective_from=date(2020, 1, 1),
    )
    revised = _two_clause_policy(
        count,
        document_id,
        "v2",
        second_seed="第二条新区间从切换日开始。",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    pipeline.apply_change(revised)
    old = pipeline.catalog.get_document(document_id, "v1")
    new = pipeline.catalog.get_document(document_id, "v2")

    assert old.status == "obsolete"
    assert old.effective_from == date(2020, 1, 1)
    assert old.effective_to == date(2023, 12, 31)
    assert new.status == "effective"
    assert new.effective_from == date(2024, 1, 1)
    assert new.effective_to is None


def test_default_lookup_drops_obsolete_and_not_yet_effective(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "filter" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条旧文默认不该被检索到。",
        effective_from=date(2020, 1, 1),
    )
    revised = _two_clause_policy(
        count,
        document_id,
        "v2",
        second_seed="第二条新文才是默认能检索到的。",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    pipeline.apply_change(revised)
    unchanged = next(
        child
        for child in pipeline.catalog.children(document_id, "v1")
        if child.clause_number == "第一条"
    )
    default_hits = pipeline.lookup(unchanged.embed_text, document_type=current.document_type, limit=5)
    future = make_policy(
        (
            make_clause(
                count,
                640,
                number="第十条",
                title_path=("差旅制度", "第六章 未生效", "第十条 预告"),
                seed="这条还没到生效日。",
            ),
        ),
        (SourceRef("公文", "future-10", "forward"),),
        effective_from=date(2099, 1, 1),
    )
    pending = pipeline.index_document(future)
    future_child = pipeline.catalog.children(future.document_id, future.version)[0]
    hidden = pipeline.lookup(future_child.embed_text, document_type=future.document_type, limit=3)

    assert pending.activated is False
    assert pipeline.catalog.get_document(future.document_id, future.version).status == "pending"
    assert default_hits
    assert {hit.version for hit in default_hits} == {"v2"}
    assert hidden == []
    pending_hits = pipeline.vectors.search(
        future_child.vector,
        limit=1,
        scalar=ScalarFilter(effective_status="pending", document_type=future.document_type),
    )
    assert pending_hits[0].child_id == future_child.child_id


def test_audit_lookup_can_include_obsolete_versions(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "audit" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条废止后审计还要能点名。",
        effective_from=date(2020, 1, 1),
    )
    revised = _two_clause_policy(
        count,
        document_id,
        "v2",
        second_seed="第二条现行文本。",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    pipeline.apply_change(revised)
    unchanged = next(child for child in pipeline.catalog.children(document_id, "v1") if child.clause_number == "第一条")
    default_hits = pipeline.lookup(unchanged.embed_text, document_type=current.document_type, limit=5)
    audit_hits = pipeline.lookup(
        unchanged.embed_text,
        document_type=current.document_type,
        limit=5,
        include_obsolete=True,
    )

    assert all(hit.version != "v1" for hit in default_hits)
    assert any(hit.version == "v1" and hit.child_id == unchanged.child_id for hit in audit_hits)


def test_queries_keep_reading_the_old_version_until_the_new_one_is_switched(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "switch" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条切换前仍然对外。",
        effective_from=date(2020, 1, 1),
    )
    revised = _two_clause_policy(
        count,
        document_id,
        "v2",
        second_seed="第二条切换后才对外。",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    old_parent = next(
        parent for parent in pipeline.catalog.parents(document_id, "v1") if parent.clause_numbers == ("第二条",)
    )
    old_text = pipeline.load_parent(document_id, old_parent.parent_id).text
    staged = pipeline.stage(revised)
    staged_parent = next(
        parent_id
        for parent_id in staged.parent_ids
        if pipeline.catalog.parent(parent_id).clause_numbers == ("第二条",)
    )

    assert pipeline.parents.active_version(document_id) == "v1"
    assert pipeline.load_parent(document_id, old_parent.parent_id).text == old_text
    assert all(pipeline.catalog.child(child_id).status == "pending" for child_id in staged.child_ids)
    preread = pipeline.parents.read_parent_version(document_id, staged_parent, "v2")
    assert preread.version == "v2"
    assert preread.text != old_text

    pipeline.activate(document_id, "v2")

    assert pipeline.parents.active_version(document_id) == "v2"
    assert pipeline.load_parent(document_id, staged_parent).version == "v2"
    assert pipeline.parents.read_parent_version(document_id, old_parent.parent_id, "v1").text == old_text


def test_failed_switch_points_queries_back_to_the_old_version(pipeline, embedder, settings, catalog, guarded_vectors):
    count = embedder.count_tokens
    document_id = "back" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条失败时要回到这一版。",
        effective_from=date(2020, 1, 1),
    )
    revised = _two_clause_policy(
        count,
        document_id,
        "v2",
        second_seed="第二条没有切成功。",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    host, secure = split_endpoint(settings.object_endpoint)
    blobs = MinioBlobStore(
        host,
        access_key=settings.object_access_key,
        secret_key=settings.object_secret_key,
        bucket=settings.object_bucket,
        secure=secure,
    )
    failing = IndexPipeline(
        embedder=embedder,
        vectors=guarded_vectors,
        parents=VersionedParentStore(_PointerFailingBlobs(blobs)),
        catalog=catalog,
        as_of=pipeline.as_of,
    )
    failing.stage(revised)

    with pytest.raises(VersionSwitchError):
        failing.activate(document_id, "v2")

    assert failing.last_rollback == (document_id, "v1")
    assert pipeline.parents.active_version(document_id) == "v1"
    assert catalog.query_version(document_id) == "v1"
    assert pipeline.catalog.get_document(document_id, "v1").status == "effective"
    assert pipeline.catalog.get_document(document_id, "v2").status == "pending"


def test_version_update_does_not_delete_then_insert(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "keep" + uuid.uuid4().hex[:6]
    current = _two_clause_policy(
        count,
        document_id,
        "v1",
        second_seed="第二条先留着，不能先删。",
        effective_from=date(2020, 1, 1),
    )
    revised = _two_clause_policy(
        count,
        document_id,
        "v2",
        second_seed="第二条新写完再切换。",
        effective_from=date(2024, 1, 1),
    )
    first = pipeline.index_document(current)
    staged = pipeline.stage(revised)
    old_ids = set(first.child_ids)
    new_ids = set(staged.child_ids)

    assert old_ids.isdisjoint(new_ids)
    assert old_ids
    for child_id in old_ids:
        assert pipeline.catalog.child(child_id).status == "effective"
    for child_id in new_ids:
        assert pipeline.catalog.child(child_id).status == "pending"
    visible = pipeline.vectors.search(
        pipeline.catalog.child(next(iter(old_ids))).vector,
        limit=5,
        scalar=ScalarFilter(effective_status="effective", document_type=current.document_type),
    )
    waiting = pipeline.vectors.search(
        pipeline.catalog.child(next(iter(new_ids))).vector,
        limit=5,
        scalar=ScalarFilter(effective_status="pending", document_type=current.document_type),
    )
    assert any(hit.child_id in old_ids for hit in visible)
    assert any(hit.child_id in new_ids for hit in waiting)
    assert pipeline.vectors.delete_calls == []


class _PointerFailingBlobs:
    def __init__(self, inner):
        self._inner = inner

    def put(self, key, body, metadata=None):
        if str(key).startswith("pointers/"):
            raise OSError("指针写入失败")
        return self._inner.put(key, body, metadata)

    def __getattr__(self, name):
        return getattr(self._inner, name)
