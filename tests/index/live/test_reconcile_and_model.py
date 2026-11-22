"""BGE-M3 真写入，以及定时哈希对账走增量。"""

import uuid
from datetime import date

import psycopg

from rag.index.embedder import MODEL_ID
from rag.index.pipeline import COLLECTION
from rag.index.types import SourceRef
from rag.storage.milvus_store import MilvusChildVectorStore
from rag.storage.types import ChildVector

from factory import make_clause, make_policy


def test_bge_m3_loads_and_writes_vectors_to_milvus(pipeline, embedder, vector_store, catalog, settings):
    policy = make_policy(
        (
            make_clause(
                embedder.count_tokens,
                640,
                number="第一条",
                title_path=("差旅制度", "第二章 报销", "第一条 模型"),
                seed="这里要写出真实的子块向量。",
            ),
        ),
        (SourceRef("制度库", "model-1", "archive"),),
        effective_from=date(2020, 1, 1),
    )
    receipt = pipeline.index_document(policy)
    child = pipeline.catalog.children(policy.document_id, policy.version)[0]
    hits = pipeline.lookup(child.embed_text, document_type=policy.document_type, limit=1)
    described = vector_store.describe_vector_index()

    assert embedder.model_id == MODEL_ID == "BAAI/bge-m3"
    assert "bge-m3" in embedder.resolved_name.lower()
    assert embedder.dimension == 1024
    assert len(child.vector) == 1024
    assert receipt.embedded_hashes
    assert hits[0].child_id == child.child_id
    assert described["index_type"] == "HNSW"
    assert described["metric_type"] == "COSINE"
    assert vector_store.server_version()
    assert catalog.database_name() == "rag"
    assert settings.milvus_uri.startswith("http://127.0.0.1")
    reread = MilvusChildVectorStore(settings.milvus_uri, collection=COLLECTION)
    assert reread.index_type() == "HNSW"


def test_change_event_updates_incrementally(pipeline, embedder):
    count = embedder.count_tokens
    document_id = "event" + uuid.uuid4().hex[:6]
    current = make_policy(
        (
            make_clause(
                count,
                640,
                number="第一条",
                title_path=("差旅制度", "第二章 报销", "第一条 范围"),
                seed="变更事件里这一条不动。",
            ),
            make_clause(
                count,
                640,
                number="第二条",
                title_path=("差旅制度", "第三章 住宿", "第二条 标准"),
                seed="变更事件会改写第二条。",
            ),
        ),
        (SourceRef("公文", "event-v1", "forward"),),
        document_id=document_id,
        effective_from=date(2020, 1, 1),
    )
    revised = make_policy(
        (
            current.clauses[0],
            make_clause(
                count,
                640,
                number="第二条",
                title_path=("差旅制度", "第三章 住宿", "第二条 标准"),
                seed="变更事件已经改写第二条。",
            ),
        ),
        (SourceRef("公文", "event-v2", "forward"),),
        document_id=document_id,
        version="v2",
        effective_from=date(2024, 1, 1),
    )
    pipeline.index_document(current)
    before = len(embedder.document_texts)
    receipt = pipeline.apply_change(revised)

    assert receipt.event_kind == "change"
    assert receipt.reused_hashes
    assert receipt.embedded_hashes
    assert len(embedder.document_texts) - before == len(receipt.embedded_hashes)
    assert pipeline.catalog.events(document_id)[-1].kind == "change"


def test_scheduled_hash_reconcile_repairs_drift_incrementally(pipeline, embedder, settings):
    count = embedder.count_tokens
    policy = make_policy(
        (
            make_clause(
                count,
                640,
                number="第一条",
                title_path=("差旅制度", "第二章 报销", "第一条 对账"),
                seed="对账时这一条不该重嵌。",
            ),
            make_clause(
                count,
                640,
                number="第二条",
                title_path=("差旅制度", "第三章 住宿", "第二条 对账"),
                seed="对账时只修这一条的哈希漂移。",
            ),
        ),
        (SourceRef("共享目录", "reconcile-1", "archive"),),
        effective_from=date(2020, 1, 1),
    )
    pipeline.index_document(policy)
    quiet = len(embedder.document_texts)
    first = pipeline.reconcile(policy.document_id)
    assert first.embedded_hashes == ()
    assert len(embedder.document_texts) == quiet

    drifted = next(child for child in pipeline.catalog.children(policy.document_id, policy.version) if child.clause_number == "第二条")
    untouched = [child for child in pipeline.catalog.children(policy.document_id, policy.version) if child.child_id != drifted.child_id]
    bogus = tuple([1.0] + [0.0] * (embedder.dimension - 1))
    with psycopg.connect(settings.metadata_dsn) as conn:
        conn.execute(
            "UPDATE index_child SET content_hash = %s WHERE child_id = %s",
            ("0" * 64, drifted.child_id),
        )
        conn.commit()
    pipeline.vectors.upsert_children(
        [
            ChildVector(
                child_id=drifted.child_id,
                vector=bogus,
                parent_id=drifted.parent_id,
                document_id=drifted.document_id,
                version=drifted.version,
                content_hash="0" * 64,
                clause_number=drifted.clause_number,
                permission=policy.permission,
                effective_status="effective",
                document_type=policy.document_type,
            )
        ]
    )
    before = len(embedder.document_texts)
    report = pipeline.reconcile(policy.document_id)
    added = embedder.document_texts[before:]
    restored = pipeline.catalog.child(drifted.child_id)

    assert report.event_kind == "reconcile"
    assert pipeline.catalog.events(policy.document_id)[-1].kind == "reconcile"
    assert added == [drifted.embed_text]
    assert restored.content_hash == drifted.content_hash
    assert restored.vector != bogus
    for child in untouched:
        assert pipeline.catalog.child(child.child_id).content_hash == child.content_hash
        assert pipeline.catalog.child(child.child_id).vector == child.vector
    hits = pipeline.lookup(drifted.embed_text, document_type=policy.document_type, limit=1)
    assert hits[0].child_id == drifted.child_id
    assert pipeline.vectors.delete_calls == []
