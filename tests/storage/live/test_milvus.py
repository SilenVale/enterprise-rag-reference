import pytest

from rag.storage.errors import ParentTextNotAllowed
from rag.storage.types import ScalarFilter

from helpers import make_child_vector, new_id, no_filter, vector


def test_child_vectors_use_hnsw_index(vector_store):
    described = vector_store.describe_vector_index()
    assert described["index_type"] == "HNSW"
    assert described["metric_type"] == "COSINE"
    assert vector_store.index_type() == "HNSW"


def test_search_returns_nearest_child(vector_store):
    near_id = new_id("near")
    far_id = new_id("far")
    rows = [
        make_child_vector(near_id, vector(1.0, 0.0), document_id=new_id("doc")),
        make_child_vector(far_id, vector(0.0, 1.0), document_id=new_id("doc")),
    ]
    vector_store.upsert_children(rows)
    try:
        hits = vector_store.search(vector(1.0, 0.0), limit=1, scalar=no_filter())
    finally:
        vector_store.delete_children([near_id, far_id])
    assert [hit.child_id for hit in hits] == [near_id]
    assert hits[0].parent_id == "parent-1"
    assert not hasattr(hits[0], "text")


def test_vector_schema_has_no_parent_full_text(vector_store):
    banned = {"parent_text", "parent_body", "full_text", "text", "body", "content"}
    names = set(vector_store.field_names())
    assert "vector" in names
    assert names.isdisjoint(banned)
    for name, max_length in vector_store.varchar_max_lengths().items():
        assert max_length <= 256, name


def test_vector_upsert_rejects_parent_full_text(vector_store):
    class Sneaky:
        parent_text = "父块全文不应该进向量库"
        child_id = "sneaky"

    with pytest.raises(ParentTextNotAllowed):
        vector_store.upsert_children([Sneaky()])


def test_permission_filter_excludes_closer_child_inside_limit(vector_store):
    secret_id = new_id("secret")
    public_id = new_id("public")
    document_id = new_id("doc")
    rows = [
        make_child_vector(
            secret_id,
            vector(1.0, 0.0),
            document_id=document_id,
            permission="secret",
        ),
        make_child_vector(
            public_id,
            vector(0.8, 0.2),
            document_id=document_id,
            permission="public",
        ),
    ]
    vector_store.upsert_children(rows)
    try:
        unfiltered = vector_store.search(vector(1.0, 0.0), limit=1, scalar=no_filter())
        filtered = vector_store.search(
            vector(1.0, 0.0),
            limit=1,
            scalar=ScalarFilter(permission="public"),
        )
        typed = vector_store.search(
            vector(1.0, 0.0),
            limit=2,
            scalar=ScalarFilter(
                permission="public",
                effective_status="effective",
                document_type="policy",
            ),
        )
    finally:
        vector_store.delete_children([secret_id, public_id])
    assert [hit.child_id for hit in unfiltered] == [secret_id]
    assert [hit.child_id for hit in filtered] == [public_id]
    assert [hit.child_id for hit in typed] == [public_id]
