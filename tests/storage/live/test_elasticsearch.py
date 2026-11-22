from rag.storage.types import ScalarFilter

from helpers import make_child_text, new_id, no_filter


def test_bm25_ranks_term_match_above_unrelated(lexical_store):
    match_id = new_id("match")
    other_id = new_id("other")
    token = match_id.replace("-", "")
    rows = [
        make_child_text(match_id, f"reimbursement {token} seat"),
        make_child_text(other_id, "unrelated budget form"),
    ]
    lexical_store.upsert_children(rows)
    try:
        hits = lexical_store.search(token, limit=5, scalar=no_filter())
    finally:
        lexical_store.delete_children([match_id, other_id])
    assert hits[0].child_id == match_id
    assert other_id not in {hit.child_id for hit in hits[:1]}
    assert hits[0].score > 0


def test_filter_context_does_not_change_bm25_score(lexical_store):
    public_id = new_id("public")
    restricted_id = new_id("restricted")
    token = public_id.replace("-", "")
    text = f"reimbursement {token} seat"
    rows = [
        make_child_text(public_id, text, permission="public"),
        make_child_text(restricted_id, text, permission="restricted"),
    ]
    lexical_store.upsert_children(rows)
    try:
        plain = lexical_store.search(token, limit=5, scalar=no_filter())
        filtered = lexical_store.search(
            token,
            limit=5,
            scalar=ScalarFilter(permission="public"),
        )
    finally:
        lexical_store.delete_children([public_id, restricted_id])
    plain_score = next(hit.score for hit in plain if hit.child_id == public_id)
    filtered_hits = [hit for hit in filtered if hit.child_id == public_id]
    assert len(filtered_hits) == 1
    assert filtered_hits[0].score == plain_score
    assert restricted_id not in {hit.child_id for hit in filtered}


def test_clause_number_matches_as_whole_token_only(lexical_store):
    exact_id = new_id("exact")
    shorter_id = new_id("shorter")
    rows = [
        make_child_text(exact_id, "seat rule", clause_number="DZ-12.3"),
        make_child_text(shorter_id, "seat rule", clause_number="DZ-12"),
    ]
    lexical_store.upsert_children(rows)
    try:
        exact = lexical_store.search_clause("DZ-12.3", limit=5, scalar=no_filter())
        shorter = lexical_store.search_clause("DZ-12", limit=5, scalar=no_filter())
        fragment = lexical_store.search_clause("12.3", limit=5, scalar=no_filter())
    finally:
        lexical_store.delete_children([exact_id, shorter_id])
    assert [hit.child_id for hit in exact] == [exact_id]
    assert [hit.child_id for hit in shorter] == [shorter_id]
    assert fragment == []


def test_lexical_mapping_has_child_text_but_not_parent_body(lexical_store):
    properties = lexical_store.mapping_properties()
    assert "text" in properties
    assert properties["clause_number"]["type"] == "keyword"
    banned = {"parent_text", "parent_body", "full_text", "body"}
    assert banned.isdisjoint(properties)
