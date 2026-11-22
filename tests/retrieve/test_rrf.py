import pytest

from rag.retrieve.params import RRF_K
from rag.retrieve.rrf import fuse
from rag.retrieve.types import RecallHit


def _hit(child_id: str, rank: int, score: float) -> RecallHit:
    return RecallHit(
        child_id=child_id,
        parent_id="p-" + child_id,
        document_id="doc",
        version="v1",
        content_hash="hash",
        clause_number="第一条",
        rank=rank,
        source_score=score,
    )


def test_rrf_sums_reciprocal_ranks_and_ignores_raw_scores():
    quiet = {
        "vector": [_hit("a", 1, 0.01), _hit("b", 2, 0.99)],
        "bm25": [_hit("b", 1, 100.0)],
        "metadata": [],
    }
    loud = {
        "vector": [_hit("a", 1, 0.99), _hit("b", 2, 0.01)],
        "bm25": [_hit("b", 1, 0.001)],
        "metadata": [],
    }
    fused_quiet = fuse(quiet)
    fused_loud = fuse(loud)
    assert [item.child_id for item in fused_quiet] == [item.child_id for item in fused_loud]
    by_id = {item.child_id: item.score for item in fused_quiet}
    assert by_id["a"] == pytest.approx(1 / (RRF_K + 1))
    assert by_id["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert by_id["b"] != pytest.approx(0.99 + 100.0)
    assert RRF_K == 60


def test_missing_path_adds_nothing_and_k_stays_60():
    fused = fuse({"vector": [_hit("a", 1, 0.2)], "bm25": [], "metadata": []})
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[0].parts[0][0] == "vector"
    assert fused[0].parts[0][1] == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        fuse({"vector": []}, k=20)
