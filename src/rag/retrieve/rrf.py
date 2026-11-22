"""RRF 只加名次。不把余弦和 BM25 分数加在一起。"""

from rag.retrieve.params import RRF_K
from rag.retrieve.types import FusedHit, RecallHit


def fuse(paths: dict[str, list[RecallHit] | tuple[RecallHit, ...]], *, k: int = RRF_K) -> list[FusedHit]:
    if k != RRF_K:
        raise ValueError("RRF 的 k 固定为 60")
    scores: dict[str, float] = {}
    parts: dict[str, list[tuple[str, float]]] = {}
    chosen: dict[str, RecallHit] = {}
    for name, hits in paths.items():
        for hit in hits:
            if hit.rank < 1:
                raise ValueError("名次从 1 开始")
            share = 1.0 / (k + hit.rank)
            scores[hit.child_id] = scores.get(hit.child_id, 0.0) + share
            parts.setdefault(hit.child_id, []).append((name, share))
            chosen.setdefault(hit.child_id, hit)
    ordered = sorted(scores, key=lambda child_id: (-scores[child_id], child_id))
    fused: list[FusedHit] = []
    for child_id in ordered:
        hit = chosen[child_id]
        fused.append(
            FusedHit(
                child_id=hit.child_id,
                parent_id=hit.parent_id,
                document_id=hit.document_id,
                version=hit.version,
                content_hash=hit.content_hash,
                clause_number=hit.clause_number,
                score=scores[child_id],
                parts=tuple(parts[child_id]),
            )
        )
    return fused
