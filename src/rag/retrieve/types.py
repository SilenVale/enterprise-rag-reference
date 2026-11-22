from dataclasses import dataclass

from rag.storage.types import ScalarFilter


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class RetrieveRequest:
    question: str
    history: tuple[Turn, ...] = ()
    role: str = "staff"
    document_type: str | None = None
    include_obsolete: bool = False
    request_id: str = ""


@dataclass(frozen=True)
class RewritePlan:
    original: str
    standalone: str | None
    subqueries: tuple[str, ...]
    model_id: str
    failed: bool = False

    def queries(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for item in (self.original, self.standalone, *self.subqueries):
            if item and item not in ordered:
                ordered.append(item)
        return tuple(ordered)


@dataclass(frozen=True)
class RecallHit:
    child_id: str
    parent_id: str
    document_id: str
    version: str
    content_hash: str
    clause_number: str
    rank: int
    source_score: float


@dataclass(frozen=True)
class PathRecall:
    name: str
    hits: tuple[RecallHit, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class FusedHit:
    child_id: str
    parent_id: str
    document_id: str
    version: str
    content_hash: str
    clause_number: str
    score: float
    parts: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RankedHit:
    child_id: str
    parent_id: str
    document_id: str
    version: str
    clause_number: str
    text: str
    rerank_score: float


@dataclass(frozen=True)
class RetrieveResult:
    request_id: str
    queries: tuple[str, ...]
    rewrite_model_id: str
    rewrite_failed: bool
    filters: tuple[ScalarFilter, ...]
    paths: tuple[PathRecall, ...]
    fused: tuple[FusedHit, ...]
    passages: tuple[RankedHit, ...]
    disposition: str
    route_label: str
    caveat: str
    handoff: bool
    generated_text: str
    latency_ms: float
    recall_wall_ms: float
    failure: str
