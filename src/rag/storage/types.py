from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarFilter:
    """下推进引擎的标量条件。空着的字段表示这次不限制。"""

    permission: str | None = None
    effective_status: str | None = None
    document_type: str | None = None


@dataclass(frozen=True)
class ChildVector:
    """子块向量。没有父块正文字段。"""

    child_id: str
    vector: tuple[float, ...]
    parent_id: str
    document_id: str
    version: str
    content_hash: str
    clause_number: str
    permission: str
    effective_status: str
    document_type: str


@dataclass(frozen=True)
class VectorHit:
    child_id: str
    parent_id: str
    document_id: str
    version: str
    content_hash: str
    clause_number: str
    score: float


@dataclass(frozen=True)
class ChildText:
    """子块文本，给倒排用。父块全文不在这里。"""

    child_id: str
    text: str
    parent_id: str
    document_id: str
    version: str
    content_hash: str
    clause_number: str
    permission: str
    effective_status: str
    document_type: str


@dataclass(frozen=True)
class TextHit:
    child_id: str
    parent_id: str
    document_id: str
    version: str
    content_hash: str
    clause_number: str
    score: float


@dataclass(frozen=True)
class ParentRecord:
    document_id: str
    parent_id: str
    version: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class CitationLog:
    request_id: str
    document_id: str
    parent_id: str
    version: str
    citation_marker: str
    outcome: str
    detail: str = ""


@dataclass(frozen=True)
class CitationRecord:
    id: int
    request_id: str
    document_id: str
    parent_id: str
    version: str
    citation_marker: str
    outcome: str
    detail: str


@dataclass(frozen=True)
class EvaluationEntry:
    suite: str
    case_key: str
    payload: dict


@dataclass(frozen=True)
class EvaluationRecord:
    id: int
    suite: str
    case_key: str
    payload: dict


@dataclass(frozen=True)
class ParseFailure:
    source_id: str
    document_ref: str
    reason: str
    attempt_count: int
    status: str = "queued"


@dataclass(frozen=True)
class ParseFailureRecord:
    id: int
    source_id: str
    document_ref: str
    reason: str
    attempt_count: int
    status: str
