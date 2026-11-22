"""索引层的输入和输出。检索、改写和生成不从这里开始。"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SourceRef:
    """一个来源的标识。同一份文本可以挂上多个来源。"""

    source_id: str
    document_ref: str
    form: str


@dataclass(frozen=True)
class ClauseItem:
    marker: str
    explanation: str


@dataclass(frozen=True)
class IndexTable:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Clause:
    """一条制度。父块按这种条款来切，不按固定窗口硬截。"""

    number: str
    title_path: tuple[str, ...]
    statement: str
    items: tuple[ClauseItem, ...] = ()
    extra: str = ""
    table: IndexTable | None = None


@dataclass(frozen=True)
class PolicyDocument:
    """交给索引的一份制度。规范化文本由条款拼出来，再算 SHA-256。"""

    document_id: str
    version: str
    document_type: str
    permission: str
    effective_from: date
    clauses: tuple[Clause, ...]
    sources: tuple[SourceRef, ...]
    effective_to: date | None = None


@dataclass(frozen=True)
class ParentChunk:
    parent_id: str
    text: str
    token_count: int
    content_hash: str
    injection_mode: str
    summary: str
    title_path: str
    clause_numbers: tuple[str, ...]


@dataclass(frozen=True)
class ChildChunk:
    child_id: str
    parent_id: str
    clause_number: str
    slot: str
    body: str
    embed_text: str
    token_count: int
    content_hash: str
    title_path: str
    kind: str


@dataclass(frozen=True)
class ChunkPlan:
    parents: tuple[ParentChunk, ...]
    children: tuple[ChildChunk, ...]


@dataclass(frozen=True)
class InjectionText:
    """命中之后准备注入的文本。生成层以后用它，这里只把文本定下来。"""

    parent_id: str
    child_id: str
    mode: str
    text: str


@dataclass(frozen=True)
class IndexReceipt:
    document_id: str
    version: str
    normalized_hash: str
    reused: bool
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    embedded_hashes: tuple[str, ...]
    reused_hashes: tuple[str, ...]
    disappeared_ids: tuple[str, ...]
    event_kind: str
    activated: bool
