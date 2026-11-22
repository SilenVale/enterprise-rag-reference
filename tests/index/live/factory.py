"""造一段能落到规定 token 区间的合成条款。"""

import uuid
from datetime import date

from rag.index.types import Clause, ClauseItem, IndexTable, PolicyDocument, SourceRef


def new_document_id() -> str:
    return "d" + uuid.uuid4().hex[:8]


def text_of(count_tokens, size: int, seed: str) -> str:
    seed = seed.strip() or "制度"
    text = seed
    while count_tokens(text) < size:
        text += seed
    low = 0
    high = len(text)
    best = text
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue
        if count_tokens(text[:mid]) >= size:
            best = text[:mid]
            high = mid - 1
        else:
            low = mid + 1
    while best and count_tokens(best) > size:
        best = best[:-1]
    while count_tokens(best) < size:
        best += "条"
        if count_tokens(best) > size:
            break
    return best


def make_clause(
    count_tokens,
    statement_tokens: int,
    *,
    number: str,
    title_path: tuple[str, ...],
    seed: str,
    items: tuple[tuple[str, int, str], ...] = (),
    extra_tokens: int = 0,
    extra_seed: str = "",
    table: IndexTable | None = None,
) -> Clause:
    statement = text_of(count_tokens, statement_tokens, seed)
    built_items = tuple(
        ClauseItem(marker, text_of(count_tokens, tokens, item_seed))
        for marker, tokens, item_seed in items
    )
    extra = ""
    if extra_tokens:
        extra = text_of(count_tokens, extra_tokens, extra_seed or "附加说明。")
    return Clause(
        number=number,
        title_path=title_path,
        statement=statement,
        items=built_items,
        extra=extra,
        table=table,
    )


def make_policy(
    clauses: tuple[Clause, ...],
    sources: tuple[SourceRef, ...],
    *,
    document_id: str | None = None,
    version: str = "v1",
    effective_from: date | None = None,
    document_type: str | None = None,
    permission: str = "public",
) -> PolicyDocument:
    document_id = document_id or new_document_id()
    return PolicyDocument(
        document_id=document_id,
        version=version,
        document_type=document_type or document_id,
        permission=permission,
        effective_from=effective_from or date(2020, 1, 1),
        clauses=clauses,
        sources=sources,
    )
