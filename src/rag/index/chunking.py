"""按条款切父块，再把父块切成只用于嵌入的子块。"""

import math
from dataclasses import dataclass

from rag.index.errors import ChunkBoundError, TableHeaderMissing
from rag.index.types import (
    ChildChunk,
    ChunkPlan,
    Clause,
    ClauseItem,
    IndexTable,
    ParentChunk,
    PolicyDocument,
)
from rag.storage.hashing import content_hash

PARENT_MIN = 512
PARENT_MAX = 1024
CHILD_MIN = 128
CHILD_MAX = 256
# 父块仍然超长时，摘要只留条款开头这一小段。材料没有写死摘要多长。
SUMMARY_TOKENS = 48
PATH_SEP = " / "


@dataclass(frozen=True)
class _Prepared:
    clause: Clause
    prose: str
    stored: str
    mode: str


@dataclass(frozen=True)
class _Piece:
    body: str
    paths: tuple[tuple[str, ...], ...]
    clause_number: str
    kind: str


def canonical_text(document: PolicyDocument) -> str:
    """整篇规范化文本。哈希对着这一份，不把来源标识算进去。"""

    lines: list[str] = []
    for clause in document.clauses:
        lines.append(PATH_SEP.join(clause.title_path))
        lines.append(clause.number.strip())
        lines.append(clause.statement.strip())
        for item in clause.items:
            lines.append(_item_line(item))
        if clause.extra.strip():
            lines.append(clause.extra.strip())
        if clause.table is not None and clause.table.header:
            lines.append("\t".join(cell.strip() for cell in clause.table.header))
            for row in clause.table.rows:
                lines.append("\t".join(cell.strip() for cell in row))
    return "\n".join(lines)


def build_chunks(document: PolicyDocument, count_tokens) -> ChunkPlan:
    prepared = [_prepare(clause, count_tokens) for clause in document.clauses]
    groups = _backfill(_group(prepared, count_tokens), count_tokens)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    child_ordinal = 0
    for parent_index, group in enumerate(groups):
        parent_id = f"{document.document_id}-{document.version}-p{parent_index:02d}"
        text = "\n".join(item.stored for item in group)
        mode = _group_mode(group)
        token_count = count_tokens(text)
        _check_parent_bounds(parent_id, mode, token_count)
        summary = _summary(group[0].clause, count_tokens) if mode == "summary_plus_child" else ""
        title = _prefix(tuple(item.clause.title_path for item in group))
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                text=text,
                token_count=token_count,
                content_hash=content_hash(text),
                injection_mode=mode,
                summary=summary,
                title_path=title,
                clause_numbers=tuple(item.clause.number for item in group),
            )
        )
        pieces = _prose_pieces(group, count_tokens) + _table_pieces(group, count_tokens)
        for ordinal, piece in enumerate(pieces):
            body_tokens = count_tokens(piece.body)
            if not (CHILD_MIN <= body_tokens <= CHILD_MAX):
                raise ChunkBoundError(
                    f"子块有 {body_tokens} token，不在 {CHILD_MIN} 到 {CHILD_MAX}"
                )
            prefix = _prefix(piece.paths)
            embed_text = f"{prefix}\n{piece.body}"
            child_id = f"{document.document_id}-{document.version}-c{child_ordinal:03d}"
            child_ordinal += 1
            children.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    clause_number=piece.clause_number,
                    slot=f"{parent_index:02d}-{ordinal:03d}",
                    body=piece.body,
                    embed_text=embed_text,
                    token_count=body_tokens,
                    content_hash=content_hash(embed_text),
                    title_path=prefix,
                    kind=piece.kind,
                )
            )
    if not children:
        raise ChunkBoundError("没有切出子块")
    return ChunkPlan(tuple(parents), tuple(children))


def _prepare(clause: Clause, count_tokens) -> _Prepared:
    _reject_broken_table(clause)
    prose_full = _render_prose(clause, include_extra=True)
    prose_degraded = _render_prose(clause, include_extra=False)
    table_text = _render_table(clause.table) if clause.table is not None else ""
    full = _join(prose_full, table_text)
    if count_tokens(full) <= PARENT_MAX:
        return _Prepared(clause, prose_full, full, "full_parent")
    if count_tokens(prose_degraded) <= PARENT_MAX:
        return _Prepared(clause, prose_degraded, prose_degraded, "clause_plus_items")
    return _Prepared(clause, prose_degraded, prose_degraded, "summary_plus_child")


def _reject_broken_table(clause: Clause) -> None:
    table = clause.table
    if table is None:
        return
    header = tuple(cell.strip() for cell in table.header)
    if not header or any(not cell for cell in header):
        raise TableHeaderMissing(clause.number)
    for row in table.rows:
        if len(row) != len(header) or any(not cell.strip() for cell in row):
            raise TableHeaderMissing(clause.number)


def _render_prose(clause: Clause, *, include_extra: bool) -> str:
    lines = [clause.number.strip(), clause.statement.strip()]
    lines.extend(_item_line(item) for item in clause.items)
    if include_extra and clause.extra.strip():
        lines.append(clause.extra.strip())
    return "\n".join(line for line in lines if line)


def _item_line(item: ClauseItem) -> str:
    return f"{item.marker.strip()} {item.explanation.strip()}".strip()


def _render_table(table: IndexTable | None) -> str:
    if table is None or not table.rows:
        return ""
    lines = ["\t".join(cell.strip() for cell in table.header)]
    lines.extend("\t".join(cell.strip() for cell in row) for row in table.rows)
    return "\n".join(lines)


def _render_row(header: tuple[str, ...], row: tuple[str, ...]) -> str:
    labels = "、".join(cell.strip() for cell in header)
    cells = "；".join(f"{name}：{value.strip()}" for name, value in zip(header, row))
    return f"表头：{labels}\n{cells}"


def _group(prepared: list[_Prepared], count_tokens) -> list[list[_Prepared]]:
    groups: list[list[_Prepared]] = []
    current: list[_Prepared] = []
    current_key: tuple[str, ...] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for item in prepared:
        key = _chapter(item.clause)
        if item.mode == "summary_plus_child":
            flush()
            groups.append([item])
            current_key = None
            continue
        if current and key != current_key:
            flush()
        if not current:
            current = [item]
            current_key = key
            continue
        candidate = "\n".join(part.stored for part in [*current, item])
        if count_tokens(candidate) > PARENT_MAX:
            flush()
            current = [item]
            current_key = key
        else:
            current.append(item)
    flush()
    return groups


def _backfill(groups: list[list[_Prepared]], count_tokens) -> list[list[_Prepared]]:
    changed = True
    while changed:
        changed = False
        index = len(groups) - 1
        while index > 0:
            group = groups[index]
            if _group_mode(group) == "summary_plus_child":
                index -= 1
                continue
            if count_tokens("\n".join(item.stored for item in group)) >= PARENT_MIN:
                index -= 1
                continue
            previous = groups[index - 1]
            same_chapter = _chapter(previous[0].clause) == _chapter(group[0].clause)
            if _group_mode(previous) == "summary_plus_child" or not same_chapter:
                index -= 1
                continue
            merged = previous + group
            if count_tokens("\n".join(item.stored for item in merged)) <= PARENT_MAX:
                groups[index - 1] = merged
                del groups[index]
                changed = True
                break
            index -= 1
    for group in groups:
        text = "\n".join(item.stored for item in group)
        _check_parent_bounds("合并后的父块", _group_mode(group), count_tokens(text))
    return groups


def _check_parent_bounds(parent_id: str, mode: str, token_count: int) -> None:
    if mode == "summary_plus_child":
        if token_count <= PARENT_MAX:
            raise ChunkBoundError(f"{parent_id} 已降到上限以内，不应再走摘要注入")
        return
    if not (PARENT_MIN <= token_count <= PARENT_MAX):
        raise ChunkBoundError(
            f"{parent_id} 有 {token_count} token，不在 {PARENT_MIN} 到 {PARENT_MAX}"
        )


def _group_mode(group: list[_Prepared]) -> str:
    modes = {item.mode for item in group}
    if "summary_plus_child" in modes:
        return "summary_plus_child"
    if "clause_plus_items" in modes:
        return "clause_plus_items"
    return "full_parent"


def _chapter(clause: Clause) -> tuple[str, ...]:
    if len(clause.title_path) > 1:
        return clause.title_path[:-1]
    return clause.title_path


def _prose_pieces(group: list[_Prepared], count_tokens) -> list[_Piece]:
    segments: list[tuple[int, int, _Prepared]] = []
    chunks: list[str] = []
    cursor = 0
    for index, item in enumerate(group):
        if index:
            chunks.append("\n")
            cursor += 1
        start = cursor
        chunks.append(item.prose)
        cursor += len(item.prose)
        segments.append((start, cursor, item))
    prose = "".join(chunks)
    bodies = split_into_range(prose, count_tokens, CHILD_MIN, CHILD_MAX)
    pieces: list[_Piece] = []
    cursor = 0
    for body in bodies:
        start = cursor
        end = cursor + len(body)
        cursor = end
        owners = [item for seg_start, seg_end, item in segments if seg_start < end and seg_end > start]
        if not owners:
            owners = [group[0]]
        paths = tuple(dict.fromkeys(item.clause.title_path for item in owners))
        pieces.append(_Piece(body, paths, owners[0].clause.number, "prose"))
    return pieces


def _table_pieces(group: list[_Prepared], count_tokens) -> list[_Piece]:
    pieces: list[_Piece] = []
    for item in group:
        table = item.clause.table
        if table is None or not table.rows:
            continue
        rendered = [_render_row(table.header, row) for row in table.rows]
        for body in _pack_labeled(rendered, count_tokens, table.header):
            pieces.append(_Piece(body, (item.clause.title_path,), item.clause.number, "table_row"))
    return pieces


def _pack_labeled(rows: list[str], count_tokens, header: tuple[str, ...]) -> list[str]:
    packed: list[str] = []
    buffer = ""
    for row in rows:
        parts = _row_parts(row, count_tokens, header)
        for part in parts:
            if not buffer:
                buffer = part
                continue
            merged = buffer + "\n" + part
            if count_tokens(merged) <= CHILD_MAX:
                buffer = merged
            else:
                packed.append(buffer)
                buffer = part
    if buffer:
        packed.append(buffer)
    if len(packed) >= 2 and count_tokens(packed[-1]) < CHILD_MIN:
        merged = packed[-2] + "\n" + packed[-1]
        if count_tokens(merged) <= CHILD_MAX:
            packed[-2] = merged
            packed.pop()
    for body in packed:
        tokens = count_tokens(body)
        if not (CHILD_MIN <= tokens <= CHILD_MAX):
            raise ChunkBoundError(f"表格行有 {tokens} token，不在 {CHILD_MIN} 到 {CHILD_MAX}")
        for label in header:
            if label.strip() not in body:
                raise TableHeaderMissing("表格行")
    return packed


def _row_parts(row: str, count_tokens, header: tuple[str, ...]) -> list[str]:
    if count_tokens(row) <= CHILD_MAX:
        return [row]
    pieces = split_into_range(row, count_tokens, CHILD_MIN, CHILD_MAX)
    labeled: list[str] = []
    for piece in pieces:
        if all(label.strip() in piece for label in header):
            labeled.append(piece)
            continue
        prefix = "表头：" + "、".join(label.strip() for label in header) + "\n"
        body = prefix + piece
        if count_tokens(body) > CHILD_MAX:
            raise ChunkBoundError("带表头之后表格行仍然超过子块上限")
        labeled.append(body)
    return labeled


def _summary(clause: Clause, count_tokens) -> str:
    text = f"{clause.number.strip()} {clause.statement.strip()}"
    if count_tokens(text) <= SUMMARY_TOKENS:
        return text
    prefix, _rest = _cut_tokens(text, SUMMARY_TOKENS, count_tokens)
    return prefix


def _prefix(paths: tuple[tuple[str, ...], ...]) -> str:
    unique = tuple(dict.fromkeys(paths))
    return "；".join(PATH_SEP.join(path) for path in unique)


def _join(prose: str, table: str) -> str:
    if prose and table:
        return prose + "\n" + table
    return prose or table


def split_into_range(text: str, count_tokens, lo: int, hi: int) -> list[str]:
    total = count_tokens(text)
    if total < lo:
        raise ChunkBoundError(f"{total} token 短于 {lo}，无法单独成块")
    if total <= hi:
        return [text]
    parts = math.ceil(total / hi)
    while parts > 1 and total / parts < lo:
        parts -= 1
    if parts < 1 or math.ceil(total / parts) > hi or total < parts * lo:
        raise ChunkBoundError(f"无法把 {total} token 切进 {lo} 到 {hi}")
    targets = _even_sizes(total, parts)
    pieces: list[str] = []
    rest = text
    for target in targets[:-1]:
        piece, rest = _cut_tokens(rest, target, count_tokens)
        pieces.append(piece)
    pieces.append(rest)
    return _balance(pieces, count_tokens, lo, hi)


def _even_sizes(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def _cut_tokens(text: str, target: int, count_tokens) -> tuple[str, str]:
    if target <= 0:
        raise ChunkBoundError("切分目标至少 1 token")
    low = 0
    high = len(text)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        tokens = count_tokens(text[:mid]) if mid else 0
        if tokens <= target:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best == 0:
        raise ChunkBoundError("切不开这一段")
    return text[:best], text[best:]


def _balance(pieces: list[str], count_tokens, lo: int, hi: int) -> list[str]:
    pieces = [piece for piece in pieces if piece]
    limit = len(pieces) * 8 + 8
    for _ in range(limit):
        counts = [count_tokens(piece) for piece in pieces]
        if all(lo <= tokens <= hi for tokens in counts):
            return pieces
        moved = False
        for index, tokens in enumerate(counts):
            if tokens > hi:
                piece = pieces[index]
                while piece and count_tokens(piece) > hi:
                    if index + 1 == len(pieces):
                        pieces.append("")
                    pieces[index + 1] = piece[-1] + pieces[index + 1]
                    piece = piece[:-1]
                pieces[index] = piece
                moved = True
                break
            if tokens < lo and index + 1 < len(pieces):
                piece = pieces[index]
                nxt = pieces[index + 1]
                while nxt and count_tokens(piece) < lo:
                    step = nxt[0]
                    if count_tokens(piece + step) > hi:
                        break
                    piece += step
                    nxt = nxt[1:]
                pieces[index] = piece
                pieces[index + 1] = nxt
                moved = True
                break
        pieces = [piece for piece in pieces if piece]
        if not moved:
            break
    counts = [count_tokens(piece) for piece in pieces]
    if not all(lo <= tokens <= hi for tokens in counts):
        raise ChunkBoundError(f"子块长度没有落入 {lo} 到 {hi}：{counts}")
    return pieces
