"""按坐标重建阅读顺序，并把跨页的同一张表拼回去。

版面模型材料没有写死。这里用栏间距把块分成左右栏。文档未写死。
"""

from collections.abc import Sequence

from rag.ingest.params import COLUMN_GAP, LINE_TOLERANCE
from rag.ingest.types import RestoredTable, TableBlock, TextBlock


def reading_order(
    blocks: Sequence[TextBlock],
    *,
    column_gap: float = COLUMN_GAP,
    line_tolerance: float = LINE_TOLERANCE,
) -> tuple[TextBlock, ...]:
    """栏内从上到下，栏与栏从左到右。输入列表的先后不算阅读顺序。"""

    if not blocks:
        return ()
    ordered = sorted(blocks, key=lambda block: (block.bbox.x0, block.bbox.y0))
    columns: list[list[TextBlock]] = [[ordered[0]]]
    for block in ordered[1:]:
        current = columns[-1]
        right_edge = max(item.bbox.x1 for item in current)
        if block.bbox.x0 - right_edge >= column_gap:
            columns.append([block])
        else:
            current.append(block)
    columns.sort(key=lambda column: min(item.bbox.x0 for item in column))
    result: list[TextBlock] = []
    for column in columns:
        result.extend(_one_block_per_line(column, line_tolerance))
    return tuple(result)


def _one_block_per_line(column: Sequence[TextBlock], line_tolerance: float) -> list[TextBlock]:
    """同一行只留最左边的一块。

    两栏被并成一栏时，右栏会和左栏挤在同一行上，于是被丢掉。
    丢掉之后字符保有率过不了闸门，调用方再换更窄的栏间距重试。
    """

    kept: list[TextBlock] = []
    for block in sorted(column, key=lambda item: (item.bbox.y0, item.bbox.x0)):
        if kept and abs(block.bbox.y0 - kept[-1].bbox.y0) <= line_tolerance:
            continue
        kept.append(block)
    return kept


def merge_continued_tables(pages: Sequence[Sequence[TableBlock]]) -> tuple[RestoredTable, ...]:
    """页尾没有结束边框，且下一页第一张表没有表头，就当成同一张表。

    合并后的第一行仍是前一页的表头。
    """

    restored: list[RestoredTable] = []
    open_rows: list[tuple[str, ...]] | None = None
    open_closed = True

    def flush() -> None:
        nonlocal open_rows
        if open_rows is not None:
            restored.append(RestoredTable(tuple(open_rows)))
            open_rows = None

    for page in pages:
        for index, table in enumerate(page):
            continues = (
                open_rows is not None
                and not open_closed
                and index == 0
                and not table.has_header
            )
            if continues:
                open_rows.extend(table.rows)
                open_closed = table.closed_bottom
                if open_closed:
                    flush()
                continue
            flush()
            open_rows = list(table.rows)
            open_closed = table.closed_bottom
            if open_closed:
                flush()
    flush()
    return tuple(restored)
