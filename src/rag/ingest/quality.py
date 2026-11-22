"""两道闸门：字符保有率，以及表格的行列数和抽查单元格。"""

from collections.abc import Sequence

from rag.ingest.params import EXAMPLE_RETENTION_THRESHOLD
from rag.ingest.types import ReferenceTable, RestoredTable


def countable(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def retention_ratio(parsed: str, reference: str) -> float:
    """解析产出的字符数，除以原文可提取字符数。"""

    ref = countable(reference)
    if not ref:
        raise ValueError("原文可提取字符为空，不能计算保有率")
    return len(countable(parsed)) / len(ref)


def character_failure(parsed: str, reference: str, threshold: float = EXAMPLE_RETENTION_THRESHOLD) -> str:
    ratio = retention_ratio(parsed, reference)
    if ratio + 1e-12 >= threshold:
        return ""
    return f"字符保有率 {ratio:.2f} 低于阈值 {threshold:.2f}"


def table_failure(
    parsed: Sequence[RestoredTable],
    expected: Sequence[ReferenceTable],
) -> str:
    if len(parsed) != len(expected):
        return f"表格张数不一致：解析 {len(parsed)}，原文 {len(expected)}"
    for index, (table, ref) in enumerate(zip(parsed, expected), start=1):
        width = len(table.rows[0]) if table.rows else 0
        if len(table.rows) != ref.row_count or width != ref.col_count:
            return (
                f"第 {index} 张表行列数不一致："
                f"解析 {len(table.rows)}x{width}，原文 {ref.row_count}x{ref.col_count}"
            )
        if any(len(row) != ref.col_count for row in table.rows):
            return f"第 {index} 张表有行列不齐"
        for row_index, col_index, expected_text in ref.samples:
            actual = table.rows[row_index][col_index]
            if actual != expected_text:
                return (
                    f"第 {index} 张表抽查单元格 ({row_index},{col_index}) "
                    f"是「{actual}」，原文是「{expected_text}」"
                )
    return ""
