from rag.ingest.params import EXAMPLE_RETENTION_THRESHOLD
from rag.ingest.quality import character_failure, table_failure
from rag.ingest.types import ReferenceTable, RestoredTable


def test_character_retention_below_example_threshold_fails():
    assert EXAMPLE_RETENTION_THRESHOLD == 0.95
    reference = "甲" * 100
    assert character_failure("甲" * 95, reference) == ""
    reason = character_failure("甲" * 94, reference)
    assert reason.startswith("字符保有率")
    assert "0.94" in reason
    assert "0.95" in reason


def test_two_column_swallowed_half_fails_character_retention():
    left = "左栏上段" * 8 + "左栏下段" * 8
    right = "右栏上段" * 8 + "右栏下段" * 8
    assert len(left) == len(right)
    reason = character_failure(left, left + right)
    assert reason.startswith("字符保有率")
    assert "0.50" in reason
    assert character_failure(left + right, left + right) == ""


def test_table_gate_checks_row_column_counts_and_sampled_cells():
    expected = (
        ReferenceTable(
            row_count=2,
            col_count=2,
            samples=((0, 0, "费用项"), (1, 1, "凭票")),
        ),
    )
    matched = RestoredTable((("费用项", "标准"), ("住宿", "凭票")))
    assert table_failure((matched,), expected) == ""

    extra_header = RestoredTable(
        (("费用项", "标准"), ("住宿", "凭票"), ("费用项", "标准"))
    )
    assert "行列数" in table_failure((extra_header,), expected)

    extra_column = RestoredTable((("费用项", "标准", "备注"), ("住宿", "凭票", "无")))
    assert "行列数" in table_failure((extra_column,), expected)

    wrong_cell = RestoredTable((("费用项", "标准"), ("住宿", "实报")))
    assert "抽查单元格" in table_failure((wrong_cell,), expected)
