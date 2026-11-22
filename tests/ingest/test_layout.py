from rag.ingest.pipeline import ingest_documents
from rag.ingest.types import BBox, ReferenceTable, TableBlock

from fixtures import GuardQueue, block, document_for, text_page

LEFT_TOP = "左栏第一条出发前审批"
LEFT_BOTTOM = "左栏第二条回来后报销"
RIGHT_TOP = "右栏第一条住宿按标准"
RIGHT_BOTTOM = "右栏第二条伙食按天数"


def test_two_column_fixture_reads_down_each_column_then_left_to_right():
    page = text_page(
        block(RIGHT_BOTTOM, 320, 140, 470, 180),
        block(LEFT_BOTTOM, 40, 140, 180, 180),
        block(RIGHT_TOP, 320, 30, 470, 70),
        block(LEFT_TOP, 40, 30, 180, 70),
    )
    reference = LEFT_TOP + LEFT_BOTTOM + RIGHT_TOP + RIGHT_BOTTOM
    document = document_for("制度库", "two-column", page, reference)
    ready = ingest_documents([document], GuardQueue()).ready_for_index()
    assert len(ready) == 1
    assert ready[0].reading_order == (LEFT_TOP, LEFT_BOTTOM, RIGHT_TOP, RIGHT_BOTTOM)
    assert ready[0].attempts[0].routes == ("text_layer",)
    assert ready[0].attempts[0].passed


def test_open_bottom_and_headerless_next_page_merge_and_restore_header():
    header = ("费用项", "标准")
    continued = ("住宿", "凭票")
    page_one = _table_page(
        TableBlock(
            rows=(header, ("市内交通", "实报")),
            bbox=BBox(40, 480, 460, 640),
            has_header=True,
            closed_bottom=False,
        )
    )
    page_two = _table_page(
        TableBlock(
            rows=(continued,),
            bbox=BBox(40, 30, 460, 90),
            has_header=False,
            closed_bottom=True,
        )
    )
    merged_rows = (header, ("市内交通", "实报"), continued)
    document = _two_page_document(page_one, page_two, merged_rows, "merge-ok")
    ready = ingest_documents([document], GuardQueue()).ready_for_index()
    assert len(ready) == 1
    table = ready[0].tables[0]
    assert table.header == header
    assert table.rows == merged_rows
    assert table.rows.count(header) == 1


def test_header_on_next_page_or_closed_bottom_stays_two_tables():
    header = ("费用项", "标准")
    body = ("市内交通", "实报")
    next_header = ("费用项", "标准")
    next_body = ("住宿", "凭票")
    open_then_headed = _pair(
        TableBlock(rows=(header, body), bbox=BBox(40, 480, 460, 640), has_header=True, closed_bottom=False),
        TableBlock(
            rows=(next_header, next_body),
            bbox=BBox(40, 30, 460, 120),
            has_header=True,
            closed_bottom=True,
        ),
        "has-header",
    )
    closed_then_headerless = _pair(
        TableBlock(rows=(header, body), bbox=BBox(40, 480, 460, 640), has_header=True, closed_bottom=True),
        TableBlock(rows=(next_body,), bbox=BBox(40, 30, 460, 90), has_header=False, closed_bottom=True),
        "closed-bottom",
    )
    for document in (open_then_headed, closed_then_headerless):
        ready = ingest_documents([document], GuardQueue()).ready_for_index()
        assert len(ready[0].tables) == 2
        assert ready[0].tables[0].rows[0] == header
        assert ready[0].tables[1].rows != ready[0].tables[0].rows


def _table_page(table: TableBlock):
    return text_page(ratio=1.0, text_tables=(table,))


def _two_page_document(page_one, page_two, merged_rows, document_ref):
    from rag.ingest.pipeline import compose_normalized
    from rag.ingest.types import RestoredTable, SourceDocument, SourceIdentity

    table = RestoredTable(merged_rows)
    return SourceDocument(
        identity=SourceIdentity("制度库", document_ref, "归档"),
        pages=(page_one, page_two),
        reference_text=compose_normalized((), (table,)),
        reference_tables=(
            ReferenceTable(
                row_count=len(merged_rows),
                col_count=2,
                samples=(
                    (0, 0, merged_rows[0][0]),
                    (0, 1, merged_rows[0][1]),
                    (len(merged_rows) - 1, 0, merged_rows[-1][0]),
                    (len(merged_rows) - 1, 1, merged_rows[-1][1]),
                ),
            ),
        ),
    )


def _pair(first: TableBlock, second: TableBlock, document_ref):
    from rag.ingest.pipeline import compose_normalized
    from rag.ingest.types import RestoredTable, SourceDocument, SourceIdentity

    tables = (RestoredTable(first.rows), RestoredTable(second.rows))
    return SourceDocument(
        identity=SourceIdentity("制度库", document_ref, "归档"),
        pages=(_table_page(first), _table_page(second)),
        reference_text=compose_normalized((), tables),
        reference_tables=(
            ReferenceTable(row_count=len(first.rows), col_count=2, samples=((0, 0, first.rows[0][0]),)),
            ReferenceTable(row_count=len(second.rows), col_count=2, samples=((0, 0, second.rows[0][0]),)),
        ),
    )
