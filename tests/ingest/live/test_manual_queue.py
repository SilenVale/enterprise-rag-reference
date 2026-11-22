import uuid

import pytest

from rag.ingest.pipeline import ingest_documents, ingest_from_sources
from rag.ingest.sources import SourceAdapter

from fixtures import block, document_for, text_page

pytestmark = pytest.mark.live


def test_repeated_failures_land_in_metadata_queue_and_not_ready_for_index(metadata_store):
    source_id = _new_id("src")
    left = "左" * 40
    right = "右" * 40
    bad_page = text_page(
        block(left, 40, 30, 180, 80),
        image_blocks=(block(left, 40, 30, 180, 80),),
    )
    bad = document_for(source_id, "half-column", bad_page, left + right)
    good_text = "完整的单栏条款"
    good = document_for(source_id, "good-column", text_page(block(good_text, 40, 40, 300, 80)), good_text)

    batch = ingest_documents([good, bad], metadata_store)
    ready_refs = [item.document_ref for item in batch.ready_for_index()]
    assert ready_refs == ["good-column"]
    assert "half-column" not in ready_refs

    rows = metadata_store.list_parse_failures(source_id)
    assert len(rows) == 1
    assert rows[0].document_ref == "half-column"
    assert rows[0].status == "queued"
    assert rows[0].attempt_count == 3
    assert rows[0].reason.startswith("字符保有率")
    assert "0.50" in rows[0].reason
    assert batch.queued[0].reason == rows[0].reason


def test_unreachable_source_is_queued_in_metadata_and_not_ready_for_index(metadata_store):
    source_id = _new_id("src")
    adapter = SourceAdapter({}, source_id=source_id, form="附件")
    batch = ingest_from_sources((adapter,), [(source_id, "missing-file")], metadata_store)

    assert batch.ready_for_index() == ()
    assert adapter.fetches == 3
    rows = metadata_store.list_parse_failures(source_id)
    assert len(rows) == 1
    assert rows[0].document_ref == "missing-file"
    assert rows[0].status == "queued"
    assert rows[0].attempt_count == 3
    assert rows[0].reason == "来源取不到文件"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
