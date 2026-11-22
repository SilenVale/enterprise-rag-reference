import pytest

from rag.ingest.errors import ManualQueueWriteError
from rag.ingest.pipeline import ingest_documents, ingest_from_sources
from rag.ingest.sources import SourceAdapter
from rag.storage.errors import MetadataWriteError

from fixtures import GuardQueue, block, document_for, text_page


def test_narrow_columns_retry_tighter_gap_instead_of_dropping():
    left = "左" * 30
    right = "右" * 30
    page = text_page(
        block(right, 115, 20, 200, 50),
        block(left, 30, 20, 100, 50),
    )
    ready = ingest_documents(
        [document_for("制度库", "narrow", page, left + right)],
        GuardQueue(),
    ).ready_for_index()
    assert len(ready) == 1
    attempts = ready[0].attempts
    assert [item.strategy for item in attempts] == ["默认分栏", "更窄栏间距"]
    assert attempts[0].retention == pytest.approx(0.5)
    assert attempts[0].failure.startswith("字符保有率")
    assert attempts[1].passed
    assert left in ready[0].normalized_text
    assert right in ready[0].normalized_text


def test_source_fetch_retries_then_succeeds_without_queue():
    source_id = "邮件"
    sentence = "转发来的同一份制度"
    page = text_page(block(sentence, 40, 40, 300, 80))

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, key):
            self.calls += 1
            if self.calls < 3:
                return None
            from rag.ingest.types import CatalogEntry

            return CatalogEntry((page,), sentence)

    catalog = Flaky()
    adapter = SourceAdapter(catalog, source_id=source_id, form="转发")
    batch = ingest_from_sources((adapter,), [(source_id, "fwd-1")], GuardQueue())
    ready = batch.ready_for_index()
    assert catalog.calls == 3
    assert adapter.fetches == 3
    assert [item.source_id for item in ready] == [source_id]
    assert ready[0].normalized_text == sentence
    assert batch.queued == ()


def test_queue_write_failure_does_not_count_file_as_ingested():
    left = "左" * 40
    right = "右" * 40
    page = text_page(block(left, 40, 40, 180, 80))
    document = document_for("制度库", "unstored", page, left + right)

    class RejectingQueue:
        def enqueue_parse_failure(self, item) -> int:
            raise MetadataWriteError("解析失败队列写不进去，对应文件不能当成已经入库")

    with pytest.raises(ManualQueueWriteError) as caught:
        ingest_documents([document], RejectingQueue())
    assert caught.value.document_ref == "unstored"
    assert caught.value.source_id == "制度库"
