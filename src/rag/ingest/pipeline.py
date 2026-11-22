"""把来源、版面、OCR、闸门和人工队列串起来。

失败先换解析参数或改走 OCR。多次仍失败，写入元数据库的解析失败队列。
这一层不计算内容哈希，不嵌入，也不检索。
"""

from collections.abc import Sequence
from typing import Protocol

from rag.ingest.errors import ManualQueueWriteError, SourceExhausted, SourceFetchError
from rag.ingest.layout import merge_continued_tables, reading_order
from rag.ingest.ocr import STAMP_AWARE_OCR, STANDARD_OCR, needs_ocr
from rag.ingest.params import DEFAULT_STRATEGIES, FETCH_ATTEMPTS, ParseParams
from rag.ingest.quality import character_failure, retention_ratio, table_failure
from rag.ingest.sources import SourceAdapter
from rag.ingest.types import (
    AcceptedDocument,
    AttemptRecord,
    IngestBatch,
    PageModel,
    QueuedDocument,
    RestoredTable,
    SourceDocument,
    TableBlock,
    TextBlock,
)
from rag.storage.errors import MetadataWriteError, StorageUnavailable
from rag.storage.types import ParseFailure


class ParseFailureQueue(Protocol):
    """元数据库上的解析失败队列。调用方传入存储层的实现，而不是内存列表。"""

    def enqueue_parse_failure(self, item: ParseFailure) -> int: ...


def ingest_from_sources(
    adapters: Sequence[SourceAdapter],
    requests: Sequence[tuple[str, str]],
    queue: ParseFailureQueue,
    *,
    strategies: Sequence[ParseParams] = DEFAULT_STRATEGIES,
    fetch_attempts: int = FETCH_ATTEMPTS,
) -> IngestBatch:
    by_source = {adapter.source_id: adapter for adapter in adapters}
    accepted: list[AcceptedDocument] = []
    queued: list[QueuedDocument] = []
    for source_id, document_ref in requests:
        adapter = by_source[source_id]
        try:
            document = _fetch_with_retries(adapter, document_ref, fetch_attempts)
        except SourceExhausted as exc:
            queued.append(
                _enqueue(
                    queue,
                    source_id,
                    document_ref,
                    "来源取不到文件",
                    exc.attempts,
                )
            )
            continue
        _collect(document, queue, strategies, accepted, queued)
    return IngestBatch(tuple(accepted), tuple(queued))


def ingest_documents(
    documents: Sequence[SourceDocument],
    queue: ParseFailureQueue,
    *,
    strategies: Sequence[ParseParams] = DEFAULT_STRATEGIES,
) -> IngestBatch:
    accepted: list[AcceptedDocument] = []
    queued: list[QueuedDocument] = []
    for document in documents:
        _collect(document, queue, strategies, accepted, queued)
    return IngestBatch(tuple(accepted), tuple(queued))


def compose_normalized(block_texts: Sequence[str], tables: Sequence[RestoredTable]) -> str:
    parts = list(block_texts)
    for table in tables:
        for row in table.rows:
            parts.append("\t".join(row))
    return "\n".join(parts)


def _fetch_with_retries(adapter: SourceAdapter, document_ref: str, attempts: int) -> SourceDocument:
    for _ in range(attempts):
        try:
            return adapter.fetch(document_ref)
        except SourceFetchError:
            continue
    raise SourceExhausted(adapter.source_id, document_ref, attempts)


def _collect(
    document: SourceDocument,
    queue: ParseFailureQueue,
    strategies: Sequence[ParseParams],
    accepted: list[AcceptedDocument],
    queued: list[QueuedDocument],
) -> None:
    attempts: list[AttemptRecord] = []
    for params in strategies:
        record, parsed = _try_parse(document, params)
        attempts.append(record)
        if record.passed and parsed is not None:
            ordered, tables = parsed
            accepted.append(
                AcceptedDocument(
                    source_id=document.identity.source_id,
                    document_ref=document.identity.document_ref,
                    form=document.identity.form,
                    normalized_text=compose_normalized([block.text for block in ordered], tables),
                    tables=tables,
                    reading_order=tuple(block.text for block in ordered),
                    attempts=tuple(attempts),
                )
            )
            return
    last = attempts[-1].failure if attempts else "没有可用的解析参数"
    queued.append(
        _enqueue(
            queue,
            document.identity.source_id,
            document.identity.document_ref,
            last,
            len(attempts),
        )
    )


def _try_parse(
    document: SourceDocument,
    params: ParseParams,
) -> tuple[AttemptRecord, tuple[tuple[TextBlock, ...], tuple[RestoredTable, ...]] | None]:
    routes: list[str] = []
    engines: list[str] = []
    ordered: list[TextBlock] = []
    page_tables: list[tuple[TableBlock, ...]] = []
    for page in document.pages:
        route, engine, blocks, tables = _load_page(page, params)
        routes.append(route)
        engines.append(engine)
        ordered.extend(
            reading_order(
                blocks,
                column_gap=params.column_gap,
                line_tolerance=params.line_tolerance,
            )
        )
        page_tables.append(tables)
    restored = merge_continued_tables(page_tables)
    text = compose_normalized([block.text for block in ordered], restored)
    retention = retention_ratio(text, document.reference_text)
    char_reason = character_failure(text, document.reference_text)
    table_reason = table_failure(restored, document.reference_tables)
    failure = char_reason or table_reason
    record = AttemptRecord(
        strategy=params.name,
        routes=tuple(routes),
        ocr_engines=tuple(engines),
        retention=retention,
        table_passed=table_reason == "",
        failure=failure,
    )
    if failure:
        return record, None
    return record, (tuple(ordered), restored)


def _load_page(
    page: PageModel,
    params: ParseParams,
) -> tuple[str, str, tuple[TextBlock, ...], tuple[TableBlock, ...]]:
    if params.ocr_mode == "stamp_aware":
        blocks, tables = STAMP_AWARE_OCR.read(page)
        return "ocr", STAMP_AWARE_OCR.name, blocks, tables
    if params.ocr_mode == "standard" or (
        params.ocr_mode == "auto" and needs_ocr(page, params.text_layer_ratio_floor)
    ):
        blocks, tables = STANDARD_OCR.read(page)
        return "ocr", STANDARD_OCR.name, blocks, tables
    return "text_layer", "", page.text_layer_blocks, page.text_tables


def _enqueue(
    queue: ParseFailureQueue,
    source_id: str,
    document_ref: str,
    reason: str,
    attempt_count: int,
) -> QueuedDocument:
    item = ParseFailure(
        source_id=source_id,
        document_ref=document_ref,
        reason=reason,
        attempt_count=attempt_count,
        status="queued",
    )
    try:
        queue.enqueue_parse_failure(item)
    except (MetadataWriteError, StorageUnavailable) as exc:
        raise ManualQueueWriteError(source_id, document_ref) from exc
    return QueuedDocument(source_id, document_ref, reason, attempt_count)
