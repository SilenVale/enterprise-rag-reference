"""数据接入层交给下一层的数据结构。这里没有向量，也没有答案。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class TextBlock:
    """版面上的一块字。阅读顺序由坐标决定，不由这块在列表里的位置决定。"""

    text: str
    bbox: BBox
    kind: str = "body"


@dataclass(frozen=True)
class TableBlock:
    """一页上的一张表。跨页合并看结束边框和下一页有没有表头。"""

    rows: tuple[tuple[str, ...], ...]
    bbox: BBox
    has_header: bool
    closed_bottom: bool


@dataclass(frozen=True)
class StampMark:
    """盖章挡住的那一段字。普通识别读不到，盖章感知识别才补得回来。"""

    covered_text: str


@dataclass(frozen=True)
class PageModel:
    """一页的文本层和图像层。文本层空着或占比过低时，走 OCR。"""

    width: float
    height: float
    text_layer_blocks: tuple[TextBlock, ...]
    text_tables: tuple[TableBlock, ...]
    text_layer_ratio: float
    has_text_layer: bool = True
    image_blocks: tuple[TextBlock, ...] = ()
    image_tables: tuple[TableBlock, ...] = ()
    stamps: tuple[StampMark, ...] = ()


@dataclass(frozen=True)
class ReferenceTable:
    """原文表格，用来对行列数和抽查单元格。"""

    row_count: int
    col_count: int
    samples: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    document_ref: str
    form: str


@dataclass(frozen=True)
class SourceDocument:
    """一个来源交进来的文件。对照文本用来算字符保有率，不去重。"""

    identity: SourceIdentity
    pages: tuple[PageModel, ...]
    reference_text: str
    reference_tables: tuple[ReferenceTable, ...] = ()


@dataclass(frozen=True)
class CatalogEntry:
    pages: tuple[PageModel, ...]
    reference_text: str
    reference_tables: tuple[ReferenceTable, ...] = ()


@dataclass(frozen=True)
class RestoredTable:
    rows: tuple[tuple[str, ...], ...]

    @property
    def header(self) -> tuple[str, ...]:
        if not self.rows:
            return ()
        return self.rows[0]


@dataclass(frozen=True)
class AttemptRecord:
    """一次解析尝试。失败原因留在这里，不把文件悄悄丢掉。"""

    strategy: str
    routes: tuple[str, ...]
    ocr_engines: tuple[str, ...]
    retention: float
    table_passed: bool
    failure: str

    @property
    def passed(self) -> bool:
        return self.failure == ""


@dataclass(frozen=True)
class AcceptedDocument:
    """过了闸门、可以交给索引层的一份。同一段文本的另一份来源仍是另一份。"""

    source_id: str
    document_ref: str
    form: str
    normalized_text: str
    tables: tuple[RestoredTable, ...]
    reading_order: tuple[str, ...]
    attempts: tuple[AttemptRecord, ...]


@dataclass(frozen=True)
class QueuedDocument:
    source_id: str
    document_ref: str
    reason: str
    attempt_count: int


@dataclass(frozen=True)
class IngestBatch:
    accepted: tuple[AcceptedDocument, ...]
    queued: tuple[QueuedDocument, ...]

    def ready_for_index(self) -> tuple[AcceptedDocument, ...]:
        """人工队列里的文件不在这份清单里。索引层只该拿走这里的结果。"""

        blocked = {(item.source_id, item.document_ref) for item in self.queued}
        return tuple(
            document
            for document in self.accepted
            if (document.source_id, document.document_ref) not in blocked
        )
