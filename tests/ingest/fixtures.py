from rag.ingest.types import (
    BBox,
    CatalogEntry,
    PageModel,
    SourceDocument,
    SourceIdentity,
    TextBlock,
)


class GuardQueue:
    """成功路径不该写队列。谁写了，测试直接失败。"""

    def enqueue_parse_failure(self, item) -> int:
        raise AssertionError(f"不该进人工队列: {item.source_id}/{item.document_ref} {item.reason}")


def identity(source_id: str, document_ref: str, form: str = "归档") -> SourceIdentity:
    return SourceIdentity(source_id, document_ref, form)


def text_page(*blocks: TextBlock, ratio: float = 1.0, **kwargs) -> PageModel:
    return PageModel(
        width=kwargs.get("width", 500),
        height=kwargs.get("height", 700),
        text_layer_blocks=blocks,
        text_tables=kwargs.get("text_tables", ()),
        text_layer_ratio=ratio,
        has_text_layer=kwargs.get("has_text_layer", True),
        image_blocks=kwargs.get("image_blocks", ()),
        image_tables=kwargs.get("image_tables", ()),
        stamps=kwargs.get("stamps", ()),
    )


def entry_for(page: PageModel, reference_text: str, reference_tables=()) -> CatalogEntry:
    return CatalogEntry((page,), reference_text, reference_tables)


def document_for(
    source_id: str,
    document_ref: str,
    page: PageModel,
    reference_text: str,
    form: str = "归档",
    reference_tables=(),
) -> SourceDocument:
    return SourceDocument(
        identity(source_id, document_ref, form),
        (page,),
        reference_text,
        reference_tables,
    )


def block(text: str, x0: float, y0: float, x1: float, y1: float) -> TextBlock:
    return TextBlock(text, BBox(x0, y0, x1, y1))
