"""文本层不够用时改走 OCR。

引擎材料没有写死。这里用可替换的区域读取器：普通档会漏掉盖章挡住的字，
盖章感知档把挡住的字补回来。文档未写死。
"""

from rag.ingest.types import PageModel, StampMark, TableBlock, TextBlock


class RegionOcr:
    def __init__(self, name: str, *, stamp_aware: bool) -> None:
        self.name = name
        self.stamp_aware = stamp_aware

    def read(self, page: PageModel) -> tuple[tuple[TextBlock, ...], tuple[TableBlock, ...]]:
        blocks: list[TextBlock] = []
        for block in page.image_blocks:
            text = _uncover(block.text, page.stamps, aware=self.stamp_aware)
            if text.strip():
                blocks.append(TextBlock(text, block.bbox, block.kind))
        tables: list[TableBlock] = []
        for table in page.image_tables:
            rows = tuple(
                tuple(_uncover(cell, page.stamps, aware=self.stamp_aware) for cell in row)
                for row in table.rows
            )
            tables.append(TableBlock(rows, table.bbox, table.has_header, table.closed_bottom))
        return tuple(blocks), tuple(tables)


def _uncover(text: str, stamps: tuple[StampMark, ...], *, aware: bool) -> str:
    if aware:
        return text
    for stamp in stamps:
        text = text.replace(stamp.covered_text, "")
    return text


# 文档未写死：不是材料点名的某一款商用引擎。
STANDARD_OCR = RegionOcr("region-reader", stamp_aware=False)
STAMP_AWARE_OCR = RegionOcr("region-reader-stamp", stamp_aware=True)


def needs_ocr(page: PageModel, text_layer_ratio_floor: float) -> bool:
    """没有文本层，或文本层占比低于选定的下限，就改走 OCR。"""

    if not page.has_text_layer:
        return True
    return page.text_layer_ratio < text_layer_ratio_floor
