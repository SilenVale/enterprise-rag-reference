"""数据接入：来源、版面、OCR、质量闸门、重试、人工队列。

这一层不嵌入、不检索、不生成。同一段规范化文本从多个来源进来时，来源标识都留着。
"""

from rag.ingest.params import EXAMPLE_RETENTION_THRESHOLD
from rag.ingest.pipeline import ingest_documents, ingest_from_sources
from rag.ingest.sources import seven_source_adapters

__all__ = [
    "EXAMPLE_RETENTION_THRESHOLD",
    "ingest_documents",
    "ingest_from_sources",
    "seven_source_adapters",
]
