"""从数据接入的产物取出索引要留下的来源标识。"""

from rag.index.types import SourceRef
from rag.ingest.types import AcceptedDocument


def source_from_accepted(document: AcceptedDocument) -> SourceRef:
    """接入层已经带上的来源。索引按哈希合并正文时，这些标识还要在。"""

    return SourceRef(
        source_id=document.source_id,
        document_ref=document.document_ref,
        form=document.form,
    )
