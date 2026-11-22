"""七个来源适配器。材料只写了七个系统和转发、归档、附件三种形态，没有写系统名字。"""

from collections.abc import Mapping

from rag.ingest.errors import SourceFetchError
from rag.ingest.types import CatalogEntry, SourceDocument, SourceIdentity

Catalog = Mapping[tuple[str, str], CatalogEntry]


class SourceAdapter:
    """把某一个来源里的文件取出来，并带上这个来源的标识。"""

    source_id: str
    form: str

    def __init__(
        self,
        catalog: Catalog,
        *,
        source_id: str | None = None,
        form: str | None = None,
    ) -> None:
        if source_id is not None:
            self.source_id = source_id
        if form is not None:
            self.form = form
        self._catalog = catalog
        self.fetches = 0

    def fetch(self, document_ref: str) -> SourceDocument:
        self.fetches += 1
        entry = self._catalog.get((self.source_id, document_ref))
        if entry is None:
            raise SourceFetchError(self.source_id, document_ref)
        return SourceDocument(
            identity=SourceIdentity(self.source_id, document_ref, self.form),
            pages=entry.pages,
            reference_text=entry.reference_text,
            reference_tables=entry.reference_tables,
        )


class PolicyArchiveAdapter(SourceAdapter):
    source_id = "制度库"
    form = "归档"


class MailForwardAdapter(SourceAdapter):
    source_id = "邮件"
    form = "转发"


class ChatForwardAdapter(SourceAdapter):
    source_id = "即时消息"
    form = "转发"


class DispatchForwardAdapter(SourceAdapter):
    source_id = "公文"
    form = "转发"


class AttachmentAdapter(SourceAdapter):
    source_id = "附件"
    form = "附件"


class ScanArchiveAdapter(SourceAdapter):
    source_id = "扫描库"
    form = "归档"


class SharedFolderAdapter(SourceAdapter):
    source_id = "共享目录"
    form = "归档"


SEVEN_ADAPTERS: tuple[type[SourceAdapter], ...] = (
    PolicyArchiveAdapter,
    MailForwardAdapter,
    ChatForwardAdapter,
    DispatchForwardAdapter,
    AttachmentAdapter,
    ScanArchiveAdapter,
    SharedFolderAdapter,
)


def seven_source_adapters(catalog: Catalog) -> tuple[SourceAdapter, ...]:
    return tuple(adapter_cls(catalog) for adapter_cls in SEVEN_ADAPTERS)
