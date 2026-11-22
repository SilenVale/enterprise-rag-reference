"""把分块、嵌入、去重、差量和版本切换串起来。

新版本先写完并校验，查询在切换前仍读旧版。更新不先删后插。
"""

from datetime import timedelta

from rag.index.catalog import IndexCatalog, StoredChild, StoredDocument
from rag.index.chunking import build_chunks, canonical_text
from rag.index.embedder import BgeM3Embedder
from rag.index.errors import IndexingError
from rag.index.types import (
    ChildChunk,
    IndexReceipt,
    InjectionText,
    PolicyDocument,
    SourceRef,
)
from rag.storage.errors import VersionSwitchError
from rag.storage.hashing import content_hash
from rag.storage.types import ChildVector, ScalarFilter, VectorHit

COLLECTION = "rag_index_children"


class IndexPipeline:
    """索引工程。向量、父块原文、账本分别走已经接通的三个服务。"""

    def __init__(
        self,
        *,
        embedder: BgeM3Embedder,
        vectors,
        parents,
        catalog: IndexCatalog,
        as_of,
    ) -> None:
        self.embedder = embedder
        self.vectors = vectors
        self.parents = parents
        self.catalog = catalog
        self.as_of = as_of
        self.last_rollback: tuple[str, str] | None = None
        if embedder.dimension:
            vectors.ensure_ready(embedder.dimension)

    def index_document(self, document: PolicyDocument, *, event_kind: str = "") -> IndexReceipt:
        receipt = self.stage(document, event_kind=event_kind)
        if receipt.reused or receipt.activated:
            return receipt
        if document.effective_from <= self.as_of:
            self.activate(document.document_id, document.version)
            return IndexReceipt(
                document_id=receipt.document_id,
                version=receipt.version,
                normalized_hash=receipt.normalized_hash,
                reused=False,
                parent_ids=receipt.parent_ids,
                child_ids=receipt.child_ids,
                embedded_hashes=receipt.embedded_hashes,
                reused_hashes=receipt.reused_hashes,
                disappeared_ids=receipt.disappeared_ids,
                event_kind=receipt.event_kind,
                activated=True,
            )
        return receipt

    def apply_change(self, document: PolicyDocument) -> IndexReceipt:
        """变更事件走增量：只嵌入哈希变了的子块。"""

        return self.index_document(document, event_kind="change")

    def stage(self, document: PolicyDocument, *, event_kind: str = "") -> IndexReceipt:
        _check_identity(document)
        normalized = canonical_text(document)
        digest = content_hash(normalized)
        existing = self.catalog.find_ready_hash(digest)
        if existing is not None:
            self.catalog.add_sources(digest, existing.document_id, document.sources)
            if event_kind:
                self.catalog.add_event(existing.document_id, existing.version, event_kind, "哈希相同，复用已有索引")
            return IndexReceipt(
                document_id=existing.document_id,
                version=existing.version,
                normalized_hash=digest,
                reused=True,
                parent_ids=(),
                child_ids=(),
                embedded_hashes=(),
                reused_hashes=(),
                disappeared_ids=(),
                event_kind=event_kind,
                activated=existing.status == "effective",
            )
        plan = build_chunks(document, self.embedder.count_tokens)
        previous = self.parents.active_version(document.document_id)
        known = self._vectors_by_hash(document.document_id, previous)
        reused: list[ChildChunk] = []
        fresh: list[ChildChunk] = []
        for child in plan.children:
            if child.content_hash in known:
                reused.append(child)
            else:
                fresh.append(child)
        embedded_vectors = self.embedder.embed_documents([child.embed_text for child in fresh])
        vector_by_id: dict[str, tuple[float, ...]] = {}
        reused_flag: dict[str, bool] = {}
        for child, vector in zip(fresh, embedded_vectors, strict=True):
            vector_by_id[child.child_id] = vector
            reused_flag[child.child_id] = False
        for child in reused:
            vector_by_id[child.child_id] = known[child.content_hash]
            reused_flag[child.child_id] = True
        rows = [
            _child_vector(document, child, vector_by_id[child.child_id], "pending")
            for child in plan.children
        ]
        for parent in plan.parents:
            self.parents.write_parent(
                document.document_id,
                document.version,
                parent.parent_id,
                parent.text,
                parent.content_hash,
            )
        self.parents.validate_version(
            document.document_id,
            document.version,
            [(parent.parent_id, parent.content_hash) for parent in plan.parents],
        )
        self.vectors.upsert_children(rows)
        self.catalog.save_staged(
            document,
            digest,
            _policy_json(document),
            [
                {
                    "parent_id": parent.parent_id,
                    "content_hash": parent.content_hash,
                    "token_count": parent.token_count,
                    "injection_mode": parent.injection_mode,
                    "summary": parent.summary,
                    "title_path": parent.title_path,
                    "clause_numbers": ",".join(parent.clause_numbers),
                }
                for parent in plan.parents
            ],
            [
                {
                    "child_id": child.child_id,
                    "parent_id": child.parent_id,
                    "content_hash": child.content_hash,
                    "clause_number": child.clause_number,
                    "slot": child.slot,
                    "token_count": child.token_count,
                    "embed_text": child.embed_text,
                    "body": child.body,
                    "title_path": child.title_path,
                    "kind": child.kind,
                    "vector_reused": reused_flag[child.child_id],
                    "vector": vector_by_id[child.child_id],
                }
                for child in plan.children
            ],
        )
        disappeared = _disappeared(self.catalog, document.document_id, previous, plan.children)
        if event_kind:
            self.catalog.add_event(
                document.document_id,
                document.version,
                event_kind,
                f"嵌入 {len(fresh)} 块，复用 {len(reused)} 块",
            )
        return IndexReceipt(
            document_id=document.document_id,
            version=document.version,
            normalized_hash=digest,
            reused=False,
            parent_ids=tuple(parent.parent_id for parent in plan.parents),
            child_ids=tuple(child.child_id for child in plan.children),
            embedded_hashes=tuple(child.content_hash for child in fresh),
            reused_hashes=tuple(child.content_hash for child in reused),
            disappeared_ids=tuple(child.child_id for child in disappeared),
            event_kind=event_kind,
            activated=False,
        )

    def activate(self, document_id: str, version: str) -> None:
        document = self.catalog.get_document(document_id, version)
        if document.effective_from > self.as_of:
            return
        current = self.parents.active_version(document_id)
        if current == version:
            return
        try:
            self.parents.switch_version(document_id, version)
        except VersionSwitchError:
            if current is not None:
                self.catalog.point_query(document_id, current)
                self.last_rollback = (document_id, current)
            raise
        previous_children = self.catalog.children(document_id, current) if current else []
        new_children = self.catalog.children(document_id, version)
        new_hashes = {child.content_hash for child in new_children}
        disappeared = [child for child in previous_children if child.content_hash not in new_hashes]
        obsolete = [child for child in previous_children if child.content_hash in new_hashes]
        effective_to = None
        if current is not None:
            effective_to = document.effective_from - timedelta(days=1)
        self.catalog.apply_activation(
            document_id,
            version,
            "effective",
            current,
            effective_to,
            [child.child_id for child in disappeared],
            [child.child_id for child in obsolete],
        )
        self._push_status(document, new_children, "effective")
        if current is not None:
            previous = self.catalog.get_document(document_id, current)
            self._push_status(previous, disappeared, "deleted")
            self._push_status(previous, obsolete, "obsolete")

    def reconcile(self, document_id: str) -> IndexReceipt:
        """定时按哈希对账。漂移的块走同一条增量嵌入，不先删后插。"""

        version = self.catalog.query_version(document_id) or self.parents.active_version(document_id)
        if version is None:
            raise IndexingError("还没有可对账的生效版本")
        document = _policy_from_json(self.catalog.load_policy(document_id, version))
        plan = build_chunks(document, self.embedder.count_tokens)
        stored = {child.slot: child for child in self.catalog.children(document_id, version)}
        drifted = [child for child in plan.children if _drifted(stored.get(child.slot), child)]
        vectors = self.embedder.embed_documents([child.embed_text for child in drifted])
        rows: list[ChildVector] = []
        for child, vector in zip(drifted, vectors, strict=True):
            current = stored.get(child.slot)
            child_id = current.child_id if current is not None else child.child_id
            status = current.status if current is not None and current.status != "deleted" else "effective"
            rows.append(_child_vector(document, _with_id(child, child_id), vector, status))
            if current is not None:
                self.catalog.update_child_embedding(
                    child_id,
                    content_hash=child.content_hash,
                    embed_text=child.embed_text,
                    body=child.body,
                    token_count=child.token_count,
                    vector=vector,
                    status=status if status != "deleted" else "effective",
                )
        if rows:
            self.vectors.upsert_children(rows)
        self.catalog.add_event(document_id, version, "reconcile", f"修复 {len(drifted)} 块")
        return IndexReceipt(
            document_id=document_id,
            version=version,
            normalized_hash=content_hash(canonical_text(document)),
            reused=False,
            parent_ids=tuple(parent.parent_id for parent in plan.parents),
            child_ids=tuple(child.child_id for child in drifted),
            embedded_hashes=tuple(child.content_hash for child in drifted),
            reused_hashes=(),
            disappeared_ids=(),
            event_kind="reconcile",
            activated=True,
        )

    def lookup(
        self,
        query: str,
        *,
        document_type: str | None = None,
        limit: int = 5,
        include_obsolete: bool = False,
    ) -> list[VectorHit]:
        """默认只留生效子块。审计可以再带上废止版本。未生效的不在这两条里。"""

        vector = self.embedder.embed_query(query)
        hits = list(
            self.vectors.search(
                vector,
                limit=limit,
                scalar=ScalarFilter(effective_status="effective", document_type=document_type),
            )
        )
        if include_obsolete:
            hits.extend(
                self.vectors.search(
                    vector,
                    limit=limit,
                    scalar=ScalarFilter(effective_status="obsolete", document_type=document_type),
                )
            )
        return hits

    def load_parent(self, document_id: str, parent_id: str):
        return self.parents.read_parent(document_id, parent_id)

    def injection_for(self, child_id: str) -> InjectionText:
        child = self.catalog.child(child_id)
        parent = self.catalog.parent(child.parent_id)
        stored = self.parents.read_parent_version(child.document_id, child.parent_id, child.version)
        if parent.injection_mode == "summary_plus_child":
            text = f"{parent.summary}\n{child.body}"
        else:
            text = stored.text
        return InjectionText(child.parent_id, child.child_id, parent.injection_mode, text)

    def _vectors_by_hash(self, document_id: str, version: str | None) -> dict[str, tuple[float, ...]]:
        if version is None:
            return {}
        found: dict[str, tuple[float, ...]] = {}
        for child in self.catalog.children(document_id, version):
            if child.status == "deleted":
                continue
            found[child.content_hash] = child.vector
        return found

    def _push_status(self, document: StoredDocument, children: list[StoredChild], status: str) -> None:
        if not children:
            return
        rows = []
        for child in children:
            current = self.catalog.child(child.child_id)
            rows.append(
                ChildVector(
                    child_id=current.child_id,
                    vector=current.vector,
                    parent_id=current.parent_id,
                    document_id=current.document_id,
                    version=current.version,
                    content_hash=current.content_hash,
                    clause_number=current.clause_number,
                    permission=document.permission,
                    effective_status=status,
                    document_type=document.document_type,
                )
            )
        self.vectors.upsert_children(rows)


def _check_identity(document: PolicyDocument) -> None:
    if not document.document_id or not document.version or not document.sources:
        raise IndexingError("文档编号、版本和来源都不能空")
    if len(document.document_id) > 40 or len(document.version) > 20:
        raise IndexingError("文档编号或版本号过长，装不进子块主键")
    if "/" in document.document_id or "/" in document.version:
        raise IndexingError("文档编号和版本号不能包含斜杠")


def _child_vector(document: PolicyDocument, child: ChildChunk, vector: tuple[float, ...], status: str) -> ChildVector:
    return ChildVector(
        child_id=child.child_id,
        vector=vector,
        parent_id=child.parent_id,
        document_id=document.document_id,
        version=document.version,
        content_hash=child.content_hash,
        clause_number=child.clause_number,
        permission=document.permission,
        effective_status=status,
        document_type=document.document_type,
    )


def _with_id(child: ChildChunk, child_id: str) -> ChildChunk:
    if child.child_id == child_id:
        return child
    return ChildChunk(
        child_id=child_id,
        parent_id=child.parent_id,
        clause_number=child.clause_number,
        slot=child.slot,
        body=child.body,
        embed_text=child.embed_text,
        token_count=child.token_count,
        content_hash=child.content_hash,
        title_path=child.title_path,
        kind=child.kind,
    )


def _disappeared(catalog: IndexCatalog, document_id: str, version: str | None, children: tuple[ChildChunk, ...]):
    if version is None:
        return []
    new_hashes = {child.content_hash for child in children}
    return [
        child
        for child in catalog.children(document_id, version)
        if child.status != "deleted" and child.content_hash not in new_hashes
    ]


def _drifted(stored: StoredChild | None, fresh: ChildChunk) -> bool:
    if stored is None:
        return True
    return stored.content_hash != fresh.content_hash or stored.status == "deleted"


def _policy_json(document: PolicyDocument) -> dict:
    return {
        "document_id": document.document_id,
        "version": document.version,
        "document_type": document.document_type,
        "permission": document.permission,
        "effective_from": document.effective_from.isoformat(),
        "effective_to": None if document.effective_to is None else document.effective_to.isoformat(),
        "sources": [
            {"source_id": source.source_id, "document_ref": source.document_ref, "form": source.form}
            for source in document.sources
        ],
        "clauses": [
            {
                "number": clause.number,
                "title_path": list(clause.title_path),
                "statement": clause.statement,
                "items": [{"marker": item.marker, "explanation": item.explanation} for item in clause.items],
                "extra": clause.extra,
                "table": None
                if clause.table is None
                else {
                    "header": list(clause.table.header),
                    "rows": [list(row) for row in clause.table.rows],
                },
            }
            for clause in document.clauses
        ],
    }


def _policy_from_json(payload: dict) -> PolicyDocument:
    from datetime import date

    from rag.index.types import Clause, ClauseItem, IndexTable

    clauses = []
    for raw in payload["clauses"]:
        table = None
        if raw["table"] is not None:
            table = IndexTable(
                header=tuple(raw["table"]["header"]),
                rows=tuple(tuple(row) for row in raw["table"]["rows"]),
            )
        clauses.append(
            Clause(
                number=raw["number"],
                title_path=tuple(raw["title_path"]),
                statement=raw["statement"],
                items=tuple(ClauseItem(item["marker"], item["explanation"]) for item in raw["items"]),
                extra=raw["extra"],
                table=table,
            )
        )
    return PolicyDocument(
        document_id=payload["document_id"],
        version=payload["version"],
        document_type=payload["document_type"],
        permission=payload["permission"],
        effective_from=date.fromisoformat(payload["effective_from"]),
        effective_to=None if payload["effective_to"] is None else date.fromisoformat(payload["effective_to"]),
        clauses=tuple(clauses),
        sources=tuple(SourceRef(item["source_id"], item["document_ref"], item["form"]) for item in payload["sources"]),
    )
