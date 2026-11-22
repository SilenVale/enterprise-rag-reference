from collections.abc import Sequence

from elasticsearch import Elasticsearch, NotFoundError

from rag.storage.errors import ParentTextNotAllowed, StorageError, StorageUnavailable
from rag.storage.types import ChildText, ScalarFilter, TextHit

MAPPINGS = {
    "properties": {
        "child_id": {"type": "keyword"},
        "text": {"type": "text"},
        "parent_id": {"type": "keyword"},
        "document_id": {"type": "keyword"},
        "version": {"type": "keyword"},
        "content_hash": {"type": "keyword"},
        "clause_number": {"type": "keyword"},
        "permission": {"type": "keyword"},
        "effective_status": {"type": "keyword"},
        "document_type": {"type": "keyword"},
    }
}


class ElasticsearchChildTextStore:
    """子块倒排。相关性用 BM25。标量条件放在 filter context，不进打分。"""

    def __init__(self, url: str, *, index: str = "rag_child_text", timeout: float = 5) -> None:
        self._index = index
        self._client = Elasticsearch(url, request_timeout=timeout)

    def check_connection(self) -> None:
        try:
            if not self._client.ping():
                raise StorageUnavailable("elasticsearch")
        except StorageUnavailable:
            raise
        except Exception as exc:
            raise StorageUnavailable("elasticsearch") from exc

    def ensure_ready(self) -> None:
        self.check_connection()
        if self._client.indices.exists(index=self._index):
            properties = self.mapping_properties()
            if "text" not in properties or properties.get("clause_number", {}).get("type") != "keyword":
                raise StorageError("已有倒排索引和子块映射不一致")
            return
        self._client.indices.create(
            index=self._index,
            settings={"index": {"similarity": {"default": {"type": "BM25"}}}},
            mappings=MAPPINGS,
        )

    def mapping_properties(self) -> dict:
        mapping = self._client.indices.get_mapping(index=self._index)
        body = mapping.body if hasattr(mapping, "body") else mapping
        return body[self._index]["mappings"]["properties"]

    def upsert_children(self, rows: Sequence[object]) -> None:
        for row in rows:
            if not isinstance(row, ChildText):
                raise ParentTextNotAllowed("倒排只接收子块文本，不接收父块全文")
            self._client.index(
                index=self._index,
                id=row.child_id,
                document={
                    "child_id": row.child_id,
                    "text": row.text,
                    "parent_id": row.parent_id,
                    "document_id": row.document_id,
                    "version": row.version,
                    "content_hash": row.content_hash,
                    "clause_number": row.clause_number,
                    "permission": row.permission,
                    "effective_status": row.effective_status,
                    "document_type": row.document_type,
                },
                refresh=False,
            )
        if rows:
            self._client.indices.refresh(index=self._index)

    def delete_children(self, child_ids: Sequence[str]) -> None:
        for child_id in child_ids:
            try:
                self._client.delete(index=self._index, id=child_id, refresh=True)
            except NotFoundError:
                continue

    def search(self, query: str, *, limit: int, scalar: ScalarFilter) -> list[TextHit]:
        filters = _filters(scalar)
        if filters:
            body: dict = {"bool": {"must": [{"match": {"text": query}}], "filter": filters}}
        else:
            body = {"match": {"text": query}}
        return self._search(body, limit)

    def search_clause(self, clause_number: str, *, limit: int, scalar: ScalarFilter) -> list[TextHit]:
        filters = _filters(scalar)
        term = {"term": {"clause_number": clause_number}}
        if filters:
            body: dict = {"bool": {"must": [term], "filter": filters}}
        else:
            body = term
        return self._search(body, limit)

    def list_matching(self, *, limit: int, scalar: ScalarFilter) -> list[TextHit]:
        """只按标量条件取出子块。没有文本查询，不走 BM25 打分。"""

        filters = _filters(scalar)
        if not filters:
            raise StorageError("元数据过滤至少要有一个标量条件")
        if limit < 1:
            raise StorageError("至少取 1 条")
        response = self._client.search(
            index=self._index,
            query={"bool": {"filter": filters}},
            size=limit,
            sort=[{"child_id": "asc"}],
        )
        hits = []
        for row in response["hits"]["hits"]:
            source = row["_source"]
            raw_score = row.get("_score")
            hits.append(
                TextHit(
                    child_id=source["child_id"],
                    parent_id=source["parent_id"],
                    document_id=source["document_id"],
                    version=source["version"],
                    content_hash=source["content_hash"],
                    clause_number=source["clause_number"],
                    score=0.0 if raw_score is None else float(raw_score),
                )
            )
        return hits

    def fetch_texts(self, child_ids: Sequence[str]) -> dict[str, str]:
        if not child_ids:
            return {}
        response = self._client.mget(index=self._index, ids=list(child_ids))
        found: dict[str, str] = {}
        for doc in response["docs"]:
            if doc.get("found") and doc.get("_source"):
                found[doc["_id"]] = doc["_source"]["text"]
        return found

    def _search(self, query: dict, limit: int) -> list[TextHit]:
        if limit < 1:
            raise StorageError("至少取 1 条")
        response = self._client.search(index=self._index, query=query, size=limit)
        hits = []
        for row in response["hits"]["hits"]:
            source = row["_source"]
            hits.append(
                TextHit(
                    child_id=source["child_id"],
                    parent_id=source["parent_id"],
                    document_id=source["document_id"],
                    version=source["version"],
                    content_hash=source["content_hash"],
                    clause_number=source["clause_number"],
                    score=float(row["_score"]),
                )
            )
        return hits


def _filters(scalar: ScalarFilter) -> list[dict]:
    parts = []
    for field, value in (
        ("permission", scalar.permission),
        ("effective_status", scalar.effective_status),
        ("document_type", scalar.document_type),
    ):
        if value is not None:
            parts.append({"term": {field: value}})
    return parts
