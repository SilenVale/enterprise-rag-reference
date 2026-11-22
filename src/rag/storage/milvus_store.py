from collections.abc import Sequence

from pymilvus import DataType, MilvusClient

from rag.storage.errors import ParentTextNotAllowed, StorageError, StorageUnavailable
from rag.storage.types import ChildVector, ScalarFilter, VectorHit

VARCHAR_LIMITS = {
    "child_id": 128,
    "parent_id": 128,
    "document_id": 128,
    "version": 64,
    "content_hash": 64,
    "clause_number": 128,
    "permission": 64,
    "effective_status": 32,
    "document_type": 64,
}

OUTPUT_FIELDS = (
    "child_id",
    "parent_id",
    "document_id",
    "version",
    "content_hash",
    "clause_number",
)


class MilvusChildVectorStore:
    """子块向量。索引类型是 HNSW。集合里不设父块正文字段。"""

    def __init__(
        self,
        uri: str,
        *,
        collection: str = "rag_child_vectors",
        m: int = 16,
        ef_construction: int = 64,
        ef: int = 64,
        timeout: float = 3,
    ) -> None:
        self._uri = uri
        self._collection = collection
        self._m = m
        self._ef_construction = ef_construction
        self._ef = ef
        self._dimension: int | None = None
        try:
            self._client = MilvusClient(uri=uri, timeout=timeout)
        except Exception as exc:
            raise StorageUnavailable("milvus") from exc

    def check_connection(self) -> None:
        try:
            self._client.list_collections()
        except Exception as exc:
            raise StorageUnavailable("milvus") from exc

    def server_version(self) -> str:
        try:
            return str(self._client.get_server_version())
        except Exception as exc:
            raise StorageUnavailable("milvus") from exc

    def ensure_ready(self, dimension: int) -> None:
        self.check_connection()
        if dimension < 2:
            raise StorageError("向量维度至少为 2")
        self._dimension = dimension
        if self._client.has_collection(self._collection):
            described = self._client.describe_collection(self._collection)
            if self._vector_dim(described) != dimension or self.index_type() != "HNSW":
                raise StorageError("已有向量集合和子块 HNSW 结构不一致")
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("child_id", DataType.VARCHAR, is_primary=True, max_length=VARCHAR_LIMITS["child_id"])
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        for name, length in VARCHAR_LIMITS.items():
            if name == "child_id":
                continue
            schema.add_field(name, DataType.VARCHAR, max_length=length)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": self._m, "efConstruction": self._ef_construction},
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )

    def index_type(self) -> str:
        return self.describe_vector_index()["index_type"]

    def describe_vector_index(self) -> dict[str, str]:
        info = self._client.describe_index(self._collection, "vector")
        return {"index_type": info["index_type"], "metric_type": info["metric_type"]}

    def field_names(self) -> list[str]:
        described = self._client.describe_collection(self._collection)
        return [field["name"] for field in described["fields"]]

    def varchar_max_lengths(self) -> dict[str, int]:
        described = self._client.describe_collection(self._collection)
        lengths: dict[str, int] = {}
        for field in described["fields"]:
            params = field.get("params") or {}
            if "max_length" in params:
                lengths[field["name"]] = int(params["max_length"])
        return lengths

    def upsert_children(self, rows: Sequence[object]) -> None:
        payload = []
        for row in rows:
            if not isinstance(row, ChildVector):
                raise ParentTextNotAllowed("向量库只接收子块向量，不接收父块全文")
            if self._dimension is not None and len(row.vector) != self._dimension:
                raise StorageError("向量维度和集合不一致")
            payload.append(
                {
                    "child_id": row.child_id,
                    "vector": list(row.vector),
                    "parent_id": row.parent_id,
                    "document_id": row.document_id,
                    "version": row.version,
                    "content_hash": row.content_hash,
                    "clause_number": row.clause_number,
                    "permission": row.permission,
                    "effective_status": row.effective_status,
                    "document_type": row.document_type,
                }
            )
        if not payload:
            return
        self._client.upsert(collection_name=self._collection, data=payload)

    def delete_children(self, child_ids: Sequence[str]) -> None:
        if not child_ids:
            return
        self._client.delete(collection_name=self._collection, ids=list(child_ids))

    def search(self, query: Sequence[float], *, limit: int, scalar: ScalarFilter) -> list[VectorHit]:
        if limit < 1:
            raise StorageError("至少取 1 条")
        arguments: dict = {
            "collection_name": self._collection,
            "data": [list(query)],
            "limit": limit,
            "output_fields": list(OUTPUT_FIELDS),
            "search_params": {"metric_type": "COSINE", "params": {"ef": self._ef}},
        }
        expression = _filter_expr(scalar)
        if expression:
            arguments["filter"] = expression
        found = self._client.search(**arguments)
        hits: list[VectorHit] = []
        for row in found[0]:
            entity = row.get("entity") or {}
            hits.append(
                VectorHit(
                    child_id=entity.get("child_id", row.get("child_id")),
                    parent_id=entity["parent_id"],
                    document_id=entity["document_id"],
                    version=entity["version"],
                    content_hash=entity["content_hash"],
                    clause_number=entity["clause_number"],
                    score=float(row["distance"]),
                )
            )
        return hits

    @staticmethod
    def _vector_dim(described: dict) -> int:
        for field in described["fields"]:
            if field["name"] == "vector":
                return int(field["params"]["dim"])
        raise StorageError("集合里没有向量字段")


def _filter_expr(scalar: ScalarFilter) -> str | None:
    parts: list[str] = []
    for field, value in (
        ("permission", scalar.permission),
        ("effective_status", scalar.effective_status),
        ("document_type", scalar.document_type),
    ):
        if value is None:
            continue
        if any(char in value for char in '"\\'):
            raise StorageError("过滤值不能包含引号")
        parts.append(f'{field} == "{value}"')
    if not parts:
        return None
    return " and ".join(parts)
