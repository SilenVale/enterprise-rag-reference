"""检索验收用的子块夹具。不读 corpus/，也不动索引层的集合。"""

import uuid

from rag.storage.hashing import content_hash
from rag.storage.types import ChildText, ChildVector


def new_type() -> str:
    return "t" + uuid.uuid4().hex[:10]


def make_row(
    text: str,
    *,
    document_type: str,
    clause_number: str = "第一条",
    permission: str = "public",
    effective_status: str = "effective",
    child_id: str | None = None,
) -> dict:
    child_id = child_id or ("c" + uuid.uuid4().hex[:16])
    return {
        "child_id": child_id,
        "text": text,
        "parent_id": "p" + child_id[1:],
        "document_id": "d" + child_id[1:9],
        "version": "v1",
        "content_hash": content_hash(text + child_id),
        "clause_number": clause_number,
        "permission": permission,
        "effective_status": effective_status,
        "document_type": document_type,
    }


def index_rows(embedder, vectors, lexical, rows: list[dict]) -> list[str]:
    unique_texts = list(dict.fromkeys(row["text"] for row in rows))
    encoded = dict(zip(unique_texts, embedder.embed_documents(unique_texts), strict=True))
    vector_rows = []
    text_rows = []
    for row in rows:
        vector_rows.append(
            ChildVector(
                child_id=row["child_id"],
                vector=encoded[row["text"]],
                parent_id=row["parent_id"],
                document_id=row["document_id"],
                version=row["version"],
                content_hash=row["content_hash"],
                clause_number=row["clause_number"],
                permission=row["permission"],
                effective_status=row["effective_status"],
                document_type=row["document_type"],
            )
        )
        text_rows.append(
            ChildText(
                child_id=row["child_id"],
                text=row["text"],
                parent_id=row["parent_id"],
                document_id=row["document_id"],
                version=row["version"],
                content_hash=row["content_hash"],
                clause_number=row["clause_number"],
                permission=row["permission"],
                effective_status=row["effective_status"],
                document_type=row["document_type"],
            )
        )
    vectors.upsert_children(vector_rows)
    lexical.upsert_children(text_rows)
    return [row["child_id"] for row in rows]


class TracingEmbedder:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.queries: list[str] = []

    def embed_query(self, text: str):
        self.queries.append(text)
        return self.inner.embed_query(text)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class TracingLexical:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple[str, object, int]] = []

    def search(self, query, *, limit, scalar):
        self.calls.append(("bm25", scalar, limit))
        return self.inner.search(query, limit=limit, scalar=scalar)

    def search_clause(self, clause_number, *, limit, scalar):
        self.calls.append(("clause", scalar, limit))
        return self.inner.search_clause(clause_number, limit=limit, scalar=scalar)

    def list_matching(self, *, limit, scalar):
        self.calls.append(("metadata", scalar, limit))
        return self.inner.list_matching(limit=limit, scalar=scalar)

    def fetch_texts(self, child_ids):
        return self.inner.fetch_texts(child_ids)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class TracingVectors:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple[object, int]] = []

    def search(self, query, *, limit, scalar):
        self.calls.append((scalar, limit))
        return self.inner.search(query, limit=limit, scalar=scalar)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Sleepy:
    """包住一路的引擎调用，用来观察三路是不是一起跑。"""

    def __init__(self, inner, delay: float, method: str) -> None:
        self.inner = inner
        self.delay = delay
        self.method = method

    def __getattr__(self, name):
        target = getattr(self.inner, name)

        def wrapped(*args, **kwargs):
            if name == self.method:
                import time

                time.sleep(self.delay)
            return target(*args, **kwargs)

        return wrapped
