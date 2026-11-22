from datetime import date

import psycopg
import pytest

from rag.index.catalog import IndexCatalog
from rag.index.embedder import BgeM3Embedder
from rag.index.pipeline import COLLECTION, IndexPipeline
from rag.storage.milvus_store import MilvusChildVectorStore
from rag.storage.object_store import open_parent_store
from rag.storage.settings import load_settings

pytestmark = pytest.mark.live

AS_OF = date(2024, 6, 1)


class GuardedVectors:
    """索引更新如果调用删除，测试直接失败。"""

    def __init__(self, inner):
        self.inner = inner
        self.delete_calls: list[list[str]] = []

    def delete_children(self, child_ids):
        self.delete_calls.append(list(child_ids))
        raise RuntimeError("索引层不能先删后插")

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def embedder():
    model = BgeM3Embedder()
    model.load()
    return model


@pytest.fixture(scope="session")
def vector_store(settings, embedder):
    store = MilvusChildVectorStore(settings.milvus_uri, collection=COLLECTION)
    store.ensure_ready(embedder.dimension)
    return store


@pytest.fixture(scope="session")
def guarded_vectors(vector_store):
    return GuardedVectors(vector_store)


@pytest.fixture(scope="session")
def parent_store(settings):
    return open_parent_store(settings)


@pytest.fixture(scope="session")
def catalog(settings):
    store = IndexCatalog(settings.metadata_dsn)
    store.ensure_schema()
    return store


@pytest.fixture(scope="session", autouse=True)
def isolated_index_tables(catalog, settings):
    with psycopg.connect(settings.metadata_dsn) as conn:
        conn.execute(
            """
            TRUNCATE
                index_document,
                index_source,
                index_parent,
                index_child,
                index_event,
                index_query_pointer
            """
        )
        conn.commit()


@pytest.fixture
def pipeline(embedder, guarded_vectors, parent_store, catalog):
    return IndexPipeline(
        embedder=embedder,
        vectors=guarded_vectors,
        parents=parent_store,
        catalog=catalog,
        as_of=AS_OF,
    )
