import pytest

from rag.storage.elasticsearch_store import ElasticsearchChildTextStore
from rag.storage.metadata_store import PostgresMetadataStore
from rag.storage.milvus_store import MilvusChildVectorStore
from rag.storage.object_store import open_parent_store
from rag.storage.settings import load_settings

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def vector_store(settings):
    store = MilvusChildVectorStore(settings.milvus_uri)
    store.ensure_ready(dimension=settings.vector_dimension)
    return store


@pytest.fixture(scope="session")
def lexical_store(settings):
    store = ElasticsearchChildTextStore(settings.elasticsearch_url)
    store.ensure_ready()
    return store


@pytest.fixture(scope="session")
def parent_store(settings):
    return open_parent_store(settings)


@pytest.fixture(scope="session")
def metadata_store(settings):
    store = PostgresMetadataStore(settings.metadata_dsn)
    store.ensure_schema()
    return store
