import pytest

from rag.index.embedder import BgeM3Embedder
from rag.retrieve.params import TEXT_INDEX, VECTOR_COLLECTION
from rag.retrieve.refusal import RefusalLog
from rag.retrieve.rerank import BgeReranker
from rag.storage.elasticsearch_store import ElasticsearchChildTextStore
from rag.storage.milvus_store import MilvusChildVectorStore
from rag.storage.settings import load_settings

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def embedder():
    model = BgeM3Embedder()
    model.load()
    return model


@pytest.fixture(scope="session")
def reranker():
    model = BgeReranker()
    model.load()
    return model


@pytest.fixture(scope="session")
def vectors(settings, embedder):
    store = MilvusChildVectorStore(settings.milvus_uri, collection=VECTOR_COLLECTION)
    store.ensure_ready(embedder.dimension)
    return store


@pytest.fixture(scope="session")
def lexical(settings):
    store = ElasticsearchChildTextStore(settings.elasticsearch_url, index=TEXT_INDEX)
    store.ensure_ready()
    return store


@pytest.fixture(scope="session")
def refusals(settings):
    log = RefusalLog(settings.metadata_dsn)
    log.ensure_schema()
    return log
