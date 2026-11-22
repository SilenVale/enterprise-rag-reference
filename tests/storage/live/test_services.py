import pytest

from rag.storage.elasticsearch_store import ElasticsearchChildTextStore
from rag.storage.errors import StorageUnavailable
from rag.storage.metadata_store import PostgresMetadataStore
from rag.storage.milvus_store import MilvusChildVectorStore
from rag.storage.object_store import MinioBlobStore
from rag.storage.probe import probe_services


def test_four_services_accept_connections(settings, vector_store):
    report = probe_services(settings)
    assert report.milvus_connected
    assert report.elasticsearch_connected
    assert report.object_storage_connected
    assert report.metadata_connected
    assert settings.milvus_uri.startswith("http://")
    assert settings.elasticsearch_url.startswith("http://")
    assert settings.object_endpoint.startswith("http://")
    assert settings.metadata_dsn.startswith("postgresql://")
    assert "milvus" in report.milvus_version.lower()
    assert vector_store.index_type() == "HNSW"
    assert report.backend == "services"


def test_unreachable_endpoint_is_not_a_successful_write(settings):
    with pytest.raises(StorageUnavailable):
        MilvusChildVectorStore("http://127.0.0.1:9").check_connection()
    elastic = ElasticsearchChildTextStore("http://127.0.0.1:9")
    blobs = MinioBlobStore(
        "127.0.0.1:9",
        access_key=settings.object_access_key,
        secret_key=settings.object_secret_key,
        bucket=settings.object_bucket,
        secure=False,
    )
    metadata = PostgresMetadataStore("postgresql://rag:rag-storage@127.0.0.1:9/rag")
    with pytest.raises(StorageUnavailable):
        elastic.check_connection()
    with pytest.raises(StorageUnavailable):
        blobs.check_connection()
    with pytest.raises(StorageUnavailable):
        metadata.check_connection()
