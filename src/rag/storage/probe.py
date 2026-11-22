from dataclasses import dataclass

from rag.storage.elasticsearch_store import ElasticsearchChildTextStore
from rag.storage.metadata_store import PostgresMetadataStore
from rag.storage.milvus_store import MilvusChildVectorStore
from rag.storage.object_store import MinioBlobStore, split_endpoint
from rag.storage.settings import StorageSettings


@dataclass(frozen=True)
class ServiceReport:
    milvus_connected: bool
    elasticsearch_connected: bool
    object_storage_connected: bool
    metadata_connected: bool
    milvus_version: str
    backend: str


def probe_services(settings: StorageSettings) -> ServiceReport:
    """连四个服务各做一次轻量调用。这里不走内存替身。"""
    milvus = MilvusChildVectorStore(settings.milvus_uri)
    milvus.check_connection()
    elastic = ElasticsearchChildTextStore(settings.elasticsearch_url)
    elastic.check_connection()
    host, secure = split_endpoint(settings.object_endpoint)
    blobs = MinioBlobStore(
        host,
        access_key=settings.object_access_key,
        secret_key=settings.object_secret_key,
        bucket=settings.object_bucket,
        secure=secure,
    )
    blobs.check_connection()
    metadata = PostgresMetadataStore(settings.metadata_dsn)
    metadata.check_connection()
    return ServiceReport(
        milvus_connected=True,
        elasticsearch_connected=True,
        object_storage_connected=True,
        metadata_connected=True,
        milvus_version=milvus.server_version(),
        backend="services",
    )
