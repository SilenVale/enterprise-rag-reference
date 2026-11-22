import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StorageSettings:
    """本机四个服务的地址。端口和库名是选定的，材料没有写死。"""

    milvus_uri: str
    elasticsearch_url: str
    object_endpoint: str
    object_access_key: str
    object_secret_key: str
    object_bucket: str
    metadata_dsn: str
    vector_dimension: int = 8


def load_settings() -> StorageSettings:
    return StorageSettings(
        milvus_uri=os.environ.get("RAG_MILVUS_URI", "http://127.0.0.1:19530"),
        elasticsearch_url=os.environ.get("RAG_ELASTICSEARCH_URL", "http://127.0.0.1:9200"),
        object_endpoint=os.environ.get("RAG_OBJECT_ENDPOINT", "http://127.0.0.1:9000"),
        object_access_key=os.environ.get("RAG_OBJECT_ACCESS_KEY", "rag"),
        object_secret_key=os.environ.get("RAG_OBJECT_SECRET_KEY", "rag-storage-secret"),
        object_bucket=os.environ.get("RAG_OBJECT_BUCKET", "rag-parents"),
        metadata_dsn=os.environ.get(
            "RAG_METADATA_DSN",
            "postgresql://rag:rag-storage@127.0.0.1:5432/rag",
        ),
        vector_dimension=int(os.environ.get("RAG_VECTOR_DIMENSION", "8")),
    )
