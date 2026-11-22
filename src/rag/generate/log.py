"""拒答和引用记录写进已经接通的元数据库。写失败就不能当成已经审计。"""

from rag.storage.errors import MetadataWriteError, StorageUnavailable
from rag.storage.metadata_store import PostgresMetadataStore
from rag.storage.types import CitationLog, CitationRecord


class GenerateLog:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._metadata = PostgresMetadataStore(dsn)

    def ensure_schema(self) -> None:
        self._metadata.ensure_schema()
        sql = """
            CREATE TABLE IF NOT EXISTS generate_refusal (
                id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                recorded_at timestamptz NOT NULL DEFAULT now(),
                request_id text NOT NULL CHECK (request_id <> ''),
                kind text NOT NULL CHECK (kind IN ('拒答', '降级')),
                detail text NOT NULL DEFAULT ''
            )
        """
        with self._connect() as conn:
            conn.execute(sql)
            conn.commit()

    def citation(self, entry: CitationLog) -> int:
        return self._metadata.append_citation(entry)

    def list_citations(self, request_id: str) -> list[CitationRecord]:
        return self._metadata.list_citations(request_id)

    def refusal(self, request_id: str, kind: str, detail: str) -> int:
        if kind not in {"拒答", "降级"}:
            raise MetadataWriteError("生成拒答日志的类别不对")
        if not request_id:
            raise MetadataWriteError("生成拒答日志没有请求编号")
        sql = """
            INSERT INTO generate_refusal (request_id, kind, detail)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        try:
            with self._connect() as conn:
                row = conn.execute(sql, (request_id, kind, detail)).fetchone()
                conn.commit()
        except StorageUnavailable:
            raise
        except MetadataWriteError:
            raise
        except Exception as exc:
            raise MetadataWriteError("生成拒答日志写不进去") from exc
        if row is None:
            raise MetadataWriteError("生成拒答日志写不进去")
        return int(row[0])

    def list_refusals(self, request_id: str) -> list[tuple[str, str]]:
        sql = """
            SELECT kind, detail
            FROM generate_refusal
            WHERE request_id = %s
            ORDER BY id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (request_id,)).fetchall()
        return [(row[0], row[1]) for row in rows]

    def _connect(self):
        import psycopg

        try:
            return psycopg.connect(self._dsn, connect_timeout=3)
        except Exception as exc:
            raise StorageUnavailable("metadata") from exc
