"""检索失败和低分拒答记一笔。知识补录队列不在这一层。"""

from rag.storage.errors import MetadataWriteError, StorageUnavailable


class RefusalLog:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def ensure_schema(self) -> None:
        sql = """
            CREATE TABLE IF NOT EXISTS retrieve_refusal (
                id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                recorded_at timestamptz NOT NULL DEFAULT now(),
                request_id text NOT NULL CHECK (request_id <> ''),
                kind text NOT NULL CHECK (kind IN ('检索失败', '低分拒答')),
                detail text NOT NULL DEFAULT ''
            )
        """
        with self._connect() as conn:
            conn.execute(sql)
            conn.commit()

    def append(self, request_id: str, kind: str, detail: str) -> int:
        if kind not in {"检索失败", "低分拒答"}:
            raise MetadataWriteError("拒答日志的类别不对")
        if not request_id:
            raise MetadataWriteError("拒答日志没有请求编号")
        sql = """
            INSERT INTO retrieve_refusal (request_id, kind, detail)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        try:
            with self._connect() as conn:
                row = conn.execute(sql, (request_id, kind, detail)).fetchone()
                conn.commit()
        except StorageUnavailable:
            raise
        except Exception as exc:
            raise MetadataWriteError("拒答日志写不进去") from exc
        if row is None:
            raise MetadataWriteError("拒答日志写不进去")
        return int(row[0])

    def list_for(self, request_id: str) -> list[tuple[str, str]]:
        sql = """
            SELECT kind, detail
            FROM retrieve_refusal
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
