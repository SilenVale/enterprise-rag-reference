from psycopg.types.json import Json

from rag.storage.errors import MetadataWriteError, StorageUnavailable
from rag.storage.types import (
    CitationLog,
    CitationRecord,
    EvaluationEntry,
    EvaluationRecord,
    ParseFailure,
    ParseFailureRecord,
)

STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS citation_log (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        request_id text NOT NULL,
        document_id text NOT NULL,
        parent_id text NOT NULL,
        version text NOT NULL,
        citation_marker text NOT NULL CHECK (citation_marker <> ''),
        outcome text NOT NULL CHECK (outcome <> ''),
        detail text NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluation_entry (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        suite text NOT NULL CHECK (suite <> ''),
        case_key text NOT NULL CHECK (case_key <> ''),
        payload jsonb NOT NULL,
        UNIQUE (suite, case_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS parse_failure_queue (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        source_id text NOT NULL CHECK (source_id <> ''),
        document_ref text NOT NULL CHECK (document_ref <> ''),
        reason text NOT NULL CHECK (reason <> ''),
        attempt_count integer NOT NULL CHECK (attempt_count >= 1),
        status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'resolved'))
    )
    """,
)


class PostgresMetadataStore:
    """引用日志、评测条目、解析失败队列。写失败就抛出，不把空结果当成成功。"""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def check_connection(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT 1")

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            for statement in STATEMENTS:
                conn.execute(statement)
            conn.commit()

    def append_citation(self, entry: CitationLog) -> int:
        if not isinstance(entry, CitationLog):
            raise MetadataWriteError("引用日志结构不对，不能当成已经审计")
        sql = """
            INSERT INTO citation_log
                (request_id, document_id, parent_id, version, citation_marker, outcome, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        row = self._write(
            sql,
            (
                entry.request_id,
                entry.document_id,
                entry.parent_id,
                entry.version,
                entry.citation_marker,
                entry.outcome,
                entry.detail,
            ),
            "引用日志写不进去，这次不能当成已经审计",
        )
        return int(row[0])

    def list_citations(self, request_id: str) -> list[CitationRecord]:
        sql = """
            SELECT id, request_id, document_id, parent_id, version, citation_marker, outcome, detail
            FROM citation_log
            WHERE request_id = %s
            ORDER BY id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (request_id,)).fetchall()
        return [
            CitationRecord(
                id=row[0],
                request_id=row[1],
                document_id=row[2],
                parent_id=row[3],
                version=row[4],
                citation_marker=row[5],
                outcome=row[6],
                detail=row[7],
            )
            for row in rows
        ]

    def save_evaluation_entry(self, entry: EvaluationEntry) -> int:
        if not isinstance(entry, EvaluationEntry):
            raise MetadataWriteError("评测条目结构不对，不能当成已经留下记录")
        sql = """
            INSERT INTO evaluation_entry (suite, case_key, payload)
            VALUES (%s, %s, %s)
            ON CONFLICT (suite, case_key)
            DO UPDATE SET payload = EXCLUDED.payload, recorded_at = now()
            RETURNING id
        """
        row = self._write(
            sql,
            (entry.suite, entry.case_key, Json(entry.payload)),
            "评测条目写不进去，这次回归不能当成已经留下记录",
        )
        return int(row[0])

    def list_evaluation_entries(self, suite: str) -> list[EvaluationRecord]:
        sql = """
            SELECT id, suite, case_key, payload
            FROM evaluation_entry
            WHERE suite = %s
            ORDER BY id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (suite,)).fetchall()
        return [
            EvaluationRecord(id=row[0], suite=row[1], case_key=row[2], payload=dict(row[3]))
            for row in rows
        ]

    def enqueue_parse_failure(self, item: ParseFailure) -> int:
        if not isinstance(item, ParseFailure):
            raise MetadataWriteError("解析失败队列结构不对，不能当成已经入库")
        sql = """
            INSERT INTO parse_failure_queue
                (source_id, document_ref, reason, attempt_count, status)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """
        row = self._write(
            sql,
            (item.source_id, item.document_ref, item.reason, item.attempt_count, item.status),
            "解析失败队列写不进去，对应文件不能当成已经入库",
        )
        return int(row[0])

    def list_parse_failures(self, source_id: str) -> list[ParseFailureRecord]:
        sql = """
            SELECT id, source_id, document_ref, reason, attempt_count, status
            FROM parse_failure_queue
            WHERE source_id = %s AND status = 'queued'
            ORDER BY id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (source_id,)).fetchall()
        return [
            ParseFailureRecord(
                id=row[0],
                source_id=row[1],
                document_ref=row[2],
                reason=row[3],
                attempt_count=row[4],
                status=row[5],
            )
            for row in rows
        ]

    def _connect(self):
        import psycopg

        try:
            return psycopg.connect(self._dsn, connect_timeout=3)
        except Exception as exc:
            raise StorageUnavailable("metadata") from exc

    def _write(self, sql: str, params: tuple, failure: str):
        try:
            with self._connect() as conn:
                row = conn.execute(sql, params).fetchone()
                conn.commit()
        except StorageUnavailable:
            raise
        except Exception as exc:
            raise MetadataWriteError(failure) from exc
        if row is None:
            raise MetadataWriteError(failure)
        return row
