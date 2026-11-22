"""索引自己的账：哈希、来源、子块状态、查询指针。写在已经通过的元数据库里。"""

from dataclasses import dataclass
from datetime import date

from psycopg.types.json import Json

from rag.index.errors import IndexingError
from rag.index.types import PolicyDocument, SourceRef
from rag.storage.errors import StorageUnavailable

STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS index_document (
        document_id text NOT NULL,
        version text NOT NULL,
        normalized_hash text NOT NULL,
        status text NOT NULL CHECK (status IN ('pending', 'effective', 'obsolete')),
        effective_from date NOT NULL,
        effective_to date,
        permission text NOT NULL,
        document_type text NOT NULL,
        policy_json jsonb NOT NULL,
        PRIMARY KEY (document_id, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_source (
        normalized_hash text NOT NULL,
        source_id text NOT NULL,
        document_ref text NOT NULL,
        form text NOT NULL,
        document_id text NOT NULL,
        PRIMARY KEY (normalized_hash, source_id, document_ref)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_parent (
        parent_id text PRIMARY KEY,
        document_id text NOT NULL,
        version text NOT NULL,
        content_hash text NOT NULL,
        token_count integer NOT NULL,
        injection_mode text NOT NULL,
        summary text NOT NULL,
        title_path text NOT NULL,
        clause_numbers text NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_child (
        child_id text PRIMARY KEY,
        document_id text NOT NULL,
        version text NOT NULL,
        parent_id text NOT NULL,
        content_hash text NOT NULL,
        clause_number text NOT NULL,
        slot text NOT NULL,
        token_count integer NOT NULL,
        embed_text text NOT NULL,
        body text NOT NULL,
        title_path text NOT NULL,
        kind text NOT NULL,
        status text NOT NULL CHECK (status IN ('pending', 'effective', 'obsolete', 'deleted')),
        vector_reused boolean NOT NULL,
        vector double precision[] NOT NULL,
        UNIQUE (document_id, version, slot)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_event (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        document_id text NOT NULL,
        version text NOT NULL,
        kind text NOT NULL CHECK (kind IN ('change', 'reconcile')),
        detail text NOT NULL,
        recorded_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_query_pointer (
        document_id text PRIMARY KEY,
        version text NOT NULL
    )
    """,
)


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    version: str
    normalized_hash: str
    status: str
    effective_from: date
    effective_to: date | None
    permission: str
    document_type: str


@dataclass(frozen=True)
class StoredChild:
    child_id: str
    document_id: str
    version: str
    parent_id: str
    content_hash: str
    clause_number: str
    slot: str
    token_count: int
    embed_text: str
    body: str
    title_path: str
    kind: str
    status: str
    vector_reused: bool
    vector: tuple[float, ...]


@dataclass(frozen=True)
class StoredParent:
    parent_id: str
    document_id: str
    version: str
    content_hash: str
    token_count: int
    injection_mode: str
    summary: str
    title_path: str
    clause_numbers: tuple[str, ...]


@dataclass(frozen=True)
class StoredEvent:
    kind: str
    version: str
    detail: str


class IndexCatalog:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            for statement in STATEMENTS:
                conn.execute(statement)
            conn.commit()

    def database_name(self) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT current_database()").fetchone()
        if row is None:
            raise IndexingError("元数据库没有返回库名")
        return str(row[0])

    def find_ready_hash(self, normalized_hash: str) -> StoredDocument | None:
        sql = """
            SELECT document_id, version, normalized_hash, status, effective_from, effective_to,
                   permission, document_type
            FROM index_document
            WHERE normalized_hash = %s AND status IN ('pending', 'effective', 'obsolete')
            ORDER BY effective_from, version
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (normalized_hash,)).fetchone()
        return _document(row) if row else None

    def list_sources(self, normalized_hash: str) -> list[SourceRef]:
        sql = """
            SELECT source_id, document_ref, form
            FROM index_source
            WHERE normalized_hash = %s
            ORDER BY source_id, document_ref
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (normalized_hash,)).fetchall()
        return [SourceRef(row[0], row[1], row[2]) for row in rows]

    def save_staged(
        self,
        document: PolicyDocument,
        normalized_hash: str,
        policy_json: dict,
        parents: list[dict],
        children: list[dict],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_document (
                    document_id, version, normalized_hash, status, effective_from, effective_to,
                    permission, document_type, policy_json
                ) VALUES (%s, %s, %s, 'pending', %s, %s, %s, %s, %s)
                """,
                (
                    document.document_id,
                    document.version,
                    normalized_hash,
                    document.effective_from,
                    document.effective_to,
                    document.permission,
                    document.document_type,
                    Json(policy_json),
                ),
            )
            for source in document.sources:
                conn.execute(
                    """
                    INSERT INTO index_source (normalized_hash, source_id, document_ref, form, document_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        normalized_hash,
                        source.source_id,
                        source.document_ref,
                        source.form,
                        document.document_id,
                    ),
                )
            for parent in parents:
                conn.execute(
                    """
                    INSERT INTO index_parent (
                        parent_id, document_id, version, content_hash, token_count,
                        injection_mode, summary, title_path, clause_numbers
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        parent["parent_id"],
                        document.document_id,
                        document.version,
                        parent["content_hash"],
                        parent["token_count"],
                        parent["injection_mode"],
                        parent["summary"],
                        parent["title_path"],
                        parent["clause_numbers"],
                    ),
                )
            for child in children:
                conn.execute(
                    """
                    INSERT INTO index_child (
                        child_id, document_id, version, parent_id, content_hash, clause_number,
                        slot, token_count, embed_text, body, title_path, kind, status,
                        vector_reused, vector
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s
                    )
                    """,
                    (
                        child["child_id"],
                        document.document_id,
                        document.version,
                        child["parent_id"],
                        child["content_hash"],
                        child["clause_number"],
                        child["slot"],
                        child["token_count"],
                        child["embed_text"],
                        child["body"],
                        child["title_path"],
                        child["kind"],
                        child["vector_reused"],
                        list(child["vector"]),
                    ),
                )
            conn.commit()

    def add_sources(self, normalized_hash: str, document_id: str, sources: tuple[SourceRef, ...]) -> None:
        with self._connect() as conn:
            for source in sources:
                conn.execute(
                    """
                    INSERT INTO index_source (normalized_hash, source_id, document_ref, form, document_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (normalized_hash, source.source_id, source.document_ref, source.form, document_id),
                )
            conn.commit()

    def get_document(self, document_id: str, version: str) -> StoredDocument:
        sql = """
            SELECT document_id, version, normalized_hash, status, effective_from, effective_to,
                   permission, document_type
            FROM index_document
            WHERE document_id = %s AND version = %s
        """
        with self._connect() as conn:
            row = conn.execute(sql, (document_id, version)).fetchone()
        if row is None:
            raise IndexingError(f"没有这份版本：{document_id} {version}")
        return _document(row)

    def children(self, document_id: str, version: str) -> list[StoredChild]:
        sql = """
            SELECT child_id, document_id, version, parent_id, content_hash, clause_number, slot,
                   token_count, embed_text, body, title_path, kind, status, vector_reused, vector
            FROM index_child
            WHERE document_id = %s AND version = %s
            ORDER BY slot
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (document_id, version)).fetchall()
        return [_child(row) for row in rows]

    def child(self, child_id: str) -> StoredChild:
        sql = """
            SELECT child_id, document_id, version, parent_id, content_hash, clause_number, slot,
                   token_count, embed_text, body, title_path, kind, status, vector_reused, vector
            FROM index_child
            WHERE child_id = %s
        """
        with self._connect() as conn:
            row = conn.execute(sql, (child_id,)).fetchone()
        if row is None:
            raise IndexingError(f"没有这个子块：{child_id}")
        return _child(row)

    def parent(self, parent_id: str) -> StoredParent:
        sql = """
            SELECT parent_id, document_id, version, content_hash, token_count, injection_mode,
                   summary, title_path, clause_numbers
            FROM index_parent
            WHERE parent_id = %s
        """
        with self._connect() as conn:
            row = conn.execute(sql, (parent_id,)).fetchone()
        if row is None:
            raise IndexingError(f"没有这个父块：{parent_id}")
        numbers = tuple(part for part in str(row[8]).split(",") if part)
        return StoredParent(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], numbers)

    def parents(self, document_id: str, version: str) -> list[StoredParent]:
        sql = """
            SELECT parent_id, document_id, version, content_hash, token_count, injection_mode,
                   summary, title_path, clause_numbers
            FROM index_parent
            WHERE document_id = %s AND version = %s
            ORDER BY parent_id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (document_id, version)).fetchall()
        found = []
        for row in rows:
            numbers = tuple(part for part in str(row[8]).split(",") if part)
            found.append(StoredParent(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], numbers))
        return found

    def load_policy(self, document_id: str, version: str) -> dict:
        sql = "SELECT policy_json FROM index_document WHERE document_id = %s AND version = %s"
        with self._connect() as conn:
            row = conn.execute(sql, (document_id, version)).fetchone()
        if row is None:
            raise IndexingError(f"没有可对账的版本：{document_id} {version}")
        return dict(row[0])

    def point_query(self, document_id: str, version: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_query_pointer (document_id, version)
                VALUES (%s, %s)
                ON CONFLICT (document_id) DO UPDATE SET version = EXCLUDED.version
                """,
                (document_id, version),
            )
            conn.commit()

    def query_version(self, document_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version FROM index_query_pointer WHERE document_id = %s",
                (document_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def apply_activation(
        self,
        document_id: str,
        version: str,
        new_status: str,
        previous: str | None,
        previous_effective_to: date | None,
        disappeared_ids: list[str],
        obsolete_ids: list[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE index_document SET status = %s WHERE document_id = %s AND version = %s",
                (new_status, document_id, version),
            )
            conn.execute(
                "UPDATE index_child SET status = %s WHERE document_id = %s AND version = %s",
                (new_status, document_id, version),
            )
            if previous is not None:
                conn.execute(
                    """
                    UPDATE index_document
                    SET status = 'obsolete', effective_to = %s
                    WHERE document_id = %s AND version = %s
                    """,
                    (previous_effective_to, document_id, previous),
                )
                if disappeared_ids:
                    conn.execute(
                        "UPDATE index_child SET status = 'deleted' WHERE child_id = ANY(%s)",
                        (disappeared_ids,),
                    )
                if obsolete_ids:
                    conn.execute(
                        "UPDATE index_child SET status = 'obsolete' WHERE child_id = ANY(%s)",
                        (obsolete_ids,),
                    )
            conn.execute(
                """
                INSERT INTO index_query_pointer (document_id, version)
                VALUES (%s, %s)
                ON CONFLICT (document_id) DO UPDATE SET version = EXCLUDED.version
                """,
                (document_id, version),
            )
            conn.commit()

    def update_child_embedding(
        self,
        child_id: str,
        *,
        content_hash: str,
        embed_text: str,
        body: str,
        token_count: int,
        vector: tuple[float, ...],
        status: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE index_child
                SET content_hash = %s, embed_text = %s, body = %s, token_count = %s,
                    vector = %s, status = %s, vector_reused = false
                WHERE child_id = %s
                """,
                (content_hash, embed_text, body, token_count, list(vector), status, child_id),
            )
            conn.commit()

    def add_event(self, document_id: str, version: str, kind: str, detail: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_event (document_id, version, kind, detail)
                VALUES (%s, %s, %s, %s)
                """,
                (document_id, version, kind, detail),
            )
            conn.commit()

    def events(self, document_id: str) -> list[StoredEvent]:
        sql = """
            SELECT kind, version, detail
            FROM index_event
            WHERE document_id = %s
            ORDER BY id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (document_id,)).fetchall()
        return [StoredEvent(row[0], row[1], row[2]) for row in rows]

    def _connect(self):
        import psycopg

        try:
            return psycopg.connect(self._dsn, connect_timeout=3)
        except Exception as exc:
            raise StorageUnavailable("metadata") from exc


def _document(row) -> StoredDocument:
    return StoredDocument(
        document_id=row[0],
        version=row[1],
        normalized_hash=row[2],
        status=row[3],
        effective_from=row[4],
        effective_to=row[5],
        permission=row[6],
        document_type=row[7],
    )


def _child(row) -> StoredChild:
    vector = row[14] or []
    return StoredChild(
        child_id=row[0],
        document_id=row[1],
        version=row[2],
        parent_id=row[3],
        content_hash=row[4],
        clause_number=row[5],
        slot=row[6],
        token_count=row[7],
        embed_text=row[8],
        body=row[9],
        title_path=row[10],
        kind=row[11],
        status=row[12],
        vector_reused=bool(row[13]),
        vector=tuple(float(value) for value in vector),
    )
