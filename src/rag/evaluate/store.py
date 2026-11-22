"""评测条目、拒答日志、补录队列和本次报告写进元数据库。"""

from psycopg.types.json import Json

from rag.evaluate.cases import dataset_version
from rag.evaluate.metrics import GateNumbers
from rag.evaluate.params import SUITE_NAME
from rag.evaluate.types import Clause, EvalCase, RefusalRow
from rag.storage.errors import MetadataWriteError, StorageUnavailable
from rag.storage.metadata_store import PostgresMetadataStore
from rag.storage.types import EvaluationEntry

STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS evaluate_refusal (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        run_id text NOT NULL CHECK (run_id <> ''),
        case_key text NOT NULL CHECK (case_key <> ''),
        kind text NOT NULL CHECK (kind IN ('知识缺失', '检索失败', '权限不足')),
        detail text NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS supplement_queue (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        run_id text NOT NULL CHECK (run_id <> ''),
        case_key text NOT NULL CHECK (case_key <> ''),
        status text NOT NULL CHECK (status IN ('queued', 'supplemented')),
        UNIQUE (run_id, case_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluation_report (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        run_id text NOT NULL UNIQUE CHECK (run_id <> ''),
        baseline boolean NOT NULL DEFAULT false,
        payload jsonb NOT NULL
    )
    """,
)

_KINDS = {"知识缺失", "检索失败", "权限不足"}


class EvaluateStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._metadata = PostgresMetadataStore(dsn)

    def ensure_schema(self) -> None:
        self._metadata.ensure_schema()
        with self._connect() as conn:
            for statement in STATEMENTS:
                conn.execute(statement)
            conn.commit()

    def replace_cases(self, cases: tuple[EvalCase, ...]) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM evaluation_entry WHERE suite = %s", (SUITE_NAME,))
                conn.commit()
        except StorageUnavailable:
            raise
        except Exception as exc:
            raise MetadataWriteError("评测条目写不进去，这次回归不能当成已经留下记录") from exc
        for case in cases:
            self._metadata.save_evaluation_entry(
                EvaluationEntry(suite=SUITE_NAME, case_key=case.case_key, payload=case_to_payload(case))
            )

    def load_cases(self) -> tuple[EvalCase, ...]:
        rows = self._metadata.list_evaluation_entries(SUITE_NAME)
        return tuple(case_from_payload(row.case_key, row.payload) for row in rows)

    def save_refusals(self, run_id: str, rows: tuple[RefusalRow, ...]) -> None:
        for row in rows:
            if row.kind not in _KINDS:
                raise MetadataWriteError("拒答日志的类别不对")
            self._write(
                """
                INSERT INTO evaluate_refusal (run_id, case_key, kind, detail)
                VALUES (%s, %s, %s, %s)
                """,
                (run_id, row.case_key, row.kind, row.detail),
                "拒答日志写不进去",
            )
            if row.queue:
                if row.kind != "知识缺失":
                    raise MetadataWriteError("只有知识缺失进入补录队列")
                self._write(
                    """
                    INSERT INTO supplement_queue (run_id, case_key, status)
                    VALUES (%s, %s, 'queued')
                    """,
                    (run_id, row.case_key),
                    "补录队列写不进去",
                )

    def list_refusals(self, run_id: str) -> list[RefusalRow]:
        sql = """
            SELECT case_key, kind, detail
            FROM evaluate_refusal
            WHERE run_id = %s
            ORDER BY id
        """
        with self._connect() as conn:
            found = conn.execute(sql, (run_id,)).fetchall()
        queued = set(self.list_queue(run_id))
        return [RefusalRow(row[0], row[1], row[2], row[0] in queued) for row in found]

    def list_queue(self, run_id: str, status: str = "queued") -> list[str]:
        sql = """
            SELECT case_key
            FROM supplement_queue
            WHERE run_id = %s AND status = %s
            ORDER BY id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (run_id, status)).fetchall()
        return [row[0] for row in rows]

    def mark_supplemented(self, run_id: str, case_key: str) -> None:
        self._write(
            """
            UPDATE supplement_queue
            SET status = 'supplemented'
            WHERE run_id = %s AND case_key = %s AND status = 'queued'
            """,
            (run_id, case_key),
            "补录状态没有写上",
        )

    def save_report(self, run_id: str, payload: dict, *, baseline: bool) -> None:
        self._write(
            """
            INSERT INTO evaluation_report (run_id, baseline, payload)
            VALUES (%s, %s, %s)
            """,
            (run_id, baseline, Json(payload)),
            "评测报告写不进去，这次回归不能当成已经留下记录",
        )

    def load_baseline(self) -> GateNumbers:
        sql = """
            SELECT payload
            FROM evaluation_report
            WHERE baseline = true
            ORDER BY id DESC
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql).fetchone()
        if row is None:
            from rag.evaluate.errors import GateError

            raise GateError("没有基线，这次回归不能当成通过")
        return numbers_from_payload(row[0])

    def _connect(self):
        import psycopg

        try:
            return psycopg.connect(self._dsn, connect_timeout=3)
        except Exception as exc:
            raise StorageUnavailable("metadata") from exc

    def _write(self, sql: str, params: tuple, failure: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(sql, params)
                conn.commit()
        except StorageUnavailable:
            raise
        except MetadataWriteError:
            raise
        except Exception as exc:
            raise MetadataWriteError(failure) from exc


def case_to_payload(case: EvalCase) -> dict:
    return {
        "dataset_version": dataset_version(),
        "category": case.category,
        "boundary": case.boundary,
        "question_type": case.question_type,
        "role": case.role,
        "domain": case.domain,
        "question": case.question,
        "clauses": [
            {
                "clause_id": clause.clause_id,
                "text": clause.text,
                "claim": clause.claim,
                "version": clause.version,
                "effective_status": clause.effective_status,
                "permission": clause.permission,
                "effective_from": clause.effective_from,
                "effective_to": clause.effective_to,
            }
            for clause in case.clauses
        ],
        "ranking": list(case.ranking),
        "gold_ids": list(case.gold_ids),
        "expected_points": list(case.expected_points),
        "forbidden_points": list(case.forbidden_points),
        "expect_refusal": case.expect_refusal,
        "permission_denied": case.permission_denied,
        "retrieval_failed": case.retrieval_failed,
        "rrf_sensitive": case.rrf_sensitive,
        "supplement_clause_id": case.supplement_clause_id,
        "fixed_note": case.fixed_note,
    }


def case_from_payload(case_key: str, payload: dict) -> EvalCase:
    clauses = tuple(
        Clause(
            clause_id=item["clause_id"],
            text=item["text"],
            claim=item["claim"],
            version=item["version"],
            effective_status=item["effective_status"],
            permission=item["permission"],
            effective_from=item["effective_from"],
            effective_to=item["effective_to"],
        )
        for item in payload["clauses"]
    )
    return EvalCase(
        case_key=case_key,
        category=payload["category"],
        boundary=payload.get("boundary", ""),
        question_type=payload["question_type"],
        role=payload["role"],
        domain=payload["domain"],
        question=payload["question"],
        clauses=clauses,
        ranking=tuple(payload["ranking"]),
        gold_ids=tuple(payload["gold_ids"]),
        expected_points=tuple(payload["expected_points"]),
        forbidden_points=tuple(payload.get("forbidden_points") or ()),
        expect_refusal=bool(payload.get("expect_refusal", False)),
        permission_denied=bool(payload.get("permission_denied", False)),
        retrieval_failed=bool(payload.get("retrieval_failed", False)),
        rrf_sensitive=bool(payload.get("rrf_sensitive", False)),
        supplement_clause_id=payload.get("supplement_clause_id") or "",
        fixed_note=payload.get("fixed_note") or "",
    )


def numbers_to_payload(numbers: GateNumbers) -> dict:
    return {
        "recall_at_5": numbers.recall_at_5,
        "hallucination_rate": numbers.hallucination_rate,
        "accuracy": numbers.accuracy,
        "refusal_correctness": numbers.refusal_correctness,
        "citation_correctness": numbers.citation_correctness,
        "over_refusal": numbers.over_refusal,
        "missed_refusal": numbers.missed_refusal,
    }


def numbers_from_payload(payload: dict) -> GateNumbers:
    return GateNumbers(
        recall_at_5=float(payload["recall_at_5"]),
        hallucination_rate=float(payload["hallucination_rate"]),
        accuracy=float(payload["accuracy"]),
        refusal_correctness=float(payload["refusal_correctness"]),
        citation_correctness=float(payload["citation_correctness"]),
        over_refusal=int(payload["over_refusal"]),
        missed_refusal=int(payload["missed_refusal"]),
    )
