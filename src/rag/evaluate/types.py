from dataclasses import dataclass


@dataclass(frozen=True)
class Clause:
    clause_id: str
    text: str
    claim: str
    version: str = "v3"
    effective_status: str = "effective"
    permission: str = "public"
    effective_from: str = "2024-01-01"
    effective_to: str | None = None


@dataclass(frozen=True)
class EvalCase:
    case_key: str
    category: str
    question_type: str
    role: str
    domain: str
    question: str
    clauses: tuple[Clause, ...]
    ranking: tuple[str, ...]
    gold_ids: tuple[str, ...]
    expected_points: tuple[str, ...]
    boundary: str = ""
    forbidden_points: tuple[str, ...] = ()
    expect_refusal: bool = False
    permission_denied: bool = False
    retrieval_failed: bool = False
    rrf_sensitive: bool = False
    supplement_clause_id: str = ""
    fixed_note: str = ""

    def clause_map(self) -> dict[str, Clause]:
        return {clause.clause_id: clause for clause in self.clauses}


@dataclass(frozen=True)
class Finding:
    """裁判写下的结构化字段。没有生成模型时，这些字段由可执行清单填。"""

    error_point: str
    error_type: str
    evidence: str


@dataclass(frozen=True)
class RunConfig:
    top_k: int = 5
    rrf_k: int = 60
    high_score: float = 0.72
    mid_score: float = 0.30
    prompt_id: str = "numbered-v1"
    generator_id: str = "local-extractive"
    supplemented: frozenset[str] = frozenset()
    fail_retrieval: bool = False


@dataclass(frozen=True)
class CaseOutcome:
    case_key: str
    category: str
    expect_refusal: bool
    gold_ids: tuple[str, ...]
    ranked_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    retrieval_scored: bool
    entered_exit: bool
    exit_failed: bool
    refused: bool
    refusal_kind: str
    queue: bool
    shipped_text: str
    checklist_passed: bool
    points_hit: int
    points_total: int
    forbidden_hit: bool
    shipped_sentences: int
    supported_sentences: int
    latency_ms: float
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class RefusalRow:
    case_key: str
    kind: str
    detail: str
    queue: bool
