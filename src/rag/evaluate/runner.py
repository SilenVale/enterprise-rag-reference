"""按参考配置跑一遍评测集。检索名次、草稿和出口校验都在这次运行里发生。"""

import time

from rag.evaluate.cases import CASES
from rag.evaluate.checklist import ChecklistEntailment, review_answer
from rag.evaluate.metrics import gate_numbers, material_metrics, prompt_metrics
from rag.evaluate.params import (
    REFERENCE_GENERATOR_ID,
    REFERENCE_PROMPT_ID,
    REFERENCE_RRF_K,
    SCORE_FULL,
    SCORE_MISS,
    SCORE_PARTIAL,
)
from rag.evaluate.types import CaseOutcome, EvalCase, RefusalRow, RunConfig
from rag.generate.exit_checks import apply_exit_checks
from rag.generate.params import CAVEAT, REFUSAL_TEXT
from rag.generate.sentences import split_draft, strip_markers
from rag.generate.types import CitedBlock

_ENTAILMENT = ChecklistEntailment()
_MIXED_EXTRA = "另外还有 200 元交通补贴。"
_UNGROUNDED_EXTRA = "额外记一笔 960 元。"


def run_suite(config: RunConfig | None = None, cases: tuple[EvalCase, ...] = CASES) -> "SuiteRun":
    active = config or RunConfig()
    outcomes = tuple(_run_case(case, active) for case in cases)
    refusals = tuple(
        RefusalRow(item.case_key, item.refusal_kind, _detail(item.refusal_kind), item.queue)
        for item in outcomes
        if item.refused
    )
    return SuiteRun(
        config=active,
        outcomes=outcomes,
        material=material_metrics(outcomes),
        prompt=prompt_metrics(outcomes),
        numbers=gate_numbers(outcomes),
        refusals=refusals,
    )


class SuiteRun:
    def __init__(self, *, config, outcomes, material, prompt, numbers, refusals) -> None:
        self.config = config
        self.outcomes = outcomes
        self.material = material
        self.prompt = prompt
        self.numbers = numbers
        self.refusals = refusals

    def outcome(self, case_key: str) -> CaseOutcome:
        for item in self.outcomes:
            if item.case_key == case_key:
                return item
        raise KeyError(case_key)

    @property
    def queued_keys(self) -> tuple[str, ...]:
        return tuple(item.case_key for item in self.refusals if item.queue)


def _detail(kind: str) -> str:
    if kind == "检索失败":
        return "检索引擎没有返回结果"
    if kind == "权限不足":
        return "这个角色看不到这条制度"
    return "现行条款没有进入上下文"


def _run_case(case: EvalCase, config: RunConfig) -> CaseOutcome:
    started = time.perf_counter()
    ranked = _ranked(case, config)
    if case.retrieval_failed or config.fail_retrieval:
        return _finish(
            case,
            ranked_ids=(),
            context_ids=(),
            retrieval_scored=False,
            refused=True,
            refusal_kind="检索失败",
            queue=False,
            shipped_text=REFUSAL_TEXT,
            started=started,
        )
    if case.permission_denied:
        return _finish(
            case,
            ranked_ids=(),
            context_ids=(),
            retrieval_scored=True,
            refused=True,
            refusal_kind="权限不足",
            queue=False,
            shipped_text=REFUSAL_TEXT,
            started=started,
        )

    hits = tuple(ranked[: config.top_k])
    gold_hit = tuple(clause_id for clause_id in case.gold_ids if clause_id in hits)
    score = _score(case.gold_ids, gold_hit)
    if score < config.mid_score or not gold_hit:
        return _finish(
            case,
            ranked_ids=hits,
            context_ids=hits,
            retrieval_scored=True,
            refused=True,
            refusal_kind="知识缺失",
            queue=True,
            shipped_text=REFUSAL_TEXT,
            started=started,
        )

    cited = _cited(case, gold_hit)
    draft = _draft(case, cited, config)
    exit_result = apply_exit_checks(draft, cited, _ENTAILMENT)
    shipped = exit_result.text
    if exit_result.action == "answer" and config.mid_score <= score < config.high_score:
        shipped = shipped + CAVEAT
    factual = _factual_text(shipped)
    findings = review_answer(factual, cited) if exit_result.action == "answer" else ()
    kept = split_draft(factual) if exit_result.action == "answer" else ()
    supported = sum(1 for sentence in kept if _sentence_supported(sentence.text, findings))
    points_hit = sum(1 for point in case.expected_points if point in factual)
    return _finish(
        case,
        ranked_ids=hits,
        context_ids=hits,
        retrieval_scored=True,
        refused=exit_result.action != "answer",
        refusal_kind="知识缺失" if exit_result.action != "answer" else "",
        queue=exit_result.action != "answer",
        shipped_text=shipped if exit_result.action == "answer" else REFUSAL_TEXT,
        started=started,
        entered_exit=True,
        exit_failed=any(item.outcome != "放行" for item in exit_result.decisions),
        checklist_passed=not findings,
        points_hit=points_hit if exit_result.action == "answer" else 0,
        forbidden_hit=any(point in factual for point in case.forbidden_points),
        shipped_sentences=len(kept),
        supported_sentences=supported,
        findings=findings,
    )


def _ranked(case: EvalCase, config: RunConfig) -> list[str]:
    ranked = list(case.ranking)
    if case.case_key in config.supplemented and case.supplement_clause_id:
        if case.supplement_clause_id not in ranked:
            ranked.insert(0, case.supplement_clause_id)
    known = case.clause_map()
    visible = []
    for clause_id in ranked:
        clause = known.get(clause_id)
        if clause is not None and clause.effective_status == "obsolete":
            continue
        if clause is not None and clause.permission == "restricted" and case.role == "实习生":
            continue
        visible.append(clause_id)
    if config.rrf_k != REFERENCE_RRF_K and case.rrf_sensitive and len(visible) > 5:
        moved = visible.pop(4)
        visible.append(moved)
    return visible


def _score(gold: tuple[str, ...], gold_hit: tuple[str, ...]) -> float:
    if not gold or not gold_hit:
        return SCORE_MISS
    if set(gold) <= set(gold_hit):
        return SCORE_FULL
    return SCORE_PARTIAL


def _cited(case: EvalCase, gold_hit: tuple[str, ...]) -> tuple[CitedBlock, ...]:
    known = case.clause_map()
    blocks: list[CitedBlock] = []
    for clause_id in gold_hit:
        clause = known[clause_id]
        blocks.append(
            CitedBlock(
                marker=len(blocks) + 1,
                document_id="synthetic",
                parent_id=clause.clause_id,
                version=clause.version,
                text=clause.text,
                effective_status=clause.effective_status,
                effective_from=clause.effective_from,
                effective_to=clause.effective_to,
            )
        )
    return tuple(blocks)


def _draft(case: EvalCase, cited: tuple[CitedBlock, ...], config: RunConfig) -> str:
    use_markers = config.prompt_id == REFERENCE_PROMPT_ID
    parts: list[str] = []
    for block in cited:
        marker = f"[{block.marker}]" if use_markers else ""
        parts.append(f"{block.text}{marker}")
    if case.case_key == "sample-mixed" and config.generator_id == REFERENCE_GENERATOR_ID and use_markers:
        parts.append(f"{_MIXED_EXTRA}[1]")
    if config.generator_id != REFERENCE_GENERATOR_ID:
        marker = "[1]" if use_markers and cited else ""
        parts.append(f"{_UNGROUNDED_EXTRA}{marker}")
    return "".join(parts)


def _factual_text(shipped: str) -> str:
    if shipped.endswith(CAVEAT):
        return shipped[: -len(CAVEAT)]
    return shipped


def _sentence_supported(sentence: str, findings: tuple) -> bool:
    body = strip_markers(sentence)
    return not any(strip_markers(item.error_point) == body or item.error_point == sentence for item in findings)


def _finish(
    case: EvalCase,
    *,
    ranked_ids: tuple[str, ...] | list[str],
    context_ids: tuple[str, ...] | list[str],
    retrieval_scored: bool,
    refused: bool,
    refusal_kind: str,
    queue: bool,
    shipped_text: str,
    started: float,
    entered_exit: bool = False,
    exit_failed: bool = False,
    checklist_passed: bool = True,
    points_hit: int = 0,
    forbidden_hit: bool = False,
    shipped_sentences: int = 0,
    supported_sentences: int = 0,
    findings: tuple = (),
) -> CaseOutcome:
    latency_ms = (time.perf_counter() - started) * 1000
    return CaseOutcome(
        case_key=case.case_key,
        category=case.category,
        expect_refusal=case.expect_refusal,
        gold_ids=case.gold_ids,
        ranked_ids=tuple(ranked_ids),
        context_ids=tuple(context_ids),
        retrieval_scored=retrieval_scored,
        entered_exit=entered_exit,
        exit_failed=exit_failed,
        refused=refused,
        refusal_kind=refusal_kind if refused else "",
        queue=bool(queue and refused and refusal_kind == "知识缺失"),
        shipped_text=shipped_text,
        checklist_passed=checklist_passed,
        points_hit=points_hit,
        points_total=len(case.expected_points),
        forbidden_hit=forbidden_hit,
        shipped_sentences=shipped_sentences,
        supported_sentences=supported_sentences,
        latency_ms=latency_ms,
        findings=findings,
    )
