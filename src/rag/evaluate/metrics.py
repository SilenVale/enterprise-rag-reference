"""两套指标分开算。拒答正确率不并进准确率的分子。"""

from dataclasses import dataclass

from rag.evaluate.params import RECALL_AT_K
from rag.evaluate.types import CaseOutcome


@dataclass(frozen=True)
class MaterialMetrics:
    """材料正文那一套。"""

    context_recall: float
    context_precision: float
    recall_at_5: float
    faithfulness: float
    answer_relevancy: float
    accuracy: float
    refusal_correctness: float
    citation_correctness: float
    latency_ms: float


@dataclass(frozen=True)
class PromptMetrics:
    """提示词里那一套。名字不并进上一套。"""

    clause_in_context: float
    recall_at_5: float
    citation_supported: float
    accuracy: float
    refusal_correctness: float
    citation_correctness: float
    latency_ms: float


@dataclass(frozen=True)
class GateNumbers:
    recall_at_5: float
    hallucination_rate: float
    accuracy: float
    refusal_correctness: float
    citation_correctness: float
    over_refusal: int
    missed_refusal: int


def _ratio(numerator: float, denominator: int, *, empty: float) -> float:
    if denominator == 0:
        return empty
    return numerator / denominator


def _recall(gold: tuple[str, ...], found: tuple[str, ...]) -> float:
    needed = set(gold)
    if not needed:
        return 1.0
    return len(needed & set(found)) / len(needed)


def recall_at_5(outcome: CaseOutcome) -> float:
    return _recall(outcome.gold_ids, outcome.ranked_ids[:RECALL_AT_K])


def context_recall(outcome: CaseOutcome) -> float:
    return _recall(outcome.gold_ids, outcome.context_ids)


def context_precision(outcome: CaseOutcome) -> float:
    needed = set(outcome.gold_ids)
    if not needed:
        return 1.0
    if not outcome.context_ids:
        return 0.0
    return len(needed & set(outcome.context_ids)) / len(set(outcome.context_ids))


def clause_in_context(outcome: CaseOutcome) -> float:
    needed = set(outcome.gold_ids)
    if not needed:
        return 1.0
    return 1.0 if needed <= set(outcome.context_ids) else 0.0


def _retrieval_pool(outcomes: tuple[CaseOutcome, ...]) -> tuple[CaseOutcome, ...]:
    return tuple(item for item in outcomes if item.retrieval_scored and item.gold_ids)


def answer_correct(outcome: CaseOutcome) -> bool:
    return (
        not outcome.refused
        and outcome.checklist_passed
        and outcome.points_total > 0
        and outcome.points_hit == outcome.points_total
        and not outcome.forbidden_hit
    )


def refusal_correct(outcome: CaseOutcome) -> bool:
    return outcome.refused and outcome.checklist_passed and not outcome.forbidden_hit


def citation_correct(outcome: CaseOutcome) -> bool:
    if outcome.refused:
        return outcome.checklist_passed and not outcome.forbidden_hit
    return outcome.checklist_passed and outcome.shipped_sentences > 0 and (
        outcome.supported_sentences == outcome.shipped_sentences
    )


def faithfulness(outcomes: tuple[CaseOutcome, ...]) -> float:
    shipped = sum(item.shipped_sentences for item in outcomes)
    if shipped == 0:
        return 1.0
    supported = sum(item.supported_sentences for item in outcomes)
    return supported / shipped


def citation_supported(outcomes: tuple[CaseOutcome, ...]) -> float:
    """只在应该作答的条目上算。拒答不记成「已被引用支持」，避免多拒答把这一个数抬高。"""
    pool = tuple(item for item in outcomes if not item.expect_refusal)
    if not pool:
        return 1.0
    hits = 0
    for item in pool:
        if item.refused:
            continue
        hits += int(
            item.checklist_passed
            and item.shipped_sentences > 0
            and item.supported_sentences == item.shipped_sentences
        )
    return hits / len(pool)


def answer_relevancy(outcomes: tuple[CaseOutcome, ...]) -> float:
    pool = tuple(item for item in outcomes if not item.expect_refusal)
    if not pool:
        return 1.0
    total = 0.0
    for item in pool:
        if item.points_total == 0:
            total += 0.0
            continue
        total += item.points_hit / item.points_total
    return total / len(pool)


def accuracy(outcomes: tuple[CaseOutcome, ...]) -> float:
    pool = tuple(item for item in outcomes if not item.expect_refusal)
    return _ratio(sum(1 for item in pool if answer_correct(item)), len(pool), empty=1.0)


def refusal_correctness(outcomes: tuple[CaseOutcome, ...]) -> float:
    pool = tuple(item for item in outcomes if item.expect_refusal)
    return _ratio(sum(1 for item in pool if refusal_correct(item)), len(pool), empty=1.0)


def citation_correctness(outcomes: tuple[CaseOutcome, ...]) -> float:
    return _ratio(sum(1 for item in outcomes if citation_correct(item)), len(outcomes), empty=1.0)


def hallucination_rate(outcomes: tuple[CaseOutcome, ...]) -> float:
    pool = tuple(item for item in outcomes if item.entered_exit)
    return _ratio(sum(1 for item in pool if item.exit_failed), len(pool), empty=0.0)


def mean_latency(outcomes: tuple[CaseOutcome, ...]) -> float:
    if not outcomes:
        return 0.0
    return sum(item.latency_ms for item in outcomes) / len(outcomes)


def _mean(values: tuple[float, ...], *, empty: float) -> float:
    if not values:
        return empty
    return sum(values) / len(values)


def material_metrics(outcomes: tuple[CaseOutcome, ...]) -> MaterialMetrics:
    pool = _retrieval_pool(outcomes)
    return MaterialMetrics(
        context_recall=_mean(tuple(context_recall(item) for item in pool), empty=1.0),
        context_precision=_mean(tuple(context_precision(item) for item in pool), empty=1.0),
        recall_at_5=_mean(tuple(recall_at_5(item) for item in pool), empty=1.0),
        faithfulness=faithfulness(outcomes),
        answer_relevancy=answer_relevancy(outcomes),
        accuracy=accuracy(outcomes),
        refusal_correctness=refusal_correctness(outcomes),
        citation_correctness=citation_correctness(outcomes),
        latency_ms=mean_latency(outcomes),
    )


def prompt_metrics(outcomes: tuple[CaseOutcome, ...]) -> PromptMetrics:
    pool = _retrieval_pool(outcomes)
    return PromptMetrics(
        clause_in_context=_mean(tuple(clause_in_context(item) for item in pool), empty=1.0),
        recall_at_5=_mean(tuple(recall_at_5(item) for item in pool), empty=1.0),
        citation_supported=citation_supported(outcomes),
        accuracy=accuracy(outcomes),
        refusal_correctness=refusal_correctness(outcomes),
        citation_correctness=citation_correctness(outcomes),
        latency_ms=mean_latency(outcomes),
    )


def gate_numbers(outcomes: tuple[CaseOutcome, ...]) -> GateNumbers:
    return GateNumbers(
        recall_at_5=prompt_metrics(outcomes).recall_at_5,
        hallucination_rate=hallucination_rate(outcomes),
        accuracy=accuracy(outcomes),
        refusal_correctness=refusal_correctness(outcomes),
        citation_correctness=citation_correctness(outcomes),
        over_refusal=sum(1 for item in outcomes if not item.expect_refusal and item.refused),
        missed_refusal=sum(1 for item in outcomes if item.expect_refusal and not item.refused),
    )
