"""门禁只卡本仓库这次的三个数：Recall@5、出口校验失败算出的幻觉率、端到端准确率。"""

from dataclasses import dataclass

from rag.evaluate.errors import GateError
from rag.evaluate.metrics import GateNumbers
from rag.evaluate.params import ACCURACY_DROP_LIMIT, HALLUCINATION_RISE_LIMIT, RECALL_DROP_LIMIT


@dataclass(frozen=True)
class GateDecision:
    blocked: bool
    reasons: tuple[str, ...]
    numbers: GateNumbers
    recall_drop_limit: float
    hallucination_rise_limit: float
    accuracy_drop_limit: float


def compare_to_baseline(baseline: GateNumbers | None, current: GateNumbers) -> GateDecision:
    if baseline is None or current is None:
        raise GateError("没有基线，这次回归不能当成通过")
    reasons: list[str] = []
    if baseline.recall_at_5 - current.recall_at_5 > RECALL_DROP_LIMIT:
        reasons.append("Recall@5 相对基线回退超过阈值")
    if current.hallucination_rate - baseline.hallucination_rate > HALLUCINATION_RISE_LIMIT:
        reasons.append("出口校验失败算出的幻觉率相对基线上升超过阈值")
    if baseline.accuracy - current.accuracy > ACCURACY_DROP_LIMIT:
        reasons.append("端到端准确率相对基线回退超过阈值")
    return GateDecision(
        blocked=bool(reasons),
        reasons=tuple(reasons),
        numbers=current,
        recall_drop_limit=RECALL_DROP_LIMIT,
        hallucination_rise_limit=HALLUCINATION_RISE_LIMIT,
        accuracy_drop_limit=ACCURACY_DROP_LIMIT,
    )
