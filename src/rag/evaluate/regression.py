"""改提示词、topK、分流阈值、RRF 的 k 或生成模型时，分层重跑并和基线比。"""

from dataclasses import dataclass, replace

from rag.evaluate.errors import GateError
from rag.evaluate.gate import GateDecision, compare_to_baseline
from rag.evaluate.metrics import GateNumbers, MaterialMetrics, PromptMetrics
from rag.evaluate.params import TRIGGERS
from rag.evaluate.runner import SuiteRun, run_suite
from rag.evaluate.types import RunConfig


@dataclass(frozen=True)
class RegressionResult:
    change: str
    blocked: bool
    reasons: tuple[str, ...]
    decision: GateDecision
    baseline: GateNumbers
    current: GateNumbers
    material: MaterialMetrics
    prompt: PromptMetrics
    checked: tuple[str, ...]
    first_layer: str


def config_for(change: str) -> RunConfig:
    if change not in TRIGGERS:
        raise GateError("这个变更不在分层回归的触发范围里")
    base = RunConfig()
    if change == "prompt":
        return replace(base, prompt_id="loose-v2")
    if change == "top_k":
        return replace(base, top_k=1)
    if change == "route_threshold":
        return replace(base, high_score=0.99, mid_score=0.95)
    if change == "rrf_k":
        return replace(base, rrf_k=20)
    return replace(base, generator_id="ungrounded")


def run_regression(change: str, config: RunConfig | None = None) -> RegressionResult:
    if change not in TRIGGERS:
        raise GateError("这个变更不在分层回归的触发范围里")
    baseline_run = run_suite(RunConfig())
    current_run = run_suite(config if config is not None else config_for(change))
    decision = compare_to_baseline(baseline_run.numbers, current_run.numbers)
    return RegressionResult(
        change=change,
        blocked=decision.blocked,
        reasons=decision.reasons,
        decision=decision,
        baseline=baseline_run.numbers,
        current=current_run.numbers,
        material=current_run.material,
        prompt=current_run.prompt,
        checked=("检索", "生成", "端到端"),
        first_layer=_first_layer(baseline_run, current_run),
    )


def _first_layer(baseline: SuiteRun, current: SuiteRun) -> str:
    retrieval_moved = (
        abs(current.material.recall_at_5 - baseline.material.recall_at_5) > 1e-9
        or abs(current.material.context_recall - baseline.material.context_recall) > 1e-9
        or abs(current.prompt.clause_in_context - baseline.prompt.clause_in_context) > 1e-9
    )
    if retrieval_moved:
        return "检索"
    generation_worse = current.numbers.hallucination_rate - baseline.numbers.hallucination_rate > 1e-9
    faithfulness_moved = abs(current.material.faithfulness - baseline.material.faithfulness) > 1e-9
    if generation_worse or faithfulness_moved:
        return "生成"
    return "端到端"
