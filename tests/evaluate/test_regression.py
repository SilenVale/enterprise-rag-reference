"""改提示词、topK、分流阈值、RRF 的 k 或生成模型，就分层回归。回退超过自定阈值就阻断。"""

import pytest

from rag.evaluate.errors import GateError
from rag.evaluate.gate import compare_to_baseline
from rag.evaluate.params import ACCURACY_DROP_LIMIT, HALLUCINATION_RISE_LIMIT, RECALL_DROP_LIMIT
from rag.evaluate.regression import config_for, run_regression
from rag.evaluate.runner import run_suite
from rag.evaluate.types import RunConfig


def test_和自己的基线比不会阻断():
    run = run_suite()
    decision = compare_to_baseline(run.numbers, run.numbers)
    assert not decision.blocked
    assert decision.reasons == ()
    assert decision.recall_drop_limit == RECALL_DROP_LIMIT
    assert decision.hallucination_rise_limit == HALLUCINATION_RISE_LIMIT
    assert decision.accuracy_drop_limit == ACCURACY_DROP_LIMIT


def test_没有基线不能当成通过():
    with pytest.raises(GateError):
        compare_to_baseline(None, run_suite().numbers)


@pytest.mark.parametrize(
    ("change", "layer", "reason_piece"),
    [
        ("top_k", "检索", "Recall@5"),
        ("rrf_k", "检索", "Recall@5"),
        ("prompt", "生成", "幻觉率"),
        ("generator", "生成", "幻觉率"),
        ("route_threshold", "端到端", "端到端准确率"),
    ],
)
def test_核心指标回退超过阈值就阻断(change, layer, reason_piece):
    result = run_regression(change)
    assert result.blocked
    assert result.checked == ("检索", "生成", "端到端")
    assert result.first_layer == layer
    assert any(reason_piece in reason for reason in result.reasons)
    assert result.current.refusal_correctness == pytest.approx(1)
    assert result.material.recall_at_5 == pytest.approx(result.prompt.recall_at_5)


def test_分流阈值只挪一点_指标不动就不阻断():
    result = run_regression("route_threshold", RunConfig(high_score=0.80))
    assert not result.blocked
    assert result.current.recall_at_5 == pytest.approx(result.baseline.recall_at_5)
    assert result.current.accuracy == pytest.approx(result.baseline.accuracy)
    assert result.current.hallucination_rate == pytest.approx(result.baseline.hallucination_rate)


def test_触发范围以外的变更直接拒绝():
    with pytest.raises(GateError):
        config_for("embedding")
    with pytest.raises(GateError):
        run_regression("chunk-size")
