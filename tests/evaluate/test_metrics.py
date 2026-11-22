"""两套指标都要算出来。准确率不把多拒答算成答对。"""

import pytest

from rag.evaluate.metrics import (
    citation_supported,
    context_recall,
    faithfulness,
    recall_at_5,
)
from rag.evaluate.regression import config_for
from rag.evaluate.runner import run_suite
from rag.evaluate.types import CaseOutcome


def _outcome(**kwargs) -> CaseOutcome:
    base = dict(
        case_key="toy",
        category="采样",
        expect_refusal=False,
        gold_ids=(),
        ranked_ids=(),
        context_ids=(),
        retrieval_scored=True,
        entered_exit=False,
        exit_failed=False,
        refused=False,
        refusal_kind="",
        queue=False,
        shipped_text="",
        checklist_passed=True,
        points_hit=0,
        points_total=0,
        forbidden_hit=False,
        shipped_sentences=0,
        supported_sentences=0,
        latency_ms=0.0,
        findings=(),
    )
    base.update(kwargs)
    return CaseOutcome(**base)


def test_上下文比前五条更长时_两个召回不是同一个数():
    outcome = _outcome(
        gold_ids=("a", "b"),
        ranked_ids=("a", "x", "y", "z", "w", "b"),
        context_ids=("a", "x", "y", "z", "w", "b"),
    )
    assert recall_at_5(outcome) == pytest.approx(0.5)
    assert context_recall(outcome) == pytest.approx(1.0)


def test_忠实度按句_引用支持按条():
    outcome = _outcome(
        expect_refusal=False,
        refused=False,
        checklist_passed=False,
        shipped_sentences=2,
        supported_sentences=1,
        points_hit=1,
        points_total=1,
    )
    assert faithfulness((outcome,)) == pytest.approx(0.5)
    assert citation_supported((outcome,)) == pytest.approx(0.0)


def test_参考运行两套指标和三个门禁数都有():
    run = run_suite()
    # 13 条计入检索：11 条命中全部该找条款，1 条只命中一半，1 条一条都没命中。
    assert run.numbers.recall_at_5 == pytest.approx(11.5 / 13)
    assert run.material.context_recall == pytest.approx(run.numbers.recall_at_5)
    assert run.material.context_precision < 1
    assert run.prompt.clause_in_context == pytest.approx(11 / 13)
    assert run.prompt.clause_in_context < run.prompt.recall_at_5
    assert run.prompt.recall_at_5 == pytest.approx(run.numbers.recall_at_5)
    # 进入出口校验的 12 条里，有 1 条草稿被数字校验拦住。
    assert run.numbers.hallucination_rate == pytest.approx(1 / 12)
    assert run.material.faithfulness == pytest.approx(1)
    assert run.prompt.citation_supported == pytest.approx(12 / 13)
    # 应该作答的 13 条里，11 条答对。不该拒却拒了的那条不进准确率的分子。
    assert run.numbers.accuracy == pytest.approx(11 / 13)
    assert run.material.accuracy == pytest.approx(run.numbers.accuracy)
    assert run.prompt.accuracy == pytest.approx(run.numbers.accuracy)
    assert run.numbers.refusal_correctness == pytest.approx(1)
    assert run.numbers.citation_correctness == pytest.approx(1)
    assert run.numbers.over_refusal == 1
    assert run.numbers.missed_refusal == 0
    assert run.material.latency_ms >= 0
    assert run.prompt.latency_ms == pytest.approx(run.material.latency_ms)


def test_全部拒答时准确率下降_拒答正确率单独保持():
    run = run_suite(config_for("route_threshold"))
    assert run.numbers.accuracy == pytest.approx(0)
    assert run.numbers.refusal_correctness == pytest.approx(1)
    assert run.numbers.over_refusal == 13
    assert run.numbers.recall_at_5 == pytest.approx(11.5 / 13)


def test_只命中一半的条目不算答对_出口拦住的错数字不算交卷():
    run = run_suite()
    partial = run.outcome("sample-partial")
    mixed = run.outcome("sample-mixed")
    assert partial.points_hit == 1 and partial.points_total == 2
    assert not partial.refused and not partial.exit_failed
    assert "25" not in partial.shipped_text and "80" in partial.shipped_text
    assert mixed.exit_failed
    assert "200" not in mixed.shipped_text and "640" in mixed.shipped_text
