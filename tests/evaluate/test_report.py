"""评测报告只写这次跑出来的数。"""

from pathlib import Path

from rag.evaluate.gate import compare_to_baseline
from rag.evaluate.report import BANNED_SNIPPETS, assert_report_clean, render_report
from rag.evaluate.runner import run_suite

ROOT = Path(__file__).resolve().parents[2]


def test_报告同时写下两套指标和三个门禁数():
    run = run_suite()
    decision = compare_to_baseline(run.numbers, run.numbers)
    text = render_report(run, decision)
    assert_report_clean(text)
    assert "Context Recall" in text and "Context Precision" in text and "Faithfulness" in text
    assert "Answer Relevancy" in text
    assert "该找的条款进上下文" in text and "答案被引用支持" in text
    assert "出口校验失败算出的幻觉率" in text
    assert "端到端准确率" in text and "拒答正确率" in text and "引用正确率" in text
    assert "文档未写死" in text
    assert "可执行清单" in text
    assert "未使用裁判模型" in text
    assert f"{run.numbers.recall_at_5:.4f}" in text
    assert f"{run.numbers.hallucination_rate:.4f}" in text
    assert f"{run.numbers.accuracy:.4f}" in text
    assert "通过" in text


def test_readme_不写材料里的那些数():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert_report_clean(text)
    assert "面试题对照" not in text
    assert "简历条目" not in text


def test_验收文件若已写好_也不写材料里的那些数():
    path = ROOT / "验收" / "06-评测门禁.md"
    if not path.exists():
        return
    assert_report_clean(path.read_text(encoding="utf-8"))
    for snippet in BANNED_SNIPPETS:
        assert snippet not in path.read_text(encoding="utf-8")
