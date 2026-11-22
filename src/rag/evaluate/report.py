"""评测报告只写这次运行算出来的数。"""

from rag.evaluate.cases import dataset_version
from rag.evaluate.checklist import HUMAN_REVIEW, calibrate, gate_authority
from rag.evaluate.errors import ReportLeak
from rag.evaluate.gate import GateDecision
from rag.evaluate.params import JUDGE_MODEL_ID
from rag.evaluate.runner import SuiteRun

# 这些只允许出现在阅读笔记里。报告、README、演示都不能带上。
BANNED_SNIPPETS = (
    "71.2%",
    "93.4%",
    "13.8%",
    "2.1%",
    "旧版制度误用事故清零",
    "18.6万",
    "18.6 万",
    "周增 800",
    "周增800",
    "2800",
    "20 分钟到 30 秒",
    "20分钟到30秒",
    "30 秒",
)


def assert_report_clean(text: str) -> None:
    for snippet in BANNED_SNIPPETS:
        if snippet in text:
            raise ReportLeak(snippet)


def render_report(run: SuiteRun, decision: GateDecision) -> str:
    counts = {name: sum(1 for item in run.outcomes if item.category == name) for name in ("采样", "边界", "badcase")}
    calibration = calibrate(HUMAN_REVIEW)
    numbers = run.numbers
    material = run.material
    prompt = run.prompt
    status = "阻断" if decision.blocked else "通过"
    reasons = "；".join(decision.reasons) if decision.reasons else "三个门禁指标都在阈值内"
    judge = "未使用裁判模型" if not JUDGE_MODEL_ID else JUDGE_MODEL_ID
    text = "\n".join(
        [
            "评测报告",
            f"评测集版本：{dataset_version()}。条目合计 {len(run.outcomes)}。采样 {counts['采样']}，边界 {counts['边界']}，已修复 badcase {counts['badcase']}。",
            "门禁以本次运行的可执行清单为准。",
            f"裁判：{judge}。门禁权威是{gate_authority()}。",
            f"人工对照 {calibration.compared} 条，与清单一致 {calibration.agreed} 条。对照清单不替换门禁。",
            "材料正文那一套：",
            f"Context Recall {_fmt(material.context_recall)}，Context Precision {_fmt(material.context_precision)}，Recall@5 {_fmt(material.recall_at_5)}。",
            f"Faithfulness {_fmt(material.faithfulness)}，Answer Relevancy {_fmt(material.answer_relevancy)}。",
            f"答案准确率 {_fmt(material.accuracy)}，拒答正确率 {_fmt(material.refusal_correctness)}，引用正确率 {_fmt(material.citation_correctness)}，耗时 {_fmt(material.latency_ms)} 毫秒。",
            "提示词那一套：",
            f"该找的条款进上下文 {_fmt(prompt.clause_in_context)}，Recall@5 {_fmt(prompt.recall_at_5)}。",
            f"答案被引用支持 {_fmt(prompt.citation_supported)}。",
            f"准确率 {_fmt(prompt.accuracy)}，拒答正确率 {_fmt(prompt.refusal_correctness)}，引用正确率 {_fmt(prompt.citation_correctness)}，耗时 {_fmt(prompt.latency_ms)} 毫秒。",
            "门禁卡住的三个数：",
            f"Recall@5 {_fmt(numbers.recall_at_5)}。",
            f"出口校验失败算出的幻觉率 {_fmt(numbers.hallucination_rate)}。",
            f"端到端准确率 {_fmt(numbers.accuracy)}。",
            f"同时报告：拒答正确率 {_fmt(numbers.refusal_correctness)}，引用正确率 {_fmt(numbers.citation_correctness)}。",
            f"不该拒却拒了 {numbers.over_refusal} 条。该拒却答了 {numbers.missed_refusal} 条。",
            f"回退阈值：Recall@5 下降 {decision.recall_drop_limit}，幻觉率上升 {decision.hallucination_rise_limit}，端到端准确率下降 {decision.accuracy_drop_limit}。文档未写死。",
            f"门禁结果：{status}。{reasons}。",
        ]
    )
    assert_report_clean(text)
    return text


def _fmt(value: float) -> str:
    return f"{value:.4f}"
