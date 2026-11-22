"""清单逐项判定，并写出错误点、错误类型、支持依据。没有裁判模型。"""

from rag.evaluate.checklist import (
    EXAM_OLD_TEXT,
    EXAM_TEXT,
    HUMAN_REVIEW,
    ChecklistEntailment,
    HumanItem,
    calibrate,
    content_supported,
    gate_authority,
    judge_sentence,
    review_answer,
)
from rag.evaluate.params import JUDGE_MODEL_ID, NUMBER_ERROR, SUPPORT_ERROR, VERSION_ERROR
from rag.generate.types import CitedBlock


def _block(marker: int, text: str, status: str = "effective") -> CitedBlock:
    end = "2024-12-31" if status == "obsolete" else None
    return CitedBlock(
        marker=marker,
        document_id="synthetic",
        parent_id=f"p{marker}",
        version="v1" if status == "obsolete" else "v3",
        text=text,
        effective_status=status,
        effective_from="2024-01-01",
        effective_to=end,
    )


def test_没有裁判模型_门禁以可执行清单为准():
    assert JUDGE_MODEL_ID == ""
    assert gate_authority() == "可执行清单"


def test_数字不在引用原文时写出三个字段():
    finding = judge_sentence("员工年度体检补贴为每人 960 元。[1]", (EXAM_TEXT,), obsolete=False)
    assert finding is not None
    assert finding.error_point
    assert finding.error_type == NUMBER_ERROR
    assert "960" in finding.evidence


def test_引用废止版本单独成一种错误():
    finding = judge_sentence(f"{EXAM_OLD_TEXT}[1]", (EXAM_OLD_TEXT,), obsolete=True, obsolete_detail="状态 obsolete")
    assert finding is not None
    assert finding.error_type == VERSION_ERROR
    assert finding.evidence


def test_句子得不到引用支持():
    finding = judge_sentence("今天天气不错，适合出门。[1]", (EXAM_TEXT,), obsolete=False)
    assert finding is not None
    assert finding.error_type == SUPPORT_ERROR
    assert finding.evidence


def test_改写后的句子不是靠整句全等才算支持():
    paraphrase = "每年三月向员工发放体检补贴，标准是每人 640 元。"
    assert paraphrase != EXAM_TEXT
    assert paraphrase not in EXAM_TEXT
    assert content_supported(paraphrase, EXAM_TEXT)
    judge = ChecklistEntailment()
    assert judge.decides_by_string_equality is False
    assert judge.label(EXAM_TEXT, paraphrase) == "支持"


def test_整段答案会同时看数字_废止和逐句支持():
    blocks = (_block(1, EXAM_TEXT), _block(2, EXAM_OLD_TEXT, "obsolete"))
    findings = review_answer(f"{EXAM_TEXT}[1]{EXAM_OLD_TEXT}[2]", blocks)
    kinds = {item.error_type for item in findings}
    assert VERSION_ERROR in kinds
    assert all(item.error_point and item.error_type and item.evidence for item in findings)


def test_人工对照和清单一致():
    result = calibrate(HUMAN_REVIEW)
    assert result.compared == len(HUMAN_REVIEW) >= 1
    assert result.disagreements == ()
    assert result.agreed == result.compared


def test_对照和人不一致时门禁仍以清单为准():
    flipped = HumanItem("extractive", f"{EXAM_TEXT}[1]", (EXAM_TEXT,), False, False, SUPPORT_ERROR)
    result = calibrate((flipped,))
    assert result.disagreements == ("extractive",)
    assert gate_authority() == "可执行清单"
