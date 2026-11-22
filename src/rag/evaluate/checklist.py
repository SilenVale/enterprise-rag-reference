"""可执行清单：数字、废止版本、每句是否被引用支持。

没有裁判模型。清单不靠整句全等，改写后仍可能判成支持。
"""

import re

from rag.evaluate.params import (
    CONTRADICTION_CUES,
    NUMBER_ERROR,
    SUPPORT_BIGRAM_HITS,
    SUPPORT_ERROR,
    VERSION_ERROR,
)
from dataclasses import dataclass

from rag.evaluate.types import Finding
from rag.generate.numbers import missing_numbers
from rag.generate.sentences import split_draft, strip_markers
from rag.generate.types import CitedBlock
from rag.generate.versions import is_obsolete, version_detail

_CJK = re.compile(r"[\u4e00-\u9fff]+")


def gate_authority() -> str:
    return "可执行清单"


def content_bigrams(text: str) -> set[str]:
    grams: set[str] = set()
    for run in _CJK.findall(strip_markers(text)):
        if len(run) < 2:
            continue
        grams.update(run[index : index + 2] for index in range(len(run) - 1))
    return grams


def contradicts(hypothesis: str, premise: str) -> bool:
    return any(cue in hypothesis and cue not in premise for cue in CONTRADICTION_CUES)


def content_supported(hypothesis: str, premise: str) -> bool:
    if contradicts(hypothesis, premise):
        return False
    shared = content_bigrams(hypothesis) & content_bigrams(premise)
    return len(shared) >= SUPPORT_BIGRAM_HITS


class ChecklistEntailment:
    """出口蕴含这一步调用的判定器。不是字符串全等，也不是另一个生成模型。"""

    decides_by_string_equality = False

    def label(self, premise: str, hypothesis: str) -> str:
        if contradicts(hypothesis, premise):
            return "矛盾"
        if content_supported(hypothesis, premise):
            return "支持"
        return "中立"


def judge_sentence(sentence: str, sources: tuple[str, ...], *, obsolete: bool, obsolete_detail: str = "") -> Finding | None:
    point = sentence.strip()
    if obsolete:
        return Finding(point, VERSION_ERROR, obsolete_detail or "这句引用的是废止版本")
    missing = missing_numbers(sentence, sources)
    if missing:
        return Finding(point, NUMBER_ERROR, "引用原文里找不到 " + "、".join(missing))
    premise = "\n".join(sources)
    if not content_supported(strip_markers(sentence), premise):
        snippet = premise.replace("\n", " ")[:40]
        return Finding(point, SUPPORT_ERROR, f"引用原文对不上这句。原文开头是：{snippet}")
    return None


def review_answer(text: str, blocks: tuple[CitedBlock, ...]) -> tuple[Finding, ...]:
    by_marker = {block.marker: block for block in blocks}
    findings: list[Finding] = []
    for sentence in split_draft(text):
        if not sentence.markers or any(marker not in by_marker for marker in sentence.markers):
            findings.append(Finding(sentence.text, SUPPORT_ERROR, "这句没有落到本次引用的编号上"))
            continue
        cited = [by_marker[marker] for marker in sentence.markers]
        obsolete = [block for block in cited if is_obsolete(block)]
        finding = judge_sentence(
            sentence.text,
            tuple(block.text for block in cited),
            obsolete=bool(obsolete),
            obsolete_detail="；".join(version_detail(block) for block in obsolete),
        )
        if finding is not None:
            findings.append(finding)
    return tuple(findings)


@dataclass(frozen=True)
class HumanItem:
    """人工对照的一条。用来看清单和人是否一致，不拿来替换门禁。"""

    item_id: str
    sentence: str
    sources: tuple[str, ...]
    obsolete: bool
    human_supported: bool
    human_error_type: str = ""


@dataclass(frozen=True)
class Calibration:
    compared: int
    agreed: int
    disagreements: tuple[str, ...]


EXAM_TEXT = "员工年度体检补贴为每人 640 元，于每年三月发放。"
EXAM_OLD_TEXT = "员工年度体检补贴为每人 480 元，于每年三月发放。"

HUMAN_REVIEW = (
    HumanItem("extractive", f"{EXAM_TEXT}[1]", (EXAM_TEXT,), False, True),
    HumanItem(
        "paraphrase",
        "每年三月向员工发放体检补贴，标准是每人 640 元。[1]",
        (EXAM_TEXT,),
        False,
        True,
    ),
    HumanItem(
        "wrong-number",
        "员工年度体检补贴为每人 960 元，于每年三月发放。[1]",
        (EXAM_TEXT,),
        False,
        False,
        NUMBER_ERROR,
    ),
    HumanItem("obsolete", f"{EXAM_OLD_TEXT}[1]", (EXAM_OLD_TEXT,), True, False, VERSION_ERROR),
    HumanItem("unsupported", "今天天气不错，适合出门。[1]", (EXAM_TEXT,), False, False, SUPPORT_ERROR),
)


def calibrate(items: tuple[HumanItem, ...] = HUMAN_REVIEW) -> Calibration:
    disagreements: list[str] = []
    for item in items:
        finding = judge_sentence(item.sentence, item.sources, obsolete=item.obsolete)
        checklist_supported = finding is None
        same_label = checklist_supported == item.human_supported
        same_type = finding is None or finding.error_type == item.human_error_type or item.human_supported
        if not same_label or (not item.human_supported and not same_type):
            disagreements.append(item.item_id)
    return Calibration(len(items), len(items) - len(disagreements), tuple(disagreements))
