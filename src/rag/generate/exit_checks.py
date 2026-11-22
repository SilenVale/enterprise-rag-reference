"""出口三道都在这里执行：数字、废止版本、蕴含。提示词不能把它们关掉。"""

from rag.generate.numbers import missing_numbers
from rag.generate.params import DOWNGRADE_TEXT, EXIT_CHECKS, REFUSAL_TEXT
from rag.generate.readability import remains_readable
from rag.generate.sentences import split_draft, strip_markers
from rag.generate.types import CitedBlock, DraftSentence, ExitResult, SentenceDecision
from rag.generate.versions import is_obsolete, version_detail


def apply_exit_checks(draft: str, blocks: tuple[CitedBlock, ...], judge) -> ExitResult:
    by_marker = {block.marker: block for block in blocks}
    sentences = split_draft(draft)
    checks_called: list[str] = []

    checks_called.append(EXIT_CHECKS[0])
    number_miss = _number_failures(sentences, by_marker)
    checks_called.append(EXIT_CHECKS[1])
    obsolete = _obsolete_markers(sentences, by_marker)
    checks_called.append(EXIT_CHECKS[2])
    unsupported = _entailment_failures(sentences, by_marker, judge)

    decisions: list[SentenceDecision] = []
    for index, sentence in enumerate(sentences):
        decisions.append(_decide(index, sentence, by_marker, number_miss, obsolete, unsupported))

    kept = [item for item in decisions if item.kept]
    removed = [item for item in decisions if not item.kept]
    kept_sentences = tuple(DraftSentence(item.text, item.markers) for item in kept)
    checks = tuple(checks_called)

    if not decisions or not kept or not remains_readable(kept_sentences):
        only_entailment = bool(removed) and all(item.outcome == "蕴含不支持" for item in removed) and bool(kept)
        action = "downgraded" if only_entailment else "refused"
        text = DOWNGRADE_TEXT if action == "downgraded" else REFUSAL_TEXT
        withheld = "降级" if action == "downgraded" else "拒答"
        adjusted = [
            SentenceDecision(item.text, item.markers, withheld, "整条没有按删句后的残稿交卷", False)
            if item.kept
            else item
            for item in decisions
        ]
        return ExitResult(
            action=action,
            text=text,
            decisions=tuple(adjusted),
            entailment_deletion_executed=False,
            checks_called=checks,
        )

    entailment_deletion_executed = any(item.outcome == "蕴含不支持" for item in removed)
    shipped = [
        SentenceDecision(item.text, item.markers, "删句", item.detail, False)
        if item.outcome == "蕴含不支持"
        else item
        for item in decisions
    ]
    return ExitResult(
        action="answer",
        text="".join(item.text for item in kept),
        decisions=tuple(shipped),
        entailment_deletion_executed=entailment_deletion_executed,
        checks_called=checks,
    )


def _number_failures(
    sentences: tuple[DraftSentence, ...],
    by_marker: dict[int, CitedBlock],
) -> dict[int, tuple[str, ...]]:
    found: dict[int, tuple[str, ...]] = {}
    for index, sentence in enumerate(sentences):
        if not _markers_known(sentence, by_marker):
            continue
        missing = missing_numbers(sentence.text, [by_marker[marker].text for marker in sentence.markers])
        if missing:
            found[index] = missing
    return found


def _obsolete_markers(
    sentences: tuple[DraftSentence, ...],
    by_marker: dict[int, CitedBlock],
) -> dict[int, str]:
    found: dict[int, str] = {}
    for index, sentence in enumerate(sentences):
        if not _markers_known(sentence, by_marker):
            continue
        hits = [by_marker[marker] for marker in sentence.markers if is_obsolete(by_marker[marker])]
        if hits:
            found[index] = "；".join(version_detail(block) for block in hits)
    return found


def _entailment_failures(
    sentences: tuple[DraftSentence, ...],
    by_marker: dict[int, CitedBlock],
    judge,
) -> dict[int, str]:
    found: dict[int, str] = {}
    for index, sentence in enumerate(sentences):
        if not _markers_known(sentence, by_marker):
            continue
        premise = "\n".join(by_marker[marker].text for marker in sentence.markers)
        hypothesis = strip_markers(sentence.text)
        label = judge.label(premise, hypothesis)
        if label != "支持":
            found[index] = label
    return found


def _decide(
    index: int,
    sentence: DraftSentence,
    by_marker: dict[int, CitedBlock],
    number_miss: dict[int, tuple[str, ...]],
    obsolete: dict[int, str],
    unsupported: dict[int, str],
) -> SentenceDecision:
    if not _markers_known(sentence, by_marker):
        return SentenceDecision(sentence.text, sentence.markers, "无编号", "事实句没有落到本次注入的编号上", False)
    if index in obsolete:
        return SentenceDecision(sentence.text, sentence.markers, "拦截废止", obsolete[index], False)
    if index in number_miss:
        detail = "引用原文里找不到 " + "、".join(number_miss[index])
        return SentenceDecision(sentence.text, sentence.markers, "拦截数字", detail, False)
    if index in unsupported:
        return SentenceDecision(sentence.text, sentence.markers, "蕴含不支持", unsupported[index], False)
    return SentenceDecision(sentence.text, sentence.markers, "放行", "数字、版本和蕴含都通过", True)


def _markers_known(sentence: DraftSentence, by_marker: dict[int, CitedBlock]) -> bool:
    return bool(sentence.markers) and all(marker in by_marker for marker in sentence.markers)
