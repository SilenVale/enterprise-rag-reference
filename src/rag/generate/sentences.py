"""把草稿切成带编号的句子。编号里的数字不当成事实数字。"""

import re

from rag.generate.types import DraftSentence

_MARKER = re.compile(r"\[\d+\]")
_MARKER_BEFORE_STOP = re.compile(r"((?:\[\d+\])+)([。！？])")


def split_draft(draft: str) -> tuple[DraftSentence, ...]:
    normalized = _MARKER_BEFORE_STOP.sub(r"\2\1", draft.strip())
    if not normalized:
        return ()
    sentences: list[DraftSentence] = []
    buffer = ""
    index = 0
    length = len(normalized)
    while index < length:
        char = normalized[index]
        buffer += char
        index += 1
        if char not in "。！？":
            continue
        markers, blob, index = _take_markers(normalized, index)
        text = buffer.strip()
        if text or markers:
            sentences.append(DraftSentence(text + blob, markers))
        buffer = ""
    tail, markers, blob = _suffix_markers(buffer)
    if tail or markers:
        sentences.append(DraftSentence(tail + blob, markers))
    return tuple(sentences)


def _take_markers(text: str, index: int) -> tuple[tuple[int, ...], str, int]:
    markers: list[int] = []
    blob = ""
    while True:
        matched = _MARKER.match(text, index)
        if not matched:
            break
        token = matched.group()
        markers.append(int(token[1:-1]))
        blob += token
        index = matched.end()
    return tuple(markers), blob, index


def _suffix_markers(text: str) -> tuple[str, tuple[int, ...], str]:
    markers: list[int] = []
    blob = ""
    remaining = text.strip()
    while True:
        matched = re.search(r"\[\d+\]$", remaining)
        if matched is None:
            break
        token = matched.group()
        markers.append(int(token[1:-1]))
        blob = token + blob
        remaining = remaining[: matched.start()].strip()
    markers.reverse()
    return remaining, tuple(markers), blob


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text).strip()


def format_markers(markers: tuple[int, ...]) -> str:
    if not markers:
        return "[?]"
    return "".join(f"[{marker}]" for marker in markers)
