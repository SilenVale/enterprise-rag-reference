"""答案里的数字必须能在所引用块的原文里找到。"""

import re

from rag.generate.sentences import strip_markers

_DATE = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2}")
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*[%％]")
_PLAIN = re.compile(r"(?<!\d)\d+(?:\.\d+)?(?!\d)")


def number_tokens(sentence: str) -> tuple[str, ...]:
    body = strip_markers(sentence)
    tokens: list[str] = []
    for match in _DATE.finditer(body):
        tokens.append(match.group())
    without_dates = _DATE.sub(" ", body)
    for match in _PERCENT.finditer(without_dates):
        tokens.append(match.group().replace(" ", ""))
    without_percents = _PERCENT.sub(" ", without_dates)
    for match in _PLAIN.finditer(without_percents):
        tokens.append(match.group())
    return tuple(tokens)


def missing_numbers(sentence: str, sources: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    blob = "\n".join(sources)
    missing = [token for token in number_tokens(sentence) if not token_in_text(token, blob)]
    return tuple(missing)


def token_in_text(token: str, source: str) -> bool:
    if token.endswith("%") or token.endswith("％"):
        alt = token.replace("%", "％") if token.endswith("%") else token.replace("％", "%")
        return token in source or alt in source
    if "年" in token or "-" in token:
        return token in source
    return re.search(rf"(?<!\d){re.escape(token)}(?!\d)", source) is not None
