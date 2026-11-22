"""把多轮指代收成独立问题，并把复合问题拆开。

改写失败时调用方仍用原查询。这个组件不是生成主模型。
规格材料没有写死，选型见 params。文档未写死。
"""

import re

from rag.retrieve.params import GENERATION_MAIN_MODEL, REWRITE_MODEL_ID
from rag.retrieve.types import RewritePlan, Turn

_CLAUSE = re.compile(
    r"第[0-9０-９一二三四五六七八九十百千零两]+条|[A-Za-z]{1,16}-\d+(?:\.\d+)*"
)
_ANAPHORA = (
    ("那一条", "{a}"),
    ("这一条", "{a}"),
    ("该条", "{a}"),
    ("它的", "{a}的"),
    ("这个规定", "{a}"),
    ("前面说的", "{a}"),
    ("上述", "{a}"),
    ("那个", "{a}"),
    ("这个", "{a}"),
    ("它", "{a}"),
)
_TRAILING = re.compile(r"(写在哪一条|在哪一条|是哪一条|是什么|怎么规定|怎么办|怎么|吗|呢|[？?])+$")
_SPLIT = re.compile(r"以及|及|和|、")


def extract_clauses(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for token in _CLAUSE.findall(text):
        if token not in found:
            found.append(token)
    return tuple(found)


def text_without_clauses(text: str) -> str:
    return " ".join(_CLAUSE.sub(" ", text).split())


class QueryRewriteModel:
    """规则改写。不加载 Qwen2.5-72B，也不占用嵌入或精排模型。"""

    def __init__(self, model_id: str = REWRITE_MODEL_ID) -> None:
        if model_id == GENERATION_MAIN_MODEL:
            raise ValueError("改写不能占用生成主模型")
        self.model_id = model_id

    def rewrite(self, question: str, history: tuple[Turn, ...]) -> RewritePlan:
        standalone = _resolve(question, history)
        base = standalone or question
        subqueries = _split_compound(base)
        if standalone == question:
            standalone = None
        return RewritePlan(
            original=question,
            standalone=standalone,
            subqueries=subqueries,
            model_id=self.model_id,
            failed=False,
        )


def _resolve(question: str, history: tuple[Turn, ...]) -> str | None:
    token = _first_anaphora(question)
    if token is None:
        return None
    topic = ""
    for turn in reversed(history):
        if turn.role == "user" and _first_anaphora(turn.text) is None:
            topic = _topic(turn.text)
            break
    clause = ""
    for turn in reversed(history):
        clauses = extract_clauses(turn.text)
        if clauses:
            clause = clauses[-1]
            break
    antecedent = _antecedent(topic, clause)
    if not antecedent:
        return None
    phrase, pattern = token
    replacement = pattern.format(a=antecedent)
    return question.replace(phrase, replacement, 1)


def _first_anaphora(question: str) -> tuple[str, str] | None:
    earliest: tuple[int, str, str] | None = None
    for phrase, pattern in _ANAPHORA:
        index = question.find(phrase)
        if index < 0:
            continue
        if earliest is None or index < earliest[0]:
            earliest = (index, phrase, pattern)
    if earliest is None:
        return None
    return earliest[1], earliest[2]


def _topic(text: str) -> str:
    cleaned = _TRAILING.sub("", text.strip()).strip(" ，,。")
    return cleaned or text.strip()


def _antecedent(topic: str, clause: str) -> str:
    if clause and clause in topic:
        return topic
    if topic and clause:
        return f"{topic}{clause}"
    return topic or clause


def _split_compound(question: str) -> tuple[str, ...]:
    if "分别" in question:
        head, tail = question.split("分别", 1)
        parts = [part.strip() for part in _SPLIT.split(head) if part.strip()]
    elif "以及" in question:
        parts = [part.strip() for part in question.split("以及") if part.strip()]
        tail = ""
    else:
        return ()
    if len(parts) < 2:
        return ()
    # 材料要求 2 到 3 个。多出来的只留前面三个。文档未写死。
    parts = parts[:3]
    if tail:
        return tuple(f"{part}{tail}" for part in parts)
    return tuple(parts)
