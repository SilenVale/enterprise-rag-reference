"""系统说明、格式和引用规则放在稳定前缀。检索块和问题放在变化段。"""

import re

from rag.generate.params import MAX_INJECTED_PARENTS
from rag.generate.types import CitedBlock, Prompt

STABLE_PREFIX = (
    "你是制度问答的作答模块。只使用本次给出的条款，不要补充条款之外的事实。\n"
    "格式：每个事实单独成句，用句号收束。\n"
    "引用：事实句末尾标上条款编号，形如 [1]。编号只能来自本次给出的条款。没有编号的事实不要写。"
)

_BLOCK_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def assign_markers(blocks: tuple[CitedBlock, ...] | list[CitedBlock]) -> tuple[CitedBlock, ...]:
    numbered: list[CitedBlock] = []
    for index, block in enumerate(blocks[:MAX_INJECTED_PARENTS], start=1):
        numbered.append(
            CitedBlock(
                marker=index,
                document_id=block.document_id,
                parent_id=block.parent_id,
                version=block.version,
                text=block.text,
                effective_status=block.effective_status,
                effective_from=block.effective_from,
                effective_to=block.effective_to,
            )
        )
    return tuple(numbered)


def render(blocks: tuple[CitedBlock, ...], question: str) -> Prompt:
    lines = [f"[{block.marker}] {block.text}" for block in blocks]
    lines.append(f"问题：{question}")
    return Prompt(prefix=STABLE_PREFIX, variable="\n".join(lines), question=question)


def parse_variable(variable: str) -> tuple[str, tuple[tuple[int, str], ...]]:
    """从变化段读回编号块和问题。本机档只消费这段，不再另接一份正文。"""

    question = ""
    blocks: list[tuple[int, str]] = []
    for raw in variable.splitlines():
        line = raw.strip()
        if line.startswith("问题："):
            question = line.removeprefix("问题：").strip()
            continue
        matched = _BLOCK_LINE.match(line)
        if matched:
            blocks.append((int(matched.group(1)), matched.group(2).strip()))
    return question, tuple(blocks)
