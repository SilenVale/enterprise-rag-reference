"""引用到废止版本时，由代码拦截，不靠提示词自觉。"""

from rag.generate.types import CitedBlock


def is_obsolete(block: CitedBlock) -> bool:
    return block.effective_status == "obsolete"


def version_detail(block: CitedBlock) -> str:
    end = block.effective_to or "未结束"
    return f"状态 {block.effective_status}，生效区间 {block.effective_from} 至 {end}"
