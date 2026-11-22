"""评测门禁：评测集、两套指标、可执行清单、回退阻断、拒答补录。

检索和生成的实现不在这一层改。
"""

from rag.evaluate.runner import run_suite

__all__ = ["run_suite"]
