class EvaluateError(Exception):
    """评测门禁没有按规则留下可回归的结果。"""


class DatasetError(EvaluateError):
    """评测集缺类、缺边界，或把条目写乱了。"""


class GateError(EvaluateError):
    """没有基线，或变更类型不在回归触发范围内。"""


class ReportLeak(EvaluateError):
    """评测报告写进了只允许出现在阅读笔记里的材料数字。"""
