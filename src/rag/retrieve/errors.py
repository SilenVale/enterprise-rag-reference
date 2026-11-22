class RetrieveError(Exception):
    """检索链路无法按规则完成这一步。"""


class RerankerUnavailable(RetrieveError):
    """点名的精排模型没有加载成功。这一层停住，不换模型。"""

    def __init__(self, model_id: str, reason: str) -> None:
        self.model_id = model_id
        super().__init__(f"精排模型 {model_id} 加载失败：{reason}")


class RewriteError(RetrieveError):
    """改写组件没有给出可用结果。调用方仍应保留原查询。"""


class RetrievalFailed(RetrieveError):
    """某一路引擎不可用。整次召回作废，不用其余路的结果充数。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
