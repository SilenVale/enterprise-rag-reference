class GenerateError(Exception):
    """生成防御没有按规则把答案放出去。"""


class EntailmentUnavailable(GenerateError):
    """选定的蕴含模型没有加载成功。这一层不能改用字符串全等蒙混过去。"""

    def __init__(self, model_id: str, reason: str) -> None:
        self.model_id = model_id
        super().__init__(f"蕴含模型 {model_id} 加载失败：{reason}")


class EntailmentRejected(GenerateError):
    """字符串全等不能当作蕴含校验。"""


class GenerationFailed(GenerateError):
    """生成档没有给出草稿。调用方仍要走出口校验，不能把未核对的文本放出去。"""


class GenerateAuditError(GenerateError):
    """引用或拒答没有写进元数据库。这次不能当成已经审计。"""
