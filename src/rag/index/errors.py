class IndexingError(Exception):
    """索引层无法按规则完成这一步。"""


class EmbeddingModelUnavailable(IndexingError):
    """点名的嵌入模型没有加载成功。这一层停住，不换模型。"""

    def __init__(self, model_id: str, reason: str) -> None:
        self.model_id = model_id
        super().__init__(f"嵌入模型 {model_id} 加载失败：{reason}")


class TableHeaderMissing(IndexingError):
    """表头补不回去。退回数据接入的表格闸门，不把破表当段落切进索引。"""

    def __init__(self, clause_number: str) -> None:
        self.clause_number = clause_number
        super().__init__(f"{clause_number} 的表格没有可用表头，不进入索引")


class ChunkBoundError(IndexingError):
    """切出来的块落不进规定的 token 区间。"""
