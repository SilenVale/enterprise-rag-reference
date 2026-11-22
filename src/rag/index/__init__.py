"""索引工程：父子分块、BGE-M3、内容哈希、增量和废止切换。

这一层不改写查询，不做三路召回，也不生成答案。
"""

from rag.index.embedder import MODEL_ID
from rag.index.pipeline import COLLECTION, IndexPipeline

__all__ = ["COLLECTION", "MODEL_ID", "IndexPipeline"]
