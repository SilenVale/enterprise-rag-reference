"""检索链路：改写、三路召回、RRF、精排和三档分流。

这一层不生成答案正文，也不做出口校验。
"""

from rag.retrieve.pipeline import RetrievePipeline
from rag.retrieve.rerank import RERANK_MODEL_ID

__all__ = ["RERANK_MODEL_ID", "RetrievePipeline"]
