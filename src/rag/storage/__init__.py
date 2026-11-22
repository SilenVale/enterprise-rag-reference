"""存储层：子块向量、倒排、父块原文和版本、引用日志与队列。

这一层不解析文件，不改写查询，不生成答案。
"""

from rag.storage.settings import StorageSettings, load_settings

__all__ = ["StorageSettings", "load_settings"]
