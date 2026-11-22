class IngestError(Exception):
    """数据接入没有把这份文件收成可交给索引的文本。"""


class SourceFetchError(IngestError):
    def __init__(self, source_id: str, document_ref: str) -> None:
        self.source_id = source_id
        self.document_ref = document_ref
        super().__init__(f"{source_id} 取不到 {document_ref}")


class SourceExhausted(IngestError):
    def __init__(self, source_id: str, document_ref: str, attempts: int) -> None:
        self.source_id = source_id
        self.document_ref = document_ref
        self.attempts = attempts
        super().__init__(f"{source_id} 取 {document_ref} 失败 {attempts} 次")


class ManualQueueWriteError(IngestError):
    """解析失败队列没写进元数据库。这份文件不能当成已经入库。"""

    def __init__(self, source_id: str, document_ref: str) -> None:
        self.source_id = source_id
        self.document_ref = document_ref
        super().__init__(
            f"{source_id}/{document_ref} 的解析失败没有写进元数据库，不能当成已经入库"
        )
