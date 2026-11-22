class StorageError(Exception):
    """存储层拒绝了这次调用，或者服务没有按约定完成。"""


class StorageUnavailable(StorageError):
    def __init__(self, service: str) -> None:
        self.service = service
        super().__init__(f"{service} 不可用")


class ParentTextNotAllowed(StorageError):
    """向量库和倒排只收子块，不收父块全文。"""


class HashMismatch(StorageError):
    """正文和调用方给出的内容哈希不一致。"""


class ActiveVersionFrozen(StorageError):
    """已经生效的版本不能原地覆盖。"""


class VersionNotValidated(StorageError):
    """新版本还没写全，或者校验还没通过。"""


class VersionSwitchError(StorageError):
    """指针没有切过去，查询仍应读旧版。"""


class OldVersionDeleteBlocked(StorageError):
    """旧版在切换完成前不能删，已留下的版本要还能读到。"""


class MetadataWriteError(StorageError):
    """元数据没有写进去。调用方不能把这次当成已经审计、已经入库或已经留下评测记录。"""
