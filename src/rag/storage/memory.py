from rag.storage.object_store import BlobObject


class MemoryBlobStore:
    """进程内的对象替身，只给单元测试用，不能当作存储层已经接通。"""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put(self, key: str, body: bytes, metadata: dict | None = None) -> None:
        self._objects[key] = bytes(body)

    def get(self, key: str) -> BlobObject | None:
        body = self._objects.get(key)
        if body is None:
            return None
        return BlobObject(body=body, metadata={})

    def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(key for key in self._objects if key.startswith(prefix))
