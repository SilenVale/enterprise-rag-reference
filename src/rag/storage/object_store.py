import io
import json
from dataclasses import dataclass

import urllib3
from minio import Minio
from minio.error import S3Error

from rag.storage.errors import (
    ActiveVersionFrozen,
    HashMismatch,
    OldVersionDeleteBlocked,
    StorageError,
    StorageUnavailable,
    VersionNotValidated,
    VersionSwitchError,
)
from rag.storage.hashing import content_hash
from rag.storage.settings import StorageSettings
from rag.storage.types import ParentRecord


@dataclass(frozen=True)
class BlobObject:
    body: bytes
    metadata: dict[str, str]


class MinioBlobStore:
    """本机对象存储。父块正文和版本指针都放在这里。"""

    def __init__(self, endpoint: str, *, access_key: str, secret_key: str, bucket: str, secure: bool) -> None:
        self._bucket = bucket
        http = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=2.0, read=10.0),
            retries=False,
        )
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            http_client=http,
        )

    def check_connection(self) -> None:
        try:
            self._client.list_buckets()
        except Exception as exc:
            raise StorageUnavailable("object_storage") from exc

    def ensure_bucket(self) -> None:
        self.check_connection()
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def put(self, key: str, body: bytes, metadata: dict | None = None) -> None:
        self._client.put_object(
            self._bucket,
            key,
            io.BytesIO(body),
            length=len(body),
        )

    def get(self, key: str) -> BlobObject | None:
        try:
            response = self._client.get_object(self._bucket, key)
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject"}:
                return None
            raise
        try:
            body = response.read()
        finally:
            response.close()
            response.release_conn()
        return BlobObject(body=body, metadata={})

    def delete(self, key: str) -> None:
        self._client.remove_object(self._bucket, key)

    def list_keys(self, prefix: str) -> list[str]:
        return [
            obj.object_name
            for obj in self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
            if obj.object_name is not None
        ]


class VersionedParentStore:
    """父块原文按版本存放。查询读指针；指针只在新版本校验通过后移动。"""

    def __init__(self, blobs: MinioBlobStore | object) -> None:
        self.blobs = blobs

    def write_parent(
        self,
        document_id: str,
        version: str,
        parent_id: str,
        text: str,
        content_hash: str,
    ) -> None:
        _token(document_id)
        _token(version)
        _token(parent_id)
        actual = content_hash_of(text)
        if actual != content_hash:
            raise HashMismatch("正文和内容哈希不一致，没有写入")
        if self._active_version(document_id) == version:
            raise ActiveVersionFrozen("已经生效的版本不能原地覆盖")
        self.blobs.put(_parent_key(document_id, version, parent_id), text.encode("utf-8"))
        manifest = self._read_manifest(document_id, version) or {"status": "writing", "parents": {}}
        manifest["status"] = "writing"
        manifest["parents"][parent_id] = actual
        self._write_manifest(document_id, version, manifest)

    def validate_version(self, document_id: str, version: str, expected: list[tuple[str, str]]) -> None:
        _token(document_id)
        _token(version)
        expected_map: dict[str, str] = {}
        for parent_id, digest in expected:
            _token(parent_id)
            if parent_id in expected_map:
                raise VersionNotValidated("同一版本里父块编号重复")
            expected_map[parent_id] = digest
        prefix = f"parents/{document_id}/{version}/"
        actual: dict[str, str] = {}
        for key in self.blobs.list_keys(prefix):
            parent_id = key[len(prefix) :]
            if not parent_id or "/" in parent_id:
                continue
            blob = self.blobs.get(key)
            if blob is None:
                raise VersionNotValidated("父块在校验时读不到")
            actual[parent_id] = content_hash_of(blob.body.decode("utf-8"))
        if actual != expected_map:
            raise VersionNotValidated("版本还没写全，或哈希对不上，指针不动")
        self._write_manifest(document_id, version, {"status": "validated", "parents": actual})

    def switch_version(self, document_id: str, version: str) -> None:
        _token(document_id)
        _token(version)
        manifest = self._read_manifest(document_id, version)
        if not manifest or manifest.get("status") != "validated":
            raise VersionNotValidated("校验通过之前不能切换")
        payload = json.dumps({"active_version": version}, ensure_ascii=False).encode("utf-8")
        try:
            self.blobs.put(f"pointers/{document_id}", payload)
        except Exception as exc:
            raise VersionSwitchError("指针没有写上，查询仍指向切换前的版本") from exc

    def read_parent(self, document_id: str, parent_id: str) -> ParentRecord:
        version = self._active_version(document_id)
        if version is None:
            raise VersionNotValidated("还没有生效版本")
        return self.read_parent_version(document_id, parent_id, version)

    def read_parent_version(self, document_id: str, parent_id: str, version: str) -> ParentRecord:
        _token(document_id)
        _token(parent_id)
        _token(version)
        blob = self.blobs.get(_parent_key(document_id, version, parent_id))
        if blob is None:
            raise VersionNotValidated("这一版没有该父块")
        text = blob.body.decode("utf-8")
        return ParentRecord(
            document_id=document_id,
            parent_id=parent_id,
            version=version,
            text=text,
            content_hash=content_hash_of(text),
        )

    def delete_version(self, document_id: str, version: str) -> None:
        _token(document_id)
        _token(version)
        if self._active_version(document_id) == version:
            raise OldVersionDeleteBlocked("当前查询指向的版本不能删")
        manifest = self._read_manifest(document_id, version)
        if manifest and manifest.get("status") == "validated":
            raise OldVersionDeleteBlocked("已经校验过的版本要留着，供切换和回读")
        prefix = f"parents/{document_id}/{version}/"
        for key in self.blobs.list_keys(prefix):
            self.blobs.delete(key)
        self.blobs.delete(f"manifests/{document_id}/{version}")

    def _active_version(self, document_id: str) -> str | None:
        blob = self.blobs.get(f"pointers/{document_id}")
        if blob is None:
            return None
        payload = json.loads(blob.body.decode("utf-8"))
        version = payload.get("active_version")
        if not isinstance(version, str) or not version:
            raise StorageError("版本指针损坏")
        return version

    def _read_manifest(self, document_id: str, version: str) -> dict | None:
        blob = self.blobs.get(f"manifests/{document_id}/{version}")
        if blob is None:
            return None
        return json.loads(blob.body.decode("utf-8"))

    def _write_manifest(self, document_id: str, version: str, manifest: dict) -> None:
        body = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        self.blobs.put(f"manifests/{document_id}/{version}", body)


def open_parent_store(settings: StorageSettings) -> VersionedParentStore:
    host, secure = split_endpoint(settings.object_endpoint)
    blobs = MinioBlobStore(
        host,
        access_key=settings.object_access_key,
        secret_key=settings.object_secret_key,
        bucket=settings.object_bucket,
        secure=secure,
    )
    blobs.ensure_bucket()
    return VersionedParentStore(blobs)


def split_endpoint(endpoint: str) -> tuple[str, bool]:
    secure = endpoint.startswith("https://")
    host = endpoint
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix) :]
    return host, secure


def content_hash_of(text: str) -> str:
    return content_hash(text)


def _parent_key(document_id: str, version: str, parent_id: str) -> str:
    return f"parents/{document_id}/{version}/{parent_id}"


def _token(value: str) -> None:
    if not value or "/" in value or value in {".", ".."}:
        raise StorageError("编号不能为空，也不能包含斜杠")
