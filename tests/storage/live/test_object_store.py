import pytest

from rag.storage.errors import (
    ActiveVersionFrozen,
    HashMismatch,
    OldVersionDeleteBlocked,
    VersionNotValidated,
    VersionSwitchError,
)
from rag.storage.hashing import content_hash


def _publish(parent_store, document_id: str, version: str, parent_id: str, text: str) -> None:
    parent_store.write_parent(
        document_id=document_id,
        version=version,
        parent_id=parent_id,
        text=text,
        content_hash=content_hash(text),
    )
    parent_store.validate_version(
        document_id,
        version,
        expected=[(parent_id, content_hash(text))],
    )
    parent_store.switch_version(document_id, version)


def test_query_reads_old_parent_until_switch(parent_store):
    document_id = new_document()
    parent_id = "parent-1"
    _publish(parent_store, document_id, "v1", parent_id, "旧版条款正文")
    parent_store.write_parent(
        document_id=document_id,
        version="v2",
        parent_id=parent_id,
        text="新版条款正文",
        content_hash=content_hash("新版条款正文"),
    )
    assert parent_store.read_parent(document_id, parent_id).text == "旧版条款正文"
    parent_store.validate_version(
        document_id,
        "v2",
        expected=[(parent_id, content_hash("新版条款正文"))],
    )
    assert parent_store.read_parent(document_id, parent_id).version == "v1"
    parent_store.switch_version(document_id, "v2")
    current = parent_store.read_parent(document_id, parent_id)
    assert current.text == "新版条款正文"
    assert current.version == "v2"


def test_switch_waits_for_full_validated_version(parent_store):
    document_id = new_document()
    parent_id = "parent-1"
    _publish(parent_store, document_id, "v1", parent_id, "仍在生效的正文")
    parent_store.write_parent(
        document_id=document_id,
        version="v2",
        parent_id=parent_id,
        text="只写了一块",
        content_hash=content_hash("只写了一块"),
    )
    with pytest.raises(VersionNotValidated):
        parent_store.validate_version(
            document_id,
            "v2",
            expected=[
                (parent_id, content_hash("只写了一块")),
                ("parent-2", content_hash("缺少的那一块")),
            ],
        )
    with pytest.raises(VersionNotValidated):
        parent_store.switch_version(document_id, "v2")
    assert parent_store.read_parent(document_id, parent_id).text == "仍在生效的正文"


def test_failed_pointer_write_keeps_query_on_old_version(parent_store, monkeypatch):
    document_id = new_document()
    parent_id = "parent-1"
    _publish(parent_store, document_id, "v1", parent_id, "切换失败时应留下的正文")
    parent_store.write_parent(
        document_id=document_id,
        version="v2",
        parent_id=parent_id,
        text="还没切过去的正文",
        content_hash=content_hash("还没切过去的正文"),
    )
    parent_store.validate_version(
        document_id,
        "v2",
        expected=[(parent_id, content_hash("还没切过去的正文"))],
    )
    original = parent_store.blobs.put

    def put(key, body, metadata=None):
        if key.startswith("pointers/"):
            raise OSError("pointer put failed")
        return original(key, body, metadata)

    monkeypatch.setattr(parent_store.blobs, "put", put)
    with pytest.raises(VersionSwitchError):
        parent_store.switch_version(document_id, "v2")
    monkeypatch.undo()
    current = parent_store.read_parent(document_id, parent_id)
    assert current.version == "v1"
    assert current.text == "切换失败时应留下的正文"
    parent_store.switch_version(document_id, "v2")
    assert parent_store.read_parent(document_id, parent_id).version == "v2"


def test_old_version_cannot_be_deleted_before_new_one_validates(parent_store):
    document_id = new_document()
    parent_id = "parent-1"
    _publish(parent_store, document_id, "v1", parent_id, "校验通过前必须还在")
    parent_store.write_parent(
        document_id=document_id,
        version="v2",
        parent_id=parent_id,
        text="新版本尚未校验",
        content_hash=content_hash("新版本尚未校验"),
    )
    with pytest.raises(OldVersionDeleteBlocked):
        parent_store.delete_version(document_id, "v1")
    assert parent_store.read_parent(document_id, parent_id).text == "校验通过前必须还在"


def test_switch_keeps_previous_version_readable(parent_store):
    document_id = new_document()
    parent_id = "parent-1"
    _publish(parent_store, document_id, "v1", parent_id, "切换后仍可取回的旧版")
    _publish(parent_store, document_id, "v2", parent_id, "当前版本")
    previous = parent_store.read_parent_version(document_id, parent_id, "v1")
    assert previous.text == "切换后仍可取回的旧版"
    with pytest.raises(OldVersionDeleteBlocked):
        parent_store.delete_version(document_id, "v1")
    assert parent_store.read_parent_version(document_id, parent_id, "v1").text == "切换后仍可取回的旧版"


def test_active_version_is_not_overwritten_in_place(parent_store):
    document_id = new_document()
    parent_id = "parent-1"
    _publish(parent_store, document_id, "v1", parent_id, "不能原地改掉的正文")
    with pytest.raises(ActiveVersionFrozen):
        parent_store.write_parent(
            document_id=document_id,
            version="v1",
            parent_id=parent_id,
            text="试图覆盖",
            content_hash=content_hash("试图覆盖"),
        )
    assert parent_store.read_parent(document_id, parent_id).text == "不能原地改掉的正文"


def test_parent_write_rejects_hash_that_does_not_match_body(parent_store):
    document_id = new_document()
    with pytest.raises(HashMismatch):
        parent_store.write_parent(
            document_id=document_id,
            version="v1",
            parent_id="parent-1",
            text="正文",
            content_hash="0" * 64,
        )
    with pytest.raises(VersionNotValidated):
        parent_store.read_parent_version(document_id, "parent-1", "v1")


def new_document() -> str:
    from helpers import new_id

    return new_id("doc")
