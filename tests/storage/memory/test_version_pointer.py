import pytest

from rag.storage.errors import OldVersionDeleteBlocked, VersionSwitchError
from rag.storage.hashing import content_hash
from rag.storage.memory import MemoryBlobStore
from rag.storage.object_store import VersionedParentStore

pytestmark = pytest.mark.memory


def test_memory_pointer_failure_keeps_old_version_not_used_for_acceptance():
    store = VersionedParentStore(MemoryBlobStore())
    document_id = "doc-memory"
    parent_id = "parent-1"
    old = "旧版"
    store.write_parent(document_id, "v1", parent_id, old, content_hash(old))
    store.validate_version(document_id, "v1", expected=[(parent_id, content_hash(old))])
    store.switch_version(document_id, "v1")
    new = "新版"
    store.write_parent(document_id, "v2", parent_id, new, content_hash(new))
    store.validate_version(document_id, "v2", expected=[(parent_id, content_hash(new))])
    original = store.blobs.put

    def put(key, body, metadata=None):
        if str(key).startswith("pointers/"):
            raise OSError("injected")
        return original(key, body, metadata)

    store.blobs.put = put
    with pytest.raises(VersionSwitchError):
        store.switch_version(document_id, "v2")
    assert store.read_parent(document_id, parent_id).text == old
    with pytest.raises(OldVersionDeleteBlocked):
        store.delete_version(document_id, "v1")
