import uuid

from rag.storage.hashing import content_hash
from rag.storage.types import ChildText, ChildVector, ScalarFilter


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def vector(x: float, y: float = 0.0, z: float = 0.0) -> tuple[float, ...]:
    values = [0.0] * 8
    values[0] = x
    values[1] = y
    values[2] = z
    return tuple(values)


def make_child_vector(child_id: str, values: tuple[float, ...], **overrides) -> ChildVector:
    fields = {
        "child_id": child_id,
        "vector": values,
        "parent_id": "parent-1",
        "document_id": "doc-1",
        "version": "v1",
        "content_hash": "hash-child",
        "clause_number": "DZ-1",
        "permission": "public",
        "effective_status": "effective",
        "document_type": "policy",
    }
    fields.update(overrides)
    return ChildVector(**fields)


def make_child_text(child_id: str, text: str, **overrides) -> ChildText:
    fields = {
        "child_id": child_id,
        "text": text,
        "parent_id": "parent-1",
        "document_id": "doc-1",
        "version": "v1",
        "content_hash": content_hash(text),
        "clause_number": "DZ-1",
        "permission": "public",
        "effective_status": "effective",
        "document_type": "policy",
    }
    fields.update(overrides)
    return ChildText(**fields)


def no_filter() -> ScalarFilter:
    return ScalarFilter()
