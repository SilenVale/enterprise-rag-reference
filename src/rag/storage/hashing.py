import hashlib


def content_hash(text: str) -> str:
    """对正文做 SHA-256。索引层对规范化文本用的是同一种算法。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
