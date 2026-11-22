from rag.generate.errors import GenerationFailed
from rag.generate.types import CitedBlock, ParentRef
from rag.storage.errors import MetadataWriteError, VersionNotValidated
from rag.storage.types import ParentRecord


EXAM = "员工年度体检补贴为每人 640 元，于每年三月发放。"
LODGING = "出差住宿报销上限为每晚 320 元，返回后十日内提交票据。"
TRAINING = "年度培训名额为每个部门 12 人，名额不作跨年结转。"
DANGLING = "因此按照前面那一句的标准发放补贴。"
# 和原文不是全等，也不是子串。用来证明蕴含不是字符串比对。
PARAPHRASE = "每年三月向员工发放体检补贴，标准是每人 640 元。"
CONTRADICTION = "员工年度体检补贴已经停发，以后不再发放。"


def parent_ref(parent_id: str, *, status: str = "effective", version: str = "v1", document_id: str = "doc-gen") -> ParentRef:
    end = "2024-12-31" if status == "obsolete" else None
    return ParentRef(
        document_id=document_id,
        parent_id=parent_id,
        version=version,
        effective_status=status,
        effective_from="2024-01-01",
        effective_to=end,
    )


def parent_record(parent_id: str, text: str, *, version: str = "v1", document_id: str = "doc-gen") -> ParentRecord:
    return ParentRecord(
        document_id=document_id,
        parent_id=parent_id,
        version=version,
        text=text,
        content_hash="hash",
    )


def cited(marker: int, parent_id: str, text: str, *, status: str = "effective") -> CitedBlock:
    end = "2024-12-31" if status == "obsolete" else None
    return CitedBlock(
        marker=marker,
        document_id="doc-gen",
        parent_id=parent_id,
        version="v1",
        text=text,
        effective_status=status,
        effective_from="2024-01-01",
        effective_to=end,
    )


class MemoryParents:
    def __init__(self, rows: dict[tuple[str, str, str], ParentRecord]) -> None:
        self.rows = rows
        self.reads: list[str] = []

    def read_parent_version(self, document_id: str, parent_id: str, version: str) -> ParentRecord:
        self.reads.append(parent_id)
        key = (document_id, parent_id, version)
        if key not in self.rows:
            raise VersionNotValidated("这一版没有该父块")
        return self.rows[key]


class MemoryAudit:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.citations = []
        self.refusals: list[tuple[str, str, str]] = []

    def citation(self, entry) -> int:
        if self.fail:
            raise MetadataWriteError("引用日志写不进去")
        self.citations.append(entry)
        return len(self.citations)

    def refusal(self, request_id: str, kind: str, detail: str) -> int:
        if self.fail:
            raise MetadataWriteError("拒答日志写不进去")
        self.refusals.append((request_id, kind, detail))
        return len(self.refusals)


class ScriptedJudge:
    decides_by_string_equality = False

    def __init__(self, labels: dict[str, str] | None = None, default: str = "支持") -> None:
        self.labels = labels or {}
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def label(self, premise: str, hypothesis: str) -> str:
        self.calls.append((premise, hypothesis))
        for key, value in self.labels.items():
            if key in hypothesis:
                return value
        return self.default


class StringEqualityJudge:
    decides_by_string_equality = True

    def label(self, premise: str, hypothesis: str) -> str:
        left = "".join(hypothesis.split())
        right = "".join(premise.split())
        if left == right or (left and left in right):
            return "支持"
        return "中立"


class FixedGenerator:
    def __init__(self, text: str, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.model_id = "fixed-draft"
        self.weights_loaded = False
        self.calls = 0

    def complete(self, prompt) -> str:
        self.calls += 1
        if self.fail:
            raise GenerationFailed("草稿没有生成出来")
        return self.text
