from dataclasses import dataclass


@dataclass(frozen=True)
class ParentRef:
    """检索先过滤之后交过来的父块指针。正文不在这里，要到对象存储取。"""

    document_id: str
    parent_id: str
    version: str
    effective_status: str
    effective_from: str
    effective_to: str | None = None


@dataclass(frozen=True)
class GenerateRequest:
    question: str
    parents: tuple[ParentRef, ...]
    disposition: str = "high"
    request_id: str = ""


@dataclass(frozen=True)
class CitedBlock:
    marker: int
    document_id: str
    parent_id: str
    version: str
    text: str
    effective_status: str
    effective_from: str
    effective_to: str | None = None


@dataclass(frozen=True)
class Prompt:
    prefix: str
    variable: str
    question: str

    def render(self) -> str:
        return self.prefix + "\n\n" + self.variable


@dataclass(frozen=True)
class DraftSentence:
    text: str
    markers: tuple[int, ...]


@dataclass(frozen=True)
class SentenceDecision:
    text: str
    markers: tuple[int, ...]
    outcome: str
    detail: str = ""
    kept: bool = False


@dataclass(frozen=True)
class ExitResult:
    action: str
    text: str
    decisions: tuple[SentenceDecision, ...]
    entailment_deletion_executed: bool
    checks_called: tuple[str, ...]


@dataclass(frozen=True)
class GenerateResult:
    request_id: str
    action: str
    text: str
    draft: str
    prefix: str
    variable: str
    stages: tuple[str, ...]
    model_id: str
    weights_loaded: bool
    decisions: tuple[SentenceDecision, ...]
    entailment_deletion_executed: bool
    checks_called: tuple[str, ...]
    checks_skipped: bool
