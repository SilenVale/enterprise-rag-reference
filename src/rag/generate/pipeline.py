"""先过滤、编号引用、出口三道按这个顺序走。拒答写进元数据库。"""

import uuid

from rag.generate.adapters import LocalExtractiveGenerator
from rag.generate.entailment import EntailmentJudge
from rag.generate.errors import EntailmentRejected, GenerateAuditError, GenerationFailed
from rag.generate.exit_checks import apply_exit_checks
from rag.generate.params import CAVEAT, DEFENSE_STAGES, REFUSAL_TEXT
from rag.generate.prompt import assign_markers, render
from rag.generate.sentences import format_markers
from rag.generate.types import CitedBlock, ExitResult, GenerateRequest, GenerateResult, ParentRef
from rag.storage.errors import MetadataWriteError, StorageError
from rag.storage.types import CitationLog


class GeneratePipeline:
    def __init__(self, *, parents, audit, generator=None, entailment: EntailmentJudge | object | None = None) -> None:
        self.parents = parents
        self.audit = audit
        self.generator = generator or LocalExtractiveGenerator()
        self.entailment = entailment if entailment is not None else EntailmentJudge()

    def run(self, request: GenerateRequest) -> GenerateResult:
        if getattr(self.entailment, "decides_by_string_equality", False):
            raise EntailmentRejected("字符串全等不能当作蕴含校验")
        request_id = request.request_id or uuid.uuid4().hex
        if request.disposition in {"low", "retrieval_failed"}:
            reason = "检索分流为低分，生成层不写答案" if request.disposition == "low" else "检索失败，生成层不写答案"
            self._refusal(request_id, "拒答", reason)
            return _result(
                request_id,
                action="refused",
                text=REFUSAL_TEXT,
                draft="",
                prefix="",
                variable="",
                stages=(DEFENSE_STAGES[0],),
                model_id=getattr(self.generator, "model_id", ""),
                weights_loaded=bool(getattr(self.generator, "weights_loaded", False)),
                exit_result=None,
            )

        stages = [DEFENSE_STAGES[0], DEFENSE_STAGES[1]]
        cited, missing = self._load_cited(request.parents)
        if missing is not None or not cited:
            detail = missing or "没有可引用的父块"
            self._audit_missing(request_id, request.parents, detail)
            self._refusal(request_id, "拒答", detail)
            return _result(
                request_id,
                action="refused",
                text=REFUSAL_TEXT,
                draft="",
                prefix="",
                variable="",
                stages=tuple(stages),
                model_id=getattr(self.generator, "model_id", ""),
                weights_loaded=bool(getattr(self.generator, "weights_loaded", False)),
                exit_result=None,
            )

        prompt = render(cited, request.question)
        draft, _failed = self._draft(prompt)
        stages.append(DEFENSE_STAGES[2])
        exit_result = apply_exit_checks(draft, cited, self.entailment)
        text = exit_result.text
        if exit_result.action == "answer" and request.disposition == "mid":
            text = text + CAVEAT
        self._persist(request_id, cited, exit_result)
        return _result(
            request_id,
            action=exit_result.action,
            text=text,
            draft=draft,
            prefix=prompt.prefix,
            variable=prompt.variable,
            stages=tuple(stages),
            model_id=getattr(self.generator, "model_id", ""),
            weights_loaded=bool(getattr(self.generator, "weights_loaded", False)),
            exit_result=exit_result,
        )

    def _load_cited(self, parents: tuple[ParentRef, ...]) -> tuple[tuple[CitedBlock, ...], str | None]:
        loaded: list[CitedBlock] = []
        for ref in parents:
            try:
                record = self.parents.read_parent_version(ref.document_id, ref.parent_id, ref.version)
            except StorageError:
                return (), f"父块 {ref.parent_id} 取不回"
            loaded.append(
                CitedBlock(
                    marker=0,
                    document_id=record.document_id,
                    parent_id=record.parent_id,
                    version=record.version,
                    text=record.text,
                    effective_status=ref.effective_status,
                    effective_from=ref.effective_from,
                    effective_to=ref.effective_to,
                )
            )
        return assign_markers(loaded), None

    def _draft(self, prompt) -> tuple[str, bool]:
        try:
            return self.generator.complete(prompt), False
        except GenerationFailed:
            return "", True

    def _persist(self, request_id: str, cited: tuple[CitedBlock, ...], exit_result: ExitResult) -> None:
        by_marker = {block.marker: block for block in cited}
        try:
            for decision in exit_result.decisions:
                markers = decision.markers or ()
                targets = [by_marker[marker] for marker in markers if marker in by_marker]
                if not targets:
                    targets = list(cited[:1])
                for block in targets:
                    self.audit.citation(
                        CitationLog(
                            request_id=request_id,
                            document_id=block.document_id,
                            parent_id=block.parent_id,
                            version=block.version,
                            citation_marker=format_markers(decision.markers),
                            outcome=decision.outcome,
                            detail=decision.detail,
                        )
                    )
            if exit_result.action == "refused":
                self._refusal(request_id, "拒答", _action_detail(exit_result))
            elif exit_result.action == "downgraded":
                self._refusal(request_id, "降级", _action_detail(exit_result))
        except MetadataWriteError as exc:
            raise GenerateAuditError(str(exc)) from exc

    def _audit_missing(self, request_id: str, parents: tuple[ParentRef, ...], detail: str) -> None:
        ref = parents[0] if parents else None
        try:
            self.audit.citation(
                CitationLog(
                    request_id=request_id,
                    document_id=ref.document_id if ref else "未取回",
                    parent_id=ref.parent_id if ref else "未取回",
                    version=ref.version if ref else "-",
                    citation_marker="[?]",
                    outcome="父块缺失",
                    detail=detail,
                )
            )
        except MetadataWriteError as exc:
            raise GenerateAuditError(str(exc)) from exc

    def _refusal(self, request_id: str, kind: str, detail: str) -> None:
        try:
            self.audit.refusal(request_id, kind, detail)
        except MetadataWriteError as exc:
            raise GenerateAuditError(str(exc)) from exc


def _action_detail(exit_result: ExitResult) -> str:
    outcomes = "、".join(dict.fromkeys(item.outcome for item in exit_result.decisions if not item.kept))
    if exit_result.entailment_deletion_executed:
        return "不支持的句子已删除"
    if not outcomes:
        return "没有可放行的句子"
    return outcomes


def _result(
    request_id: str,
    *,
    action: str,
    text: str,
    draft: str,
    prefix: str,
    variable: str,
    stages: tuple[str, ...],
    model_id: str,
    weights_loaded: bool,
    exit_result: ExitResult | None,
) -> GenerateResult:
    return GenerateResult(
        request_id=request_id,
        action=action,
        text=text,
        draft=draft,
        prefix=prefix,
        variable=variable,
        stages=stages,
        model_id=model_id,
        weights_loaded=weights_loaded,
        decisions=() if exit_result is None else exit_result.decisions,
        entailment_deletion_executed=False if exit_result is None else exit_result.entailment_deletion_executed,
        checks_called=() if exit_result is None else exit_result.checks_called,
        checks_skipped=False,
    )
