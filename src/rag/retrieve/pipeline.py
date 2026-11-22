"""把改写、三路召回、RRF、精排和分流串起来。不写答案正文。"""

import uuid

from rag.index.embedder import MODEL_ID as EMBED_MODEL_ID
from rag.index.errors import EmbeddingModelUnavailable
from rag.retrieve.errors import RerankerUnavailable, RetrievalFailed, RewriteError
from rag.retrieve.params import CAVEAT, GENERATION_MAIN_MODEL, INTERN_PERMISSION, RERANK_POOL, RERANK_TOP_K
from rag.retrieve.recall import TripleRecall
from rag.retrieve.rerank import RERANK_MODEL_ID, ROUTE_LABELS, BgeReranker, tier_of
from rag.retrieve.rewrite import QueryRewriteModel
from rag.retrieve.rrf import fuse
from rag.retrieve.types import FusedHit, RankedHit, RetrieveRequest, RetrieveResult, RewritePlan
from rag.storage.types import ScalarFilter


class RetrievePipeline:
    def __init__(self, *, embedder, vectors, lexical, reranker: BgeReranker, refusals, rewriter=None) -> None:
        self.embedder = embedder
        self.vectors = vectors
        self.lexical = lexical
        self.reranker = reranker
        self.refusals = refusals
        self.rewriter = rewriter or QueryRewriteModel()
        self.recall = TripleRecall(embedder, vectors, lexical)

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        self._require_models()
        request_id = request.request_id or uuid.uuid4().hex
        plan = self._rewrite(request)
        filters = _filters_for(request)
        try:
            bundle = self.recall.run(plan.queries(), filters)
        except RetrievalFailed as exc:
            self.refusals.append(request_id, "检索失败", str(exc))
            return _failed(request_id, plan, filters, str(exc))
        fused = fuse({path.name: path.hits for path in bundle.paths})
        passages = self._rerank(plan, fused)
        disposition = "low" if not passages else tier_of(passages[0].rerank_score)
        if disposition == "low":
            detail = "没有候选" if not passages else f"最高精排分 {passages[0].rerank_score:.4f}"
            self.refusals.append(request_id, "低分拒答", detail)
        return RetrieveResult(
            request_id=request_id,
            queries=plan.queries(),
            rewrite_model_id=plan.model_id,
            rewrite_failed=plan.failed,
            filters=filters,
            paths=bundle.paths,
            fused=tuple(fused),
            passages=tuple(passages),
            disposition=disposition,
            route_label=ROUTE_LABELS[disposition],
            caveat=CAVEAT if disposition == "mid" else "",
            handoff=disposition == "low",
            generated_text="",
            latency_ms=bundle.latency_ms,
            recall_wall_ms=bundle.wall_ms,
            failure="",
        )

    def _require_models(self) -> None:
        if self.reranker.model_id != RERANK_MODEL_ID or not self.reranker.ready:
            raise RerankerUnavailable(self.reranker.model_id, "精排模型未加载，检索层停住")
        name = getattr(self.embedder, "resolved_name", "")
        model_id = getattr(self.embedder, "model_id", "")
        if model_id != EMBED_MODEL_ID or "bge-m3" not in str(name).lower():
            raise EmbeddingModelUnavailable(str(model_id), "向量一路只使用已加载的 BGE-M3")
        if self.rewriter.model_id == GENERATION_MAIN_MODEL:
            raise RewriteError("改写不能占用生成主模型")

    def _rewrite(self, request: RetrieveRequest) -> RewritePlan:
        try:
            return self.rewriter.rewrite(request.question, request.history)
        except Exception:
            return RewritePlan(
                original=request.question,
                standalone=None,
                subqueries=(),
                model_id=getattr(self.rewriter, "model_id", ""),
                failed=True,
            )

    def _rerank(self, plan: RewritePlan, fused: list[FusedHit]) -> list[RankedHit]:
        pool = fused[:RERANK_POOL]
        texts = self.lexical.fetch_texts([hit.child_id for hit in pool])
        query = plan.standalone or plan.original
        pairs: list[tuple[str, str]] = []
        kept: list[FusedHit] = []
        for hit in pool:
            text = texts.get(hit.child_id, "")
            if not text:
                continue
            kept.append(hit)
            pairs.append((query, text))
        if not pairs:
            return []
        scores = self.reranker.score(pairs)
        ranked = [
            RankedHit(
                child_id=hit.child_id,
                parent_id=hit.parent_id,
                document_id=hit.document_id,
                version=hit.version,
                clause_number=hit.clause_number,
                text=text,
                rerank_score=score,
            )
            for hit, text, score in zip(kept, (item[1] for item in pairs), scores, strict=True)
        ]
        ranked.sort(key=lambda item: (-item.rerank_score, item.child_id))
        return ranked[:RERANK_TOP_K]


def _filters_for(request: RetrieveRequest) -> tuple[ScalarFilter, ...]:
    if request.role == "intern" or request.role not in {"staff", "auditor"}:
        permission = INTERN_PERMISSION
    else:
        permission = None
    statuses = ["effective"]
    if request.include_obsolete:
        statuses.append("obsolete")
    return tuple(
        ScalarFilter(
            permission=permission,
            effective_status=status,
            document_type=request.document_type,
        )
        for status in statuses
    )


def _failed(request_id: str, plan: RewritePlan, filters, reason: str) -> RetrieveResult:
    return RetrieveResult(
        request_id=request_id,
        queries=plan.queries(),
        rewrite_model_id=plan.model_id,
        rewrite_failed=plan.failed,
        filters=filters,
        paths=(),
        fused=(),
        passages=(),
        disposition="retrieval_failed",
        route_label=ROUTE_LABELS["retrieval_failed"],
        caveat="",
        handoff=False,
        generated_text="",
        latency_ms=0.0,
        recall_wall_ms=0.0,
        failure=reason,
    )
