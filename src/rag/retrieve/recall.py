"""三路并发召回。标量条件推进引擎，不在结果上再滤一遍权限。"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from rag.retrieve.errors import RetrievalFailed
from rag.retrieve.params import PATH_TOP_K
from rag.retrieve.rewrite import extract_clauses, text_without_clauses
from rag.retrieve.types import PathRecall, RecallHit
from rag.storage.types import ScalarFilter

VECTOR = "vector"
BM25 = "bm25"
METADATA = "metadata"


class RecallBundle:
    def __init__(self, paths: tuple[PathRecall, ...], wall_ms: float) -> None:
        self.paths = paths
        self.wall_ms = wall_ms
        self.latency_ms = max(path.elapsed_ms for path in paths)


class TripleRecall:
    def __init__(self, embedder, vectors, lexical) -> None:
        self.embedder = embedder
        self.vectors = vectors
        self.lexical = lexical

    def run(self, queries: tuple[str, ...], scalars: tuple[ScalarFilter, ...]) -> RecallBundle:
        if not queries:
            raise RetrievalFailed("没有可检索的查询")
        if not scalars:
            raise RetrievalFailed("没有下推进引擎的过滤条件")
        started = time.perf_counter()
        tasks = {
            VECTOR: lambda: self._vector(queries, scalars),
            BM25: lambda: self._bm25(queries, scalars),
            METADATA: lambda: self._metadata(scalars),
        }
        results: dict[str, PathRecall] = {}
        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {name: pool.submit(fn) for name, fn in tasks.items()}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            first = errors[0]
            raise RetrievalFailed(str(first)) from first
        wall_ms = (time.perf_counter() - started) * 1000
        ordered = tuple(results[name] for name in (VECTOR, BM25, METADATA))
        return RecallBundle(ordered, wall_ms)

    def _vector(self, queries: tuple[str, ...], scalars: tuple[ScalarFilter, ...]) -> PathRecall:
        started = time.perf_counter()
        per_query: list[list[RecallHit]] = []
        for query in queries:
            vector = self.embedder.embed_query(query)
            groups = []
            for scalar in scalars:
                rows = self.vectors.search(vector, limit=PATH_TOP_K, scalar=scalar)
                groups.append(_from_engine(rows))
            per_query.append(merge_by_best_rank(groups))
        hits = tuple(merge_by_best_rank(per_query)[:PATH_TOP_K])
        return PathRecall(VECTOR, hits, (time.perf_counter() - started) * 1000)

    def _bm25(self, queries: tuple[str, ...], scalars: tuple[ScalarFilter, ...]) -> PathRecall:
        started = time.perf_counter()
        per_query: list[list[RecallHit]] = []
        for query in queries:
            groups: list[list[RecallHit]] = []
            prose = text_without_clauses(query)
            clauses = extract_clauses(query)
            for scalar in scalars:
                if prose:
                    groups.append(_from_engine(self.lexical.search(prose, limit=PATH_TOP_K, scalar=scalar)))
                for clause in clauses:
                    groups.append(
                        _from_engine(self.lexical.search_clause(clause, limit=PATH_TOP_K, scalar=scalar))
                    )
            per_query.append(merge_by_best_rank(groups) if groups else [])
        hits = tuple(merge_by_best_rank(per_query)[:PATH_TOP_K])
        return PathRecall(BM25, hits, (time.perf_counter() - started) * 1000)

    def _metadata(self, scalars: tuple[ScalarFilter, ...]) -> PathRecall:
        """权限、生效状态、文档类型在引擎内过滤。这一路不看词面，也不看向量。"""

        started = time.perf_counter()
        found: dict[str, RecallHit] = {}
        for scalar in scalars:
            for hit in _from_engine(self.lexical.list_matching(limit=PATH_TOP_K, scalar=scalar)):
                found.setdefault(hit.child_id, hit)
        ordered = [found[key] for key in sorted(found)][:PATH_TOP_K]
        hits = tuple(replace(hit, rank=index) for index, hit in enumerate(ordered, start=1))
        return PathRecall(METADATA, hits, (time.perf_counter() - started) * 1000)


def merge_by_best_rank(groups: list[list[RecallHit]]) -> list[RecallHit]:
    best: dict[str, tuple[int, RecallHit]] = {}
    for group in groups:
        for hit in group:
            current = best.get(hit.child_id)
            if current is None or hit.rank < current[0]:
                best[hit.child_id] = (hit.rank, hit)
    ordered = sorted(best.values(), key=lambda item: (item[0], item[1].child_id))
    return [replace(hit, rank=index) for index, (_, hit) in enumerate(ordered, start=1)]


def _from_engine(rows) -> list[RecallHit]:
    hits: list[RecallHit] = []
    for index, row in enumerate(rows, start=1):
        hits.append(
            RecallHit(
                child_id=row.child_id,
                parent_id=row.parent_id,
                document_id=row.document_id,
                version=row.version,
                content_hash=row.content_hash,
                clause_number=row.clause_number,
                rank=index,
                source_score=float(row.score),
            )
        )
    return hits
