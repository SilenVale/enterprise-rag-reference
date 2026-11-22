"""精排只用材料点名的 bge-reranker-v2-m3。加载失败不换模型。"""

import os
from pathlib import Path

from rag.retrieve.errors import RerankerUnavailable
from rag.retrieve.params import HIGH_SCORE, MID_SCORE

_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(_ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"

ROUTE_LABELS = {
    "high": "生成答案并引用",
    "mid": "生成答案并附上可能不完整、建议人工核实",
    "low": "拒答并转人工",
    "retrieval_failed": "检索失败",
}


class BgeReranker:
    def __init__(self, model_id: str = RERANK_MODEL_ID) -> None:
        self.model_id = model_id
        self.resolved_name = ""
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None and "bge-reranker-v2-m3" in self.resolved_name.lower()

    def load(self) -> None:
        if self.model_id != RERANK_MODEL_ID:
            raise RerankerUnavailable(self.model_id, "检索层精排只加载 bge-reranker-v2-m3")
        try:
            import torch
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, device="cpu")
            self.resolved_name = _resolved_name(self._model)
            self._sigmoid = torch.nn.Sigmoid()
        except RerankerUnavailable:
            raise
        except Exception as exc:
            self._model = None
            raise RerankerUnavailable(self.model_id, str(exc)) from exc
        if "bge-reranker-v2-m3" not in self.resolved_name.lower():
            self._model = None
            raise RerankerUnavailable(self.model_id, "加载到的不是 bge-reranker-v2-m3")

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not self.ready or self._model is None:
            raise RerankerUnavailable(self.model_id, "模型还没有加载")
        if not pairs:
            return []
        try:
            array = self._model.predict(
                [(left, right) for left, right in pairs],
                activation_fn=self._sigmoid,
                show_progress_bar=False,
                batch_size=16,
            )
        except Exception as exc:
            raise RerankerUnavailable(self.model_id, str(exc)) from exc
        values = array.tolist()
        if isinstance(values, float):
            values = [values]
        return [float(value) for value in values]


def tier_of(score: float, *, high: float = HIGH_SCORE, mid: float = MID_SCORE) -> str:
    if high <= mid:
        raise ValueError("高分线必须高于中分线")
    if score >= high:
        return "high"
    if score >= mid:
        return "mid"
    return "low"


def _resolved_name(model: object) -> str:
    names: list[str] = []
    inner = getattr(model, "model", None)
    if inner is not None:
        for module in inner.modules():
            config = getattr(module, "config", None)
            name = getattr(config, "name_or_path", "") if config is not None else ""
            if name:
                names.append(str(name))
                break
    tokenizer = getattr(model, "tokenizer", None)
    name = getattr(tokenizer, "name_or_path", "") if tokenizer is not None else ""
    if name:
        names.append(str(name))
    return names[0] if names else ""
