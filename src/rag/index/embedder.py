"""子块嵌入。模型用材料点名的 BGE-M3，加载失败就不换别的模型。"""

import os
from pathlib import Path

from rag.index.errors import EmbeddingModelUnavailable

_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(_ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "BAAI/bge-m3"
DENSE_DIMENSION = 1024


class BgeM3Embedder:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.document_texts: list[str] = []
        self._model = None
        self._tokenizer = None
        self.resolved_name = ""
        self.dimension = 0

    def load(self) -> None:
        if self.model_id != MODEL_ID:
            raise EmbeddingModelUnavailable(self.model_id, "索引层只加载 BGE-M3")
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device="cpu")
            self._tokenizer = self._model.tokenizer
            self.dimension = int(_embedding_dimension(self._model))
            self.resolved_name = _resolved_name(self._model)
        except EmbeddingModelUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingModelUnavailable(self.model_id, str(exc)) from exc
        if self.dimension != DENSE_DIMENSION:
            raise EmbeddingModelUnavailable(
                self.model_id,
                f"稠密向量维度是 {self.dimension}，不是 BGE-M3 的 {DENSE_DIMENSION}",
            )
        if "bge-m3" not in self.resolved_name.lower():
            raise EmbeddingModelUnavailable(self.model_id, "加载到的不是 BGE-M3")

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise EmbeddingModelUnavailable(self.model_id, "模型还没有加载")
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
        """只记录真正写入索引的子块。查询向量不记在这里。"""

        if not texts:
            return []
        self.document_texts.extend(texts)
        return self._encode(texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        if self._model is None:
            raise EmbeddingModelUnavailable(self.model_id, "模型还没有加载")
        try:
            array = self._model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingModelUnavailable(self.model_id, str(exc)) from exc
        return [tuple(float(value) for value in row) for row in array]


def _embedding_dimension(model) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def _resolved_name(model: object) -> str:
    try:
        first = model[0]
        auto = getattr(first, "auto_model", None)
        if auto is not None:
            return str(auto.config.name_or_path)
    except Exception:
        return MODEL_ID
    return MODEL_ID
