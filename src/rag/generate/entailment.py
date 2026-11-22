"""句子和它所引用的块做蕴含判定。材料没有写死模型名字。

判定只看模型给出的支持、中立、矛盾。字符串全等不是这条校验。
"""

import os
from pathlib import Path

from rag.generate.errors import EntailmentUnavailable
from rag.generate.params import ENTAILMENT_MODEL_ID

_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(_ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_TO_ZH = {
    "entailment": "支持",
    "neutral": "中立",
    "contradiction": "矛盾",
}


class EntailmentJudge:
    """默认蕴含模型。名字是本仓库选的，文档未写死。"""

    decides_by_string_equality = False

    def __init__(self, model_id: str = ENTAILMENT_MODEL_ID) -> None:
        self.model_id = model_id
        self.resolved_name = ""
        self._tokenizer = None
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None and self.resolved_name == self.model_id

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self._model.eval()
            self.resolved_name = self.model_id
        except Exception as exc:
            self._model = None
            raise EntailmentUnavailable(self.model_id, str(exc)) from exc
        labels = set(self._canonical(name) for name in self._model.config.id2label.values())
        if labels != {"entailment", "neutral", "contradiction"}:
            self._model = None
            raise EntailmentUnavailable(self.model_id, "标签不是支持、中立、矛盾这一套")

    def label(self, premise: str, hypothesis: str) -> str:
        if self._model is None or self._tokenizer is None:
            raise EntailmentUnavailable(self.model_id, "模型还没有加载")
        english = self._predict(premise, hypothesis)
        return _TO_ZH[english]

    def _predict(self, premise: str, hypothesis: str) -> str:
        assert self._tokenizer is not None and self._model is not None
        try:
            encoded = self._tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            index = int(self._torch.argmax(logits, dim=-1).item())
            raw = self._model.config.id2label[index]
        except EntailmentUnavailable:
            raise
        except Exception as exc:
            raise EntailmentUnavailable(self.model_id, str(exc)) from exc
        return self._canonical(raw)

    @staticmethod
    def _canonical(raw: str) -> str:
        text = str(raw).lower()
        if "entail" in text or "蕴含" in str(raw):
            return "entailment"
        if "contrad" in text or "矛盾" in str(raw):
            return "contradiction"
        if "neutral" in text or "中立" in str(raw):
            return "neutral"
        raise EntailmentUnavailable(ENTAILMENT_MODEL_ID, f"无法识别的标签 {raw}")
