"""两个生成档。引用格式和出口校验不在这里分叉，由调用方共用。"""

import json
import re
import urllib.error
import urllib.request

from rag.generate.errors import GenerationFailed
from rag.generate.params import LOCAL_MODEL_ID, LOCAL_OVERLAP_CHARS, VLLM_MODEL_ID
from rag.generate.prompt import parse_variable
from rag.generate.types import Prompt

_STOP = "，。！？、；： \n\t"


class LocalExtractiveGenerator:
    """本机可跑的一档。从编号块里摘出和问题重合的句子，不加载 72B。文档未写死。"""

    def __init__(self) -> None:
        self.model_id = LOCAL_MODEL_ID
        self.weights_loaded = False

    def complete(self, prompt: Prompt) -> str:
        question, blocks = parse_variable(prompt.variable)
        if not blocks:
            raise GenerationFailed("变化段里没有编号块")
        chosen: list[str] = []
        for marker, text in blocks:
            sentence = _best_sentence(text, question)
            if sentence is None:
                continue
            if sentence[-1:] not in "。！？":
                sentence += "。"
            chosen.append(f"{sentence}[{marker}]")
        return "".join(chosen)


class VllmConfig:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = VLLM_MODEL_ID,
        enable_prefix_caching: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.enable_prefix_caching = enable_prefix_caching
        self.timeout_seconds = timeout_seconds


class VllmGenerator:
    """vLLM 档。配置保留 Qwen2.5-72B 和前缀缓存。这个类不在本机加载权重。"""

    def __init__(self, config: VllmConfig | None = None, transport=None) -> None:
        self.config = config or VllmConfig()
        if self.config.model != VLLM_MODEL_ID:
            raise GenerationFailed("vLLM 档的模型配置必须保留 Qwen2.5-72B")
        self.model_id = self.config.model
        self.weights_loaded = False
        self._transport = transport

    def request_payload(self, prompt: Prompt) -> dict:
        return {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt.prefix},
                {"role": "user", "content": prompt.variable},
            ],
            "extra_body": {"enable_prefix_caching": self.config.enable_prefix_caching},
        }

    def complete(self, prompt: Prompt) -> str:
        payload = self.request_payload(prompt)
        try:
            if self._transport is not None:
                data = self._transport(payload)
            else:
                data = _post_json(self.config, payload)
            return data["choices"][0]["message"]["content"]
        except GenerationFailed:
            raise
        except Exception as exc:
            raise GenerationFailed("vLLM 没有返回草稿") from exc


def _post_json(config: VllmConfig, payload: dict) -> dict:
    url = config.base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise GenerationFailed("vLLM 没有返回草稿") from exc


def _best_sentence(text: str, question: str) -> str | None:
    parts = re.split(r"(?<=[。！？])", text)
    best_score = 0
    best = None
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        score = len({char for char in sentence if char not in _STOP and char in question})
        if score > best_score:
            best_score = score
            best = sentence
    if best is None or best_score < LOCAL_OVERLAP_CHARS:
        return None
    return best
