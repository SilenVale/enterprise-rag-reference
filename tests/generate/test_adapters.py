"""本机档和 vLLM 档共用引用格式与出口校验。验收不加载 72B。"""

import inspect

from rag.generate.adapters import LocalExtractiveGenerator, VllmConfig, VllmGenerator
from rag.generate.errors import GenerationFailed
from rag.generate.params import VLLM_MODEL_ID
from rag.generate.pipeline import GeneratePipeline
from rag.generate.prompt import STABLE_PREFIX
from rag.generate.types import GenerateRequest

from support import EXAM, LODGING, TRAINING, MemoryAudit, MemoryParents, ScriptedJudge, parent_record, parent_ref


def test_本机档和vLLM档对同一份坏草稿做出相同的数字拦截且不加载72B():
    bad = "出差住宿报销上限为每晚 1750 元。[2]"
    seen = {}

    def transport(payload):
        seen["payload"] = payload
        return {"choices": [{"message": {"content": bad}}]}

    vllm = VllmGenerator(VllmConfig(enable_prefix_caching=True), transport=transport)
    local = LocalExtractiveGenerator()
    assert vllm.model_id == VLLM_MODEL_ID
    assert vllm.weights_loaded is False
    assert local.weights_loaded is False
    assert "from_pretrained" not in inspect.getsource(VllmGenerator)
    assert "Qwen2.5-72B" == VLLM_MODEL_ID

    vllm_result = _run(vllm)
    assert seen["payload"]["model"] == "Qwen2.5-72B"
    assert seen["payload"]["messages"][0]["content"] == STABLE_PREFIX
    assert seen["payload"]["messages"][0]["role"] == "system"
    assert "[1]" in seen["payload"]["messages"][1]["content"]
    assert "问题：" in seen["payload"]["messages"][1]["content"]
    assert seen["payload"]["extra_body"]["enable_prefix_caching"] is True
    assert vllm_result.weights_loaded is False
    assert vllm_result.action == "refused"
    assert "1750" not in vllm_result.text

    from support import FixedGenerator

    local_result = _run(FixedGenerator(bad))
    assert local_result.action == vllm_result.action
    assert local_result.text == vllm_result.text
    assert local_result.checks_called == vllm_result.checks_called
    assert [item.outcome for item in local_result.decisions] == [item.outcome for item in vllm_result.decisions]


def test_本机档从编号块摘出事实句并在句末带上编号():
    generator = LocalExtractiveGenerator()
    result = _run(generator, question="体检补贴每人是多少？")
    assert result.model_id == "local-extractive"
    assert result.weights_loaded is False
    assert result.action == "answer"
    assert "[1]" in result.text
    assert "640" in result.text
    assert "1750" not in result.text


def test_vLLM配置换成别的模型名会在发请求之前拒绝():
    try:
        VllmGenerator(VllmConfig(model="Qwen2.5-7B"))
    except GenerationFailed as exc:
        assert "Qwen2.5-72B" in str(exc)
    else:
        raise AssertionError("vLLM 档必须保留 Qwen2.5-72B")


def _run(generator, question: str = "住宿报销上限是多少？"):
    store = MemoryParents(
        {
            ("doc-gen", "p-exam", "v1"): parent_record("p-exam", EXAM),
            ("doc-gen", "p-lodging", "v1"): parent_record("p-lodging", LODGING),
            ("doc-gen", "p-training", "v1"): parent_record("p-training", TRAINING),
        }
    )
    pipeline = GeneratePipeline(parents=store, audit=MemoryAudit(), generator=generator, entailment=ScriptedJudge())
    return pipeline.run(
        GenerateRequest(
            question=question,
            parents=(parent_ref("p-exam"), parent_ref("p-lodging"), parent_ref("p-training")),
            request_id="req-adapter",
        )
    )
