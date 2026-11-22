"""编号引用是第二道。稳定前缀和变化段要能分开。"""

import pytest

from rag.generate.errors import EntailmentRejected, GenerateAuditError
from rag.generate.params import DEFENSE_STAGES
from rag.generate.pipeline import GeneratePipeline
from rag.generate.prompt import STABLE_PREFIX
from rag.generate.prompt import render
from rag.generate.types import GenerateRequest

from support import EXAM, LODGING, TRAINING, FixedGenerator, MemoryAudit, MemoryParents, ScriptedJudge, StringEqualityJudge, cited, parent_record, parent_ref


def _blocks():
    return (
        cited(1, "p-exam", EXAM),
        cited(2, "p-lodging", LODGING),
        cited(3, "p-training", TRAINING),
    )


def test_命中父块以一二三编号注入且事实句末尾带回编号():
    prompt = render(_blocks(), "体检补贴、住宿报销和培训名额分别是多少？")
    assert "[1]" in prompt.variable
    assert "[2]" in prompt.variable
    assert "[3]" in prompt.variable
    assert prompt.variable.index("[1]") < prompt.variable.index("[2]") < prompt.variable.index("[3]")
    assert EXAM in prompt.variable
    assert LODGING in prompt.variable
    assert TRAINING in prompt.variable
    draft = f"{EXAM}[1]{LODGING}[2]{TRAINING}[3]"
    pipeline = _pipeline(draft)
    result = pipeline.run(_request())
    assert result.action == "answer"
    assert result.text.endswith("[3]")
    assert "[1]" in result.text and "[2]" in result.text and "[3]" in result.text
    assert result.stages == DEFENSE_STAGES
    assert result.prefix == STABLE_PREFIX
    assert "[1]" in result.variable and "体检补贴" in result.variable


def test_系统说明格式约束和引用规则放在稳定前缀_检索块和问题放在变化段():
    first = render(_blocks(), "体检补贴是多少？")
    second = render((cited(1, "p-other", "夜班交通补贴为每人 15 元。"),), "夜班交通补贴怎么发？")
    assert first.prefix == second.prefix == STABLE_PREFIX
    assert "640" not in first.prefix
    assert "夜班交通补贴" not in first.prefix
    assert "体检补贴是多少" not in first.prefix
    assert f"[1] {EXAM}" not in first.prefix
    assert "问题：" not in first.prefix
    assert "问题：体检补贴是多少？" in first.variable
    assert "问题：夜班交通补贴怎么发？" in second.variable
    assert first.variable != second.variable


def test_先过滤是第一道_低分和检索失败都不生成正文():
    generator = FixedGenerator(f"{EXAM}[1]")
    for disposition, detail_prefix in (("low", "低分"), ("retrieval_failed", "检索失败")):
        audit = MemoryAudit()
        pipeline = GeneratePipeline(
            parents=_store(),
            audit=audit,
            generator=generator,
            entailment=ScriptedJudge(),
        )
        result = pipeline.run(_request(disposition=disposition))
        assert result.stages == (DEFENSE_STAGES[0],)
        assert result.action == "refused"
        assert "640" not in result.text
        assert generator.calls == 0
        assert audit.refusals[-1][1] == "拒答"
        assert detail_prefix in audit.refusals[-1][2]


def test_先过滤只取已过滤父块_对象存储里的其他父块不注入():
    store = _store()
    store.rows[("doc-gen", "p-secret", "v1")] = parent_record("p-secret", "受限制度补贴为每人 9999 元。")
    audit = MemoryAudit()
    pipeline = GeneratePipeline(
        parents=store,
        audit=audit,
        generator=FixedGenerator(f"{EXAM}[1]"),
        entailment=ScriptedJudge(),
    )
    result = pipeline.run(_request())
    assert store.reads == ["p-exam", "p-lodging", "p-training"]
    assert "p-secret" not in store.reads
    assert "9999" not in result.variable
    assert "受限制度" not in result.variable
    assert result.stages[0] == "先过滤"


def test_父块取不回时不生成没有出处的句子并写拒答():
    store = _store()
    del store.rows[("doc-gen", "p-lodging", "v1")]
    generator = FixedGenerator(f"{EXAM}[1]")
    audit = MemoryAudit()
    pipeline = GeneratePipeline(parents=store, audit=audit, generator=generator, entailment=ScriptedJudge())
    result = pipeline.run(_request())
    assert generator.calls == 0
    assert result.action == "refused"
    assert result.stages == ("先过滤", "编号引用")
    assert "640" not in result.text
    assert audit.citations[-1].outcome == "父块缺失"
    assert audit.refusals[-1][1] == "拒答"


def test_引用日志写不进去就不能当成已经作答():
    pipeline = GeneratePipeline(
        parents=_store(),
        audit=MemoryAudit(fail=True),
        generator=FixedGenerator(f"{EXAM}[1]{LODGING}[2]{TRAINING}[3]"),
        entailment=ScriptedJudge(),
    )
    with pytest.raises(GenerateAuditError):
        pipeline.run(_request())


def test_字符串全等的蕴含器不能用来通过生成():
    pipeline = GeneratePipeline(
        parents=_store(),
        audit=MemoryAudit(),
        generator=FixedGenerator(EXAM + "[1]"),
        entailment=StringEqualityJudge(),
    )
    with pytest.raises(EntailmentRejected):
        pipeline.run(_request())


def _store() -> MemoryParents:
    return MemoryParents(
        {
            ("doc-gen", "p-exam", "v1"): parent_record("p-exam", EXAM),
            ("doc-gen", "p-lodging", "v1"): parent_record("p-lodging", LODGING),
            ("doc-gen", "p-training", "v1"): parent_record("p-training", TRAINING),
        }
    )


def _request(**overrides) -> GenerateRequest:
    fields = {
        "question": "体检补贴、住宿报销和培训名额分别是多少？",
        "parents": (parent_ref("p-exam"), parent_ref("p-lodging"), parent_ref("p-training")),
        "disposition": "high",
        "request_id": "req-numbered",
    }
    fields.update(overrides)
    return GenerateRequest(**fields)


def _pipeline(draft: str) -> GeneratePipeline:
    return GeneratePipeline(
        parents=_store(),
        audit=MemoryAudit(),
        generator=FixedGenerator(draft),
        entailment=ScriptedJudge(),
    )
