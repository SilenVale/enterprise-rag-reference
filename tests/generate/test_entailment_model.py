"""蕴含用选定的模型做支持、中立、矛盾。字符串全等不能算通过。"""

import pytest

from rag.generate.entailment import EntailmentJudge
from rag.generate.exit_checks import apply_exit_checks
from rag.generate.params import ENTAILMENT_MODEL_ID
from rag.generate.pipeline import GeneratePipeline
from rag.generate.types import GenerateRequest

from support import (
    CONTRADICTION,
    EXAM,
    LODGING,
    PARAPHRASE,
    FixedGenerator,
    MemoryAudit,
    MemoryParents,
    StringEqualityJudge,
    cited,
    parent_record,
    parent_ref,
)


@pytest.fixture(scope="module")
def judge():
    model = EntailmentJudge()
    model.load()
    return model


def test_蕴含模型没有写死名字_加载的是仓库选定的那一个且不是字符串全等(judge):
    assert judge.model_id == ENTAILMENT_MODEL_ID
    assert judge.resolved_name == ENTAILMENT_MODEL_ID
    assert judge.decides_by_string_equality is False
    assert judge.ready


def test_字符串全等不能当作蕴含校验_改写后的句子仍可由模型判为支持(judge):
    assert PARAPHRASE != EXAM
    assert PARAPHRASE not in EXAM
    equality = StringEqualityJudge()
    assert equality.label(EXAM, PARAPHRASE) != "支持"
    assert judge.label(EXAM, PARAPHRASE) == "支持"
    assert judge.label(EXAM, EXAM) == "支持"
    assert judge.label(EXAM, CONTRADICTION) != "支持"
    unrelated = "夜班员工每周一可以在食堂领取一顿免费夜宵。"
    assert judge.label(EXAM, unrelated) != "支持"


def test_真实蕴含模型不支持的句子会被删掉_全等判断做不到这一步(judge):
    blocks = (cited(1, "p-exam", EXAM), cited(2, "p-lodging", LODGING))
    draft = f"{PARAPHRASE}[1]{CONTRADICTION}[1]{LODGING}[2]"
    result = apply_exit_checks(draft, blocks, judge)
    assert result.action == "answer"
    assert "640" in result.text
    assert "320" in result.text
    assert "停发" not in result.text
    assert result.entailment_deletion_executed is True
    store = MemoryParents(
        {
            ("doc-gen", "p-exam", "v1"): parent_record("p-exam", EXAM),
            ("doc-gen", "p-lodging", "v1"): parent_record("p-lodging", LODGING),
        }
    )
    pipeline = GeneratePipeline(
        parents=store,
        audit=MemoryAudit(),
        generator=FixedGenerator(draft),
        entailment=judge,
    )
    answered = pipeline.run(
        GenerateRequest(
            question="体检补贴和住宿报销是多少？",
            parents=(parent_ref("p-exam"), parent_ref("p-lodging")),
            request_id="req-nli",
        )
    )
    assert answered.action == "answer"
    assert answered.checks_skipped is False
    assert "停发" not in answered.text
    assert answered.model_id == "fixed-draft"
