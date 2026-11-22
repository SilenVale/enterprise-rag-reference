"""拒答和引用写进已经通过的元数据库，父块从对象存储取。"""

import pytest

from rag.generate.adapters import LocalExtractiveGenerator
from rag.generate.entailment import EntailmentJudge
from rag.generate.params import DOWNGRADE_TEXT
from rag.generate.pipeline import GeneratePipeline
from rag.generate.types import GenerateRequest
from rag.storage.hashing import content_hash

from support import CONTRADICTION, DANGLING, EXAM, LODGING, PARAPHRASE, FixedGenerator, ScriptedJudge, parent_ref


@pytest.fixture(scope="module")
def judge():
    model = EntailmentJudge()
    model.load()
    return model


def test_父块从对象存储注入编号_拒答和放行都写进元数据库(parent_store, generate_log, judge):
    document_id, exam_id, lodging_id = _ids("live")
    _write(parent_store, document_id, exam_id, EXAM)
    _write(parent_store, document_id, lodging_id, LODGING)
    request_id = document_id + "-ok"
    pipeline = GeneratePipeline(
        parents=parent_store,
        audit=generate_log,
        generator=LocalExtractiveGenerator(),
        entailment=judge,
    )
    result = pipeline.run(
        GenerateRequest(
            question="体检补贴每人是多少，住宿报销上限是多少？",
            parents=(
                parent_ref(exam_id, document_id=document_id),
                parent_ref(lodging_id, document_id=document_id),
            ),
            request_id=request_id,
        )
    )
    assert result.action == "answer"
    assert "[1]" in result.variable and "[2]" in result.variable
    assert result.text.endswith("[2]") or "[2]" in result.text
    assert "640" in result.text
    citations = generate_log.list_citations(request_id)
    assert citations
    assert {row.outcome for row in citations} == {"放行"}
    assert generate_log.list_refusals(request_id) == []

    missing_id = document_id + "-missing"
    missing = GeneratePipeline(
        parents=parent_store,
        audit=generate_log,
        generator=FixedGenerator(f"{EXAM}[1]"),
        entailment=ScriptedJudge(),
    ).run(
        GenerateRequest(
            question="体检补贴是多少？",
            parents=(parent_ref("missing-parent", document_id=document_id),),
            request_id=missing_id,
        )
    )
    assert missing.action == "refused"
    assert generate_log.list_citations(missing_id)[0].outcome == "父块缺失"
    assert generate_log.list_refusals(missing_id)[0][0] == "拒答"


def test_数字和废止的拦截写进元数据库(parent_store, generate_log, judge):
    document_id, exam_id, lodging_id = _ids("guard")
    _write(parent_store, document_id, exam_id, EXAM)
    _write(parent_store, document_id, lodging_id, LODGING)
    number_id = document_id + "-number"
    number = GeneratePipeline(
        parents=parent_store,
        audit=generate_log,
        generator=FixedGenerator("员工年度体检补贴为每人 1750 元。[1]"),
        entailment=judge,
    ).run(
        GenerateRequest(
            question="体检补贴是多少？",
            parents=(parent_ref(exam_id, document_id=document_id),),
            request_id=number_id,
        )
    )
    assert number.action == "refused"
    assert "1750" not in number.text
    assert "拦截数字" in {row.outcome for row in generate_log.list_citations(number_id)}
    assert generate_log.list_refusals(number_id)[0][0] == "拒答"

    obsolete_id = document_id + "-obsolete"
    obsolete = GeneratePipeline(
        parents=parent_store,
        audit=generate_log,
        generator=FixedGenerator(f"{EXAM}[1]{LODGING}[2]"),
        entailment=judge,
    ).run(
        GenerateRequest(
            question="体检补贴和住宿报销是多少？",
            parents=(
                parent_ref(exam_id, document_id=document_id, status="obsolete"),
                parent_ref(lodging_id, document_id=document_id),
            ),
            request_id=obsolete_id,
        )
    )
    assert obsolete.action == "answer"
    assert "640" not in obsolete.text
    assert "320" in obsolete.text
    outcomes = {row.outcome for row in generate_log.list_citations(obsolete_id)}
    assert "拦截废止" in outcomes
    assert "放行" in outcomes
    assert generate_log.list_refusals(obsolete_id) == []


def test_真实蕴含不支持的句子在仍可读时写入删句_读不通时只记降级(parent_store, generate_log, judge):
    document_id, exam_id, lodging_id = _ids("nli")
    _write(parent_store, document_id, exam_id, EXAM)
    _write(parent_store, document_id, lodging_id, LODGING)
    kept_id = document_id + "-kept"
    kept = GeneratePipeline(
        parents=parent_store,
        audit=generate_log,
        generator=FixedGenerator(f"{PARAPHRASE}[1]{CONTRADICTION}[1]{LODGING}[2]"),
        entailment=judge,
    ).run(
        GenerateRequest(
            question="体检补贴和住宿报销是多少？",
            parents=(
                parent_ref(exam_id, document_id=document_id),
                parent_ref(lodging_id, document_id=document_id),
            ),
            request_id=kept_id,
        )
    )
    assert PARAPHRASE not in EXAM
    assert kept.action == "answer"
    assert kept.entailment_deletion_executed is True
    assert "停发" not in kept.text
    assert "640" in kept.text
    assert "删句" in {row.outcome for row in generate_log.list_citations(kept_id)}
    assert generate_log.list_refusals(kept_id) == []

    dropped_id = document_id + "-drop"
    dropped = GeneratePipeline(
        parents=parent_store,
        audit=generate_log,
        generator=FixedGenerator(f"{CONTRADICTION}[1]{DANGLING}[2]"),
        entailment=ScriptedJudge({"已经停发": "矛盾", "因此按照": "支持"}),
    ).run(
        GenerateRequest(
            question="体检补贴是多少？",
            parents=(
                parent_ref(exam_id, document_id=document_id),
                parent_ref(lodging_id, document_id=document_id),
            ),
            request_id=dropped_id,
        )
    )
    assert dropped.entailment_deletion_executed is False
    assert dropped.action == "downgraded"
    assert dropped.text == DOWNGRADE_TEXT
    assert "因此按照" not in dropped.text
    outcomes = {row.outcome for row in generate_log.list_citations(dropped_id)}
    assert "删句" not in outcomes
    assert generate_log.list_refusals(dropped_id)[0][0] == "降级"


def _ids(prefix: str) -> tuple[str, str, str]:
    import uuid

    token = uuid.uuid4().hex[:10]
    return f"{prefix}-{token}", f"p-exam-{token}", f"p-stay-{token}"


def _write(store, document_id: str, parent_id: str, text: str) -> None:
    store.write_parent(document_id, "v1", parent_id, text, content_hash(text))
