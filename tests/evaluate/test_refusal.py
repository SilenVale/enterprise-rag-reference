"""拒答分成知识缺失、检索失败、权限不足。只有知识缺失去补录。"""

from rag.evaluate.metrics import answer_correct
from rag.evaluate.runner import run_suite
from rag.evaluate.types import RunConfig


def test_三类拒答都出现_而且不会混进同一个队列():
    run = run_suite()
    kinds = {item.kind for item in run.refusals}
    assert kinds == {"知识缺失", "检索失败", "权限不足"}
    queued = set(run.queued_keys)
    assert queued == {item.case_key for item in run.refusals if item.kind == "知识缺失"}
    assert "sample-outage" not in queued
    assert "boundary-permission" not in queued
    assert "sample-gap" in queued


def test_知识缺失补录前用例失败_补录后再跑通过():
    before = run_suite()
    gap = before.outcome("sample-gap")
    assert gap.refused and gap.refusal_kind == "知识缺失" and gap.queue
    assert not answer_correct(gap)
    after = run_suite(RunConfig(supplemented=frozenset({"sample-gap"})))
    filled = after.outcome("sample-gap")
    assert answer_correct(filled)
    assert filled.case_key not in after.queued_keys
    assert "150" in filled.shipped_text


def test_废止版本和受限条款不进上下文():
    run = run_suite()
    obsolete = run.outcome("boundary-obsolete")
    permission = run.outcome("boundary-permission")
    version = run.outcome("boundary-version")
    assert obsolete.refused and "exam-old" not in obsolete.context_ids
    assert permission.refused and permission.refusal_kind == "权限不足"
    assert "restricted-bonus" not in permission.context_ids
    assert "640" in version.shipped_text and "480" not in version.shipped_text
    assert "exam-old" not in version.context_ids


def test_检索失败不拿已经写好的名次充数():
    run = run_suite()
    outage = run.outcome("sample-outage")
    assert outage.refusal_kind == "检索失败"
    assert outage.context_ids == ()
    assert outage.retrieval_scored is False
    assert not outage.queue
