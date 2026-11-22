"""评测条目、拒答和补录队列写进元数据库。内存结果不能代替这一步。"""

import uuid

import pytest

from rag.evaluate.cases import CASES, validate_dataset
from rag.evaluate.gate import compare_to_baseline
from rag.evaluate.metrics import answer_correct
from rag.evaluate.runner import run_suite
from rag.evaluate.store import EvaluateStore, numbers_to_payload
from rag.evaluate.types import RunConfig
from rag.storage.errors import StorageUnavailable


def test_元数据库连不上就不能当成已经留下记录():
    offline = EvaluateStore("postgresql://rag:rag-storage@127.0.0.1:1/rag")
    with pytest.raises(StorageUnavailable):
        offline.ensure_schema()


def test_评测集写入后再读回_三类和边界还在(store):
    store.replace_cases(CASES)
    loaded = store.load_cases()
    validate_dataset(loaded)
    by_key = {case.case_key: case for case in loaded}
    assert set(by_key) == {case.case_key for case in CASES}
    for case in CASES:
        assert by_key[case.case_key] == case


def test_拒答日志和补录队列能按运行编号读回(store):
    run_id = uuid.uuid4().hex
    run = run_suite()
    store.save_refusals(run_id, run.refusals)
    stored = store.list_refusals(run_id)
    assert {(row.case_key, row.kind) for row in stored} == {(row.case_key, row.kind) for row in run.refusals}
    assert set(store.list_queue(run_id)) == set(run.queued_keys)
    assert "sample-outage" not in store.list_queue(run_id)
    assert "boundary-permission" not in store.list_queue(run_id)
    store.mark_supplemented(run_id, "sample-gap")
    assert "sample-gap" not in store.list_queue(run_id)
    assert "sample-gap" in store.list_queue(run_id, status="supplemented")


def test_基线写进元数据库后_topK_回退会阻断(store):
    store.replace_cases(CASES)
    loaded = store.load_cases()
    baseline = run_suite(cases=loaded)
    run_id = uuid.uuid4().hex
    store.save_report(run_id, numbers_to_payload(baseline.numbers), baseline=True)
    current = run_suite(RunConfig(top_k=1), cases=loaded)
    decision = compare_to_baseline(store.load_baseline(), current.numbers)
    assert decision.blocked
    assert any("Recall@5" in reason for reason in decision.reasons)
    filled = run_suite(RunConfig(supplemented=frozenset({"sample-gap"})), cases=loaded)
    assert answer_correct(filled.outcome("sample-gap"))
