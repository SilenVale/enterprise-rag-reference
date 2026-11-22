import pytest

from rag.retrieve.params import CAVEAT, HIGH_SCORE, MID_SCORE, RERANK_POOL, RERANK_TOP_K
from rag.retrieve.pipeline import RetrievePipeline
from rag.retrieve.rerank import tier_of
from rag.retrieve.types import RetrieveRequest

from fixtures import index_rows, make_row, new_type

pytestmark = pytest.mark.live


@pytest.fixture
def corpus(embedder, vectors, lexical):
    created: list[str] = []

    def add(rows: list[dict]) -> None:
        created.extend(index_rows(embedder, vectors, lexical, rows))

    yield add
    if created:
        vectors.delete_children(created)
        lexical.delete_children(created)


def test_pool_size_sits_between_50_and_100():
    assert 50 <= RERANK_POOL <= 100
    assert RERANK_TOP_K == 5
    assert HIGH_SCORE > MID_SCORE


def test_real_reranker_separates_three_tiers(reranker):
    question = "差旅住宿每晚上限是多少"
    scores = {
        "high": reranker.score([(question, "差旅住宿每晚上限为六百元。")])[0],
        "mid": reranker.score([(question, "本条讨论差旅住宿报销，上限金额未完整给出。")])[0],
        "low": reranker.score([(question, "门禁卡丢失后应当在当日挂失并补办。")])[0],
    }
    assert 0.0 < scores["low"] < scores["mid"] < scores["high"] < 1.0
    assert tier_of(scores["high"]) == "high"
    assert tier_of(scores["mid"]) == "mid"
    assert tier_of(scores["low"]) == "low"


def test_pipeline_routes_a_direct_match_without_writing_an_answer(
    corpus, embedder, vectors, lexical, reranker, refusals
):
    document_type = new_type()
    text = "差旅住宿每晚上限为六百元。"
    corpus([make_row(text, document_type=document_type, clause_number="第三条")])
    request_id = new_type()
    result = RetrievePipeline(
        embedder=embedder, vectors=vectors, lexical=lexical, reranker=reranker, refusals=refusals
    ).retrieve(RetrieveRequest("差旅住宿每晚上限是多少", document_type=document_type, request_id=request_id))
    assert result.generated_text == ""
    assert result.passages
    assert len(result.passages) <= RERANK_TOP_K
    assert result.disposition == tier_of(result.passages[0].rerank_score)
    assert result.disposition == "high"
    assert result.route_label == "生成答案并引用"
    assert result.caveat == ""
    assert result.handoff is False
    assert result.passages[0].text == text
    assert refusals.list_for(request_id) == []


def test_rerank_keeps_five_passages(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    corpus(
        [
            make_row(
                f"差旅住宿每晚上限为六百元，补充说明第{index}款。",
                document_type=document_type,
                clause_number=f"第{index}条",
            )
            for index in range(1, 9)
        ]
    )
    result = RetrievePipeline(
        embedder=embedder, vectors=vectors, lexical=lexical, reranker=reranker, refusals=refusals
    ).retrieve(RetrieveRequest("差旅住宿每晚上限是多少", document_type=document_type, request_id=new_type()))
    assert len(result.passages) == RERANK_TOP_K
    assert result.generated_text == ""
    scores = [item.rerank_score for item in result.passages]
    assert scores == sorted(scores, reverse=True)


def test_low_score_refuses_and_hands_off(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    corpus([make_row("门禁卡丢失后应当在当日挂失并补办。", document_type=document_type)])
    request_id = new_type()
    result = RetrievePipeline(
        embedder=embedder, vectors=vectors, lexical=lexical, reranker=reranker, refusals=refusals
    ).retrieve(
        RetrieveRequest("量子引力波段的食堂菜谱如何编排", document_type=document_type, request_id=request_id)
    )
    assert result.generated_text == ""
    assert result.disposition == "low"
    assert result.handoff is True
    assert result.route_label == "拒答并转人工"
    assert result.caveat == ""
    assert refusals.list_for(request_id)[0][0] == "低分拒答"


def test_mid_score_keeps_the_caveat(corpus, embedder, vectors, lexical, reranker, refusals):
    document_type = new_type()
    text = "本条讨论差旅住宿报销，上限金额未完整给出。"
    corpus([make_row(text, document_type=document_type, clause_number="第四条")])
    result = RetrievePipeline(
        embedder=embedder, vectors=vectors, lexical=lexical, reranker=reranker, refusals=refusals
    ).retrieve(RetrieveRequest("差旅住宿每晚上限是多少", document_type=document_type, request_id=new_type()))
    assert result.generated_text == ""
    assert result.passages
    assert result.disposition == "mid"
    assert result.caveat == CAVEAT
    assert "可能不完整、建议人工核实" in result.route_label
    assert result.handoff is False
