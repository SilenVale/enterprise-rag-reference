import pytest

from rag.retrieve.errors import RerankerUnavailable
from rag.retrieve.params import GENERATION_MAIN_MODEL, REWRITE_MODEL_ID
from rag.retrieve.rerank import BgeReranker
from rag.retrieve.rewrite import QueryRewriteModel
from rag.retrieve.types import Turn

HISTORY = (
    Turn("user", "差旅住宿标准写在哪一条"),
    Turn("assistant", "写在差旅制度第三条"),
)


def test_followup_pronoun_becomes_a_standalone_question():
    plan = QueryRewriteModel().rewrite("那一条的上限是多少", HISTORY)
    assert plan.standalone == "差旅住宿标准第三条的上限是多少"
    assert plan.queries()[0] == "那一条的上限是多少"
    assert plan.standalone in plan.queries()
    assert plan.failed is False


def test_possessive_pronoun_uses_history():
    plan = QueryRewriteModel().rewrite("它的上限是多少", HISTORY)
    assert plan.standalone == "差旅住宿标准第三条的上限是多少"


def test_compound_question_splits_into_two_or_three():
    two = QueryRewriteModel().rewrite("住宿上限和报销时限分别是什么", ())
    assert two.subqueries == ("住宿上限是什么", "报销时限是什么")
    assert two.queries()[0] == "住宿上限和报销时限分别是什么"
    three = QueryRewriteModel().rewrite("住宿上限、报销时限和座席等级分别是什么", ())
    assert three.subqueries == ("住宿上限是什么", "报销时限是什么", "座席等级是什么")
    via = QueryRewriteModel().rewrite("住宿上限以及报销时限", ())
    assert via.subqueries == ("住宿上限", "报销时限")
    capped = QueryRewriteModel().rewrite("甲、乙、丙和丁分别是什么", ())
    assert capped.subqueries == ("甲是什么", "乙是什么", "丙是什么")
    assert all("丁" not in item for item in capped.subqueries)


def test_topic_change_is_not_rewritten_from_history():
    plan = QueryRewriteModel().rewrite("食堂补贴怎么发放", HISTORY)
    assert plan.standalone is None
    assert plan.subqueries == ()
    assert plan.queries() == ("食堂补贴怎么发放",)
    assert "差旅" not in plan.queries()[0]


def test_rewrite_model_is_not_the_generation_model():
    model = QueryRewriteModel()
    assert model.model_id == REWRITE_MODEL_ID
    assert model.model_id != GENERATION_MAIN_MODEL
    with pytest.raises(ValueError):
        QueryRewriteModel(GENERATION_MAIN_MODEL)


def test_reranker_refuses_a_substitute_model_before_download():
    with pytest.raises(RerankerUnavailable):
        BgeReranker("BAAI/bge-reranker-base").load()
