"""本地能跑完的评测集。三类都有条目，边界盖住五个场景。

条数按这个仓库的夹具来，不抄材料里的规模。
"""

from rag.evaluate.errors import DatasetError
from rag.evaluate.params import DATASET_VERSION
from rag.evaluate.types import Clause, EvalCase

REQUIRED_CATEGORIES = ("采样", "边界", "badcase")
REQUIRED_BOUNDARIES = ("新旧版本", "废止制度", "跨条款", "权限", "表格里的数字")


def _clause(
    clause_id: str,
    text: str,
    *,
    status: str = "effective",
    permission: str = "public",
    version: str = "v3",
    end: str | None = None,
) -> Clause:
    return Clause(
        clause_id=clause_id,
        text=text,
        claim=text,
        version=version,
        effective_status=status,
        permission=permission,
        effective_from="2024-01-01",
        effective_to=end,
    )


EXAM = _clause("exam-new", "员工年度体检补贴为每人 640 元，于每年三月发放。")
EXAM_OLD = _clause(
    "exam-old",
    "员工年度体检补贴为每人 480 元，于每年三月发放。",
    status="obsolete",
    version="v1",
    end="2024-12-31",
)
LODGING = _clause("lodging", "出差住宿报销上限为每晚 320 元。")
RECEIPT = _clause("receipt", "差旅票据应当在返回后十日内提交。")
TRAINING = _clause("training", "年度培训名额为每个部门 12 人，名额不作跨年结转。")
BADGE = _clause("badge", "门禁卡丢失后应当在当日挂失并补办。")
CANTEEN = _clause("canteen", "员工食堂午餐供应时间为十一时至十三时。")
COMM = _clause("comm", "员工通讯补贴为每人每月 80 元，随工资发放。")
MEAL = _clause("meal", "员工餐补为每人每个工作日 25 元。")
TABLE = _clause("table-stay", "住宿上限表：一线城市每晚 450 元，其他城市每晚 260 元。")
REMOTE = _clause("remote", "远程办公补贴为每人每月 150 元，按实际远程天数折算。")
EDGE = _clause("edge", "档案查阅应当提前一个工作日预约。")
RESTRICTED = _clause(
    "restricted-bonus",
    "专项激励的计提比例为基本工资的 15%，仅限授权岗位查阅。",
    permission="restricted",
)


def _case(case_key: str, category: str, **kwargs) -> EvalCase:
    return EvalCase(case_key=case_key, category=category, **kwargs)


CASES: tuple[EvalCase, ...] = (
    _case(
        "sample-amount",
        "采样",
        question_type="金额",
        role="员工",
        domain="福利",
        question="员工年度体检补贴每人多少",
        clauses=(EXAM, LODGING),
        ranking=("exam-new", "lodging"),
        gold_ids=("exam-new",),
        expected_points=("640",),
    ),
    _case(
        "sample-clause",
        "采样",
        question_type="条款号",
        role="员工",
        domain="差旅",
        question="差旅制度里住宿报销上限是多少",
        clauses=(LODGING, TRAINING),
        ranking=("lodging", "training"),
        gold_ids=("lodging",),
        expected_points=("320",),
    ),
    _case(
        "sample-procedure",
        "采样",
        question_type="流程",
        role="实习生",
        domain="行政",
        question="门禁卡丢失后怎么处理",
        clauses=(CANTEEN, BADGE),
        ranking=("canteen", "badge"),
        gold_ids=("badge",),
        expected_points=("当日挂失",),
    ),
    _case(
        "sample-definition",
        "采样",
        question_type="定义",
        role="审计",
        domain="人事",
        question="培训名额能不能跨年结转",
        clauses=(TRAINING,),
        ranking=("training",),
        gold_ids=("training",),
        expected_points=("不作跨年结转",),
    ),
    _case(
        "sample-partial",
        "采样",
        question_type="复合",
        role="员工",
        domain="福利",
        question="通讯补贴和餐补分别是多少",
        clauses=(COMM, MEAL, EXAM, LODGING, TRAINING, BADGE),
        ranking=("comm", "exam-new", "lodging", "training", "badge", "meal"),
        gold_ids=("comm", "meal"),
        expected_points=("80", "25"),
    ),
    _case(
        "sample-mixed",
        "采样",
        question_type="金额",
        role="员工",
        domain="福利",
        question="体检补贴的标准是什么",
        clauses=(EXAM, LODGING),
        ranking=("exam-new", "lodging"),
        gold_ids=("exam-new",),
        expected_points=("640",),
        forbidden_points=("200",),
    ),
    _case(
        "sample-rrf",
        "采样",
        question_type="流程",
        role="员工",
        domain="行政",
        question="查档案要提前多久预约",
        clauses=(EDGE,),
        ranking=("d1", "d2", "d3", "d4", "edge", "d5"),
        gold_ids=("edge",),
        expected_points=("预约",),
        rrf_sensitive=True,
    ),
    _case(
        "sample-gap",
        "采样",
        question_type="知识缺失",
        role="员工",
        domain="人事",
        question="远程办公补贴每人每月多少",
        clauses=(REMOTE, CANTEEN),
        ranking=(),
        gold_ids=("remote",),
        expected_points=("150",),
        supplement_clause_id="remote",
    ),
    _case(
        "sample-outage",
        "采样",
        question_type="检索失败",
        role="员工",
        domain="差旅",
        question="住宿报销上限是多少",
        clauses=(LODGING,),
        ranking=("lodging",),
        gold_ids=("lodging",),
        expected_points=("320",),
        expect_refusal=True,
        retrieval_failed=True,
    ),
    _case(
        "boundary-version",
        "边界",
        boundary="新旧版本",
        question_type="金额",
        role="员工",
        domain="福利",
        question="现在的体检补贴是多少",
        clauses=(EXAM, EXAM_OLD, LODGING),
        ranking=("exam-new", "lodging"),
        gold_ids=("exam-new",),
        expected_points=("640",),
        forbidden_points=("480",),
    ),
    _case(
        "boundary-obsolete",
        "边界",
        boundary="废止制度",
        question_type="废止",
        role="员工",
        domain="福利",
        question="已经废止的体检补贴标准还能不能拿来回答",
        clauses=(EXAM_OLD,),
        ranking=("exam-old",),
        gold_ids=(),
        expected_points=(),
        expect_refusal=True,
    ),
    _case(
        "boundary-cross",
        "边界",
        boundary="跨条款",
        question_type="复合",
        role="员工",
        domain="差旅",
        question="住宿上限是多少，票据要在几天内提交",
        clauses=(LODGING, RECEIPT, EXAM),
        ranking=("lodging", "receipt", "exam-new"),
        gold_ids=("lodging", "receipt"),
        expected_points=("320", "十日"),
    ),
    _case(
        "boundary-permission",
        "边界",
        boundary="权限",
        question_type="权限",
        role="实习生",
        domain="薪酬",
        question="专项激励的计提比例是多少",
        clauses=(RESTRICTED, CANTEEN),
        ranking=("restricted-bonus", "canteen"),
        gold_ids=(),
        expected_points=(),
        expect_refusal=True,
        permission_denied=True,
    ),
    _case(
        "boundary-table",
        "边界",
        boundary="表格里的数字",
        question_type="表格",
        role="员工",
        domain="差旅",
        question="一线城市的住宿上限是多少",
        clauses=(TABLE, LODGING),
        ranking=("table-stay", "lodging"),
        gold_ids=("table-stay",),
        expected_points=("450",),
        forbidden_points=("710",),
    ),
    _case(
        "badcase-number",
        "badcase",
        question_type="金额",
        role="员工",
        domain="福利",
        question="体检补贴每人多少",
        clauses=(EXAM,),
        ranking=("exam-new",),
        gold_ids=("exam-new",),
        expected_points=("640",),
        forbidden_points=("960",),
        fixed_note="这条曾经把金额写成引用里没有的数字，修复后只保留原文里的金额。",
    ),
    _case(
        "badcase-table",
        "badcase",
        question_type="表格",
        role="员工",
        domain="差旅",
        question="一线城市住宿上限是多少",
        clauses=(TABLE,),
        ranking=("table-stay",),
        gold_ids=("table-stay",),
        expected_points=("450",),
        forbidden_points=("710",),
        fixed_note="这条曾经把表里两档上限加总后回答，修复后只回答被问到的那一档。",
    ),
    _case(
        "badcase-obsolete",
        "badcase",
        question_type="废止",
        role="员工",
        domain="福利",
        question="废止文本里的体检补贴金额是多少",
        clauses=(EXAM_OLD,),
        ranking=("exam-old",),
        gold_ids=(),
        expected_points=(),
        expect_refusal=True,
        fixed_note="这条曾经把废止版本的金额写进答案，修复后不再引用它。",
    ),
)


def validate_dataset(cases: tuple[EvalCase, ...] = CASES) -> None:
    categories = {case.category for case in cases}
    missing = [name for name in REQUIRED_CATEGORIES if name not in categories]
    if missing:
        raise DatasetError("评测集缺了类别：" + "、".join(missing))
    boundaries = {case.boundary for case in cases if case.category == "边界"}
    missing_boundary = [name for name in REQUIRED_BOUNDARIES if name not in boundaries]
    if missing_boundary:
        raise DatasetError("边界没有盖全：" + "、".join(missing_boundary))
    if any(case.category == "边界" and not case.boundary for case in cases):
        raise DatasetError("边界条目没有标明盖的是哪一种")
    seen: set[str] = set()
    for case in cases:
        if case.case_key in seen:
            raise DatasetError(f"评测条目重复：{case.case_key}")
        seen.add(case.case_key)
        known = {clause.clause_id for clause in case.clauses}
        unknown_gold = [clause_id for clause_id in case.gold_ids if clause_id not in known]
        if unknown_gold:
            raise DatasetError(f"{case.case_key} 的该找条款不在本条夹具里")
        if case.supplement_clause_id and case.supplement_clause_id not in known:
            raise DatasetError(f"{case.case_key} 的补录条款不在本条夹具里")
        if not case.expect_refusal and not case.expected_points:
            raise DatasetError(f"{case.case_key} 应该作答，却没有要核对的要点")


def dataset_version() -> str:
    return DATASET_VERSION
