"""评测集要有三类，边界要盖住五个场景。规模按本地夹具，不抄材料里的条数。"""

import pytest

from rag.evaluate.cases import CASES, REQUIRED_BOUNDARIES, REQUIRED_CATEGORIES, validate_dataset
from rag.evaluate.errors import DatasetError
from rag.evaluate.params import DATASET_VERSION


def test_三类都有条目_并且本地跑得完():
    validate_dataset(CASES)
    counts = {name: sum(1 for case in CASES if case.category == name) for name in REQUIRED_CATEGORIES}
    assert all(counts[name] >= 1 for name in REQUIRED_CATEGORIES)
    assert len(CASES) < 40
    assert len({case.case_key for case in CASES}) == len(CASES)


def test_边界至少覆盖新旧版本_废止_跨条款_权限和表格数字():
    covered = {case.boundary for case in CASES if case.category == "边界"}
    assert set(REQUIRED_BOUNDARIES) <= covered


def test_采样按问题类型_角色和文档域分开():
    samples = [case for case in CASES if case.category == "采样"]
    assert len({case.question_type for case in samples}) >= 3
    assert {"员工", "实习生", "审计"} <= {case.role for case in samples}
    assert len({case.domain for case in samples}) >= 3


def test_已修复的_badcase_留着修复说明():
    fixed = [case for case in CASES if case.category == "badcase"]
    assert fixed
    assert all(case.fixed_note for case in fixed)


def test_评测集带版本号():
    assert DATASET_VERSION


def test_缺了某一类就不能当成评测集通过():
    kept = tuple(case for case in CASES if case.category != "边界")
    with pytest.raises(DatasetError):
        validate_dataset(kept)


def test_边界少盖一种就不能当成评测集通过():
    kept = tuple(case for case in CASES if case.boundary != "表格里的数字")
    with pytest.raises(DatasetError):
        validate_dataset(kept)
