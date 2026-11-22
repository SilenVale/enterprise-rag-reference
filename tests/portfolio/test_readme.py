"""作品集只核对九段 README 和已经通过的端到端终端记录。"""

from pathlib import Path

from rag.evaluate.report import assert_report_clean

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DEMO = (ROOT / "docs" / "demo" / "e2e-terminal.txt").read_text(encoding="utf-8")

SECTIONS = (
    "## 1. 一句话定位",
    "## 2. 它解决什么问题",
    "## 3. 演示",
    "## 4. 架构图",
    "## 5. 核心行为",
    "## 6. 取舍",
    "## 7. 快速开始",
    "## 8. 验收结果",
    "## 9. 已知局限",
)


def _section(title: str) -> str:
    start = README.index(title)
    rest = README[start + len(title) :]
    nxt = len(README)
    for other in SECTIONS:
        if other == title:
            continue
        pos = README.find(other)
        if pos > start:
            nxt = min(nxt, pos)
    return README[start:nxt]


def test_一句话定位和六层图():
    assert README.index("这个仓库公开的是一套制度问答的全链路架构") < README.index(SECTIONS[0])
    positions = [README.index(title) for title in SECTIONS]
    assert positions == sorted(positions)
    opening = _section(SECTIONS[0])
    assert "制度问答全链路" in opening
    assert "```mermaid" in opening
    for name in ("数据接入", "索引工程", "检索链路", "生成防御", "评测门禁", "存储"):
        assert name in opening
    assert "向量、BM25、权限过滤三路并发" in opening
    assert "子块负责找准，父块负责把完整条款交给生成" in README
    assert "权限和生效状态在检索引擎里先滤掉" in README
    assert "不代表某一家机构的制度" in README
    assert "材料里的召回率、幻觉率和文档规模不作为本仓库的结果" in README


def test_它解决四类问题():
    body = _section(SECTIONS[1])
    for phrase in ("解析会吞字", "旧版还在被引用", "金额会被说错", "不该看的制度会被召回"):
        assert phrase in body


def test_演示命令录制步骤和终端证据():
    body = _section(SECTIONS[2])
    assert "python -m rag.demo" in body
    assert "录制步骤" in body
    assert "docs/demo/e2e-terminal.txt" in body
    assert "终端记录" in body
    assert "不是屏幕录制" in body
    assert ".mp4" not in README and ".webm" not in README
    for step in ("闸门", "增量切换", "先过滤", "三档分流", "出口拦截"):
        assert step in body
    for evidence in (
        "解析失败进入人工队列",
        "修订后查询在切换前仍读旧版",
        "废止版本默认检索不到",
        "后过滤=否",
        "分成高中低三档",
        "错误数字被拦住",
        "废止引用被拦住",
    ):
        assert evidence in DEMO
    assert DEMO.strip().endswith("通过")
    assert_report_clean(DEMO)


def test_架构图标出存储和字段():
    body = _section(SECTIONS[3])
    assert "```mermaid" in body
    for name in ("数据接入", "索引工程", "检索链路", "生成防御", "评测门禁", "存储"):
        assert name in body
    for name in ("Milvus", "Elasticsearch", "对象存储", "元数据库", "parent_id", "内容哈希", "版本", "权限"):
        assert name in body


def test_核心行为三个小标题():
    body = _section(SECTIONS[4])
    for title in ("### 子块找、父块答", "### 权限和生效状态先过滤", "### 出口用代码校验"):
        assert title in body


def test_取舍写了三件():
    body = _section(SECTIONS[5])
    assert "RRF" in body
    assert "加权" in body
    assert "60" in body
    assert "下推进引擎" in body or "推进" in body
    assert "提示词" in body
    assert "权限" in body


def test_快速开始可以复制():
    body = _section(SECTIONS[6])
    for line in (
        "python3 -m venv .venv",
        '.venv/bin/pip install -e ".[dev]"',
        "bash deploy/start-storage.sh",
        ".venv/bin/python -m rag.demo",
    ):
        assert line in body


def test_验收结果是本仓库这次的数():
    body = _section(SECTIONS[7])
    assert "这是本仓库这次跑出来的" in body
    for number in ("0.8846", "0.0833", "0.8462", "1.0000"):
        assert number in body
    assert "Recall@5" in body
    assert "出口校验失败算出的幻觉率" in body
    assert "端到端准确率" in body
    assert "拒答正确率" in body
    assert "引用正确率" in body
    for passed in ("20 passed", "16 passed", "23 passed", "27 passed", "29 passed", "40 passed", "15 passed"):
        assert passed in body
    assert_report_clean(README)
    assert "面试题对照" not in README
    assert "简历条目" not in README


def test_已知局限和全部验收命令():
    body = _section(SECTIONS[8])
    assert "合成制度" in body
    assert "local-extractive" in body
    assert "生产规模不在这个仓库的结果里" in body
    command = README.split("## 全部验收", 1)[1]
    for path in (
        "tests/storage/live",
        "tests/ingest",
        "tests/index/live",
        "tests/retrieve",
        "tests/generate",
        "tests/evaluate",
        "tests/demo",
        "tests/portfolio",
    ):
        assert path in command
    assert "python -m rag.demo" in command
    assert "bash deploy/start-storage.sh" in command
