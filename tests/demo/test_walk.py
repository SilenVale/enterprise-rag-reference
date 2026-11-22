"""端到端那条命令要在本机四个服务上走完每一格。"""

import pytest

from rag.demo.walk import run_demo

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def report():
    result = run_demo(echo=False)
    if not result.passed:
        pytest.fail("\n".join(result.lines))
    return result


def _line(report, demo: str) -> str:
    matches = []
    for line in report.lines:
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 3 and parts[2] == demo:
            matches.append(line)
    assert len(matches) == 1
    return matches[0]


def test_四个服务都连着本机实现(report):
    text = "\n".join(report.lines)
    assert "MilvusChildVectorStore" in text
    assert "ElasticsearchChildTextStore" in text
    assert "MinioBlobStore" in text
    assert "PostgresMetadataStore" in text
    assert "127.0.0.1:19530" in text
    assert "127.0.0.1:9200" in text
    assert "127.0.0.1:9000" in text
    assert "127.0.0.1:5432" in text
    assert "memory" not in text


def test_解析失败进入人工队列(report):
    line = _line(report, "解析失败进入人工队列")
    assert "失败处置。重试，然后人工队列" in line
    assert "状态=queued" in line
    assert "尝试=3" in line
    assert "进入索引=否" in line
    assert "队列=元数据库" in line


def test_子块命中后取回父块(report):
    line = _line(report, "子块命中后取回父块")
    assert "父子语义分块。子块检索，父块注入" in line
    assert "取回=对象存储" in line
    assert "向量命中含正文=否" in line


def test_同一内容哈希不重复入库(report):
    line = _line(report, "同一内容哈希不重复入库")
    assert "内容寻址去重。按块做差量" in line
    assert "复用=是" in line
    assert "再次嵌入=0" in line


def test_修订后查询在切换前仍读旧版(report):
    line = _line(report, "修订后查询在切换前仍读旧版")
    assert "废止制度打标。保留可追溯" in line
    assert "指针=v1" in line
    assert "检索版本=v1" in line
    assert "新版状态=pending" in line


def test_废止版本默认检索不到(report):
    line = _line(report, "废止版本默认检索不到")
    assert "废止制度打标。保留可追溯" in line
    assert "默认版本=v2" in line
    assert "废止版本=未出现" in line
    assert "审计可带出=是" in line


def test_无权限的角色检索不到受限制度(report):
    line = _line(report, "无权限的角色检索不到受限制度")
    assert "三路并发召回。向量、BM25 稀疏、元数据过滤" in line
    assert "角色=intern" in line
    assert "引擎权限=public" in line
    assert "受限命中=0" in line
    assert "后过滤=否" in line


def test_原问题和改写都参与召回(report):
    line = _line(report, "原问题和改写都参与召回")
    assert "查询改写。指代消解，复合问题拆解" in line
    assert "原问题=那一条的上限是多少" in line
    assert "改写=差旅住宿标准第三条的上限是多少" in line
    assert "Qwen2.5-72B" not in line


def test_RRF之后精排分成高中低三档(report):
    fused = _line(report, "融合用名次")
    assert "RRF 融合。免分数归一化" in fused
    assert "k=60" in fused
    ranked = _line(report, "RRF之后精排")
    assert "Cross-Encoder 精排。bge-reranker-v2-m3" in ranked
    assert "bge-reranker-v2-m3" in ranked
    tiers = _line(report, "分成高中低三档")
    assert "三档分流。高分作答，中分提示，低分拒答" in tiers
    assert "高=生成答案并引用" in tiers
    assert "中=生成答案并附上可能不完整、建议人工核实" in tiers
    assert "低=拒答并转人工" in tiers


def test_答案带编号且没有加载生成主模型(report):
    line = _line(report, "答案带编号")
    assert "编号注入提示词。逐句标注来源" in line
    assert "编号=[1][2]" in line
    assert "模型=local-extractive" in line
    assert "权重已加载=否" in line
    skipped = _line(report, "未加载")
    assert "Qwen2.5-72B" in skipped
    assert "权重=false" in skipped


def test_错误数字被拦住(report):
    line = _line(report, "错误数字被拦住")
    assert "数值一致性校验。数字必须见于引用原文" in line
    assert "处置=拦截数字" in line
    assert "交卷含错误数字=否" in line
    assert "引用日志=元数据库" in line


def test_废止引用被拦住(report):
    line = _line(report, "废止引用被拦住")
    assert "版本一致性拦截。全景图写废止版本直接拒答" in line
    assert "处置=拦截废止" in line
    assert "废止金额交卷=否" in line


def test_蕴含不支持的句子被删掉(report):
    line = _line(report, "蕴含不支持的句子被删掉")
    assert "NLI 支持性校验。全景图写不通过删句" in line
    assert "处置=删句" in line
    assert "不支持句交卷=否" in line
    assert "整条降级=否" in line


def test_评测回退会阻断(report):
    line = _line(report, "评测回退会阻断")
    assert "CI 回归门禁。指标回退阻断发布" in line
    assert "阻断=是" in line
    assert "依据=元数据库基线" in line
    assert "Recall@5" in line


def test_结果是通过(report):
    assert report.lines[-1].endswith("通过")
    assert report.passed
