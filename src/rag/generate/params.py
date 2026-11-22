"""写死的引用格式，以及材料留白、由本仓库选定的参数。"""

# 材料点名的生产模型。配置里保留这个名字。验收不在本机加载它。
VLLM_MODEL_ID = "Qwen2.5-72B"

# 本机演示和验收用的生成档，不下载 72B。文档未写死。
LOCAL_MODEL_ID = "local-extractive"

# 蕴含模型。材料没有写死名字。文档未写死。
ENTAILMENT_MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

# 删句之后还算读得通的最短正文长度。悬空连词单独成句时不算读得通。文档未写死。
MIN_READABLE_CHARS = 12
DANGLING_PREFIXES = ("因此", "所以", "但是", "然而", "并且", "同时", "此外", "另外")

# 一次最多把多少个已过滤父块编上号。材料用 [1] [2] [3] 举例，没有封死条数。文档未写死。
MAX_INJECTED_PARENTS = 5

# 本机档从父块里摘句时，至少要和问题重合这么多个字。文档未写死。
LOCAL_OVERLAP_CHARS = 2

CAVEAT = "可能不完整、建议人工核实。"
REFUSAL_TEXT = "无法根据已引用的条款给出可核对的答案，已转人工。"
DOWNGRADE_TEXT = "删去得不到支持的句子后答案读不通。可能不完整、建议人工核实。"

DEFENSE_STAGES = ("先过滤", "编号引用", "出口三道")
EXIT_CHECKS = ("数字", "版本", "蕴含")
