"""材料留白、由本仓库选定的评测参数。这些数都标注文档未写死。"""

# 没有另接生成模型当裁判。门禁以可执行清单为准。
JUDGE_MODEL_ID = ""

# 评测集版本。调参用的例子不放进这一版。文档未写死。
DATASET_VERSION = "local-1"
SUITE_NAME = "gate-local"

# 参考配置。topK 是送进上下文的条数。RRF 的 k 与检索层写死的 60 对齐，这里只在它被改动时重跑。
DEFAULT_TOP_K = 5
REFERENCE_RRF_K = 60
REFERENCE_HIGH_SCORE = 0.72
REFERENCE_MID_SCORE = 0.30
REFERENCE_PROMPT_ID = "numbered-v1"
REFERENCE_GENERATOR_ID = "local-extractive"

# 召回到全部该找条款、只召回到一部分、一条都没有时的示意分。文档未写死。
SCORE_FULL = 0.90
SCORE_PARTIAL = 0.55
SCORE_MISS = 0.10

# 和引用原文至少共享这么多个二字片段，才算这句被引用支持。文档未写死。
SUPPORT_BIGRAM_HITS = 2

# 三个门禁指标相对本仓库基线的回退上限。超过就阻断。文档未写死。
RECALL_DROP_LIMIT = 0.02
HALLUCINATION_RISE_LIMIT = 0.02
ACCURACY_DROP_LIMIT = 0.02

RECALL_AT_K = 5

TRIGGERS = ("prompt", "top_k", "route_threshold", "rrf_k", "generator")

NUMBER_ERROR = "数字不在原文"
VERSION_ERROR = "引用废止版本"
SUPPORT_ERROR = "句子未被引用支持"

CONTRADICTION_CUES = ("停发", "不再", "取消", "不予")
