"""写死的检索参数，以及材料留白、由本仓库选定的参数。"""

# 三路各取这么多。融合公式里的 k。精排最终留下的条数。这些是写死的。
PATH_TOP_K = 50
RRF_K = 60
RERANK_TOP_K = 5

# 精排吃融合后的一段候选。材料给了 50 到 100，具体取 80。文档未写死。
RERANK_POOL = 80

# 精排分是 sigmoid 之后的 0 到 1。高、中分界和中、低分界还没有评测集可标定。
# 下面两个数只为这一层的分流演示。文档未写死。
HIGH_SCORE = 0.72
MID_SCORE = 0.30

# 改写不占用生成主模型。规格材料没有写死，这里用规则改写器。文档未写死。
REWRITE_MODEL_ID = "rule-rewrite"
GENERATION_MAIN_MODEL = "Qwen2.5-72B"

# 实习生只能看这个可见范围标签。角色到标签的映射材料没有写死。文档未写死。
INTERN_PERMISSION = "public"

CAVEAT = "可能不完整、建议人工核实"

VECTOR_COLLECTION = "rag_retrieve_vectors"
TEXT_INDEX = "rag_retrieve_text"
