from dataclasses import dataclass

# 材料举的例子是低于 95% 判失败。95% 是示例，不是写死的线。文档未写死。
EXAMPLE_RETENTION_THRESHOLD = 0.95

# 下面这些数材料都没有写。文档未写死。
TEXT_LAYER_RATIO_FLOOR = 0.5
COLUMN_GAP = 36.0
TIGHTER_COLUMN_GAP = 12.0
LINE_TOLERANCE = 4.0
FETCH_ATTEMPTS = 3


@dataclass(frozen=True)
class ParseParams:
    """一次解析用的参数。失败后换成下一组，或者改走 OCR。"""

    name: str
    column_gap: float
    ocr_mode: str
    line_tolerance: float = LINE_TOLERANCE
    text_layer_ratio_floor: float = TEXT_LAYER_RATIO_FLOOR


# 先按默认栏间距读文本层；不行就收紧栏间距；再不行就改走盖章感知识别。
DEFAULT_STRATEGIES: tuple[ParseParams, ...] = (
    ParseParams("默认分栏", column_gap=COLUMN_GAP, ocr_mode="auto"),
    ParseParams("更窄栏间距", column_gap=TIGHTER_COLUMN_GAP, ocr_mode="auto"),
    ParseParams("盖章感知识别", column_gap=TIGHTER_COLUMN_GAP, ocr_mode="stamp_aware"),
)
