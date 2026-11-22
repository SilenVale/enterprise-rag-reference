"""删句之后还读不读得通。读不通就不把挖空的句子交出去。"""

from rag.generate.params import DANGLING_PREFIXES, MIN_READABLE_CHARS
from rag.generate.sentences import strip_markers
from rag.generate.types import DraftSentence


def sentence_body(sentence: DraftSentence) -> str:
    return strip_markers(sentence.text).strip().strip("。！？").strip()


def remains_readable(sentences: tuple[DraftSentence, ...] | list[DraftSentence]) -> bool:
    if not sentences:
        return False
    for index, sentence in enumerate(sentences):
        if not sentence.markers:
            return False
        body = sentence_body(sentence)
        if len(body) < MIN_READABLE_CHARS:
            return False
        if index == 0 and body.startswith(DANGLING_PREFIXES):
            return False
    return True
