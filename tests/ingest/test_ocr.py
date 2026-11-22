import pytest

from rag.ingest.pipeline import ingest_documents
from rag.ingest.types import StampMark

from fixtures import GuardQueue, block, document_for, text_page


def test_page_without_text_layer_goes_through_ocr_and_quality_gate():
    sentence = "扫描页没有文本层时改走识别"
    page = text_page(
        ratio=0.0,
        has_text_layer=False,
        image_blocks=(block(sentence, 40, 40, 360, 100),),
    )
    ready = ingest_documents(
        [document_for("扫描库", "scan-1", page, sentence, form="归档")],
        GuardQueue(),
    ).ready_for_index()
    assert len(ready) == 1
    attempt = ready[0].attempts[0]
    assert attempt.routes == ("ocr",)
    assert attempt.ocr_engines == ("region-reader",)
    assert attempt.retention == pytest.approx(1.0)
    assert attempt.passed
    assert ready[0].normalized_text == sentence


def test_low_text_layer_ratio_goes_through_ocr():
    full = "完整条款正文" * 12
    page = text_page(
        block("残", 40, 40, 70, 60),
        ratio=0.2,
        image_blocks=(block(full, 40, 40, 400, 200),),
    )
    ready = ingest_documents(
        [document_for("扫描库", "thin-text", page, full)],
        GuardQueue(),
    ).ready_for_index()
    assert len(ready) == 1
    assert len(ready[0].attempts) == 1
    assert ready[0].attempts[0].routes == ("ocr",)
    assert ready[0].normalized_text == full


def test_stamp_occlusion_partial_ocr_fails_gate_then_stamp_aware_ocr_passes():
    visible = "甲" * 40
    covered = "章" * 20
    tail = "乙" * 40
    full = visible + covered + tail
    page = text_page(
        block(visible, 40, 40, 200, 80),
        ratio=0.2,
        image_blocks=(block(full, 40, 40, 420, 200),),
        stamps=(StampMark(covered),),
    )
    ready = ingest_documents(
        [document_for("扫描库", "stamped", page, full)],
        GuardQueue(),
    ).ready_for_index()
    assert len(ready) == 1
    attempts = ready[0].attempts
    assert len(attempts) == 3
    assert attempts[0].routes == ("ocr",)
    assert attempts[0].ocr_engines == ("region-reader",)
    assert attempts[0].retention == pytest.approx(0.8)
    assert attempts[0].failure.startswith("字符保有率")
    assert attempts[1].ocr_engines == ("region-reader",)
    assert attempts[1].passed is False
    assert attempts[2].routes == ("ocr",)
    assert attempts[2].ocr_engines == ("region-reader-stamp",)
    assert attempts[2].passed
    assert covered in ready[0].normalized_text
