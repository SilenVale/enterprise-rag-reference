import pytest

from rag.storage.errors import MetadataWriteError
from rag.storage.types import CitationLog, EvaluationEntry, ParseFailure

from helpers import new_id


def test_metadata_roundtrip_citation_eval_and_parse_failure(metadata_store):
    request_id = new_id("req")
    suite = new_id("suite")
    source_id = new_id("src")
    citation_id = metadata_store.append_citation(
        CitationLog(
            request_id=request_id,
            document_id="doc-1",
            parent_id="parent-1",
            version="v1",
            citation_marker="[1]",
            outcome="recorded",
            detail="引用已记下",
        )
    )
    metadata_store.save_evaluation_entry(
        EvaluationEntry(suite=suite, case_key="case-1", payload={"kind": "boundary"})
    )
    metadata_store.save_evaluation_entry(
        EvaluationEntry(suite=suite, case_key="case-1", payload={"kind": "boundary", "revised": True})
    )
    failure_id = metadata_store.enqueue_parse_failure(
        ParseFailure(
            source_id=source_id,
            document_ref="file-1",
            reason="字符保有率不够",
            attempt_count=2,
        )
    )
    citations = metadata_store.list_citations(request_id)
    entries = metadata_store.list_evaluation_entries(suite)
    failures = metadata_store.list_parse_failures(source_id=source_id)
    assert citation_id > 0
    assert [row.citation_marker for row in citations] == ["[1]"]
    assert len(entries) == 1
    assert entries[0].payload["revised"] is True
    assert failure_id > 0
    assert failures[0].document_ref == "file-1"
    assert failures[0].status == "queued"


def test_rejected_metadata_write_is_visible_and_absent(metadata_store):
    request_id = new_id("req")
    source_id = new_id("src")
    with pytest.raises(MetadataWriteError):
        metadata_store.append_citation(
            CitationLog(
                request_id=request_id,
                document_id="doc-1",
                parent_id="parent-1",
                version="v1",
                citation_marker="",
                outcome="recorded",
            )
        )
    with pytest.raises(MetadataWriteError):
        metadata_store.enqueue_parse_failure(
            ParseFailure(
                source_id=source_id,
                document_ref="file-missing",
                reason="写不进去",
                attempt_count=0,
            )
        )
    with pytest.raises(MetadataWriteError):
        metadata_store.save_evaluation_entry(
            EvaluationEntry(suite="", case_key="missing-case", payload={"kind": "boundary"})
        )
    assert metadata_store.list_citations(request_id) == []
    assert metadata_store.list_parse_failures(source_id=source_id) == []
    assert metadata_store.list_evaluation_entries("") == []
