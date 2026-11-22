from rag.ingest.pipeline import ingest_from_sources
from rag.ingest.sources import SEVEN_ADAPTERS, seven_source_adapters
from rag.ingest.types import CatalogEntry

from fixtures import block, text_page


def test_seven_source_adapters_deliver_files_with_their_own_source_ids():
    pages = {}
    catalog = {}
    for index, adapter_cls in enumerate(SEVEN_ADAPTERS, start=1):
        sentence = f"{adapter_cls.source_id}交来的第{index}份制度"
        page = text_page(block(sentence, 40, 40, 300, 80))
        pages[adapter_cls.source_id] = sentence
        catalog[(adapter_cls.source_id, f"file-{index}")] = CatalogEntry((page,), sentence)

    adapters = seven_source_adapters(catalog)
    assert len(adapters) == 7
    assert len({type(adapter) for adapter in adapters}) == 7
    assert {adapter.form for adapter in adapters} == {"转发", "归档", "附件"}

    batch = ingest_from_sources(
        adapters,
        [(adapter.source_id, f"file-{index}") for index, adapter in enumerate(adapters, start=1)],
        _no_queue(),
    )
    ready = batch.ready_for_index()
    assert len(ready) == 7
    assert [(item.source_id, item.form, item.normalized_text) for item in ready] == [
        (adapter.source_id, adapter.form, pages[adapter.source_id]) for adapter in adapters
    ]


def test_same_normalized_text_from_multiple_sources_keeps_each_source_id():
    sentence = "差旅返回后十日内提交报销"
    page = text_page(block(sentence, 40, 40, 320, 90))
    entry = CatalogEntry((page,), sentence)
    catalog = {
        ("制度库", "policy-9"): entry,
        ("邮件", "policy-9"): entry,
        ("附件", "policy-9"): entry,
    }
    wanted = {"制度库", "邮件", "附件"}
    adapters = [adapter for adapter in seven_source_adapters(catalog) if adapter.source_id in wanted]
    batch = ingest_from_sources(
        adapters,
        [("制度库", "policy-9"), ("邮件", "policy-9"), ("附件", "policy-9")],
        _no_queue(),
    )
    ready = batch.ready_for_index()
    assert [item.source_id for item in ready] == ["制度库", "邮件", "附件"]
    assert len(ready) == 3
    assert {item.normalized_text for item in ready} == {sentence}
    assert batch.queued == ()


def _no_queue():
    from fixtures import GuardQueue

    return GuardQueue()
