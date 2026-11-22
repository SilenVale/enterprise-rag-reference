import pytest

from rag.evaluate.store import EvaluateStore
from rag.storage.settings import load_settings

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def store():
    item = EvaluateStore(load_settings().metadata_dsn)
    item.ensure_schema()
    return item
