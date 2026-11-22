import sys
from pathlib import Path

import pytest

from rag.storage.metadata_store import PostgresMetadataStore
from rag.storage.settings import load_settings

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

pytestmark = pytest.mark.live


@pytest.fixture
def metadata_store():
    store = PostgresMetadataStore(load_settings().metadata_dsn)
    store.ensure_schema()
    return store
