import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.generate.log import GenerateLog
from rag.storage.object_store import open_parent_store
from rag.storage.settings import load_settings

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def parent_store(settings):
    return open_parent_store(settings)


@pytest.fixture(scope="session")
def generate_log(settings):
    log = GenerateLog(settings.metadata_dsn)
    log.ensure_schema()
    return log
