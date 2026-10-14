"""
Pytest setup: use a separate, temporary SQLite database so tests never touch
your real fraud_detection.db.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

TEST_DB = Path(tempfile.gettempdir()) / "creditguard_test.db"
if TEST_DB.exists():
    TEST_DB.unlink()
os.environ["FRAUD_DB_PATH"] = str(TEST_DB)  # must be set BEFORE config.py is imported

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make backend/ importable

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:  # "with" runs the startup code (load model, create tables)
        yield test_client
