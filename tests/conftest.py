"""Pytest configuration for deterministic database backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import top-level modules
# like schema_migrations, config, utils, scripts, etc.
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Force tests to run against a local SQLite database regardless of .env settings.
TEST_DB_DIR = Path(__file__).parent / "_tmp"
TEST_DB_DIR.mkdir(exist_ok=True)
TEST_DB_PATH = TEST_DB_DIR / "test_suite.db"

# Reset the file each session to avoid stale data bleeding into tests.
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()
TEST_DB_PATH.touch()

os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("SQLITE_DB_PATH", str(TEST_DB_PATH))


# Shared fixture for clearing NBA cache before tests that use TestClient
def clear_nba_cache():
    """Clear NBA endpoint cache to avoid cross-test pollution."""
    from api.cache import nba_cache
    nba_cache.invalidate_all()
