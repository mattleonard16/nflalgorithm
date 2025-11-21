"""Pytest configuration for deterministic database backend."""

from __future__ import annotations

import os
from pathlib import Path

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
