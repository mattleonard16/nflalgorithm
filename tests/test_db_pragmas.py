"""Connection-level durability and concurrency settings for the SQLite backend."""

from __future__ import annotations

import sqlite3

import pytest

from config import config
from schema_migrations import MigrationManager
from utils.db import get_connection


@pytest.fixture()
def sqlite_database(tmp_path, monkeypatch) -> str:
    db_path = str(tmp_path / "pragmas.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    return db_path


def test_get_connection_sets_busy_timeout(sqlite_database) -> None:
    with get_connection() as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


def test_get_connection_sets_synchronous_normal(sqlite_database) -> None:
    # NORMAL is 1. FULL (2) fsyncs every commit and is the sqlite3 default.
    with get_connection() as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_get_connection_enforces_foreign_keys(sqlite_database) -> None:
    with get_connection() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrated_database_reports_wal_journal_mode(sqlite_database) -> None:
    MigrationManager(sqlite_database).run()
    # Journal mode is a persistent property of the file, so read it on a
    # connection that never set it.
    with sqlite3.connect(sqlite_database) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(journal_mode).lower() == "wal"
