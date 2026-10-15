"""The MySQL index branch must build every index the SQLite branch builds.

`_ensure_indexes` returns from its MySQL branch before the SQLite list, so an
index added to only one branch silently does not exist on the other (CLAUDE.md
Notes). MySQL also rejects an index on an unbounded TEXT column without a
prefix length, which would fail the production migration outright.
"""

from __future__ import annotations

import re

import pytest

import schema_migrations
from schema_migrations import MigrationManager, _mysql_compatible_ddl

_SQLITE_INDEX = re.compile(
    r"CREATE INDEX IF NOT EXISTS (\w+)\s+ON (\w+)\s*\(([^)]*)\)", re.IGNORECASE
)
_MYSQL_INDEX = re.compile(r"CREATE INDEX `(\w+)` ON `(\w+)` \((.*)\)$", re.IGNORECASE)


class _RecordingCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: str, params: tuple | None = None) -> None:
        self.statements.append(" ".join(statement.split()))

    def fetchone(self):
        return None

    def fetchall(self):
        return []


def _bare(columns: str) -> list[str]:
    out = []
    for col in columns.split(","):
        col = col.strip().lower().split("(")[0].strip("` ")
        out.append(col.removesuffix(" desc").strip())
    return out


def _indexes(backend: str, monkeypatch) -> dict[str, tuple[str, str]]:
    monkeypatch.setattr(schema_migrations, "get_backend", lambda: backend)
    cursor = _RecordingCursor()
    MigrationManager("ignored.db")._ensure_indexes(cursor)
    pattern = _SQLITE_INDEX if backend == "sqlite" else _MYSQL_INDEX
    found = {}
    for statement in cursor.statements:
        match = pattern.search(statement)
        if match:
            found[match.group(1)] = (match.group(2), match.group(3))
    return found


def _mysql_text_columns() -> dict[str, set[str]]:
    """Columns that stay unbounded TEXT in the MySQL DDL, per table."""
    text: dict[str, set[str]] = {}
    for ddl in MigrationManager("ignored.db")._ddl_statements():
        table = re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", ddl)
        if not table:
            continue
        translated = _mysql_compatible_ddl(ddl)
        text[table.group(1)] = set(
            re.findall(r"^\s*(\w+)\s+TEXT\b", translated, flags=re.IGNORECASE | re.MULTILINE)
        )
    return text


def test_mysql_builds_every_sqlite_index(monkeypatch) -> None:
    sqlite = _indexes("sqlite", monkeypatch)
    mysql = _indexes("mysql", monkeypatch)

    assert sqlite, "no SQLite indexes parsed"
    assert sorted(set(sqlite) - set(mysql)) == []


def test_mysql_indexes_cover_the_same_columns(monkeypatch) -> None:
    sqlite = _indexes("sqlite", monkeypatch)
    mysql = _indexes("mysql", monkeypatch)

    for name, (table, columns) in mysql.items():
        if name not in sqlite:
            continue
        sqlite_table, sqlite_columns = sqlite[name]
        assert table == sqlite_table, name
        assert _bare(columns) == _bare(sqlite_columns), name


def test_mysql_text_index_columns_have_prefix_lengths(monkeypatch) -> None:
    text_columns = _mysql_text_columns()
    missing_prefix = []
    for name, (table, columns) in _indexes("mysql", monkeypatch).items():
        for col in columns.split(","):
            bare = col.strip().split("(")[0].strip("` ")
            if bare in text_columns.get(table, set()) and "(" not in col:
                missing_prefix.append(f"{name}.{bare}")
    assert missing_prefix == []


@pytest.mark.parametrize("name", ["idx_player_stats_latest"])
def test_mysql_index_has_no_desc_that_would_rebuild_every_run(monkeypatch, name) -> None:
    # information_schema reports bare column names, so a DESC spelling would
    # never compare equal and the index would be dropped on every migration.
    _table, columns = _indexes("mysql", monkeypatch)[name]
    assert "desc" not in columns.lower()
