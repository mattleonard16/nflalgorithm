"""Regression coverage for explicit SQLite migration targets."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

import schema_migrations
from config import config
from schema_migrations import MigrationManager, _mysql_compatible_ddl


def test_migration_manager_honors_explicit_sqlite_path(tmp_path, monkeypatch) -> None:
    configured_path = tmp_path / "configured.db"
    requested_path = tmp_path / "requested.db"

    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(configured_path))
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", str(configured_path))

    MigrationManager(requested_path).run()

    assert requested_path.exists()
    assert not configured_path.exists()
    with sqlite3.connect(requested_path) as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='materialized_value_view'"
        ).fetchone()
    assert table == ("materialized_value_view",)

    with sqlite3.connect(requested_path) as conn:
        retired_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ncaab_%'"
            ).fetchall()
        }
    assert retired_tables == set()

    with sqlite3.connect(requested_path) as conn:
        roster_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nfl_roster_players'"
        ).fetchone()
        context_table = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='nfl_player_context_snapshots'"
        ).fetchone()
        context_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(nfl_player_context_snapshots)").fetchall()
        }
        stat_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(player_stats_enhanced)").fetchall()
        }
        dim_columns = {row[1] for row in conn.execute("PRAGMA table_info(player_dim)").fetchall()}
    assert roster_table == ("nfl_roster_players",)
    assert context_table == ("nfl_player_context_snapshots",)
    assert {
        "depth_rank",
        "is_starter",
        "injury_status",
        "practice_status",
        "expected_snap_percentage",
        "expected_targets",
        "is_rookie",
        "is_new_team",
        "uncertainty_multiplier",
        "captured_at",
    }.issubset(context_columns)
    assert "gsis_id" in stat_columns
    assert "gsis_id" in dim_columns


def test_migration_manager_keeps_non_sqlite_connection_path(monkeypatch) -> None:
    connection = object()
    migrated: list[object] = []

    @contextmanager
    def fake_connection() -> Iterator[object]:
        yield connection

    def record_migration(self: MigrationManager, active_connection: object) -> None:
        migrated.append(active_connection)

    monkeypatch.setattr(schema_migrations, "get_backend", lambda: "mysql")
    monkeypatch.setattr(schema_migrations, "get_connection", fake_connection)
    monkeypatch.setattr(MigrationManager, "_run_on_connection", record_migration)

    MigrationManager("ignored-for-mysql.db").run()

    assert migrated == [connection]


def test_mysql_index_refresh_is_idempotent_without_if_not_exists(monkeypatch) -> None:
    class RecordingCursor:
        def __init__(self) -> None:
            self.statements: list[tuple[str, tuple | None]] = []

        def execute(self, statement: str, params: tuple | None = None) -> None:
            self.statements.append((statement, params))

        def fetchone(self):
            return None

    cursor = RecordingCursor()
    monkeypatch.setattr(schema_migrations, "get_backend", lambda: "mysql")

    MigrationManager("ignored-for-mysql.db")._ensure_indexes(cursor)

    create_statements = [
        statement for statement, _params in cursor.statements if "CREATE INDEX" in statement
    ]
    assert create_statements
    assert all("IF NOT EXISTS" not in statement for statement in create_statements)
    assert any("idx_pipeline_jobs_claim" in statement for statement in create_statements)
    assert any("idx_pipeline_jobs_stale" in statement for statement in create_statements)


def test_mysql_ddl_bounds_key_text_and_translates_auto_increment() -> None:
    ddl = """
        CREATE TABLE example (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            market TEXT NOT NULL,
            notes TEXT,
            UNIQUE(event_id, market)
        )
    """

    translated = _mysql_compatible_ddl(ddl)

    assert "id BIGINT PRIMARY KEY AUTO_INCREMENT" in translated
    assert "event_id VARCHAR(128) NOT NULL" in translated
    assert "market VARCHAR(64) NOT NULL" in translated
    assert "notes TEXT" in translated


def test_migration_fences_running_jobs_without_claim_tokens(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "legacy-running.db"
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(database_path))
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", str(database_path))
    MigrationManager(database_path).run()
    with sqlite3.connect(database_path) as conn:
        conn.execute("""
            INSERT INTO pipeline_runs
                (run_id, season, week, status, stages_requested, stages_completed,
                 started_at, updated_at)
            VALUES ('run', 2026, 1, 'running', 6, 0, 'now', 'now')
            """)
        conn.execute("""
            INSERT INTO pipeline_jobs
                (job_id, run_id, job_type, payload_json, status, priority, attempts,
                 max_attempts, available_at, worker_id, cancel_requested, source,
                 created_at, updated_at)
            VALUES ('job', 'run', 'nfl_weekly', '{}', 'running', 0, 1, 3,
                    'now', 'legacy-worker', 0, 'scheduler', 'now', 'now')
            """)
        conn.execute("""
            INSERT INTO pipeline_stage_runs
                (run_id, stage_name, ordinal, status, attempt, started_at)
            VALUES ('run', 'odds', 1, 'running', 1, 'now')
            """)
        conn.commit()

    MigrationManager(database_path).run()

    with sqlite3.connect(database_path) as conn:
        job = conn.execute(
            "SELECT status, worker_id, claim_token, last_error FROM pipeline_jobs"
        ).fetchone()
        run = conn.execute("SELECT status, finished_at FROM pipeline_runs").fetchone()
        stage = conn.execute(
            "SELECT status, finished_at, error_message FROM pipeline_stage_runs"
        ).fetchone()
    assert job == (
        "retry_scheduled",
        None,
        None,
        "legacy worker lease fenced during claim-token migration",
    )
    assert run == ("queued", None)
    assert stage[0] == "failed"
    assert stage[1] is not None
    assert stage[2] == "legacy worker lease fenced during claim-token migration"
