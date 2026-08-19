"""Interrupted table-swap migrations must never silently discard history.

SQLite DDL autocommits, so a rename-based primary-key rebuild that dies partway
leaves the original table stranded under its `_old` name. Recreating an empty
table beside it would read as a healthy migration while every prior row was
unreachable.
"""

from __future__ import annotations

import sqlite3

import pytest

from config import config
from schema_migrations import MigrationManager


@pytest.fixture()
def migrated_database(tmp_path, monkeypatch) -> str:
    db_path = str(tmp_path / "crash-recovery.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()
    return db_path


def _seed_stage_run(db_path: str, run_id: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, season, week, status, started_at)
            VALUES (?, 2026, 1, 'succeeded', '2026-09-01T00:00:00+00:00')
            """,
            (run_id,),
        )
        conn.execute(
            """
            INSERT INTO pipeline_stage_runs (run_id, stage_name, ordinal, status, attempt)
            VALUES (?, 'ingest', 1, 'succeeded', 1)
            """,
            (run_id,),
        )


def _simulate_crash_after_rename(db_path: str) -> None:
    """Reproduce a death between the RENAME and the CREATE."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE pipeline_stage_runs RENAME TO _pipeline_stage_runs_old")


def test_stranded_history_fails_loud_instead_of_being_overwritten(migrated_database) -> None:
    _seed_stage_run(migrated_database, "run-stranded")
    _simulate_crash_after_rename(migrated_database)

    with pytest.raises(RuntimeError) as excinfo:
        MigrationManager(migrated_database).run()

    message = str(excinfo.value)
    assert "_pipeline_stage_runs_old" in message
    assert "1 stranded row" in message
    # The operator needs the exact recovery statement, not just a diagnosis.
    assert "RENAME TO pipeline_stage_runs" in message

    # The orphan must still hold its rows: the failed run may not destroy them.
    with sqlite3.connect(migrated_database) as conn:
        stranded = conn.execute("SELECT COUNT(*) FROM _pipeline_stage_runs_old").fetchone()[0]
    assert stranded == 1


def test_documented_recovery_restores_history(migrated_database) -> None:
    _seed_stage_run(migrated_database, "run-recovered")
    _simulate_crash_after_rename(migrated_database)

    # Apply exactly what the error message instructs.
    with sqlite3.connect(migrated_database) as conn:
        conn.execute("DROP TABLE IF EXISTS pipeline_stage_runs")
        conn.execute("ALTER TABLE _pipeline_stage_runs_old RENAME TO pipeline_stage_runs")

    MigrationManager(migrated_database).run()

    with sqlite3.connect(migrated_database) as conn:
        rows = conn.execute("SELECT run_id, stage_name FROM pipeline_stage_runs").fetchall()
        orphans = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = '_pipeline_stage_runs_old'"
        ).fetchall()
    assert rows == [("run-recovered", "ingest")]
    assert orphans == []


def test_empty_orphan_beside_populated_table_does_not_block(migrated_database) -> None:
    # A leftover empty `_old` table is debris from a swap that already
    # completed; it must not wedge every future migration run.
    _seed_stage_run(migrated_database, "run-live")
    with sqlite3.connect(migrated_database) as conn:
        conn.execute("CREATE TABLE _pipeline_stage_runs_old AS SELECT * FROM pipeline_stage_runs")
        conn.execute("DELETE FROM _pipeline_stage_runs_old")

    MigrationManager(migrated_database).run()

    with sqlite3.connect(migrated_database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_stage_runs").fetchone()[0] == 1


def test_stage_run_pk_rebuild_is_atomic(migrated_database) -> None:
    # Rebuild from the pre-attempt PK shape and confirm the swap leaves no
    # intermediate table behind and loses no rows.
    with sqlite3.connect(migrated_database) as conn:
        conn.execute("DROP TABLE pipeline_stage_runs")
        conn.execute("""
            CREATE TABLE pipeline_stage_runs (
                run_id VARCHAR(36) NOT NULL,
                stage_name VARCHAR(64) NOT NULL,
                ordinal INTEGER NOT NULL,
                status VARCHAR(32) NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1,
                started_at VARCHAR(40),
                finished_at VARCHAR(40),
                result_json TEXT,
                error_message TEXT,
                PRIMARY KEY (run_id, stage_name)
            )
            """)
    _seed_stage_run(migrated_database, "run-rebuild")

    MigrationManager(migrated_database).run()

    with sqlite3.connect(migrated_database) as conn:
        primary_key = [
            row[1] for row in conn.execute("PRAGMA table_info(pipeline_stage_runs)") if row[5]
        ]
        rows = conn.execute("SELECT COUNT(*) FROM pipeline_stage_runs").fetchone()[0]
        orphans = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = '_pipeline_stage_runs_old'"
        ).fetchall()
    assert set(primary_key) == {"run_id", "attempt", "stage_name"}
    assert rows == 1
    assert orphans == []
