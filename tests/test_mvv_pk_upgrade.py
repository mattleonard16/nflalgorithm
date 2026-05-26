"""Brownfield-upgrade test for T0 #4.

Existing `materialized_value_view` tables were created with a narrow PK that
did not include `side`. After T0 #4 added the `side` column via ALTER, the PK
on existing DBs still didn't include `side`, so `ON CONFLICT(..., side)` would
fail and under-side rows could collide. This test recreates the old shape,
runs MigrationManager, and verifies the PK widens.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from config import config
from schema_migrations import MigrationManager


def _make_old_shape_db(path: Path) -> None:
    """Create materialized_value_view with the pre-T0-#4 narrow PK."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE materialized_value_view (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                line REAL NOT NULL,
                price INTEGER NOT NULL,
                mu REAL NOT NULL,
                sigma REAL NOT NULL,
                p_win REAL NOT NULL,
                edge_percentage REAL NOT NULL,
                expected_roi REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                stake REAL NOT NULL,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (season, week, player_id, market, sportsbook, event_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO materialized_value_view
            (season, week, player_id, event_id, market, sportsbook, line, price,
             mu, sigma, p_win, edge_percentage, expected_roi, kelly_fraction,
             stake, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (2024, 1, "p1", "evt1", "rushing_yards", "BookA", 70.5, -110,
             95.0, 10.0, 0.6, 0.1, 0.05, 0.02, 20.0, "2024-09-01T00:00:00"),
        )
        conn.commit()


@pytest.fixture
def isolated_db(monkeypatch):
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp))
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", str(tmp))
    yield tmp
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass


def test_pk_widens_to_include_side_on_upgrade(isolated_db):
    tmp = isolated_db
    _make_old_shape_db(tmp)

    MigrationManager(tmp).run()

    with sqlite3.connect(tmp) as conn:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='materialized_value_view'"
        ).fetchone()[0]
        assert "PRIMARY KEY" in ddl
        pk_clause = ddl.split("PRIMARY KEY", 1)[1]
        assert "side" in pk_clause, f"side not in PK: {ddl}"

        row = conn.execute(
            "SELECT player_id, side FROM materialized_value_view WHERE player_id='p1'"
        ).fetchone()
        assert row is not None, "data lost during rebuild"
        assert row[0] == "p1"
        assert row[1] == "over"

        conn.execute(
            """
            INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, side, mu, sigma, p_win, edge_percentage,
             expected_roi, kelly_fraction, stake, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (2024, 1, "p1", "evt1", "MIA", "rushing_yards", "BookA", 70.5, -110,
             "under", 95.0, 10.0, 0.4, 0.05, 0.02, 0.01, 10.0,
             "2024-09-01T00:00:00"),
        )
        conn.commit()
        n = conn.execute(
            "SELECT COUNT(*) FROM materialized_value_view WHERE player_id='p1'"
        ).fetchone()[0]
        assert n == 2, "under-side row should coexist with over-side row"


def test_idempotent_on_fresh_db(isolated_db):
    tmp = isolated_db
    MigrationManager(tmp).run()
    MigrationManager(tmp).run()

    with sqlite3.connect(tmp) as conn:
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='materialized_value_view'"
        ).fetchone()[0]
        assert "side" in ddl.split("PRIMARY KEY", 1)[1]
