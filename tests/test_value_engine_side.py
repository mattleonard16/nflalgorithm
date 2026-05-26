"""Tests for T0 #4: side column end-to-end through value engine + grading."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from config import config
from materialized_value_view import materialize_week
from schema_migrations import MigrationManager
from utils.grading import grade_bet
from value_betting_engine import rank_weekly_value


@pytest.fixture
def temp_db():
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    orig_path = config.database.path
    orig_backend = config.database.backend
    env_backend = os.environ.get("DB_BACKEND")
    env_sqlite_path = os.environ.get("SQLITE_DB_PATH")
    os.environ["DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(tmp)
    config.database.backend = "sqlite"
    config.database.path = str(tmp)

    MigrationManager(tmp).run()

    with sqlite3.connect(tmp) as conn:
        conn.execute(
            """
            INSERT INTO weekly_projections
            (season, week, player_id, team, opponent, market, mu, sigma,
             model_version, featureset_hash, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (2024, 1, "p1", "MIA", "NE", "rushing_yards", 95.0, 10.0,
             "v1", "hash1", "2024-09-01T00:00:00"),
        )
        conn.execute(
            """
            INSERT INTO weekly_odds
            (event_id, season, week, player_id, market, sportsbook, line, price, as_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("evt1", 2024, 1, "p1", "rushing_yards", "BookA", 70.5, -110,
             "2024-09-01T00:00:00"),
        )
        conn.commit()

    yield str(tmp)

    config.database.path = orig_path
    config.database.backend = orig_backend
    if env_backend is not None:
        os.environ["DB_BACKEND"] = env_backend
    else:
        os.environ.pop("DB_BACKEND", None)
    if env_sqlite_path is not None:
        os.environ["SQLITE_DB_PATH"] = env_sqlite_path
    else:
        os.environ.pop("SQLITE_DB_PATH", None)
    tmp_path = Path(tmp)
    tmp_path.unlink(missing_ok=True)


def test_rank_weekly_value_emits_side_column(temp_db):
    df = rank_weekly_value(2024, 1, min_edge=0.0, place=False)
    assert "side" in df.columns
    assert (df["side"] == "over").all()


def test_materialize_week_persists_side(temp_db):
    materialize_week(2024, 1, min_edge=0.0)
    with sqlite3.connect(temp_db) as conn:
        rows = conn.execute(
            "SELECT side FROM materialized_value_view WHERE season=? AND week=?",
            (2024, 1),
        ).fetchall()
    assert rows, "expected at least one materialized row"
    assert all(r[0] == "over" for r in rows)


def test_materialized_view_side_not_null_default_over(temp_db):
    """Schema invariant: side is NOT NULL DEFAULT 'over'."""
    with sqlite3.connect(temp_db) as conn:
        conn.execute(
            """
            INSERT INTO materialized_value_view
            (season, week, player_id, event_id, market, sportsbook, line, price,
             mu, sigma, p_win, edge_percentage, expected_roi, kelly_fraction,
             stake, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (2024, 2, "p1", "evt2", "rushing_yards", "BookA", 70.0, -110,
             75.0, 10.0, 0.6, 0.05, 0.03, 0.02, 20.0, "2024-09-01T00:00:00"),
        )
        side = conn.execute(
            "SELECT side FROM materialized_value_view WHERE event_id='evt2'"
        ).fetchone()[0]
    assert side == "over"


def test_materialized_view_pk_allows_both_sides(temp_db):
    """Same player+market+book+event with different side = two rows (T0 #4 PK)."""
    with sqlite3.connect(temp_db) as conn:
        for side in ("over", "under"):
            conn.execute(
                """
                INSERT INTO materialized_value_view
                (season, week, player_id, event_id, market, sportsbook, line,
                 price, side, mu, sigma, p_win, edge_percentage, expected_roi,
                 kelly_fraction, stake, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (2024, 3, "p1", "evt3", "rushing_yards", "BookA", 70.0, -110,
                 side, 75.0, 10.0, 0.6, 0.05, 0.03, 0.02, 20.0,
                 "2024-09-01T00:00:00"),
            )
        count = conn.execute(
            "SELECT COUNT(*) FROM materialized_value_view WHERE event_id='evt3'"
        ).fetchone()[0]
    assert count == 2


def test_grade_bet_under_pathway_works():
    """T0 #4: under bets must be gradable (not stuck on hardcoded 'over')."""
    assert grade_bet(60.0, 70.5, "under") == "win"
    assert grade_bet(80.0, 70.5, "under") == "loss"
    assert grade_bet(70.5, 70.5, "under") == "push"


def test_grade_bet_over_pathway_still_works():
    assert grade_bet(80.0, 70.5, "over") == "win"
    assert grade_bet(60.0, 70.5, "over") == "loss"
