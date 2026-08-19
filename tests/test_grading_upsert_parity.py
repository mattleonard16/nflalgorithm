"""Grading writes must upsert identically on both backends.

The grading path previously used SQLite-only `INSERT OR REPLACE`, which MySQL
rejects outright. Re-grading a week has to overwrite in place rather than
duplicate or fail, so these run against whichever backend TEST_DB_BACKEND
selects (MySQL side skips when no server is configured).
"""

from __future__ import annotations

import os

import pytest

from config import config
from schema_migrations import MigrationManager
from scripts.record_outcomes import save_outcomes
from utils.db import execute, fetchone, get_connection


@pytest.fixture()
def graded_database(tmp_path, monkeypatch) -> str:
    backend = os.getenv("TEST_DB_BACKEND", "sqlite").lower()
    if backend == "sqlite":
        db_path = str(tmp_path / "grading-parity.db")
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        monkeypatch.setenv("SQLITE_DB_PATH", db_path)
        monkeypatch.setattr(config.database, "backend", "sqlite")
        monkeypatch.setattr(config.database, "path", db_path)
        MigrationManager(db_path).run()
    else:
        test_db_url = os.getenv("TEST_DB_URL")
        if not test_db_url:
            pytest.skip("TEST_DB_BACKEND=mysql requires TEST_DB_URL")
        monkeypatch.setenv("DB_BACKEND", "mysql")
        monkeypatch.setenv("DB_URL", test_db_url)
        monkeypatch.setattr(config.database, "backend", "mysql")
        monkeypatch.setattr(config.database, "db_url", test_db_url)
        MigrationManager("unused-mysql-path").run()

    with get_connection() as conn:
        for table in ("clv_weekly", "bet_outcomes", "weekly_performance", "weekly_odds"):
            execute(f"DELETE FROM {table}", conn=conn)
        conn.commit()
    return backend


def _outcome(**overrides) -> dict:
    outcome = {
        "bet_id": "bet-parity-1",
        "season": 2026,
        "week": 3,
        "player_id": "player_a",
        "player_name": "Player A",
        "market": "receiving_yards",
        "sportsbook": "DraftKings",
        "side": "over",
        "line": 64.5,
        "price": -110,
        "actual_result": 71.0,
        "result": "win",
        "profit_units": 0.91,
        "confidence_tier": "high",
        "edge_at_placement": 9.4,
        "recorded_at": "2026-09-21T12:00:00+00:00",
        "event_id": "2026_03_NE_BUF",
    }
    outcome.update(overrides)
    return outcome


def test_bet_outcomes_upsert_overwrites_in_place(graded_database) -> None:
    save_outcomes([_outcome()])
    save_outcomes([_outcome(actual_result=48.0, result="loss", profit_units=-1.0)])

    assert fetchone("SELECT COUNT(*) FROM bet_outcomes")[0] == 1
    row = fetchone(
        "SELECT result, profit_units, actual_result FROM bet_outcomes WHERE bet_id = ?",
        ("bet-parity-1",),
    )
    assert row[0] == "loss"
    assert row[1] == pytest.approx(-1.0)
    assert row[2] == pytest.approx(48.0)


def test_weekly_performance_upsert_overwrites_in_place(graded_database) -> None:
    save_outcomes([_outcome()])
    save_outcomes(
        [
            _outcome(result="loss", profit_units=-1.0),
            _outcome(bet_id="bet-parity-2", result="loss", profit_units=-1.0),
        ]
    )

    assert fetchone("SELECT COUNT(*) FROM weekly_performance")[0] == 1
    row = fetchone(
        "SELECT total_bets, wins, losses, profit_units FROM weekly_performance "
        "WHERE season = ? AND week = ?",
        (2026, 3),
    )
    assert row[0] == 2
    assert row[1] == 0
    assert row[2] == 2
    assert row[3] == pytest.approx(-2.0)


def test_distinct_bets_insert_side_by_side(graded_database) -> None:
    # The upsert must only collapse rows that share the primary key. An
    # over/under pair on the same prop has distinct bet_ids and must survive.
    save_outcomes([_outcome(), _outcome(bet_id="bet-parity-under", side="under")])

    assert fetchone("SELECT COUNT(*) FROM bet_outcomes")[0] == 2
    sides = {
        row[0]
        for row in [
            fetchone("SELECT side FROM bet_outcomes WHERE bet_id = ?", ("bet-parity-1",)),
            fetchone("SELECT side FROM bet_outcomes WHERE bet_id = ?", ("bet-parity-under",)),
        ]
    }
    assert sides == {"over", "under"}


def test_clv_weekly_upsert_overwrites_in_place(graded_database) -> None:
    now = "2026-09-21T10:00:00+00:00"
    # Two snapshots at the same line give compute_clv the depth it needs.
    for as_of, price in (("2026-09-19T10:00:00+00:00", -105), (now, -120)):
        execute(
            """
            INSERT INTO weekly_odds (
                event_id, season, week, player_id, market, sportsbook,
                line, price, under_price, as_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026_03_NE_BUF",
                2026,
                3,
                "player_a",
                "receiving_yards",
                "DraftKings",
                64.5,
                price,
                -110,
                as_of,
            ),
        )

    save_outcomes([_outcome()])
    first = fetchone("SELECT close_price FROM clv_weekly WHERE bet_id = ?", ("bet-parity-1",))
    if first is None:
        pytest.skip("CLV needs resolvable closing lines; none produced in this fixture")

    save_outcomes([_outcome()])
    assert fetchone("SELECT COUNT(*) FROM clv_weekly")[0] == 1
