"""Line accuracy analysis in scripts/backfill_line_accuracy.py.

tests/test_backfill_line_accuracy.py covers how opening and closing lines are
picked from `weekly_odds`. This file covers what the script does with them:
joining lines to actuals and projections, and the hit-rate, playoff-split and
edge-decay numbers the report and `line_accuracy_history` are built from.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import config
from schema_migrations import MigrationManager
from scripts.backfill_line_accuracy import (
    build_accuracy_dataset,
    compute_edge_decay,
    compute_hit_rate_by_market,
    compute_season_type_accuracy,
    load_actual_stats,
    load_closing_lines,
    load_opening_lines,
    load_projections,
)
from utils.db import execute

SEASON = 2025


def _row(**overrides) -> dict:
    row = {
        "season": SEASON,
        "week": 5,
        "player_id": "BUF_receiver_a",
        "market": "receiving_yards",
        "sportsbook": "DraftKings",
        "line": 60.5,
        "actual": 70.0,
        "mu": 65.0,
        "sigma": 20.0,
        "open_line": 58.5,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def db(tmp_path, monkeypatch) -> str:
    db_path = str(tmp_path / "line-accuracy.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()
    return db_path


def _odds(player_id: str, market: str, line: float, as_of: str) -> None:
    execute(
        "INSERT INTO weekly_odds "
        "(event_id, season, week, player_id, market, sportsbook, line, price, as_of) "
        "VALUES ('2025_05_NE_BUF', ?, 5, ?, ?, 'DraftKings', ?, -110, ?)",
        (SEASON, player_id, market, line, as_of),
    )


def _stat(player_id: str, **stats: float) -> None:
    columns = ["player_id", "season", "week", "name", "team", "position", *stats]
    execute(
        f"INSERT INTO player_stats_enhanced ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        (player_id, SEASON, 5, "Test Player", "BUF", "RB", *stats.values()),
    )


def _dataset_from_db() -> pd.DataFrame:
    seasons = [SEASON]
    return build_accuracy_dataset(
        load_closing_lines(seasons),
        load_opening_lines(seasons),
        load_actual_stats(seasons),
        load_projections(seasons),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_each_closing_line_is_matched_to_the_actual_for_its_own_market(db):
    _odds("BUF_back_a", "rushing_yards", 55.5, "2025-10-03T12:00:00+00:00")
    _odds("BUF_back_a", "rushing_yards", 58.5, "2025-10-05T12:00:00+00:00")
    _odds("BUF_back_a", "receiving_yards", 20.5, "2025-10-05T12:00:00+00:00")
    _stat("BUF_back_a", rushing_yards=70.0, receiving_yards=12.0)

    dataset = _dataset_from_db().set_index("market")

    assert dataset.loc["rushing_yards", ["line", "actual"]].tolist() == [58.5, 70.0]
    assert dataset.loc["receiving_yards", ["line", "actual"]].tolist() == [20.5, 12.0]


def test_a_passing_yards_line_reaches_the_accuracy_dataset(db):
    _odds("BUF_qb_a", "passing_yards", 240.5, "2025-10-05T12:00:00+00:00")
    _stat("BUF_qb_a", passing_yards=281.0)

    dataset = _dataset_from_db()

    assert dataset["market"].tolist() == ["passing_yards"]


def test_a_line_without_a_projection_stays_in_the_dataset_with_no_mu():
    closing = pd.DataFrame([_row()]).drop(columns=["actual", "mu", "sigma", "open_line"])
    actuals = pd.DataFrame(
        [{"player_id": "BUF_receiver_a", "season": SEASON, "week": 5, "receiving_yards": 70.0}]
    )
    projections = pd.DataFrame(
        [
            {
                "season": SEASON,
                "week": 5,
                "player_id": "BUF_other",
                "market": "receiving_yards",
                "mu": 65.0,
                "sigma": 20.0,
            }
        ]
    )

    dataset = build_accuracy_dataset(closing, pd.DataFrame(), actuals, projections)

    assert len(dataset) == 1
    assert pd.isna(dataset.loc[0, "mu"])


def test_each_sportsbook_close_gets_that_sportsbooks_opening_line():
    closing = pd.DataFrame(
        [
            _row(sportsbook="DraftKings", line=60.5),
            _row(sportsbook="FanDuel", line=61.5),
        ]
    ).drop(columns=["actual", "mu", "sigma", "open_line"])
    opening = pd.DataFrame(
        [
            _row(sportsbook="DraftKings", open_line=57.5),
            _row(sportsbook="FanDuel", open_line=59.5),
        ]
    )
    actuals = pd.DataFrame(
        [{"player_id": "BUF_receiver_a", "season": SEASON, "week": 5, "receiving_yards": 70.0}]
    )

    dataset = build_accuracy_dataset(closing, opening, actuals, pd.DataFrame())

    assert dict(zip(dataset["sportsbook"], dataset["open_line"])) == {
        "DraftKings": 57.5,
        "FanDuel": 59.5,
    }


# ---------------------------------------------------------------------------
# Report numbers
# ---------------------------------------------------------------------------


def test_an_over_hit_needs_the_stat_strictly_above_the_line():
    df = pd.DataFrame(
        [
            _row(line=60.5, actual=61.0),
            _row(line=60.0, actual=60.0),
            _row(line=60.5, actual=40.0),
        ]
    )

    assert compute_hit_rate_by_market(df)["receiving_yards"] == {
        "total_lines": 3,
        "over_hits": 1,
        "hit_rate": 0.3333,
    }


def test_week_18_counts_as_regular_season_and_week_19_as_playoffs():
    df = pd.DataFrame([_row(week=18), _row(week=19), _row(week=19)])

    result = compute_season_type_accuracy(df)

    assert (result["regular_season"]["count"], result["playoffs"]["count"]) == (1, 2)


def test_model_beat_line_rate_is_the_share_of_rows_where_the_model_was_closer():
    # Actual 70 against a 60 line: the line misses by 10 on every row.
    df = pd.DataFrame(
        [
            _row(line=60.0, actual=70.0, mu=68.0),
            _row(line=60.0, actual=70.0, mu=75.0),
            _row(line=60.0, actual=70.0, mu=50.0),
            # A tie is not a win for the model.
            _row(line=60.0, actual=70.0, mu=80.0),
        ]
    )

    result = compute_season_type_accuracy(df)

    assert result["regular_season"]["model_beat_line_rate"] == 0.5


def test_edge_decay_is_the_share_of_the_opening_edge_gone_by_the_close():
    # Model 60, open 50, close 55: half of the 10-yard opening edge is gone.
    df = pd.DataFrame([_row(mu=60.0, open_line=50.0, line=55.0)])

    decay = compute_edge_decay(df)["receiving_yards"]

    assert (decay["avg_decay"], decay["decay_pct"]) == (5.0, 0.5)
