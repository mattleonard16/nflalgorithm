"""Sanity filters in materialized_value_view.materialize_week.

The 7.5 sigma floor and the ``sigma >= mu`` "ungrounded" screen are yardage
rules. Applied to every market they dropped every receptions bet (mu ~5.5) and
every anytime-touchdown bet (mu < 1) from the card. These tests run the real
materialization path against a throwaway SQLite database, with the private
``value_betting_engine`` replaced by a stub that returns a fixed ranked frame.
"""

from __future__ import annotations

import importlib
import sys
import types

import pandas as pd
import pytest

from config import config
from schema_migrations import MigrationManager
from utils.db import read_dataframe

SEASON = 2026
WEEK = 3


def _ranked_row(**overrides) -> dict:
    row = {
        "season": SEASON,
        "week": WEEK,
        "player_id": "BUF_player",
        "event_id": "2026_03_NE_BUF",
        "team": "BUF",
        "team_odds": None,
        "market": "receiving_yards",
        "sportsbook": "DraftKings",
        "line": 60.5,
        "price": -110,
        "side": "over",
        "mu": 66.0,
        "sigma": 20.0,
        "p_win": 0.58,
        "implied_prob": 0.5,
        "implied_prob_under": 0.5,
        "edge_percentage": 0.08,
        "expected_roi": 0.06,
        "kelly_fraction": 0.02,
        "stake": 10.0,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def mvv(tmp_path, monkeypatch):
    """Import materialized_value_view against a stub engine and a temp DB.

    The module binds ``rank_weekly_value`` at import, so it is imported fresh
    here and evicted afterwards; a later test importing the real engine must
    not get a module bound to this stub.
    """
    db_path = str(tmp_path / "mvv.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()

    stub = types.ModuleType("value_betting_engine")
    stub.rank_weekly_value = lambda season, week, threshold: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "value_betting_engine", stub)

    saved = sys.modules.pop("materialized_value_view", None)
    try:
        yield importlib.import_module("materialized_value_view")
    finally:
        sys.modules.pop("materialized_value_view", None)
        if saved is not None:
            sys.modules["materialized_value_view"] = saved


def _materialize(mvv, monkeypatch, rows: list[dict]) -> pd.DataFrame:
    monkeypatch.setattr(mvv, "rank_weekly_value", lambda *_: pd.DataFrame(rows))
    mvv.materialize_week(SEASON, WEEK)
    return read_dataframe(
        "SELECT player_id, market, mu, sigma FROM materialized_value_view "
        "WHERE season = ? AND week = ?",
        params=(SEASON, WEEK),
    ).set_index("player_id")


def test_count_markets_survive_while_an_ungrounded_yardage_row_is_dropped(mvv, monkeypatch):
    stored = _materialize(
        mvv,
        monkeypatch,
        [
            _ranked_row(
                player_id="BUF_receptions", market="receptions", line=4.5, mu=5.5, sigma=1.8
            ),
            _ranked_row(
                player_id="BUF_td", market="anytime_touchdown", line=0.5, mu=0.45, sigma=0.5
            ),
            # The no-history signature: a floor sigma over a collapsed mean.
            _ranked_row(player_id="BUF_ungrounded", line=4.5, mu=6.0, sigma=12.0),
            _ranked_row(player_id="BUF_grounded"),
        ],
    )

    assert set(stored.index) == {"BUF_receptions", "BUF_td", "BUF_grounded"}
    # The yardage floor does not leak into count markets.
    assert stored.loc["BUF_receptions", "sigma"] == pytest.approx(1.8)
    assert stored.loc["BUF_td", "sigma"] == pytest.approx(0.5)


def test_the_sigma_floor_still_applies_to_yardage(mvv, monkeypatch):
    stored = _materialize(
        mvv, monkeypatch, [_ranked_row(player_id="BUF_tight", line=60.5, mu=62.0, sigma=3.0)]
    )

    assert stored.loc["BUF_tight", "sigma"] == pytest.approx(7.5)


def test_a_volatile_starter_whose_spread_exceeds_his_mean_stays_on_the_card(mvv, monkeypatch):
    # An RB1's receiving line: 22 yards on average with a 26-yard spread. That
    # is a boom-or-bust receiver with history, not a missing projection.
    stored = _materialize(
        mvv,
        monkeypatch,
        [
            _ranked_row(player_id="BUF_rb1", line=19.5, mu=22.0, sigma=26.0),
            _ranked_row(player_id="BUF_collapsed", line=19.5, mu=1.6, sigma=37.5),
            _ranked_row(
                player_id="BUF_qb_collapsed",
                market="passing_yards",
                line=220.5,
                mu=60.0,
                sigma=64.5,
            ),
        ],
    )

    assert set(stored.index) == {"BUF_rb1"}
