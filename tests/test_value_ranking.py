"""Value ranking: the public baseline, and the switch that prefers the private engine."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from config import config
from utils import value_ranking
from utils.db import execute

SEASON, WEEK = 2026, 5
EVENT = "2026_05_MIA_BUF"
KICKOFF = "2026-10-11T17:00:00+00:00"


@pytest.fixture()
def db(matrix_database, monkeypatch):
    for table in ("weekly_odds", "weekly_projections", "games"):
        execute(f"DELETE FROM {table}")
    execute(
        "INSERT INTO games (game_id, season, week, home_team, away_team, kickoff_utc, game_date) "
        "VALUES (?, ?, ?, 'BUF', 'MIA', ?, '2026-10-11')",
        (EVENT, SEASON, WEEK, KICKOFF),
    )
    monkeypatch.setattr(config.features, "no_vig_enabled", True)
    monkeypatch.setattr(config.features, "kelly_cap_enabled", True)
    return matrix_database


def _projection(player_id: str, mu: float, sigma: float = 20.0) -> None:
    execute(
        "INSERT INTO weekly_projections (season, week, player_id, team, opponent, market, mu, "
        "sigma, model_version, featureset_hash, generated_at) "
        "VALUES (?, ?, ?, 'BUF', 'MIA', 'receiving_yards', ?, ?, 'v', 'h', '2026-10-10')",
        (SEASON, WEEK, player_id, mu, sigma),
    )


def _quote(player_id: str, line: float, as_of: str, price: int = -110, under: int = -110) -> None:
    execute(
        "INSERT INTO weekly_odds (event_id, season, week, player_id, market, sportsbook, line, "
        "price, under_price, as_of) VALUES (?, ?, ?, ?, 'receiving_yards', 'DraftKings', ?, ?, ?, ?)",
        (EVENT, SEASON, WEEK, player_id, line, price, under, as_of),
    )


def test_prices_both_sides_and_keeps_only_bets_that_clear_the_edge(db) -> None:
    _projection("wr1", mu=90.0)
    _quote("wr1", line=60.5, as_of="2026-10-10T12:00:00+00:00")

    bets = value_ranking.rank_weekly_value_baseline(SEASON, WEEK, min_edge=0.05)

    assert bets["side"].tolist() == ["over"]
    bet = bets.iloc[0]
    # Equal prices on both sides de-vig to a fair coin.
    assert bet["implied_prob"] == pytest.approx(0.5)
    assert bet["edge_percentage"] == pytest.approx(bet["p_win"] - 0.5)


def test_a_post_kickoff_quote_is_never_priced(db) -> None:
    _projection("wr1", mu=90.0)
    _quote("wr1", line=60.5, as_of="2026-10-11T18:00:00+00:00")

    assert value_ranking.rank_weekly_value_baseline(SEASON, WEEK).empty


def test_stakes_follow_the_capped_kelly_fraction(db) -> None:
    _projection("wr1", mu=150.0)
    _quote("wr1", line=40.5, as_of="2026-10-10T12:00:00+00:00", price=+150, under=-200)

    bet = value_ranking.rank_weekly_value_baseline(SEASON, WEEK).iloc[0]

    assert bet["kelly_fraction"] == pytest.approx(config.betting.max_kelly)
    assert bet["stake"] == pytest.approx(config.betting.max_kelly * config.betting.bankroll)


def test_an_installed_private_engine_wins(monkeypatch) -> None:
    card = pd.DataFrame({"player_id": ["private"]})
    engine = types.ModuleType("value_betting_engine")
    setattr(engine, "rank_weekly_value", lambda season, week, min_edge: card)
    monkeypatch.setitem(sys.modules, "value_betting_engine", engine)

    assert value_ranking.rank_weekly_value(SEASON, WEEK) is card


def test_a_broken_dependency_inside_the_engine_is_not_hidden(monkeypatch) -> None:
    def broken(name: str):
        raise ModuleNotFoundError(name="missing_dependency")

    monkeypatch.setattr(value_ranking, "import_module", broken)

    with pytest.raises(ModuleNotFoundError):
        value_ranking.rank_weekly_value(SEASON, WEEK)


def test_a_deployment_that_requires_the_private_engine_fails_without_it(monkeypatch) -> None:
    def missing_engine(name: str):
        raise ModuleNotFoundError(name="value_betting_engine")

    monkeypatch.setattr(value_ranking, "import_module", missing_engine)
    monkeypatch.setattr(config.features, "require_private_models", True)

    with pytest.raises(RuntimeError, match="NFL_REQUIRE_PRIVATE_MODELS"):
        value_ranking.rank_weekly_value(SEASON, WEEK)
