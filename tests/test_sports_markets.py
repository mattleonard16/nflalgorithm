"""Unit tests for NFL markets registration, Poisson anytime TD pricing, and grading."""

from __future__ import annotations

import math
import pandas as pd
import pytest
from scipy.stats import norm

from sports.markets import get_sport
from sports.nfl import MARKETS, MARKET_MIN_EXPECTED_VOLUME
from utils.nfl_markets import melt_actuals
from utils.nfl_sigma import SIGMA_DEFAULTS, SIGMA_FLOORS, compute_player_sigma
from value_betting_engine import prob_over


def test_nfl_markets_registration() -> None:
    nfl = get_sport("nfl")
    assert "anytime_touchdown" in nfl.markets
    assert "receptions" in nfl.markets
    assert "anytime_touchdown" in MARKETS
    assert "receptions" in MARKETS


def test_anytime_touchdown_market_spec() -> None:
    nfl = get_sport("nfl")
    spec = nfl.markets["anytime_touchdown"]
    assert spec.stat_column == "anytime_td"
    assert spec.unit == "touchdowns"
    assert set(spec.positions) == {"RB", "WR", "TE", "QB"}


def test_receptions_market_spec() -> None:
    nfl = get_sport("nfl")
    spec = nfl.markets["receptions"]
    assert spec.stat_column == "receptions"
    assert spec.unit == "receptions"
    assert set(spec.positions) == {"WR", "TE", "RB"}


def test_market_min_volume_floors() -> None:
    assert MARKET_MIN_EXPECTED_VOLUME["receptions"] == 1.5
    assert MARKET_MIN_EXPECTED_VOLUME["anytime_touchdown"] == 2.5


def test_sigma_floors_and_defaults() -> None:
    assert SIGMA_FLOORS[("receptions", None)] == 1.4
    assert SIGMA_DEFAULTS[("receptions", None)] == 2.2
    assert SIGMA_FLOORS[("anytime_touchdown", None)] == 0.35
    assert SIGMA_DEFAULTS[("anytime_touchdown", None)] == 0.48

    sigma_rec = compute_player_sigma([], market="receptions", position="WR")
    assert sigma_rec == 2.2

    sigma_td = compute_player_sigma([], market="anytime_touchdown", position="RB")
    assert sigma_td == 0.48


def test_poisson_probability_for_anytime_touchdown() -> None:
    mu = 0.65
    sigma = 0.45
    line = 0.5

    # Anytime TD uses Poisson survival: P(X >= 1) = 1 - exp(-mu)
    p_td = prob_over(mu, sigma, line, market="anytime_touchdown")
    expected_poisson = 1.0 - math.exp(-mu)
    assert p_td == pytest.approx(expected_poisson)

    # Continuous market uses normal distribution
    p_norm = prob_over(mu, sigma, line, market="rushing_yards")
    expected_norm = float(1.0 - norm.cdf(line, loc=mu, scale=sigma))
    assert p_norm == pytest.approx(expected_norm)
    assert p_td != p_norm


def test_melt_actuals_synthesizes_anytime_td() -> None:
    actuals = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "player_id": "P1",
                "rushing_yards": 80.0,
                "receiving_yards": 0.0,
                "passing_yards": 0.0,
                "receptions": 0,
                "targets": 0,
                "rushing_tds": 1,
                "receiving_tds": 0,
            },
            {
                "season": 2025,
                "week": 1,
                "player_id": "P2",
                "rushing_yards": 0.0,
                "receiving_yards": 50.0,
                "passing_yards": 0.0,
                "receptions": 4,
                "targets": 6,
                "rushing_tds": 0,
                "receiving_tds": 0,
            },
        ]
    )

    melted = melt_actuals(actuals)
    p1_td = melted[(melted["player_id"] == "P1") & (melted["market"] == "anytime_touchdown")]
    p2_td = melted[(melted["player_id"] == "P2") & (melted["market"] == "anytime_touchdown")]
    p2_rec = melted[(melted["player_id"] == "P2") & (melted["market"] == "receptions")]

    assert not p1_td.empty
    assert p1_td.iloc[0]["actual"] == 1

    assert not p2_td.empty
    assert p2_td.iloc[0]["actual"] == 0

    assert not p2_rec.empty
    assert p2_rec.iloc[0]["actual"] == 4
