"""Unit tests for NFL markets registration, Poisson anytime TD pricing, and grading."""

from __future__ import annotations

import math
import pandas as pd
import pytest
from scipy.stats import gamma, norm, poisson

from sports.markets import get_sport
from sports.nfl import MARKETS, MARKET_MIN_EXPECTED_VOLUME
from utils.nfl_markets import DATABASE_STAT_COLUMNS, melt_actuals, synthesize_anytime_td
from utils.nfl_sigma import SIGMA_DEFAULTS, SIGMA_FLOORS, compute_player_sigma
from utils.nfl_markets import GAMMA_MARKETS, prob_over


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
    assert MARKET_MIN_EXPECTED_VOLUME["anytime_touchdown"] == 0.5


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

    # A continuous market does not get the Poisson branch. `receptions` is the
    # one used here because it is still normal-priced; the yardage markets moved
    # to gamma (see the GAMMA_MARKETS tests below).
    p_norm = prob_over(mu, sigma, line, market="receptions")
    expected_norm = float(1.0 - norm.cdf(line, loc=mu, scale=sigma))
    assert p_norm == pytest.approx(expected_norm)
    assert p_td != p_norm


def test_gamma_markets_are_the_yardage_markets_only() -> None:
    # Receptions and targets are counts with no walk-forward rows to calibrate
    # against, and anytime_touchdown prices off Poisson. Adding one here without
    # measuring it first is the mistake this guards.
    assert GAMMA_MARKETS == {"passing_yards", "rushing_yards", "receiving_yards"}


@pytest.mark.parametrize("market", sorted(GAMMA_MARKETS))
def test_gamma_preserves_mu_and_sigma(market: str) -> None:
    # Method of moments: only the shape of the curve changes, so sigma keeps the
    # meaning utils/nfl_sigma.py calibrated it to. A swapped shape/scale passes
    # the "under 0.5" test below but fails this one.
    mu, sigma = 62.0, 31.0
    shape, scale = (mu / sigma) ** 2, sigma**2 / mu
    assert gamma.mean(a=shape, scale=scale) == pytest.approx(mu)
    assert gamma.std(a=shape, scale=scale) == pytest.approx(sigma)

    expected = float(gamma.sf(55.5, a=shape, scale=scale))
    assert prob_over(mu, sigma, 55.5, market=market) == pytest.approx(expected)


@pytest.mark.parametrize("market", sorted(GAMMA_MARKETS))
def test_yardage_line_at_mu_prices_below_even(market: str) -> None:
    # The defect this change fixes. Weekly yardage is right-skewed, so a player
    # clears his own mean less than half the time. The normal curve said exactly
    # 50% and overstated every over by 4 to 11 percentage points.
    mu, sigma = 62.0, 31.0
    p_gamma = prob_over(mu, sigma, mu, market=market)
    p_normal = prob_over(mu, sigma, mu, market="receptions")

    assert p_normal == pytest.approx(0.5)
    assert p_gamma < 0.47
    assert p_gamma > 0.35


@pytest.mark.parametrize("market", sorted(GAMMA_MARKETS))
def test_prob_over_falls_as_the_line_rises(market: str) -> None:
    prices = [prob_over(62.0, 31.0, line, market=market) for line in (20.5, 61.5, 99.5)]
    assert prices == sorted(prices, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in prices)


@pytest.mark.parametrize("mu", [0.0, -5.0])
def test_yardage_falls_back_to_normal_when_mu_is_not_positive(mu: float) -> None:
    # Gamma needs a positive mean. A zero-volume projection must still price.
    p = prob_over(mu, 20.0, 30.5, market="receiving_yards")
    assert p == pytest.approx(float(1.0 - norm.cdf(30.5, loc=mu, scale=20.0)))


@pytest.mark.parametrize("market", ["receiving_yards", "receptions", None])
def test_zero_sigma_prices_a_point_mass_rather_than_nan(market: str | None) -> None:
    # scipy divides by the scale and returns NaN here. A NaN probability does not
    # raise: it flows into edge, fails the >= threshold comparison, and the row
    # vanishes from the card with no error logged anywhere.
    assert prob_over(40.0, 0.0, 30.5, market=market) == 1.0
    assert prob_over(20.0, 0.0, 30.5, market=market) == 0.0
    assert prob_over(30.5, 0.0, 30.5, market=market) == 0.0
    assert not math.isnan(prob_over(40.0, -1.0, 30.5, market=market))


def test_touchdown_line_selects_the_threshold_not_just_one_plus() -> None:
    # The market name must not override the line. A 1.5 TD line asks for 2+,
    # and answering P(X >= 1) there roughly doubles the price.
    mu = 1.2
    p_one_plus = prob_over(mu, 1.0, 0.5, market="anytime_touchdown")
    p_two_plus = prob_over(mu, 1.0, 1.5, market="anytime_touchdown")

    assert p_one_plus == pytest.approx(1.0 - math.exp(-mu))
    assert p_two_plus == pytest.approx(float(poisson.sf(1, mu)))
    assert p_two_plus < p_one_plus


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


def test_database_stat_columns_are_physical_only() -> None:
    # anytime_td is virtual and must never appear in a SELECT against
    # player_stats_enhanced; its two source columns must.
    assert "anytime_td" not in DATABASE_STAT_COLUMNS
    assert "rushing_tds" in DATABASE_STAT_COLUMNS
    assert "receiving_tds" in DATABASE_STAT_COLUMNS
    assert DATABASE_STAT_COLUMNS == sorted(DATABASE_STAT_COLUMNS)


def test_synthesize_anytime_td_counts_both_td_columns() -> None:
    df = pd.DataFrame([{"rushing_tds": 2, "receiving_tds": 1}])
    assert synthesize_anytime_td(df)["anytime_td"].iloc[0] == 3


def test_synthesize_anytime_td_tolerates_a_missing_column() -> None:
    # A partial select used to return scalar 0 here and crash on .fillna.
    rush_only = synthesize_anytime_td(pd.DataFrame([{"rushing_tds": 1}]))
    assert rush_only["anytime_td"].iloc[0] == 1

    rec_only = synthesize_anytime_td(pd.DataFrame([{"receiving_tds": 2}]))
    assert rec_only["anytime_td"].iloc[0] == 2


def test_synthesize_anytime_td_leaves_td_less_frames_alone() -> None:
    df = pd.DataFrame([{"rushing_yards": 80.0}])
    result = synthesize_anytime_td(df)
    assert "anytime_td" not in result.columns


def test_melt_actuals_reports_td_counts_not_binary_flags() -> None:
    actuals = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "player_id": "P1",
                "rushing_tds": 2,
                "receiving_tds": 1,
            }
        ]
    )
    melted = melt_actuals(actuals)
    td = melted[melted["market"] == "anytime_touchdown"]
    assert not td.empty
    assert td.iloc[0]["actual"] == 3
