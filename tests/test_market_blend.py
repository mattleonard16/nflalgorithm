"""Rules for combining the model's mean with the mean a book's price implies.

Every test here runs in CI: `utils/market_blend.py` takes the de-vigged
probability as an argument rather than importing the private no-vig helper, so
nothing in this file needs `value_betting_engine`.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from utils.market_blend import (
    EARLY_SEASON_WEEKS,
    MIN_RELIABLE_SHAPE,
    blend,
    blend_quotes,
    blend_weight,
    gamma_shape,
    implied_mean,
    model_has_skill,
    skew_is_reliable,
)
from utils.nfl_markets import prob_over


def test_implied_mean_round_trips_through_the_pricing_curve() -> None:
    """The whole point: re-pricing at the recovered mean reproduces the price."""
    for market in ("receiving_yards", "rushing_yards", "passing_yards", "receptions"):
        for p in (0.35, 0.50, 0.62):
            mean = implied_mean(line=45.5, sigma=20.0, p_over=p, market=market)
            assert prob_over(mean, 20.0, 45.5, market=market) == pytest.approx(p, abs=1e-6)


def test_recovered_mean_sits_above_a_coin_flip_line_on_a_skewed_market() -> None:
    """A gamma's median is below its mean, so a book pricing the line at even
    money is claiming a mean *above* that line. Blending toward the raw line
    instead of this number is what dragged a whole card to the under."""
    mean = implied_mean(line=30.0, sigma=20.0, p_over=0.5, market="receiving_yards")
    assert mean > 30.0
    # The normal curve has no skew, so there the line *is* the mean.
    assert implied_mean(line=30.0, sigma=20.0, p_over=0.5, market="receptions") == pytest.approx(
        30.0, abs=1e-4
    )


def test_recovered_mean_rises_with_the_price() -> None:
    means = [
        implied_mean(line=50.5, sigma=22.0, p_over=p, market="receiving_yards")
        for p in (0.30, 0.40, 0.50, 0.60, 0.70)
    ]
    assert means == sorted(means)
    assert len(set(means)) == len(means)


def test_a_price_implying_a_non_positive_mean_is_not_blended() -> None:
    """A normal-priced low-count market can quote an over so cheap that no
    positive mean reproduces it. That is a row to drop, not a mean to average
    into the projection."""
    assert math.isnan(implied_mean(line=0.5, sigma=3.0, p_over=0.001, market="receptions"))


def test_implied_mean_rejects_input_that_cannot_have_a_solution() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        implied_mean(line=40.5, sigma=20.0, p_over=0.0, market="receiving_yards")
    with pytest.raises(ValueError, match="strictly inside"):
        implied_mean(line=40.5, sigma=20.0, p_over=1.0, market="receiving_yards")
    with pytest.raises(ValueError, match="sigma must be positive"):
        implied_mean(line=40.5, sigma=0.0, p_over=0.5, market="receiving_yards")
    with pytest.raises(ValueError, match="line must be non-negative"):
        implied_mean(line=-1.0, sigma=20.0, p_over=0.5, market="receiving_yards")


def test_a_raw_two_sided_price_recovers_a_higher_mean_than_the_devigged_one() -> None:
    """Guards the docstring's warning. Both sides of a -110/-110 quote imply
    52.4%, so skipping the de-vig hands this function 52.4% instead of 50% and
    the recovered mean comes out too high."""
    devigged = implied_mean(line=40.5, sigma=18.0, p_over=0.5, market="receiving_yards")
    raw = implied_mean(line=40.5, sigma=18.0, p_over=110 / 210, market="receiving_yards")
    assert raw > devigged


def test_blend_endpoints_and_midpoint() -> None:
    assert blend(30.0, 50.0, weight=0.0) == pytest.approx(30.0)
    assert blend(30.0, 50.0, weight=1.0) == pytest.approx(50.0)
    assert blend(30.0, 50.0, weight=0.5) == pytest.approx(40.0)


def test_blend_rejects_a_weight_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
        blend(30.0, 50.0, weight=1.5)
    with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
        blend(30.0, 50.0, weight=-0.1)


def test_configured_weight_is_the_equal_weight_baseline() -> None:
    """Moving this off 0.5 needs a season of CLV behind it, per the config
    comment, so a silent change should fail here first."""
    assert blend_weight() == pytest.approx(0.5)


def test_gamma_shape_matches_the_pricing_parameterisation() -> None:
    assert gamma_shape(40.0, 20.0) == pytest.approx(4.0)
    assert gamma_shape(20.0, 20.0) == pytest.approx(1.0)
    assert math.isnan(gamma_shape(0.0, 20.0))
    assert math.isnan(gamma_shape(40.0, 0.0))


def test_skew_screen_rejects_a_sigma_at_or_above_mu_on_a_gamma_market() -> None:
    """Shape below 1 is where the gamma's median was measured to be in the wrong
    place by 5 to 14 points. See MIN_RELIABLE_SHAPE."""
    assert MIN_RELIABLE_SHAPE == 1.0
    assert skew_is_reliable(50.0, 20.0, "receiving_yards") is True
    assert skew_is_reliable(20.0, 20.0, "receiving_yards") is True  # shape exactly 1.0
    assert skew_is_reliable(20.0, 25.0, "receiving_yards") is False
    assert skew_is_reliable(20.0, 40.0, "rushing_yards") is False


def test_skew_screen_passes_markets_that_are_not_gamma_priced() -> None:
    """Receptions prices off a normal curve and anytime TD off Poisson, so
    neither has this shape to screen on; the caller's other rules apply."""
    assert skew_is_reliable(3.0, 5.0, "receptions") is True
    assert skew_is_reliable(0.4, 2.0, "anytime_touchdown") is True
    assert skew_is_reliable(3.0, 5.0, None) is True


def _quotes() -> pd.DataFrame:
    """Two books on one player at different lines, plus a second player.

    The different lines are the point: pooling the lines themselves would
    compare two different bets, so the recovered means are what get pooled.
    """
    return pd.DataFrame(
        [
            {"player_id": "A", "market": "receiving_yards", "line": 45.5, "mu": 30.0,
             "sigma": 20.0, "p_market_over": 0.50},
            {"player_id": "A", "market": "receiving_yards", "line": 49.5, "mu": 30.0,
             "sigma": 20.0, "p_market_over": 0.46},
            {"player_id": "B", "market": "receiving_yards", "line": 20.5, "mu": 60.0,
             "sigma": 20.0, "p_market_over": 0.52},
        ]
    )


def test_blend_quotes_pools_the_recovered_means_per_player_and_market() -> None:
    out = blend_quotes(_quotes(), weight=0.5)

    a = out[out["player_id"] == "A"]
    expected = a["market_mean"].median()
    assert a["market_mean_consensus"].nunique() == 1
    assert a["market_mean_consensus"].iloc[0] == pytest.approx(expected)
    # Both of A's rows blend against the same consensus even though the books
    # posted different lines.
    assert a["mu_blended"].nunique() == 1
    assert a["mu_blended"].iloc[0] == pytest.approx(0.5 * 30.0 + 0.5 * expected)


def test_blend_quotes_moves_the_projection_toward_the_market() -> None:
    out = blend_quotes(_quotes(), weight=0.5)
    for _, row in out.iterrows():
        lo, hi = sorted((row["mu"], row["market_mean_consensus"]))
        assert lo <= row["mu_blended"] <= hi
    # A is projected well below the market and B well above, so the blend must
    # pull them in opposite directions rather than always one way.
    assert out.loc[out["player_id"] == "A", "mu_blended"].iloc[0] > 30.0
    assert out.loc[out["player_id"] == "B", "mu_blended"].iloc[0] < 60.0


def test_blend_quotes_keeps_unsolvable_rows_as_nan_rather_than_dropping_them() -> None:
    """Silently shortening a card hides coverage loss, so the caller decides."""
    frame = _quotes()
    frame.loc[len(frame)] = {
        "player_id": "C", "market": "receptions", "line": 0.5, "mu": 2.0,
        "sigma": 3.0, "p_market_over": 0.001,
    }
    out = blend_quotes(frame, weight=0.5)

    assert len(out) == 4
    bad = out[out["player_id"] == "C"]
    assert bad["market_mean"].isna().all()
    assert bad["mu_blended"].isna().all()
    assert bad["skew_reliable"].tolist() == [False]
    assert out["skew_reliable"].dtype == bool
    assert out[out["player_id"] == "A"]["mu_blended"].notna().all()


def test_blend_quotes_ignores_a_failed_row_when_pooling_its_players_consensus() -> None:
    """One unusable quote must not drag its own player's consensus."""
    frame = _quotes()
    frame.loc[len(frame)] = {
        "player_id": "A", "market": "receiving_yards", "line": 45.5, "mu": 30.0,
        "sigma": None, "p_market_over": 0.50,
    }
    out = blend_quotes(frame, weight=0.5)

    solvable = out[(out["player_id"] == "A") & out["market_mean"].notna()]
    assert out.loc[out["player_id"] == "A", "market_mean_consensus"].dropna().iloc[
        0
    ] == pytest.approx(solvable["market_mean"].median())


def test_blend_quotes_screens_the_skew_on_the_blended_mean() -> None:
    """The screen has to read the mean that will actually be priced. Blending
    raises this player's mean from 30 to over 45, which carries the gamma shape
    from below 1 to above it, so screening on the raw mu would drop a row the
    engine is about to price correctly."""
    frame = _quotes()
    frame.loc[:, "sigma"] = 35.0
    out = blend_quotes(frame, weight=0.5)

    a = out[out["player_id"] == "A"]
    assert gamma_shape(30.0, 35.0) < MIN_RELIABLE_SHAPE
    assert a["mu_blended"].iloc[0] > 35.0
    assert bool(a["skew_reliable"].iloc[0]) is True


def test_blend_quotes_requires_the_columns_it_prices_from() -> None:
    with pytest.raises(ValueError, match="blend_quotes needs columns"):
        blend_quotes(pd.DataFrame({"player_id": ["A"], "market": ["receiving_yards"]}))


def test_blend_quotes_rejects_a_weight_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
        blend_quotes(_quotes(), weight=2.0)


def test_early_season_screen_rejects_passing_yards_in_the_first_four_weeks() -> None:
    """Measured on 2025: weeks 1-4 passing mu correlates 0.21 with the actual and
    its MAE is worse than predicting the league mean. See EARLY_SEASON_WEEKS."""
    assert EARLY_SEASON_WEEKS == 4
    for week in (1, 2, 3, 4):
        assert model_has_skill("passing_yards", week) is False
    for week in (5, 12, 18):
        assert model_has_skill("passing_yards", week) is True


def test_early_season_screen_leaves_the_other_markets_alone() -> None:
    """Receiving and rushing correlate 0.45 to 0.63 in every week of 2025, week 1
    included, so muting the whole card early would throw away working markets."""
    for market in ("receiving_yards", "rushing_yards", "receptions", "anytime_touchdown"):
        assert model_has_skill(market, 1) is True


def test_early_season_screen_passes_a_week_it_cannot_read() -> None:
    """A null week cannot be screened on; the caller's other rules still apply."""
    assert model_has_skill("passing_yards", None) is True
    assert model_has_skill("passing_yards", "not-a-week") is True


def test_blend_quotes_flags_early_season_passing_quotes() -> None:
    frame = _quotes()
    frame["week"] = 1
    frame.loc[len(frame)] = {
        "player_id": "Q", "market": "passing_yards", "line": 233.5, "mu": 146.0,
        "sigma": 60.0, "p_market_over": 0.50, "week": 1,
    }
    out = blend_quotes(frame, weight=0.5)

    assert out["model_skilled"].dtype == bool
    assert bool(out.loc[out["player_id"] == "Q", "model_skilled"].iloc[0]) is False
    assert out.loc[out["player_id"] != "Q", "model_skilled"].all()
    # The skew screen is a separate question and must not be clobbered by this one.
    assert bool(out.loc[out["player_id"] == "Q", "skew_reliable"].iloc[0]) is True


def test_blend_quotes_without_a_week_column_does_not_silently_screen() -> None:
    """A frame with no week cannot answer the question, so every row stays in and
    the module logs it. Defaulting to False would drop a whole card on a schema
    change, so the safe default is True plus a warning."""
    out = blend_quotes(_quotes(), weight=0.5)
    assert out["model_skilled"].all()
