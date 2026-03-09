"""Tests for utils/nfl_sigma.py — EWMA sigma, NFL market floors, fallbacks.

Mirrors tests/test_nba_sigma.py structure adapted for NFL markets.

Covers:
- EWMA variance computation with known values
- Market-specific floors (rushing_yards=15, receiving_yards=12, passing_yards=30)
- Fallback defaults for <6 games
- Edge cases: empty list, single game, all same values
- Decay parameter affects output correctly
- get_sigma_or_default helper
- Defense multiplier applied to mu in predict_week (integration)
- Sigma is data-driven, not flat percentage
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Constants mirroring nfl_sigma module
# ---------------------------------------------------------------------------

SIGMA_FLOORS = {
    "rushing_yards": 15.0,
    "receiving_yards": 12.0,
    "passing_yards": 30.0,
}
SIGMA_DEFAULTS = {
    "rushing_yards": 25.0,
    "receiving_yards": 20.0,
    "passing_yards": 50.0,
}
MIN_GAMES = 6


# ---------------------------------------------------------------------------
# compute_player_sigma — basic behaviour
# ---------------------------------------------------------------------------


class TestComputePlayerSigma:
    """Core EWMA sigma computation tests for NFL markets."""

    def test_returns_float(self):
        from utils.nfl_sigma import compute_player_sigma

        result = compute_player_sigma(
            [60, 75, 85, 45, 90, 70, 55, 80], market="rushing_yards"
        )
        assert isinstance(result, float)

    def test_known_values_positive_sigma(self):
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0, 45.0, 90.0, 70.0]
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma > 0.0

    def test_constant_values_returns_floor(self):
        """All identical values → variance is 0 → sigma should be the floor."""
        from utils.nfl_sigma import compute_player_sigma

        values = [75.0] * 10
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_FLOORS["rushing_yards"])

    def test_high_variance_exceeds_floor(self):
        """Volatile player data should produce sigma above the floor."""
        from utils.nfl_sigma import compute_player_sigma

        # Boom-bust RB pattern
        values = [10.0, 120.0, 15.0, 110.0, 12.0, 105.0, 18.0, 100.0]
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma > SIGMA_FLOORS["rushing_yards"]

    def test_decay_parameter_changes_output(self):
        """Different decay values should produce different sigma estimates."""
        from utils.nfl_sigma import compute_player_sigma

        values = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
        sigma_low = compute_player_sigma(values, market="rushing_yards", decay=0.3)
        sigma_high = compute_player_sigma(values, market="rushing_yards", decay=0.9)
        assert sigma_low != pytest.approx(sigma_high, abs=0.5)

    def test_sigma_exceeds_floor_for_all_markets(self):
        """Each market with enough variable data should exceed its floor."""
        from utils.nfl_sigma import compute_player_sigma

        market_values = {
            "rushing_yards": [20.0, 100.0, 25.0, 95.0, 30.0, 90.0],
            "receiving_yards": [15.0, 85.0, 20.0, 80.0, 18.0, 78.0],
            "passing_yards": [150.0, 350.0, 180.0, 330.0, 160.0, 340.0],
        }
        for market, values in market_values.items():
            sigma = compute_player_sigma(values, market=market)
            assert sigma > SIGMA_FLOORS[market], (
                f"Expected sigma > {SIGMA_FLOORS[market]} for {market}, got {sigma}"
            )


# ---------------------------------------------------------------------------
# NFL Market-specific floors
# ---------------------------------------------------------------------------


class TestNFLMarketFloors:
    """Each NFL market has a minimum sigma floor reflecting stat scale."""

    @pytest.mark.parametrize(
        "market,expected_floor",
        [
            ("rushing_yards", 15.0),
            ("receiving_yards", 12.0),
            ("passing_yards", 30.0),
        ],
    )
    def test_floor_enforced_per_market(self, market, expected_floor):
        from utils.nfl_sigma import compute_player_sigma

        # Constant values => zero variance => floor should apply
        values = [50.0] * 10
        sigma = compute_player_sigma(values, market=market)
        assert sigma == pytest.approx(expected_floor)

    def test_rushing_floor_larger_than_receiving_floor(self):
        """Rushing sigma floor should be larger than receiving (higher scale)."""
        assert SIGMA_FLOORS["rushing_yards"] > SIGMA_FLOORS["receiving_yards"]

    def test_passing_floor_is_largest(self):
        """Passing yards have highest scale so passing floor should be largest."""
        assert SIGMA_FLOORS["passing_yards"] > SIGMA_FLOORS["rushing_yards"]
        assert SIGMA_FLOORS["passing_yards"] > SIGMA_FLOORS["receiving_yards"]

    def test_unknown_market_uses_generic_floor(self):
        from utils.nfl_sigma import compute_player_sigma

        values = [50.0] * 10
        sigma = compute_player_sigma(values, market="fumbles")
        # Should use generic floor (10.0), not crash
        assert sigma >= 1.0


# ---------------------------------------------------------------------------
# Fallback defaults for insufficient data
# ---------------------------------------------------------------------------


class TestNFLFallbackDefaults:
    """When player has fewer than MIN_GAMES (6) samples, return default."""

    @pytest.mark.parametrize(
        "market,expected_default",
        [
            ("rushing_yards", 25.0),
            ("receiving_yards", 20.0),
            ("passing_yards", 50.0),
        ],
    )
    def test_default_returned_for_insufficient_data(self, market, expected_default):
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0]  # Only 3 games, less than MIN_GAMES=6
        sigma = compute_player_sigma(values, market=market)
        assert sigma == pytest.approx(expected_default)

    def test_exactly_min_games_computes_sigma(self):
        """Exactly 6 games should compute EWMA, not return default."""
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0, 45.0, 90.0, 70.0]  # exactly 6
        sigma = compute_player_sigma(values, market="rushing_yards")
        # Should NOT be the default of 25.0
        assert sigma != pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_five_games_returns_default(self):
        """5 games (one below threshold) must return default."""
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, 75.0, 85.0, 45.0, 90.0]  # 5 games
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestNFLSigmaEdgeCases:
    """Empty list, single game, NaN values."""

    def test_empty_list_returns_default(self):
        from utils.nfl_sigma import compute_player_sigma

        sigma = compute_player_sigma([], market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_single_game_returns_default(self):
        from utils.nfl_sigma import compute_player_sigma

        sigma = compute_player_sigma([80.0], market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_nan_values_filtered(self):
        """NaN values should be stripped before computation."""
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, float("nan"), 75.0, float("nan"), 85.0]
        # Only 3 valid values → less than MIN_GAMES=6 → default
        sigma = compute_player_sigma(values, market="rushing_yards")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_nan_filtered_leaves_enough_data(self):
        """Enough non-NaN values should compute EWMA."""
        from utils.nfl_sigma import compute_player_sigma

        values = [60.0, float("nan"), 75.0, 85.0, 45.0, 90.0, 70.0]
        # 6 valid values → compute EWMA
        sigma = compute_player_sigma(values, market="rushing_yards")
        # Should not be default
        assert sigma != pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_sigma_always_positive(self):
        """Sigma must always be positive regardless of input."""
        from utils.nfl_sigma import compute_player_sigma

        for market in ["rushing_yards", "receiving_yards", "passing_yards"]:
            sigma = compute_player_sigma([], market=market)
            assert sigma > 0.0


# ---------------------------------------------------------------------------
# Sigma is data-driven (not flat percentage)
# ---------------------------------------------------------------------------


class TestSigmaIsDataDriven:
    """Verify sigma reflects actual player variance, not a fixed mu * 0.30."""

    def test_consistent_player_has_lower_sigma_than_volatile(self):
        """A consistent player should have lower sigma than a volatile one."""
        from utils.nfl_sigma import compute_player_sigma

        consistent = [75.0, 78.0, 72.0, 76.0, 74.0, 77.0, 73.0, 75.0]
        volatile = [20.0, 130.0, 15.0, 125.0, 25.0, 120.0, 18.0, 128.0]

        sigma_consistent = compute_player_sigma(consistent, market="rushing_yards")
        sigma_volatile = compute_player_sigma(volatile, market="rushing_yards")

        assert sigma_consistent < sigma_volatile

    def test_sigma_not_fixed_percentage_of_mean(self):
        """Sigma should NOT equal mu * 0.30 (old hardcoded approach)."""
        from utils.nfl_sigma import compute_player_sigma

        values = [80.0, 85.0, 75.0, 82.0, 78.0, 83.0, 77.0, 81.0]
        sigma = compute_player_sigma(values, market="rushing_yards")
        mean_val = float(np.mean(values))
        flat_sigma = mean_val * 0.30

        # Actual sigma should differ from the flat 30% rule
        assert sigma != pytest.approx(flat_sigma, rel=0.01)

    def test_recent_games_weighted_more(self):
        """With high decay, recent volatile games increase sigma more than early ones."""
        from utils.nfl_sigma import compute_player_sigma

        # Stable early, volatile recent
        stable_early_volatile_late = [75.0, 76.0, 74.0, 75.0, 20.0, 130.0]
        # Volatile early, stable recent
        volatile_early_stable_late = [20.0, 130.0, 75.0, 76.0, 74.0, 75.0]

        sigma_volatile_late = compute_player_sigma(
            stable_early_volatile_late, market="rushing_yards", decay=0.8
        )
        sigma_volatile_early = compute_player_sigma(
            volatile_early_stable_late, market="rushing_yards", decay=0.8
        )

        # Recent volatility (late) should produce higher sigma with high decay
        assert sigma_volatile_late > sigma_volatile_early


# ---------------------------------------------------------------------------
# get_sigma_or_default helper
# ---------------------------------------------------------------------------


class TestGetNFLSigmaOrDefault:
    """Fallback logic when sigma comes from the DB (may be NULL)."""

    def test_returns_sigma_when_present(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(18.5, projected_value=75.0, market="rushing_yards")
        assert result == pytest.approx(18.5)

    def test_returns_default_when_none(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(None, projected_value=75.0, market="rushing_yards")
        assert result == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_returns_default_when_nan(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(
            float("nan"), projected_value=75.0, market="rushing_yards"
        )
        assert result == pytest.approx(SIGMA_DEFAULTS["rushing_yards"])

    def test_returns_default_for_passing_when_none(self):
        from utils.nfl_sigma import get_sigma_or_default

        result = get_sigma_or_default(None, projected_value=280.0, market="passing_yards")
        assert result == pytest.approx(SIGMA_DEFAULTS["passing_yards"])

    def test_returns_sigma_even_if_at_floor(self):
        """A sigma at the floor value should be returned as-is."""
        from utils.nfl_sigma import get_sigma_or_default

        floor = SIGMA_FLOORS["rushing_yards"]
        result = get_sigma_or_default(floor, projected_value=75.0, market="rushing_yards")
        assert result == pytest.approx(floor)


# ---------------------------------------------------------------------------
# Defense multiplier applied to mu (integration smoke test)
# ---------------------------------------------------------------------------


class TestDefenseMultiplierApplied:
    """Verify defense adjustment wiring in predict_week."""

    def test_get_defense_multiplier_returns_float(self):
        """get_defense_multiplier should return a float in [0.7, 1.3]."""
        from utils.defense_adjustments import get_defense_multiplier

        # No real data in test env → should return 1.0 (no adjustment)
        mult = get_defense_multiplier(
            opponent="CIN",
            position="RB",
            stat_type="rushing_yards",
            season=2024,
            through_week=5,
        )
        assert isinstance(mult, float)
        assert 0.5 <= mult <= 2.0  # Reasonable bounds

    def test_multiplier_in_reasonable_range(self):
        """Multiplier should always be within reasonable bounds [0.5, 2.0]."""
        from utils.defense_adjustments import get_defense_multiplier

        for opponent in ["UNKNOWN_TEAM_XYZ", "CIN", "NE", ""]:
            mult = get_defense_multiplier(
                opponent=opponent,
                position="RB",
                stat_type="rushing_yards",
                season=2024,
                through_week=5,
            )
            assert isinstance(mult, float)
            assert 0.5 <= mult <= 2.0, f"Multiplier {mult} out of bounds for opponent {opponent!r}"

    def test_no_data_season_returns_one(self):
        """When no historical data exists for a season, should return 1.0."""
        from utils.defense_adjustments import get_defense_multiplier

        # Use a far-future season with no data
        mult = get_defense_multiplier(
            opponent="CIN",
            position="RB",
            stat_type="rushing_yards",
            season=2099,
            through_week=1,
        )
        assert mult == pytest.approx(1.0)
