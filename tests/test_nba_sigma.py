"""Tests for utils/nba_sigma.py — EWMA sigma, market floors, fallbacks.

Covers:
- EWMA variance computation with known values
- Market-specific floors (pts=2.5, reb=1.5, ast=1.2, fg3m=0.8)
- Fallback defaults for <8 games
- Edge cases: empty list, single game, all same values
- Decay parameter affects output correctly
- get_sigma_or_default helper
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGMA_FLOORS = {"pts": 2.5, "reb": 1.5, "ast": 1.2, "fg3m": 0.8}
SIGMA_DEFAULTS = {"pts": 5.5, "reb": 3.0, "ast": 2.8, "fg3m": 1.8}
MIN_GAMES = 8


# ---------------------------------------------------------------------------
# compute_player_sigma — basic behaviour
# ---------------------------------------------------------------------------


class TestComputePlayerSigma:
    """Core EWMA sigma computation tests."""

    def test_returns_float(self):
        from utils.nba_sigma import compute_player_sigma

        result = compute_player_sigma([20, 22, 25, 18, 30, 24, 19, 27], market="pts")
        assert isinstance(result, float)

    def test_known_values_positive_sigma(self):
        from utils.nba_sigma import compute_player_sigma

        values = [20.0, 22.0, 25.0, 18.0, 30.0, 24.0, 19.0, 27.0]
        sigma = compute_player_sigma(values, market="pts")
        assert sigma > 0.0

    def test_constant_values_returns_floor(self):
        """All identical values → variance is 0 → sigma should be the floor."""
        from utils.nba_sigma import compute_player_sigma

        values = [25.0] * 10
        sigma = compute_player_sigma(values, market="pts")
        assert sigma == pytest.approx(SIGMA_FLOORS["pts"])

    def test_high_variance_exceeds_floor(self):
        """Volatile player data should produce sigma above the floor."""
        from utils.nba_sigma import compute_player_sigma

        values = [10.0, 40.0, 12.0, 38.0, 11.0, 39.0, 13.0, 37.0]
        sigma = compute_player_sigma(values, market="pts")
        assert sigma > SIGMA_FLOORS["pts"]

    def test_decay_parameter_changes_output(self):
        """Different decay values should produce different sigma estimates."""
        from utils.nba_sigma import compute_player_sigma

        values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
        sigma_low_decay = compute_player_sigma(values, market="pts", decay=0.3)
        sigma_high_decay = compute_player_sigma(values, market="pts", decay=0.9)
        assert sigma_low_decay != pytest.approx(sigma_high_decay, abs=0.01)


# ---------------------------------------------------------------------------
# Market-specific floors
# ---------------------------------------------------------------------------


class TestMarketFloors:
    """Each market has a minimum sigma floor."""

    @pytest.mark.parametrize(
        "market,expected_floor",
        [("pts", 2.5), ("reb", 1.5), ("ast", 1.2), ("fg3m", 0.8)],
    )
    def test_floor_enforced_per_market(self, market, expected_floor):
        from utils.nba_sigma import compute_player_sigma

        # Constant values => zero variance => floor should apply
        values = [10.0] * 10
        sigma = compute_player_sigma(values, market=market)
        assert sigma == pytest.approx(expected_floor)

    def test_unknown_market_uses_generic_floor(self):
        from utils.nba_sigma import compute_player_sigma

        values = [10.0] * 10
        sigma = compute_player_sigma(values, market="blocks")
        # Should use generic floor (2.0), not crash
        assert sigma >= 1.0


# ---------------------------------------------------------------------------
# Fallback defaults for insufficient data
# ---------------------------------------------------------------------------


class TestFallbackDefaults:
    """When player has fewer than MIN_GAMES (8) samples, return default."""

    @pytest.mark.parametrize(
        "market,expected_default",
        [("pts", 5.5), ("reb", 3.0), ("ast", 2.8), ("fg3m", 1.8)],
    )
    def test_default_returned_for_insufficient_data(self, market, expected_default):
        from utils.nba_sigma import compute_player_sigma

        values = [20.0, 22.0, 25.0]  # Only 3 games, less than MIN_GAMES
        sigma = compute_player_sigma(values, market=market)
        assert sigma == pytest.approx(expected_default)

    def test_exactly_min_games_computes_sigma(self):
        """Exactly 8 games should compute EWMA, not return default."""
        from utils.nba_sigma import compute_player_sigma

        values = [20.0, 22.0, 25.0, 18.0, 30.0, 24.0, 19.0, 27.0]
        sigma = compute_player_sigma(values, market="pts")
        # Should NOT be the default of 5.5
        assert sigma != pytest.approx(SIGMA_DEFAULTS["pts"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSigmaEdgeCases:
    """Empty list, single game, NaN values."""

    def test_empty_list_returns_default(self):
        from utils.nba_sigma import compute_player_sigma

        sigma = compute_player_sigma([], market="pts")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["pts"])

    def test_single_game_returns_default(self):
        from utils.nba_sigma import compute_player_sigma

        sigma = compute_player_sigma([30.0], market="pts")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["pts"])

    def test_nan_values_filtered(self):
        """NaN values should be stripped before computation."""
        from utils.nba_sigma import compute_player_sigma

        values = [20.0, float("nan"), 22.0, float("nan"), 25.0]
        # Only 3 valid values → less than MIN_GAMES → default
        sigma = compute_player_sigma(values, market="pts")
        assert sigma == pytest.approx(SIGMA_DEFAULTS["pts"])


# ---------------------------------------------------------------------------
# get_sigma_or_default helper
# ---------------------------------------------------------------------------


class TestGetSigmaOrDefault:
    """Fallback logic when sigma comes from the DB (may be NULL)."""

    def test_returns_sigma_when_present(self):
        from utils.nba_sigma import get_sigma_or_default

        result = get_sigma_or_default(4.5, projected_value=25.0, market="pts")
        assert result == pytest.approx(4.5)

    def test_returns_default_when_none(self):
        from utils.nba_sigma import get_sigma_or_default

        result = get_sigma_or_default(None, projected_value=25.0, market="pts")
        assert result == pytest.approx(SIGMA_DEFAULTS["pts"])

    def test_returns_default_when_nan(self):
        from utils.nba_sigma import get_sigma_or_default

        result = get_sigma_or_default(float("nan"), projected_value=25.0, market="pts")
        assert result == pytest.approx(SIGMA_DEFAULTS["pts"])
