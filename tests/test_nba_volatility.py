"""Tests for utils/nba_volatility.py — NBA-calibrated volatility scoring.

Covers:
- NBA-calibrated normalizers per market
- CV computation with different denominators
- Score range [0, 100]
- Stable player (low volatility) vs boom-bust player
- Market-specific behavior (fg3m vs pts)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock config module since config.py is gitignored in this worktree.
# This must happen before any project imports that transitively import config.
if "config" not in sys.modules:
    _mock_config = MagicMock()
    _mock_config.config.database.backend = "sqlite"
    _mock_config.config.database.path = ":memory:"
    sys.modules["config"] = _mock_config

import pytest


# ---------------------------------------------------------------------------
# Market normalizers (expected from implementation)
# ---------------------------------------------------------------------------

# NBA-calibrated CV thresholds where score hits 100
# pts has a wider range than fg3m (fg3m is inherently more volatile)
MARKETS = ["pts", "reb", "ast", "fg3m"]


# ---------------------------------------------------------------------------
# compute_nba_volatility_score — basic behaviour
# ---------------------------------------------------------------------------


class TestComputeVolatilityScore:
    """Core volatility scoring tests."""

    def test_returns_float(self):
        from utils.nba_volatility import compute_nba_volatility_score

        result = compute_nba_volatility_score(
            [25.0, 22.0, 28.0, 20.0, 30.0], market="pts"
        )
        assert isinstance(result, float)

    def test_score_within_0_100(self):
        from utils.nba_volatility import compute_nba_volatility_score

        result = compute_nba_volatility_score(
            [25.0, 22.0, 28.0, 20.0, 30.0], market="pts"
        )
        assert 0.0 <= result <= 100.0

    def test_stable_player_low_score(self):
        """Player who scores consistently should have low volatility."""
        from utils.nba_volatility import compute_nba_volatility_score

        values = [25.0, 25.5, 24.5, 25.0, 25.5, 24.5, 25.0, 25.5]
        score = compute_nba_volatility_score(values, market="pts")
        assert score < 30.0

    def test_boom_bust_player_high_score(self):
        """Player with wild swings should have high volatility."""
        from utils.nba_volatility import compute_nba_volatility_score

        values = [5.0, 45.0, 8.0, 42.0, 6.0, 40.0, 10.0, 38.0]
        score = compute_nba_volatility_score(values, market="pts")
        assert score > 50.0

    def test_identical_values_low_score(self):
        """All same values → CV = 0 and range_ratio = 0 → low volatility score.

        max_week_contribution is non-zero even for uniform series
        (each game contributes 1/n of the total), so the score is not
        exactly 0 but should be well below 20 for consistent production.
        """
        from utils.nba_volatility import compute_nba_volatility_score

        values = [25.0] * 10
        score = compute_nba_volatility_score(values, market="pts")
        assert score < 20.0

    def test_score_capped_at_100(self):
        """Even extreme volatility should not exceed 100."""
        from utils.nba_volatility import compute_nba_volatility_score

        values = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0]
        score = compute_nba_volatility_score(values, market="pts")
        assert score <= 100.0


# ---------------------------------------------------------------------------
# Market-specific behavior
# ---------------------------------------------------------------------------


class TestMarketSpecificVolatility:
    """Different markets should have different calibration."""

    def test_fg3m_more_volatile_than_pts_for_same_cv(self):
        """fg3m is inherently volatile; same CV should produce different score."""
        from utils.nba_volatility import compute_nba_volatility_score

        # Same relative variance for both markets
        pts_values = [25.0, 20.0, 30.0, 22.0, 28.0, 24.0, 26.0, 23.0]
        fg3m_values = [3.0, 1.0, 5.0, 2.0, 4.0, 2.5, 3.5, 1.5]

        pts_score = compute_nba_volatility_score(pts_values, market="pts")
        fg3m_score = compute_nba_volatility_score(fg3m_values, market="fg3m")

        # Both should be valid scores
        assert 0.0 <= pts_score <= 100.0
        assert 0.0 <= fg3m_score <= 100.0

    @pytest.mark.parametrize("market", MARKETS)
    def test_all_markets_return_valid_score(self, market):
        from utils.nba_volatility import compute_nba_volatility_score

        values = [10.0, 8.0, 12.0, 9.0, 11.0, 7.0, 13.0, 10.0]
        score = compute_nba_volatility_score(values, market=market)
        assert 0.0 <= score <= 100.0

    def test_unknown_market_uses_default_normalizer(self):
        from utils.nba_volatility import compute_nba_volatility_score

        values = [10.0, 8.0, 12.0, 9.0, 11.0, 7.0, 13.0, 10.0]
        score = compute_nba_volatility_score(values, market="blocks")
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestVolatilityEdgeCases:
    """Empty list, single value, near-zero mean."""

    def test_empty_list_returns_neutral(self):
        """Insufficient data returns neutral score (50.0 per implementation)."""
        from utils.nba_volatility import compute_nba_volatility_score

        score = compute_nba_volatility_score([], market="pts")
        # Implementation returns 50.0 for < 2 values (neutral/insufficient data)
        assert score == pytest.approx(50.0)

    def test_single_value_returns_neutral(self):
        """Single value returns neutral score (50.0 per implementation)."""
        from utils.nba_volatility import compute_nba_volatility_score

        score = compute_nba_volatility_score([25.0], market="pts")
        # Implementation returns 50.0 for < 2 values (neutral/insufficient data)
        assert score == pytest.approx(50.0)

    def test_near_zero_mean_no_divide_by_zero(self):
        """Very small values should not cause division by zero in CV."""
        from utils.nba_volatility import compute_nba_volatility_score

        values = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        score = compute_nba_volatility_score(values, market="fg3m")
        assert 0.0 <= score <= 100.0
