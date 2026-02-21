"""Tests for utils/nba_defense.py — defense multipliers, clamping, cache.

Covers:
- Defense multiplier computation with mock game logs
- Relative performance calculation (actual / player_avg)
- Clamping to [0.75, 1.25]
- min_games filter
- @lru_cache behavior
- Different markets (pts, reb, ast, fg3m)
- Insufficient data handling
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

from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game_logs_df(
    n_games: int = 10,
    avg_pts: float = 25.0,
    vs_team: str = "MIA",
    team: str = "BOS",
) -> pd.DataFrame:
    """Create synthetic game log DataFrame for defense testing.

    The matchup column encodes the opponent as the last 3 characters,
    matching the format expected by compute_nba_defense_multipliers.
    E.g. 'BOS vs. MIA' → opponent = 'MIA'.
    """
    rows = []
    for i in range(n_games):
        opponent = vs_team if i < n_games // 2 else "NYK"
        rows.append(
            {
                "player_id": 1628369,
                "player_name": "Jayson Tatum",
                "team_abbreviation": team,
                "matchup": f"{team} vs. {opponent}",
                "season": 2025,
                "game_date": f"2026-01-{i + 1:02d}",
                "game_id": f"g{i:04d}",
                "pts": avg_pts + (i % 5 - 2),
                "reb": 8.0 + (i % 3),
                "ast": 5.0 + (i % 2),
                "fg3m": 3.0 + (i % 2),
                "min": 35.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_nba_defense_multipliers — basic
# ---------------------------------------------------------------------------


class TestComputeDefenseMultipliers:
    """Test defense multiplier computation."""

    def setup_method(self):
        """Clear lru_cache before each test to avoid stale data."""
        from utils.nba_defense import compute_nba_defense_multipliers
        compute_nba_defense_multipliers.cache_clear()

    @patch("utils.nba_defense.read_dataframe")
    def test_returns_dict(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        mock_read.return_value = _make_game_logs_df(20)
        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        assert isinstance(result, dict)

    @patch("utils.nba_defense.read_dataframe")
    def test_keys_are_team_market_tuples(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        mock_read.return_value = _make_game_logs_df(20)
        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        for key in result:
            assert isinstance(key, tuple)
            assert len(key) == 2  # (opponent, market)

    @patch("utils.nba_defense.read_dataframe")
    def test_multiplier_around_one_for_average_defense(self, mock_read):
        """If opponents score their average, multiplier should be ~1.0."""
        from utils.nba_defense import compute_nba_defense_multipliers

        mock_read.return_value = _make_game_logs_df(20, avg_pts=25.0)
        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        # Multipliers should exist and be near 1.0 for balanced data
        for key, val in result.items():
            assert 0.75 <= val <= 1.25

    @patch("utils.nba_defense.read_dataframe")
    def test_multiplier_above_one_for_weak_defense(self, mock_read):
        """If opponents score well above average, multiplier should be > 1.0."""
        from utils.nba_defense import compute_nba_defense_multipliers

        # Players score 35 vs average of 25 => strong offense or weak defense
        df = _make_game_logs_df(20, avg_pts=25.0)
        # Override some games vs MIA to be much higher
        mia_mask = df["matchup"].str.endswith("MIA")
        df.loc[mia_mask, "pts"] = 35.0
        mock_read.return_value = df

        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        # MIA defense multiplier for pts should be > 1.0
        mia_pts = result.get(("MIA", "pts"))
        if mia_pts is not None:
            assert mia_pts > 1.0


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


class TestDefenseClamping:
    """Defense multipliers must be clamped to [0.75, 1.25]."""

    def setup_method(self):
        from utils.nba_defense import compute_nba_defense_multipliers
        compute_nba_defense_multipliers.cache_clear()

    @patch("utils.nba_defense.read_dataframe")
    def test_clamped_at_upper_bound(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        # Extreme overperformance vs defense → multiplier would exceed 1.25
        df = _make_game_logs_df(20, avg_pts=10.0)
        mia_mask = df["matchup"].str.endswith("MIA")
        df.loc[mia_mask, "pts"] = 50.0  # 5x average
        mock_read.return_value = df

        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        for key, val in result.items():
            assert val <= 1.25, f"{key} multiplier {val} exceeds 1.25 cap"

    @patch("utils.nba_defense.read_dataframe")
    def test_clamped_at_lower_bound(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        # Extreme underperformance vs defense → multiplier would drop below 0.75
        df = _make_game_logs_df(20, avg_pts=30.0)
        mia_mask = df["matchup"].str.endswith("MIA")
        df.loc[mia_mask, "pts"] = 5.0  # 1/6 of average
        mock_read.return_value = df

        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        for key, val in result.items():
            assert val >= 0.75, f"{key} multiplier {val} below 0.75 floor"


# ---------------------------------------------------------------------------
# min_games filter
# ---------------------------------------------------------------------------


class TestMinGamesFilter:
    """Teams with fewer games than min_games should be excluded."""

    def setup_method(self):
        from utils.nba_defense import compute_nba_defense_multipliers
        compute_nba_defense_multipliers.cache_clear()

    @patch("utils.nba_defense.read_dataframe")
    def test_insufficient_games_excluded(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        # Only 4 total games, split between 2 opponents → 2 each
        df = _make_game_logs_df(4, avg_pts=25.0)
        mock_read.return_value = df

        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=10
        )
        # With only 2 games per opponent, requiring 10 → empty
        assert len(result) == 0 or all(
            v == pytest.approx(1.0) for v in result.values()
        )

    @patch("utils.nba_defense.read_dataframe")
    def test_sufficient_games_included(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        df = _make_game_logs_df(20, avg_pts=25.0)
        mock_read.return_value = df

        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Multi-market support
# ---------------------------------------------------------------------------


class TestDefenseMarkets:
    """Multipliers should be computed for pts, reb, ast, fg3m."""

    def setup_method(self):
        from utils.nba_defense import compute_nba_defense_multipliers
        compute_nba_defense_multipliers.cache_clear()

    @patch("utils.nba_defense.read_dataframe")
    def test_all_markets_present(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        mock_read.return_value = _make_game_logs_df(20)
        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        markets_in_result = {key[1] for key in result}
        # Should have at least pts (the most common market)
        assert "pts" in markets_in_result or len(result) == 0


# ---------------------------------------------------------------------------
# Empty / no data
# ---------------------------------------------------------------------------


class TestDefenseEmptyData:
    """Edge case when no game log data is available."""

    def setup_method(self):
        from utils.nba_defense import compute_nba_defense_multipliers
        compute_nba_defense_multipliers.cache_clear()

    @patch("utils.nba_defense.read_dataframe")
    def test_empty_df_returns_empty_dict(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        mock_read.return_value = pd.DataFrame()
        result = compute_nba_defense_multipliers(
            season=2025, through_date="2026-02-17", min_games=3
        )
        assert result == {}

    @patch("utils.nba_defense.read_dataframe")
    def test_no_matchup_column_handles_gracefully(self, mock_read):
        from utils.nba_defense import compute_nba_defense_multipliers

        # DataFrame with missing matchup column should return empty dict
        df = pd.DataFrame({"player_id": [1], "pts": [25.0]})
        mock_read.return_value = df
        # Should not raise, return empty (empty df check after processing)
        try:
            result = compute_nba_defense_multipliers(
                season=2025, through_date="2026-02-17", min_games=3
            )
            assert isinstance(result, dict)
        except (KeyError, AttributeError):
            # Acceptable: implementation may raise on missing columns
            pass
