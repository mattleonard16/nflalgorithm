"""Tests for utils/nba_usage.py — usage features, fga_share, spike detection.

Covers:
- compute_usage_features() with known team/player data
- fga_share calculation (player FGA / team FGA)
- min_share calculation
- usage_delta (L5 - L10 trend)
- is_usage_spike() with threshold detection
- Edge cases: no team data, single game, zero FGA
"""

from __future__ import annotations

import math

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_player_df(n_games: int = 10) -> pd.DataFrame:
    """Create a synthetic player game log DataFrame.

    Each row represents one player in one game.  The team totals are
    derived from the same DataFrame inside compute_usage_features, so
    we include only this player's data (they are the only player on the
    'team' for testing purposes).
    """
    return pd.DataFrame(
        {
            "player_id": [1628369] * n_games,
            "player_name": ["Jayson Tatum"] * n_games,
            "team_abbreviation": ["BOS"] * n_games,
            "season": [2025] * n_games,
            "game_date": [f"2026-01-{i + 1:02d}" for i in range(n_games)],
            "game_id": [f"00225{i:05d}" for i in range(n_games)],
            "fga": [20.0 + (i % 3) for i in range(n_games)],
            "min": [34.0 + (i % 4) for i in range(n_games)],
            "pts": [25.0 + (i % 8) for i in range(n_games)],
        }
    )


# ---------------------------------------------------------------------------
# compute_usage_features — basic behaviour
# ---------------------------------------------------------------------------


class TestComputeUsageFeatures:
    """Test compute_usage_features() DataFrame transformations."""

    def test_returns_dataframe(self):
        from utils.nba_usage import compute_usage_features

        df = _make_player_df()
        result = compute_usage_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_fga_share_column(self):
        from utils.nba_usage import compute_usage_features

        df = _make_player_df()
        result = compute_usage_features(df)
        assert "fga_share" in result.columns

    def test_fga_share_calculation_correct(self):
        """When player is the only player on team, fga_share = 1.0."""
        from utils.nba_usage import compute_usage_features

        df = _make_player_df(1)
        df["fga"] = [20.0]
        result = compute_usage_features(df)
        # With only one player on BOS in this game, team_fga = player_fga
        assert result.iloc[0]["fga_share"] == pytest.approx(1.0)

    def test_fga_share_with_two_players(self):
        """fga_share = player_fga / team_fga when multiple players share a game."""
        from utils.nba_usage import compute_usage_features

        df = pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["Player A", "Player B"],
                "team_abbreviation": ["BOS", "BOS"],
                "season": [2025, 2025],
                "game_date": ["2026-01-01", "2026-01-01"],
                "game_id": ["g0001", "g0001"],
                "fga": [20.0, 80.0],
                "min": [34.0, 34.0],
                "pts": [25.0, 30.0],
            }
        )
        result = compute_usage_features(df)
        player_a = result[result["player_id"] == 1].iloc[0]
        assert player_a["fga_share"] == pytest.approx(0.20)

    def test_adds_min_share_column(self):
        from utils.nba_usage import compute_usage_features

        df = _make_player_df()
        result = compute_usage_features(df)
        assert "min_share" in result.columns

    def test_min_share_calculation_correct(self):
        """When player is the only player, min_share = 1.0."""
        from utils.nba_usage import compute_usage_features

        df = _make_player_df(1)
        df["min"] = [36.0]
        result = compute_usage_features(df)
        assert result.iloc[0]["min_share"] == pytest.approx(1.0)

    def test_min_share_with_two_players(self):
        """min_share = player_min / team_min."""
        from utils.nba_usage import compute_usage_features

        df = pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["Player A", "Player B"],
                "team_abbreviation": ["BOS", "BOS"],
                "season": [2025, 2025],
                "game_date": ["2026-01-01", "2026-01-01"],
                "game_id": ["g0001", "g0001"],
                "fga": [20.0, 20.0],
                "min": [36.0, 204.0],
                "pts": [25.0, 30.0],
            }
        )
        result = compute_usage_features(df)
        player_a = result[result["player_id"] == 1].iloc[0]
        assert player_a["min_share"] == pytest.approx(36.0 / 240.0)

    def test_adds_usage_delta_column(self):
        """usage_delta tracks L5 - L10 trend in fga_share."""
        from utils.nba_usage import compute_usage_features

        df = _make_player_df(12)
        result = compute_usage_features(df)
        assert "usage_delta" in result.columns

    def test_usage_delta_positive_for_increasing_share(self):
        """When recent fga_share is higher, usage_delta should be positive."""
        from utils.nba_usage import compute_usage_features

        n = 12
        df = _make_player_df(n)
        # Increasing FGA over time: early games low, recent high.
        # Because the player is the only player, team_fga == player_fga so
        # fga_share is always 1.0.  We need two players to create genuine
        # share variation.  Build a two-player dataset where player 1 starts
        # low and ends high.
        rows = []
        for i in range(n):
            p1_fga = 15.0 if i < 7 else 25.0
            p2_fga = 90.0 - p1_fga
            for pid, name, fga in [(1, "P1", p1_fga), (2, "P2", p2_fga)]:
                rows.append(
                    {
                        "player_id": pid,
                        "player_name": name,
                        "team_abbreviation": "BOS",
                        "season": 2025,
                        "game_date": f"2026-01-{i + 1:02d}",
                        "game_id": f"g{i:04d}",
                        "fga": fga,
                        "min": 34.0,
                        "pts": 25.0,
                    }
                )
        two_player_df = pd.DataFrame(rows)
        result = compute_usage_features(two_player_df)
        p1_rows = result[result["player_id"] == 1].sort_values("game_date")
        last_delta = p1_rows.iloc[-1]["usage_delta"]
        assert last_delta > 0.0


# ---------------------------------------------------------------------------
# is_usage_spike
# ---------------------------------------------------------------------------


class TestIsUsageSpike:
    """Test spike detection for sudden usage changes."""

    def test_spike_detected_when_above_threshold(self):
        from utils.nba_usage import is_usage_spike

        # L5 avg = 0.30, L10 avg = 0.20 → delta = 0.10 > 0.08 threshold
        assert is_usage_spike(fga_share_l5=0.30, fga_share_l10=0.20, threshold=0.08)

    def test_no_spike_when_below_threshold(self):
        from utils.nba_usage import is_usage_spike

        # L5 avg = 0.22, L10 avg = 0.20 → delta = 0.02 < 0.15 threshold
        assert not is_usage_spike(fga_share_l5=0.22, fga_share_l10=0.20, threshold=0.15)

    def test_no_spike_when_none_l5(self):
        from utils.nba_usage import is_usage_spike

        assert not is_usage_spike(fga_share_l5=None, fga_share_l10=0.20)

    def test_no_spike_when_none_l10(self):
        from utils.nba_usage import is_usage_spike

        assert not is_usage_spike(fga_share_l5=0.30, fga_share_l10=None)

    def test_default_threshold_used(self):
        """Default threshold is 0.15."""
        from utils.nba_usage import is_usage_spike

        # Delta = 0.20 > 0.15 default threshold
        assert is_usage_spike(fga_share_l5=0.40, fga_share_l10=0.20)

    def test_exact_threshold_not_spike(self):
        """Equality is not a spike (strictly greater than)."""
        from utils.nba_usage import is_usage_spike

        assert not is_usage_spike(fga_share_l5=0.35, fga_share_l10=0.20, threshold=0.15)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestUsageEdgeCases:
    def test_zero_team_fga_no_division_error(self):
        """When team FGA is 0, fga_share should be 0 (not error).

        The implementation clips team_fga to a minimum of 1 to avoid
        division by zero, so a player with 20 FGA in a game where the
        team total is 0 (impossible in practice but possible in test data
        when FGA=0) should be handled gracefully.
        """
        from utils.nba_usage import compute_usage_features

        df = pd.DataFrame(
            {
                "player_id": [1],
                "player_name": ["Test"],
                "team_abbreviation": ["BOS"],
                "season": [2025],
                "game_date": ["2026-01-01"],
                "game_id": ["g1"],
                "fga": [0.0],
                "min": [34.0],
                "pts": [25.0],
            }
        )
        result = compute_usage_features(df)
        # fga=0, team_fga=0; clip(1) → 0/1 = 0.0
        assert result.iloc[0]["fga_share"] == pytest.approx(0.0)

    def test_single_game_has_nan_or_zero_delta(self):
        """With only 1 game, usage_delta should be NaN or 0."""
        from utils.nba_usage import compute_usage_features

        df = pd.DataFrame(
            {
                "player_id": [1],
                "player_name": ["Test"],
                "team_abbreviation": ["BOS"],
                "season": [2025],
                "game_date": ["2026-01-01"],
                "game_id": ["g1"],
                "fga": [20.0],
                "min": [34.0],
                "pts": [25.0],
            }
        )
        result = compute_usage_features(df)
        delta = result.iloc[0]["usage_delta"]
        assert delta == 0.0 or math.isnan(delta)
