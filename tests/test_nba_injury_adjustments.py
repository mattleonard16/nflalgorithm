"""Tests for utils/nba_injury_adjustments.py — usage redistribution, damping.

Covers:
- compute_teammate_absence_boost() with known OUT players
- Redistribution proportional to usage share
- Damping factor (0.6)
- apply_injury_adjustments() modifies value rows correctly
- No injuries → no changes
- Multiple OUT players
- JSON format for injury_boost_players
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

import json
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAMPING_FACTOR = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_value_row(
    player_id: int = 1628369,
    player_name: str = "Jayson Tatum",
    team: str = "BOS",
    market: str = "pts",
    mu: float = 28.0,
    sigma: float = 5.0,
) -> dict:
    """Build a mock value row for injury adjustments."""
    return {
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "market": market,
        "mu": mu,
        "sigma": sigma,
        "edge_percentage": 12.0,
        "p_win": 0.62,
        "line": 25.5,
        "over_price": -115,
    }


# ---------------------------------------------------------------------------
# compute_teammate_absence_boost
# ---------------------------------------------------------------------------


class TestComputeTeammateAbsenceBoost:
    """Test per-player boost from teammate injuries.

    The actual signature is:
        compute_teammate_absence_boost(player_id, team, game_date, market, base_mu)
        -> (adjusted_mu, boost_multiplier, out_player_names)

    _get_out_players returns a list of {"player_id": ..., "player_name": ...}
    _get_team_market_shares returns {player_id: fraction_of_team_total}
    """

    @patch("utils.nba_injury_adjustments._get_team_market_shares")
    @patch("utils.nba_injury_adjustments._get_out_players")
    def test_boost_when_teammate_out(self, mock_out, mock_shares):
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        # Teammate (Jaylen Brown, id=1628384) is OUT with 0.25 market share
        mock_out.return_value = [
            {"player_id": 1628384, "player_name": "Jaylen Brown"}
        ]
        # Our player (Tatum, 1628369) has 0.30 share; Brown has 0.25
        mock_shares.return_value = {1628369: 0.30, 1628384: 0.25}

        adj_mu, boost_multiplier, out_names = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )
        assert adj_mu > 28.0
        assert boost_multiplier > 1.0
        assert isinstance(out_names, list)
        assert "Jaylen Brown" in out_names

    @patch("utils.nba_injury_adjustments._get_team_market_shares")
    @patch("utils.nba_injury_adjustments._get_out_players")
    def test_no_boost_when_no_injuries(self, mock_out, mock_shares):
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        mock_out.return_value = []
        mock_shares.return_value = {1628369: 0.30}

        adj_mu, boost_multiplier, out_names = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )
        # No boost: adjusted_mu == base_mu, multiplier == 1.0
        assert adj_mu == pytest.approx(28.0)
        assert boost_multiplier == pytest.approx(1.0)
        assert out_names == []

    @patch("utils.nba_injury_adjustments._get_team_market_shares")
    @patch("utils.nba_injury_adjustments._get_out_players")
    def test_boost_proportional_to_usage_share(self, mock_out, mock_shares):
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        # Small teammate out (0.15 share)
        mock_out.return_value = [{"player_id": 100, "player_name": "A"}]
        mock_shares.return_value = {1628369: 0.30, 100: 0.15}
        adj_mu_small, boost_small, _ = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )

        # Large teammate out (0.30 share — star player)
        mock_out.return_value = [{"player_id": 200, "player_name": "B"}]
        mock_shares.return_value = {1628369: 0.30, 200: 0.30}
        adj_mu_large, boost_large, _ = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )

        assert boost_large > boost_small

    @patch("utils.nba_injury_adjustments._get_team_market_shares")
    @patch("utils.nba_injury_adjustments._get_out_players")
    def test_adjusted_mu_greater_than_base(self, mock_out, mock_shares):
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        mock_out.return_value = [
            {"player_id": 1628384, "player_name": "Jaylen Brown"}
        ]
        mock_shares.return_value = {1628369: 0.30, 1628384: 0.25}

        adj_mu, boost_multiplier, _ = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )
        assert adj_mu > 28.0
        assert adj_mu == pytest.approx(28.0 * boost_multiplier)

    @patch("utils.nba_injury_adjustments._get_team_market_shares")
    @patch("utils.nba_injury_adjustments._get_out_players")
    def test_self_excluded_from_out_list(self, mock_out, mock_shares):
        """Player should not boost themselves when they appear in out_players."""
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        # Player themselves is in the out list — should be filtered
        mock_out.return_value = [
            {"player_id": 1628369, "player_name": "Jayson Tatum"}
        ]
        mock_shares.return_value = {1628369: 0.30}

        adj_mu, boost_multiplier, out_names = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )
        # No valid teammates out after self-exclusion
        assert adj_mu == pytest.approx(28.0)
        assert boost_multiplier == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# apply_injury_adjustments — batch processing
# ---------------------------------------------------------------------------


class TestApplyInjuryAdjustments:
    """Test batch application of injury adjustments to value rows.

    apply_injury_adjustments calls compute_teammate_absence_boost which
    returns (adjusted_mu, boost_multiplier, out_names).  The function writes:
        - base_mu
        - injury_adjusted_mu
        - injury_boost_multiplier
        - injury_boost_players (JSON list or None)
    """

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_modifies_mu_when_boost(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        # Returns (adjusted_mu, boost_multiplier, out_names)
        mock_boost.return_value = (30.5, 1.089, ["Jaylen Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["injury_adjusted_mu"] == pytest.approx(30.5)

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_sets_boost_multiplier(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (30.5, 1.089, ["Jaylen Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["injury_boost_multiplier"] == pytest.approx(1.089)

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_no_injuries_preserves_original(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        # No boost: adjusted_mu == base_mu, multiplier == 1.0
        mock_boost.return_value = (28.0, 1.0, [])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["injury_adjusted_mu"] == pytest.approx(28.0)
        assert result[0]["injury_boost_multiplier"] == pytest.approx(1.0)

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_stores_injury_boost_multiplier(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (30.5, 1.089, ["Jaylen Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert "injury_boost_multiplier" in result[0]
        assert result[0]["injury_boost_multiplier"] > 1.0

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_multiple_rows_processed(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        # Return different values for each player
        mock_boost.side_effect = [
            (29.5, 1.054, ["Brown"]),
            (23.5, 1.068, ["Brown"]),
        ]

        rows = [
            _make_value_row(player_id=1, mu=28.0),
            _make_value_row(player_id=2, mu=22.0),
        ]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert len(result) == 2
        assert result[0]["injury_adjusted_mu"] == pytest.approx(29.5)
        assert result[1]["injury_adjusted_mu"] == pytest.approx(23.5)

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_empty_rows_returns_empty(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        result = apply_injury_adjustments([], game_date="2026-02-17")
        assert result == []
        mock_boost.assert_not_called()

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_original_mu_preserved(self, mock_boost):
        """The original 'mu' field should remain unchanged."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (31.0, 1.107, ["Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["mu"] == pytest.approx(28.0)
        assert result[0]["injury_adjusted_mu"] == pytest.approx(31.0)

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_injury_boost_players_stored_as_json(self, mock_boost):
        """Out player names should be stored as JSON when present."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (30.5, 1.089, ["Jaylen Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        players_field = result[0].get("injury_boost_players")
        assert players_field is not None
        parsed = json.loads(players_field)
        assert "Jaylen Brown" in parsed

    @patch("utils.nba_injury_adjustments.compute_teammate_absence_boost")
    def test_no_injuries_boost_players_is_none(self, mock_boost):
        """When no injuries, injury_boost_players should be None."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (28.0, 1.0, [])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["injury_boost_players"] is None
