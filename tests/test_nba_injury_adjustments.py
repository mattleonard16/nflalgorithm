"""Tests for utils/nba_injury_adjustments.py — usage redistribution, damping.

Covers all 10 required test cases:
1. test_no_injuries_no_boost
2. test_single_injury_boosts_teammate
3. test_boost_multiplier_capped
4. test_sigma_inflated_on_boost
5. test_sigma_not_inflated_without_boost
6. test_damping_factor_applied
7. test_usage_redistribution_proportional
8. test_immutability
9. test_empty_input
10. test_player_not_on_team_no_boost

Plus pre-existing coverage:
- compute_teammate_absence_boost() with known OUT players
- Redistribution proportional to usage share
- apply_injury_adjustments() modifies value rows correctly
- Multiple OUT players
- JSON format for injury_boost_players
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List
from unittest.mock import MagicMock, patch

# Mock config module since config.py is gitignored in this worktree.
# This must happen before any project imports that transitively import config.
if "config" not in sys.modules:
    _mock_config = MagicMock()
    _mock_config.config.database.backend = "sqlite"
    _mock_config.config.database.path = ":memory:"
    sys.modules["config"] = _mock_config

import pytest

# ---------------------------------------------------------------------------
# Constants — imported from the module under test so we stay DRY.
# ---------------------------------------------------------------------------

from utils.nba_injury_adjustments import (
    DAMPING_FACTOR,
    MAX_BOOST_MULTIPLIER,
    SIGMA_INFLATION,
)

# Patch targets live in the module under test.
_OUT_PLAYERS_PATH = "utils.nba_injury_adjustments._get_out_players"
_MARKET_SHARES_PATH = "utils.nba_injury_adjustments._get_team_market_shares"
_BOOST_PATH = "utils.nba_injury_adjustments.compute_teammate_absence_boost"


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
    """Build a mock value row that uses the 'mu' key (not projected_value)."""
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


def _make_projected_row(
    player_id: int = 10,
    team: str = "BOS",
    market: str = "pts",
    projected_value: float = 30.0,
    sigma: float = 5.0,
) -> dict:
    """Build a mock value row that uses the 'projected_value' key."""
    return {
        "player_id": player_id,
        "team": team,
        "market": market,
        "projected_value": projected_value,
        "sigma": sigma,
    }


# ---------------------------------------------------------------------------
# compute_teammate_absence_boost — core algorithm tests
# ---------------------------------------------------------------------------


class TestComputeTeammateAbsenceBoost:
    """Tests for the per-player boost computation function.

    Signature:
        compute_teammate_absence_boost(player_id, team, game_date, market, base_mu)
        -> (adjusted_mu, boost_multiplier, out_player_names)

    DB helpers mocked:
        _get_out_players(team, game_date) -> list[{"player_id", "player_name"}]
        _get_team_market_shares(team, market, season) -> {player_id: fraction}
    """

    # -----------------------------------------------------------------------
    # 1. No injuries → no boost
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_no_injuries_no_boost(self, mock_out, mock_shares):
        """When no teammates are OUT, adjusted_mu == base_mu, multiplier == 1.0,
        and no injury flags are set."""
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        mock_out.return_value = []
        mock_shares.return_value = {1628369: 0.30}

        adj_mu, multiplier, out_names = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )

        assert adj_mu == pytest.approx(28.0), "adjusted_mu must equal base_mu when no injuries"
        assert multiplier == pytest.approx(1.0), "boost_multiplier must be 1.0 when no injuries"
        assert out_names == [], "out_names must be empty when no injuries"

    # -----------------------------------------------------------------------
    # 2. Single injury boosts teammate
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_single_injury_boosts_teammate(self, mock_out, mock_shares):
        """Player A (30% usage) and Player B (25% usage) on same team.
        Player B is OUT. Player A should get a positive usage_bump.

        Verifies:
        - adjusted_mu > base_mu
        - injury_boost_multiplier > 1.0
        - Boost is proportional: freed_share * DAMPING_FACTOR * player_proportion
        """
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        player_a_id = 1628369
        player_b_id = 1628384
        base_mu = 28.0

        mock_out.return_value = [{"player_id": player_b_id, "player_name": "Jaylen Brown"}]
        mock_shares.return_value = {player_a_id: 0.30, player_b_id: 0.25}

        adj_mu, multiplier, out_names = compute_teammate_absence_boost(
            player_id=player_a_id,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=base_mu,
        )

        assert adj_mu > base_mu, "adjusted_mu must be greater than base_mu after injury"
        assert multiplier > 1.0, "boost_multiplier must exceed 1.0 after injury"
        assert "Jaylen Brown" in out_names

        # Verify the exact boost is proportional:
        # freed_share = 0.25, active_total = 1.0 - 0.25 = 0.75
        # player_proportion = 0.30 / 0.75 = 0.40
        # redistribution = 0.25 * DAMPING_FACTOR * 0.40
        freed_share = 0.25
        active_total = 1.0 - freed_share
        player_proportion = 0.30 / active_total
        expected_redistribution = freed_share * DAMPING_FACTOR * player_proportion
        expected_multiplier = 1.0 + expected_redistribution

        assert multiplier == pytest.approx(expected_multiplier, abs=1e-9), (
            f"Expected multiplier {expected_multiplier:.6f}, got {multiplier:.6f}"
        )
        assert adj_mu == pytest.approx(base_mu * expected_multiplier, abs=1e-9)

    # -----------------------------------------------------------------------
    # 3. Boost multiplier capped at MAX_BOOST_MULTIPLIER
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_boost_multiplier_capped(self, mock_out, mock_shares):
        """With many starters OUT (4 players totaling 80% usage), boost_multiplier
        must not exceed MAX_BOOST_MULTIPLIER (1.50)."""
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        active_id = 1
        # 4 starters OUT, each with 20% share → 80% freed
        out_players = [
            {"player_id": i, "player_name": f"Starter {i}"}
            for i in range(2, 6)
        ]
        market_shares: Dict[int, float] = {
            active_id: 0.15,
            2: 0.20,
            3: 0.20,
            4: 0.20,
            5: 0.20,
        }

        mock_out.return_value = out_players
        mock_shares.return_value = market_shares

        _, multiplier, _ = compute_teammate_absence_boost(
            player_id=active_id,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=10.0,
        )

        assert multiplier <= MAX_BOOST_MULTIPLIER, (
            f"boost_multiplier {multiplier:.4f} exceeds MAX_BOOST_MULTIPLIER "
            f"{MAX_BOOST_MULTIPLIER}"
        )

    # -----------------------------------------------------------------------
    # 6. Damping factor applied to freed share
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_damping_factor_applied(self, mock_out, mock_shares):
        """Verify the redistribution uses DAMPING_FACTOR (0.6).

        If 20% usage is freed, only 12% (20% * 0.6) is available for
        redistribution — not the full 20%.
        """
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        active_id = 1
        out_id = 2
        # OUT player holds exactly 20% of team production; active has 50%.
        mock_out.return_value = [{"player_id": out_id, "player_name": "Out Player"}]
        mock_shares.return_value = {active_id: 0.50, out_id: 0.20}

        _, multiplier, _ = compute_teammate_absence_boost(
            player_id=active_id,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=10.0,
        )

        # Damped calculation:
        # freed_share = 0.20
        # active_total = 0.80, player_proportion = 0.50 / 0.80 = 0.625
        # redistribution (damped) = 0.20 * DAMPING_FACTOR * 0.625
        # redistribution (full)   = 0.20 * 1.0           * 0.625
        player_proportion = 0.50 / (1.0 - 0.20)
        expected_damped = 1.0 + 0.20 * DAMPING_FACTOR * player_proportion
        expected_full = 1.0 + 0.20 * 1.0 * player_proportion

        assert multiplier == pytest.approx(expected_damped, abs=1e-9), (
            f"Expected damped multiplier {expected_damped:.6f}, got {multiplier:.6f}"
        )
        assert multiplier < expected_full, (
            "Multiplier with damping must be less than the undamped version"
        )

    # -----------------------------------------------------------------------
    # 7. Usage redistribution is proportional between active players
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_usage_redistribution_proportional(self, mock_out, mock_shares):
        """Two active players: 30% and 20% usage. A 10% usage player is OUT.
        Player with 30% must receive a larger multiplier than player with 20%."""
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        high_share_id = 10
        low_share_id = 20
        out_id = 30

        out_players = [{"player_id": out_id, "player_name": "Out Player"}]
        market_shares = {high_share_id: 0.30, low_share_id: 0.20, out_id: 0.10}

        mock_out.return_value = out_players
        mock_shares.return_value = market_shares

        _, multiplier_high, _ = compute_teammate_absence_boost(
            player_id=high_share_id,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=30.0,
        )

        mock_out.return_value = out_players
        mock_shares.return_value = market_shares

        _, multiplier_low, _ = compute_teammate_absence_boost(
            player_id=low_share_id,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=20.0,
        )

        assert multiplier_high > multiplier_low, (
            "Player with 30% usage must receive a larger boost multiplier than 20% player"
        )

    # -----------------------------------------------------------------------
    # 10. Player on different team → no boost
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_player_not_on_team_no_boost(self, mock_out, mock_shares):
        """OUT player on a different team must not affect the active player's mu.

        _get_out_players is scoped to a team, so a cross-team OUT player
        would never appear in the results for the active player's team.
        Simulated here by returning an empty list.
        """
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        # The OUT player is on LAL; the active player is on BOS.
        # _get_out_players("BOS", ...) returns [] — no BOS players are OUT.
        mock_out.return_value = []
        mock_shares.return_value = {1: 0.30}

        adj_mu, multiplier, out_names = compute_teammate_absence_boost(
            player_id=1,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=20.0,
        )

        assert adj_mu == pytest.approx(20.0)
        assert multiplier == pytest.approx(1.0)
        assert out_names == []

    # -----------------------------------------------------------------------
    # Pre-existing: self excluded from out list
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_self_excluded_from_out_list(self, mock_out, mock_shares):
        """Player should not boost themselves when they appear in out_players."""
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        mock_out.return_value = [
            {"player_id": 1628369, "player_name": "Jayson Tatum"}
        ]
        mock_shares.return_value = {1628369: 0.30}

        adj_mu, multiplier, out_names = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )

        assert adj_mu == pytest.approx(28.0)
        assert multiplier == pytest.approx(1.0)

    # -----------------------------------------------------------------------
    # Pre-existing: boost proportional to star vs bench player's freed share
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_boost_proportional_to_freed_share_magnitude(self, mock_out, mock_shares):
        """Larger freed share from star player OUT → larger boost than bench OUT."""
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        # Bench player out (0.15 share)
        mock_out.return_value = [{"player_id": 100, "player_name": "Bench"}]
        mock_shares.return_value = {1628369: 0.30, 100: 0.15}
        _, boost_small, _ = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )

        # Star player out (0.30 share)
        mock_out.return_value = [{"player_id": 200, "player_name": "Star"}]
        mock_shares.return_value = {1628369: 0.30, 200: 0.30}
        _, boost_large, _ = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )

        assert boost_large > boost_small

    # -----------------------------------------------------------------------
    # Pre-existing: adj_mu == base_mu * boost_multiplier
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_adjusted_mu_equals_base_times_multiplier(self, mock_out, mock_shares):
        """adjusted_mu must always equal base_mu * boost_multiplier."""
        from utils.nba_injury_adjustments import compute_teammate_absence_boost

        mock_out.return_value = [{"player_id": 1628384, "player_name": "Jaylen Brown"}]
        mock_shares.return_value = {1628369: 0.30, 1628384: 0.25}

        adj_mu, multiplier, _ = compute_teammate_absence_boost(
            player_id=1628369,
            team="BOS",
            game_date="2026-02-17",
            market="pts",
            base_mu=28.0,
        )

        assert adj_mu == pytest.approx(28.0 * multiplier)


# ---------------------------------------------------------------------------
# apply_injury_adjustments — batch processing
# ---------------------------------------------------------------------------


class TestApplyInjuryAdjustments:
    """Tests for the batch adjustment function.

    Tests 4, 5, 8, 9 are exercised via apply_injury_adjustments directly,
    using lower-level patches (_get_out_players / _get_team_market_shares)
    rather than mocking compute_teammate_absence_boost, to verify the full
    sigma / immutability / projected_value plumbing.
    """

    # -----------------------------------------------------------------------
    # 4. Sigma inflated on boost (+20%)
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_sigma_inflated_on_boost(self, mock_out, mock_shares):
        """When a boost is applied, sigma must be inflated by SIGMA_INFLATION (20%).
        If base sigma is 5.0, adjusted sigma should be 6.0."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        base_sigma = 5.0
        rows = [_make_projected_row(player_id=10, projected_value=30.0, sigma=base_sigma)]

        mock_out.return_value = [{"player_id": 20, "player_name": "Injured Player"}]
        mock_shares.return_value = {10: 0.30, 20: 0.25}

        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        row = result[0]
        assert row["injury_boost_multiplier"] > 1.0, "Precondition: boost must be applied"
        expected_sigma = base_sigma * (1 + SIGMA_INFLATION)  # 5.0 * 1.20 = 6.0
        assert row["sigma"] == pytest.approx(expected_sigma, abs=1e-9), (
            f"Expected sigma {expected_sigma}, got {row['sigma']}"
        )

    # -----------------------------------------------------------------------
    # 5. Sigma NOT inflated without boost
    # -----------------------------------------------------------------------

    @patch(_OUT_PLAYERS_PATH)
    def test_sigma_not_inflated_without_boost(self, mock_out):
        """When no injuries, sigma must remain exactly unchanged."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        base_sigma = 5.0
        rows = [_make_projected_row(player_id=1, projected_value=25.0, sigma=base_sigma)]
        mock_out.return_value = []

        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["sigma"] == pytest.approx(base_sigma), (
            "sigma must not change when no injury boost is applied"
        )

    # -----------------------------------------------------------------------
    # 8. Immutability — original dicts must not be mutated
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_immutability(self, mock_out, mock_shares):
        """Input list of dicts must NOT be mutated. Original dicts unchanged."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        original_row = _make_projected_row(player_id=10, projected_value=30.0, sigma=5.0)
        original_snapshot = dict(original_row)
        input_rows = [original_row]

        mock_out.return_value = [{"player_id": 20, "player_name": "Injured Player"}]
        mock_shares.return_value = {10: 0.30, 20: 0.25}

        result = apply_injury_adjustments(input_rows, game_date="2026-02-17")

        # Confirm a boost was applied so the mutation check is meaningful.
        assert result[0]["injury_boost_multiplier"] > 1.0, "Precondition: boost applied"

        # The input dict must be exactly as it was before the call.
        assert original_row == original_snapshot, (
            "apply_injury_adjustments must not mutate the original input dicts"
        )

    # -----------------------------------------------------------------------
    # 9. Empty input
    # -----------------------------------------------------------------------

    def test_empty_input(self):
        """apply_injury_adjustments([]) must return []."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        result = apply_injury_adjustments([], game_date="2026-02-17")
        assert result == []

    # -----------------------------------------------------------------------
    # Pre-existing: modifies mu when boost applied
    # -----------------------------------------------------------------------

    @patch(_BOOST_PATH)
    def test_modifies_mu_when_boost(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (30.5, 1.089, ["Jaylen Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["injury_adjusted_mu"] == pytest.approx(30.5)

    @patch(_BOOST_PATH)
    def test_sets_boost_multiplier(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (30.5, 1.089, ["Jaylen Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["injury_boost_multiplier"] == pytest.approx(1.089)

    @patch(_BOOST_PATH)
    def test_no_injuries_preserves_original(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (28.0, 1.0, [])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["injury_adjusted_mu"] == pytest.approx(28.0)
        assert result[0]["injury_boost_multiplier"] == pytest.approx(1.0)

    @patch(_BOOST_PATH)
    def test_multiple_rows_processed(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

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

    @patch(_BOOST_PATH)
    def test_empty_rows_returns_empty(self, mock_boost):
        from utils.nba_injury_adjustments import apply_injury_adjustments

        result = apply_injury_adjustments([], game_date="2026-02-17")
        assert result == []
        mock_boost.assert_not_called()

    @patch(_BOOST_PATH)
    def test_original_mu_preserved(self, mock_boost):
        """The original 'mu' field should remain unchanged in the result row."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (31.0, 1.107, ["Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["mu"] == pytest.approx(28.0)
        assert result[0]["injury_adjusted_mu"] == pytest.approx(31.0)

    @patch(_BOOST_PATH)
    def test_injury_boost_players_stored_as_json(self, mock_boost):
        """Out player names should be stored as a JSON list string when present."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (30.5, 1.089, ["Jaylen Brown"])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        players_field = result[0].get("injury_boost_players")
        assert players_field is not None
        parsed = json.loads(players_field)
        assert "Jaylen Brown" in parsed

    @patch(_BOOST_PATH)
    def test_no_injuries_boost_players_is_none(self, mock_boost):
        """When no injuries, injury_boost_players should be None."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        mock_boost.return_value = (28.0, 1.0, [])

        rows = [_make_value_row(mu=28.0)]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert result[0]["injury_boost_players"] is None

    # -----------------------------------------------------------------------
    # Additional: rows missing required fields get placeholder injury fields
    # -----------------------------------------------------------------------

    def test_row_missing_player_id_gets_placeholder_fields(self):
        """Rows without player_id receive neutral placeholder injury fields."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        rows = [{"team": "BOS", "market": "pts", "projected_value": 20.0}]
        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        assert len(result) == 1
        row = result[0]
        assert row["injury_boost_multiplier"] == pytest.approx(1.0)
        assert row["injury_adjusted_mu"] == row["base_mu"]
        assert row["injury_boost_players"] is None

    # -----------------------------------------------------------------------
    # Additional: projected_value updated to injury_adjusted_mu after boost
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_projected_value_updated_to_adjusted_mu(self, mock_out, mock_shares):
        """When boost is applied, projected_value in the result row should equal
        injury_adjusted_mu."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        rows = [_make_projected_row(player_id=10, projected_value=30.0, sigma=5.0)]
        mock_out.return_value = [{"player_id": 20, "player_name": "Injured Player"}]
        mock_shares.return_value = {10: 0.30, 20: 0.25}

        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        row = result[0]
        assert row["projected_value"] == pytest.approx(row["injury_adjusted_mu"]), (
            "projected_value must be updated to injury_adjusted_mu after boost"
        )
        assert row["projected_value"] > 30.0, "projected_value must increase after boost"

    # -----------------------------------------------------------------------
    # Additional: multi-team batch — only same-team players boosted
    # -----------------------------------------------------------------------

    @patch(_MARKET_SHARES_PATH)
    @patch(_OUT_PLAYERS_PATH)
    def test_batch_only_same_team_gets_boost(self, mock_out, mock_shares):
        """In a mixed batch, only rows on the injured team receive a boost."""
        from utils.nba_injury_adjustments import apply_injury_adjustments

        rows = [
            _make_projected_row(player_id=10, team="BOS", projected_value=30.0),
            _make_projected_row(player_id=99, team="LAL", projected_value=20.0),
        ]

        out_players_bos = [{"player_id": 20, "player_name": "Injured BOS Player"}]
        market_shares_bos = {10: 0.35, 20: 0.30}

        def fake_out_players(team: str, game_date: str) -> List[Dict]:
            return out_players_bos if team == "BOS" else []

        def fake_market_shares(team: str, market: str, season: int) -> Dict[int, float]:
            return market_shares_bos if team == "BOS" else {}

        mock_out.side_effect = fake_out_players
        mock_shares.side_effect = fake_market_shares

        result = apply_injury_adjustments(rows, game_date="2026-02-17")

        bos_row = next(r for r in result if r["player_id"] == 10)
        lal_row = next(r for r in result if r["player_id"] == 99)

        assert bos_row["injury_boost_multiplier"] > 1.0, "BOS player must be boosted"
        assert lal_row["injury_boost_multiplier"] == pytest.approx(1.0), (
            "LAL player must NOT be boosted"
        )
