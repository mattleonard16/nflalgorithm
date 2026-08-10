"""Tests for target share gating and volatility scoring.

Covers:
- Target share calculation (team-level and individual)
- Low-volume gating and confidence capping
- Volatility scoring (consistent vs boom-bust players)
- Sigma widening for high-volatility players
"""

from __future__ import annotations

import pandas as pd

from utils.target_share import (
    calculate_target_share,
    compute_team_target_shares,
    is_low_volume,
    cap_confidence_for_low_volume,
)
from utils.volatility_scoring import (
    apply_volatility_widening,
    coefficient_of_variation,
    max_week_contribution,
    range_ratio,
    compute_volatility_score,
    volatility_score_or_none,
    widen_sigma_for_volatility,
)


# ======================================================================
# Writer-side scoring: unmeasurable must stay unmeasured
# ======================================================================


def test_short_history_scores_none_rather_than_a_neutral_50():
    """Storing 50.0 for an unmeasurable player is the bug this prevents.

    Persisted, it is indistinguishable from a real 50 and earns an
    unearned 7.5% sigma penalty downstream.
    """
    assert volatility_score_or_none([]) is None
    assert volatility_score_or_none([80.0]) is None
    assert volatility_score_or_none([80.0, 20.0, 60.0]) is None

    # ...whereas compute_volatility_score answers a number for the same input.
    assert compute_volatility_score([80.0]) == 50.0


def test_sufficient_history_scores_and_separates_steady_from_boom_bust():
    steady = volatility_score_or_none([50.0, 52.0, 48.0, 51.0, 49.0])
    boom_bust = volatility_score_or_none([5.0, 0.0, 140.0, 3.0, 10.0])

    assert steady is not None and boom_bust is not None
    assert boom_bust > steady


def test_nans_do_not_count_toward_the_minimum():
    """A missed week is absent data, not a measurement."""
    import numpy as np

    assert volatility_score_or_none([50.0, np.nan, 48.0, np.nan]) is None
    assert volatility_score_or_none([50.0, 48.0, 52.0, 47.0]) is not None


def test_scored_players_receive_distinguishable_widening():
    """End-to-end: writer and reader together must vary sigma by player.

    Guards the regression this pair was built to fix. The model that writes
    volatility_score is gitignored, so CI cannot see the write itself — but
    it can pin the contract: scoring a mix of steady and boom-bust players
    and widening them must NOT collapse onto one multiplier the way an
    all-NULL column plus fillna(50.0) did.
    """
    steady = [50.0, 52.0, 48.0, 51.0, 49.0]
    boom_bust = [5.0, 0.0, 140.0, 3.0, 10.0]
    unmeasurable = [40.0]

    scores = pd.Series(
        [volatility_score_or_none(h) for h in (steady, boom_bust, unmeasurable)]
    )
    sigma = pd.Series([10.0, 10.0, 10.0])

    widened, missing = apply_volatility_widening(sigma, scores, penalty_weight=0.15)

    assert missing == 1
    assert widened.iloc[2] == 10.0                    # unmeasurable: untouched
    assert widened.iloc[1] > widened.iloc[0] > 10.0   # boom-bust widened most
    assert widened.nunique() == 3                     # not one flat multiplier


# ======================================================================
# Column-wise widening: a missing score is not a middling score
# ======================================================================


def test_widening_leaves_unscored_rows_untouched_and_counts_them():
    """An absent score must not be priced as risk.

    The NFL model has never written volatility_score, so treating a missing
    value as 50 silently multiplied every sigma by 1.075.
    """
    sigma = pd.Series([10.0, 20.0, 30.0])
    scores = pd.Series([None, 0.0, 100.0])

    widened, missing = apply_volatility_widening(
        sigma, scores, penalty_weight=0.15
    )

    assert missing == 1
    assert widened.iloc[0] == 10.0   # no score -> no widening
    assert widened.iloc[1] == 20.0   # measured zero volatility -> no widening
    assert widened.iloc[2] == 30.0 * 1.15


def test_widening_reports_every_row_when_the_column_is_absent():
    sigma = pd.Series([10.0, 12.0])

    widened, missing = apply_volatility_widening(sigma, None)

    assert missing == 2
    assert list(widened) == [10.0, 12.0]


def test_measured_midpoint_still_earns_its_penalty():
    """A real score of 50 is information and must still widen sigma.

    This is what separates the fix from simply disabling the feature.
    """
    widened, missing = apply_volatility_widening(
        pd.Series([10.0]), pd.Series([50.0]), penalty_weight=0.15
    )

    assert missing == 0
    assert widened.iloc[0] == 10.0 * 1.075


# ======================================================================
# Target share calculation
# ======================================================================


class TestCalculateTargetShare:
    def test_normal_calculation(self):
        assert calculate_target_share(8.0, 40.0) == 0.2

    def test_zero_team_targets_returns_zero(self):
        assert calculate_target_share(5.0, 0.0) == 0.0

    def test_negative_team_targets_returns_zero(self):
        assert calculate_target_share(5.0, -10.0) == 0.0

    def test_negative_player_targets_returns_zero(self):
        assert calculate_target_share(-1.0, 40.0) == 0.0

    def test_capped_at_one(self):
        result = calculate_target_share(50.0, 40.0)
        assert result == 1.0

    def test_all_targets_to_one_player(self):
        assert calculate_target_share(40.0, 40.0) == 1.0


class TestComputeTeamTargetShares:
    def _make_stats(self):
        return pd.DataFrame([
            {"player_id": "WR1", "team": "KC", "position": "WR",
             "targets": 10.0, "season": 2025, "week": 5, "target_share": 0.0},
            {"player_id": "WR2", "team": "KC", "position": "WR",
             "targets": 6.0, "season": 2025, "week": 5, "target_share": 0.0},
            {"player_id": "TE1", "team": "KC", "position": "TE",
             "targets": 4.0, "season": 2025, "week": 5, "target_share": 0.0},
            {"player_id": "QB1", "team": "KC", "position": "QB",
             "targets": 0.0, "season": 2025, "week": 5, "target_share": 0.0},
        ])

    def test_shares_sum_for_receivers(self):
        stats = self._make_stats()
        result = compute_team_target_shares(stats, 2025, 5)
        receivers = result[result["position"].isin({"WR", "TE"})]
        total_share = receivers["target_share"].sum()
        assert abs(total_share - 1.0) < 0.01

    def test_wr1_gets_half(self):
        stats = self._make_stats()
        result = compute_team_target_shares(stats, 2025, 5)
        wr1_share = result.loc[result["player_id"] == "WR1", "target_share"].iloc[0]
        assert abs(wr1_share - 0.5) < 0.01

    def test_empty_df_returns_empty(self):
        result = compute_team_target_shares(pd.DataFrame(), 2025, 5)
        assert result.empty

    def test_original_not_mutated(self):
        stats = self._make_stats()
        original_shares = stats["target_share"].copy()
        compute_team_target_shares(stats, 2025, 5)
        pd.testing.assert_series_equal(stats["target_share"], original_shares)


# ======================================================================
# Low-volume gating
# ======================================================================


class TestIsLowVolume:
    def test_below_threshold(self):
        assert is_low_volume(0.05, threshold=0.08) is True

    def test_at_threshold(self):
        assert is_low_volume(0.08, threshold=0.08) is False

    def test_above_threshold(self):
        assert is_low_volume(0.15, threshold=0.08) is False

    def test_zero_share(self):
        assert is_low_volume(0.0, threshold=0.08) is True


class TestCapConfidence:
    def test_above_threshold_no_cap(self):
        result = cap_confidence_for_low_volume(0.9, 0.15, threshold=0.08)
        assert result == 0.9

    def test_at_threshold_no_cap(self):
        result = cap_confidence_for_low_volume(0.9, 0.08, threshold=0.08)
        assert result == 0.9

    def test_below_threshold_scales_down(self):
        result = cap_confidence_for_low_volume(0.9, 0.04, threshold=0.08)
        assert abs(result - 0.45) < 0.01

    def test_zero_share_gives_zero_confidence(self):
        result = cap_confidence_for_low_volume(0.9, 0.0, threshold=0.08)
        assert result == 0.0

    def test_zero_threshold_no_crash(self):
        result = cap_confidence_for_low_volume(0.9, 0.05, threshold=0.0)
        assert result == 0.9


# ======================================================================
# Volatility scoring components
# ======================================================================


class TestCoefficientOfVariation:
    def test_identical_values(self):
        assert coefficient_of_variation([50, 50, 50, 50]) == 0.0

    def test_varied_values(self):
        cv = coefficient_of_variation([20, 80, 40, 60])
        assert cv > 0.3

    def test_single_value(self):
        assert coefficient_of_variation([50]) == 0.0

    def test_empty(self):
        assert coefficient_of_variation([]) == 0.0


class TestMaxWeekContribution:
    def test_even_distribution(self):
        result = max_week_contribution([25, 25, 25, 25])
        assert abs(result - 0.25) < 0.01

    def test_one_dominant(self):
        result = max_week_contribution([100, 10, 10, 10])
        assert result > 0.7

    def test_all_zeros(self):
        assert max_week_contribution([0, 0, 0]) == 0.0


class TestRangeRatio:
    def test_identical_values(self):
        assert range_ratio([50, 50, 50]) == 0.0

    def test_wide_range(self):
        result = range_ratio([10, 100, 50])
        assert result > 1.0

    def test_single_value(self):
        assert range_ratio([42]) == 0.0


# ======================================================================
# Volatility score (composite)
# ======================================================================


class TestComputeVolatilityScore:
    def test_consistent_player_low_score(self):
        """Player averaging ~60 yards with little variance."""
        yards = [58, 62, 60, 57, 63, 59, 61, 60]
        score = compute_volatility_score(yards)
        assert score < 30, f"Consistent player should score < 30, got {score}"

    def test_boom_bust_player_high_score(self):
        """Player with extreme variance (0 yards or 150 yards)."""
        yards = [5, 150, 10, 140, 8, 130, 15, 145]
        score = compute_volatility_score(yards)
        assert score > 60, f"Boom-bust player should score > 60, got {score}"

    def test_single_data_point_neutral(self):
        """Insufficient data returns neutral 50."""
        assert compute_volatility_score([80]) == 50.0

    def test_score_in_range(self):
        yards = [40, 60, 80, 20, 100]
        score = compute_volatility_score(yards)
        assert 0 <= score <= 100

    def test_empty_returns_neutral(self):
        assert compute_volatility_score([]) == 50.0

    def test_all_zeros_returns_low(self):
        score = compute_volatility_score([0, 0, 0, 0])
        assert score < 10


# ======================================================================
# Sigma widening
# ======================================================================


class TestWidenSigma:
    def test_zero_volatility_no_change(self):
        result = widen_sigma_for_volatility(20.0, 0.0, penalty_weight=0.15)
        assert result == 20.0

    def test_max_volatility_full_penalty(self):
        result = widen_sigma_for_volatility(20.0, 100.0, penalty_weight=0.15)
        assert abs(result - 23.0) < 0.01

    def test_mid_volatility_partial_penalty(self):
        result = widen_sigma_for_volatility(20.0, 50.0, penalty_weight=0.15)
        expected = 20.0 * (1.0 + 0.15 * 0.5)
        assert abs(result - expected) < 0.01

    def test_sigma_never_shrinks(self):
        result = widen_sigma_for_volatility(20.0, 25.0, penalty_weight=0.15)
        assert result >= 20.0

    def test_uses_config_default(self):
        """When no penalty_weight is passed, falls back to config."""
        result = widen_sigma_for_volatility(20.0, 100.0)
        # config.betting.volatility_penalty_weight = 0.15
        assert abs(result - 23.0) < 0.01
