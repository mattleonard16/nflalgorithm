"""Tests for the confidence scoring and tiered play selection engine.

Covers:
- Score computation with known inputs
- Tier assignment at boundaries
- Filtering by minimum tier
- Edge cases (zero sigma, missing data, NaN values)
- Bulk scoring of DataFrames
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from confidence_engine import (
    compute_confidence_score,
    assign_tier,
    filter_by_confidence,
    score_plays,
    _score_edge,
    _score_stability,
    _score_volume,
    _score_volatility,
)


# ======================================================================
# Component scoring
# ======================================================================


class TestScoreEdge:
    def test_zero_edge(self):
        assert _score_edge(0.0) == 0.0

    def test_negative_edge(self):
        assert _score_edge(-0.05) == 0.0

    def test_max_edge(self):
        assert _score_edge(0.25) == 100.0

    def test_above_max_capped(self):
        assert _score_edge(0.50) == 100.0

    def test_midpoint(self):
        result = _score_edge(0.125)
        assert abs(result - 50.0) < 0.01

    def test_small_edge(self):
        result = _score_edge(0.05)
        assert abs(result - 20.0) < 0.01


class TestScoreStability:
    def test_zero_sigma_perfect_stability(self):
        result = _score_stability(100.0, 0.0)
        assert result == 100.0

    def test_sigma_equals_mu(self):
        """CV of 1.0 should give 0."""
        result = _score_stability(50.0, 50.0)
        assert result == 0.0

    def test_sigma_half_mu(self):
        """CV of 0.5 should give 50."""
        result = _score_stability(100.0, 50.0)
        assert abs(result - 50.0) < 0.01

    def test_zero_mu_returns_neutral(self):
        result = _score_stability(0.0, 10.0)
        assert result == 50.0

    def test_negative_mu_returns_neutral(self):
        result = _score_stability(-5.0, 10.0)
        assert result == 50.0

    def test_negative_sigma_returns_neutral(self):
        result = _score_stability(100.0, -5.0)
        assert result == 50.0

    def test_very_high_sigma_clamped(self):
        result = _score_stability(50.0, 100.0)
        assert result == 0.0


class TestScoreVolume:
    def test_zero_share(self):
        assert _score_volume(0.0) == 0.0

    def test_negative_share(self):
        assert _score_volume(-0.1) == 0.0

    def test_max_share(self):
        assert _score_volume(0.30) == 100.0

    def test_above_max_capped(self):
        assert _score_volume(0.50) == 100.0

    def test_midpoint(self):
        result = _score_volume(0.15)
        assert abs(result - 50.0) < 0.01


class TestScoreVolatility:
    def test_zero_volatility_full_confidence(self):
        assert _score_volatility(0.0) == 100.0

    def test_max_volatility_zero_confidence(self):
        assert _score_volatility(100.0) == 0.0

    def test_mid_volatility(self):
        assert _score_volatility(50.0) == 50.0

    def test_above_100_clamped(self):
        assert _score_volatility(150.0) == 0.0

    def test_below_0_clamped(self):
        assert _score_volatility(-20.0) == 100.0


# ======================================================================
# Composite confidence score
# ======================================================================


class TestComputeConfidenceScore:
    def test_premium_play(self):
        """High edge, stable, good volume, low volatility -> high score."""
        row = {
            "edge_percentage": 0.20,
            "mu": 80.0,
            "sigma": 12.0,
            "target_share": 0.28,
            "volatility_score": 15.0,
        }
        score = compute_confidence_score(row)
        assert score >= 80, f"Premium play should score >= 80, got {score}"

    def test_pass_play(self):
        """Low edge, unstable, low volume, high volatility -> low score."""
        row = {
            "edge_percentage": 0.02,
            "mu": 30.0,
            "sigma": 30.0,
            "target_share": 0.03,
            "volatility_score": 85.0,
        }
        score = compute_confidence_score(row)
        assert score < 60, f"Bad play should score < 60, got {score}"

    def test_dict_input(self):
        row = {"edge_percentage": 0.10, "mu": 60.0, "sigma": 15.0}
        score = compute_confidence_score(row)
        assert 0 <= score <= 100

    def test_series_input(self):
        row = pd.Series({
            "edge_percentage": 0.10,
            "mu": 60.0,
            "sigma": 15.0,
            "target_share": 0.20,
            "volatility_score": 30.0,
        })
        score = compute_confidence_score(row)
        assert 0 <= score <= 100

    def test_missing_optional_fields(self):
        """Missing target_share and volatility_score should use defaults."""
        row = {"edge_percentage": 0.12, "mu": 70.0, "sigma": 15.0}
        score = compute_confidence_score(row)
        assert 0 <= score <= 100

    def test_all_zeros(self):
        row = {
            "edge_percentage": 0.0,
            "mu": 0.0,
            "sigma": 0.0,
            "target_share": 0.0,
            "volatility_score": 0.0,
        }
        score = compute_confidence_score(row)
        # edge=0, stability=50 (mu=0), volume=0, volatility inverted=100
        # 0.35*0 + 0.25*50 + 0.20*0 + 0.20*100 = 12.5 + 20 = 32.5
        assert abs(score - 32.5) < 0.01

    def test_zero_sigma(self):
        """Zero sigma should not crash and should indicate high stability."""
        row = {
            "edge_percentage": 0.15,
            "mu": 80.0,
            "sigma": 0.0,
            "target_share": 0.25,
            "volatility_score": 20.0,
        }
        score = compute_confidence_score(row)
        assert score > 70, f"Zero sigma should boost score, got {score}"

    def test_nan_values_use_defaults(self):
        row = {
            "edge_percentage": float("nan"),
            "mu": float("nan"),
            "sigma": float("nan"),
            "target_share": float("nan"),
            "volatility_score": float("nan"),
        }
        score = compute_confidence_score(row)
        assert 0 <= score <= 100
        assert not math.isnan(score)

    def test_score_range(self):
        """Score should always be between 0 and 100."""
        test_cases = [
            {"edge_percentage": 1.0, "mu": 200.0, "sigma": 1.0,
             "target_share": 1.0, "volatility_score": 0.0},
            {"edge_percentage": -1.0, "mu": -100.0, "sigma": 500.0,
             "target_share": -1.0, "volatility_score": 200.0},
        ]
        for row in test_cases:
            score = compute_confidence_score(row)
            assert 0 <= score <= 100, f"Score {score} out of range for {row}"

    def test_weights_sum_to_one(self):
        """The four weights should sum to 1.0."""
        from confidence_engine import _get_weights
        weights = _get_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001


# ======================================================================
# Tier assignment
# ======================================================================


class TestAssignTier:
    def test_premium_boundary(self):
        assert assign_tier(90.0) == "Premium"

    def test_premium_max(self):
        assert assign_tier(100.0) == "Premium"

    def test_strong_boundary(self):
        assert assign_tier(75.0) == "Strong"

    def test_strong_upper(self):
        assert assign_tier(89.99) == "Strong"

    def test_marginal_boundary(self):
        assert assign_tier(60.0) == "Marginal"

    def test_marginal_upper(self):
        assert assign_tier(74.99) == "Marginal"

    def test_pass_boundary(self):
        assert assign_tier(59.99) == "Pass"

    def test_pass_zero(self):
        assert assign_tier(0.0) == "Pass"

    def test_above_100(self):
        assert assign_tier(105.0) == "Premium"


# ======================================================================
# Filtering by tier
# ======================================================================


class TestFilterByConfidence:
    def _make_plays(self):
        return pd.DataFrame([
            {"player_id": "P1", "confidence_score": 95.0, "confidence_tier": "Premium"},
            {"player_id": "P2", "confidence_score": 82.0, "confidence_tier": "Strong"},
            {"player_id": "P3", "confidence_score": 68.0, "confidence_tier": "Marginal"},
            {"player_id": "P4", "confidence_score": 45.0, "confidence_tier": "Pass"},
        ])

    def test_default_min_tier_2(self):
        """Default min_tier=2 keeps Premium and Strong."""
        plays = self._make_plays()
        result = filter_by_confidence(plays)
        assert len(result) == 2
        assert set(result["confidence_tier"]) == {"Premium", "Strong"}

    def test_min_tier_1_only_premium(self):
        plays = self._make_plays()
        result = filter_by_confidence(plays, min_tier=1)
        assert len(result) == 1
        assert result.iloc[0]["confidence_tier"] == "Premium"

    def test_min_tier_3_includes_marginal(self):
        plays = self._make_plays()
        result = filter_by_confidence(plays, min_tier=3)
        assert len(result) == 3
        assert "Pass" not in set(result["confidence_tier"])

    def test_min_tier_4_includes_all(self):
        plays = self._make_plays()
        result = filter_by_confidence(plays, min_tier=4)
        assert len(result) == 4

    def test_empty_dataframe(self):
        result = filter_by_confidence(pd.DataFrame())
        assert result.empty

    def test_missing_tier_column_returns_copy(self):
        plays = pd.DataFrame([{"player_id": "P1", "value": 10}])
        result = filter_by_confidence(plays)
        assert len(result) == 1

    def test_original_not_mutated(self):
        plays = self._make_plays()
        original_len = len(plays)
        filter_by_confidence(plays, min_tier=1)
        assert len(plays) == original_len


# ======================================================================
# Bulk scoring
# ======================================================================


class TestScorePlays:
    def test_adds_columns(self):
        df = pd.DataFrame([
            {"edge_percentage": 0.20, "mu": 80.0, "sigma": 12.0,
             "target_share": 0.25, "volatility_score": 10.0},
            {"edge_percentage": 0.05, "mu": 40.0, "sigma": 35.0,
             "target_share": 0.04, "volatility_score": 80.0},
        ])
        result = score_plays(df)
        assert "confidence_score" in result.columns
        assert "confidence_tier" in result.columns
        assert len(result) == 2

    def test_empty_df(self):
        result = score_plays(pd.DataFrame())
        assert "confidence_score" in result.columns
        assert "confidence_tier" in result.columns
        assert result.empty

    def test_original_not_mutated(self):
        df = pd.DataFrame([
            {"edge_percentage": 0.15, "mu": 70.0, "sigma": 15.0},
        ])
        cols_before = set(df.columns)
        score_plays(df)
        assert set(df.columns) == cols_before

    def test_scores_are_valid(self):
        df = pd.DataFrame([
            {"edge_percentage": 0.10, "mu": 60.0, "sigma": 20.0,
             "target_share": 0.15, "volatility_score": 40.0},
        ])
        result = score_plays(df)
        score = result.iloc[0]["confidence_score"]
        tier = result.iloc[0]["confidence_tier"]
        assert 0 <= score <= 100
        assert tier in {"Premium", "Strong", "Marginal", "Pass"}

    def test_tier_matches_score(self):
        df = pd.DataFrame([
            {"edge_percentage": 0.22, "mu": 85.0, "sigma": 10.0,
             "target_share": 0.28, "volatility_score": 10.0},
        ])
        result = score_plays(df)
        score = result.iloc[0]["confidence_score"]
        tier = result.iloc[0]["confidence_tier"]
        assert tier == assign_tier(score)
