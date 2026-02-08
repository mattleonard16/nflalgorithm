"""Tests for TE market bias analysis module.

Covers:
- Bias calculation with synthetic data
- Playoff vs regular season classification
- Statistical significance tests
- Adjustment computation logic
- Report generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.te_market_bias import (
    PLAYOFF_WEEKS,
    REGULAR_SEASON_WEEKS,
    classify_game_type,
    compute_bias_metrics,
    compute_suggested_adjustment,
    build_report,
    merge_projections_with_actuals,
    run_significance_tests,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_projections(
    n: int,
    weeks: list[int],
    mu_base: float = 45.0,
    line_base: float = 48.0,
    season: int = 2024,
) -> pd.DataFrame:
    """Build synthetic projection/odds rows for TEs."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        week = weeks[i % len(weeks)]
        rows.append({
            "season": season,
            "week": week,
            "player_id": f"TE_{i}",
            "team": "KC",
            "mu": mu_base + rng.normal(0, 5),
            "sigma": 12.0,
            "line": line_base + rng.normal(0, 3),
            "sportsbook": "SimBook",
        })
    return pd.DataFrame(rows)


def _make_actuals(
    n: int,
    weeks: list[int],
    actual_base: float = 40.0,
    season: int = 2024,
) -> pd.DataFrame:
    """Build synthetic actual yards for TEs."""
    rng = np.random.default_rng(99)
    rows = []
    for i in range(n):
        week = weeks[i % len(weeks)]
        rows.append({
            "player_id": f"TE_{i}",
            "season": season,
            "week": week,
            "actual_yards": actual_base + rng.normal(0, 15),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# classify_game_type
# ---------------------------------------------------------------------------


class TestClassifyGameType:

    def test_regular_season_weeks(self):
        for week in range(1, 19):
            assert classify_game_type(week) == "regular"

    def test_playoff_weeks(self):
        for week in range(19, 23):
            assert classify_game_type(week) == "playoff"

    def test_constants_are_disjoint(self):
        assert PLAYOFF_WEEKS & REGULAR_SEASON_WEEKS == set()


# ---------------------------------------------------------------------------
# merge_projections_with_actuals
# ---------------------------------------------------------------------------


class TestMergeProjectionsWithActuals:

    def test_merge_produces_expected_columns(self):
        proj = _make_projections(5, [10, 11, 12, 13, 14])
        actuals = _make_actuals(5, [10, 11, 12, 13, 14])
        merged = merge_projections_with_actuals(proj, actuals)

        assert not merged.empty
        for col in ("game_type", "line_diff", "mu_diff", "hit_over"):
            assert col in merged.columns, f"Missing column: {col}"

    def test_merge_empty_projections(self):
        merged = merge_projections_with_actuals(pd.DataFrame(), _make_actuals(3, [1, 2, 3]))
        assert merged.empty

    def test_merge_empty_actuals(self):
        merged = merge_projections_with_actuals(_make_projections(3, [1, 2, 3]), pd.DataFrame())
        assert merged.empty

    def test_line_diff_sign(self):
        """When line > actual, line_diff should be positive (overpriced)."""
        proj = pd.DataFrame([{
            "season": 2024, "week": 5, "player_id": "TE_X",
            "team": "BUF", "mu": 50.0, "sigma": 10.0,
            "line": 55.0, "sportsbook": "SimBook",
        }])
        actuals = pd.DataFrame([{
            "player_id": "TE_X", "season": 2024, "week": 5,
            "actual_yards": 40.0,
        }])
        merged = merge_projections_with_actuals(proj, actuals)
        assert merged.iloc[0]["line_diff"] == pytest.approx(15.0)
        assert merged.iloc[0]["hit_over"] == 0

    def test_hit_over_flag(self):
        """When actual > line, hit_over should be 1."""
        proj = pd.DataFrame([{
            "season": 2024, "week": 5, "player_id": "TE_Y",
            "team": "SF", "mu": 40.0, "sigma": 10.0,
            "line": 35.0, "sportsbook": "SimBook",
        }])
        actuals = pd.DataFrame([{
            "player_id": "TE_Y", "season": 2024, "week": 5,
            "actual_yards": 50.0,
        }])
        merged = merge_projections_with_actuals(proj, actuals)
        assert merged.iloc[0]["hit_over"] == 1


# ---------------------------------------------------------------------------
# compute_bias_metrics
# ---------------------------------------------------------------------------


class TestComputeBiasMetrics:

    def test_empty_dataframe(self):
        metrics = compute_bias_metrics(pd.DataFrame(), "Empty")
        assert metrics["sample_size"] == 0
        assert metrics["mean_line_diff"] is None

    def test_positive_bias_detected(self):
        """Lines consistently higher than actuals -> positive mean_line_diff."""
        df = pd.DataFrame({
            "line_diff": [5.0, 8.0, 3.0, 6.0, 4.0],
            "mu_diff": [3.0, 5.0, 1.0, 4.0, 2.0],
            "hit_over": [0, 0, 0, 0, 0],
        })
        metrics = compute_bias_metrics(df, "Overpriced TEs")
        assert metrics["sample_size"] == 5
        assert metrics["mean_line_diff"] > 0
        assert metrics["over_hit_rate"] == 0.0

    def test_negative_bias(self):
        """Lines consistently lower than actuals -> negative mean_line_diff."""
        df = pd.DataFrame({
            "line_diff": [-5.0, -3.0, -7.0],
            "mu_diff": [-4.0, -2.0, -6.0],
            "hit_over": [1, 1, 1],
        })
        metrics = compute_bias_metrics(df, "Underpriced TEs")
        assert metrics["mean_line_diff"] < 0
        assert metrics["over_hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# run_significance_tests
# ---------------------------------------------------------------------------


class TestSignificanceTests:

    def test_insufficient_data_returns_none(self):
        reg = pd.DataFrame({"line_diff": [1.0], "hit_over": [0]})
        plo = pd.DataFrame({"line_diff": [2.0], "hit_over": [1]})
        results = run_significance_tests(reg, plo)
        assert results["t_test"] is None
        assert results["chi_squared"] is None

    def test_identical_distributions_not_significant(self):
        """Same distribution should not be significant."""
        rng = np.random.default_rng(42)
        n = 50
        reg = pd.DataFrame({
            "line_diff": rng.normal(0, 5, n),
            "hit_over": rng.binomial(1, 0.5, n),
        })
        plo = pd.DataFrame({
            "line_diff": rng.normal(0, 5, n),
            "hit_over": rng.binomial(1, 0.5, n),
        })
        results = run_significance_tests(reg, plo)
        assert results["t_test"] is not None
        # Similar distributions should usually not be significant
        # (not guaranteed, but with this seed it's not)
        assert results["t_test"]["p_value"] > 0.01

    def test_divergent_distributions_detected(self):
        """Very different distributions should be caught."""
        n = 100
        # Regular: mostly overs (90%), near-zero line_diff
        reg_hit = np.array([1] * 90 + [0] * 10)
        reg = pd.DataFrame({
            "line_diff": np.concatenate([np.full(90, -2.0), np.full(10, 1.0)]),
            "hit_over": reg_hit,
        })
        # Playoff: mostly unders (90%), large positive line_diff (overpriced)
        plo_hit = np.array([0] * 90 + [1] * 10)
        plo = pd.DataFrame({
            "line_diff": np.concatenate([np.full(90, 10.0), np.full(10, -1.0)]),
            "hit_over": plo_hit,
        })
        results = run_significance_tests(reg, plo)
        assert results["t_test"] is not None
        assert results["t_test"]["significant_at_05"] is True
        assert results["chi_squared"] is not None
        assert results["chi_squared"]["significant_at_05"] is True


# ---------------------------------------------------------------------------
# compute_suggested_adjustment
# ---------------------------------------------------------------------------


class TestComputeSuggestedAdjustment:

    def test_no_adjustment_with_small_sample(self):
        metrics = {"sample_size": 3, "mean_line_diff": 5.0}
        assert compute_suggested_adjustment(metrics, {}) is None

    def test_no_adjustment_when_bias_small(self):
        metrics = {"sample_size": 30, "mean_line_diff": 1.0}
        assert compute_suggested_adjustment(metrics, {}) is None

    def test_negative_adjustment_when_overpriced(self):
        metrics = {"sample_size": 30, "mean_line_diff": 4.5}
        adj = compute_suggested_adjustment(metrics, {})
        assert adj is not None
        assert adj < 0
        assert adj == pytest.approx(-4.5, abs=0.1)

    def test_none_when_mean_diff_is_none(self):
        metrics = {"sample_size": 30, "mean_line_diff": None}
        assert compute_suggested_adjustment(metrics, {}) is None

    def test_threshold_boundary(self):
        """Exactly 1.5 should not trigger adjustment."""
        metrics = {"sample_size": 30, "mean_line_diff": 1.5}
        assert compute_suggested_adjustment(metrics, {}) is None

    def test_just_above_threshold(self):
        """Just above 1.5 should trigger adjustment."""
        metrics = {"sample_size": 30, "mean_line_diff": 1.6}
        adj = compute_suggested_adjustment(metrics, {})
        assert adj is not None
        assert adj == pytest.approx(-1.6, abs=0.1)


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


class TestBuildReport:

    def test_report_structure(self):
        report = build_report(
            overall=compute_bias_metrics(pd.DataFrame(), "Overall"),
            regular=compute_bias_metrics(pd.DataFrame(), "Regular Season"),
            playoff=compute_bias_metrics(pd.DataFrame(), "Playoff"),
            significance={"t_test": None, "chi_squared": None},
            adjustment=None,
            seasons=[2024, 2025],
        )
        assert report["report_type"] == "te_market_bias_analysis"
        assert report["seasons_analyzed"] == [2024, 2025]
        assert "generated_at" in report
        assert "overall_metrics" in report
        assert "regular_season_metrics" in report
        assert "playoff_metrics" in report
        assert "statistical_tests" in report
        assert "suggested_adjustment" in report

    def test_report_with_adjustment(self):
        report = build_report(
            overall={"label": "Overall", "sample_size": 100, "mean_line_diff": 3.0,
                      "median_line_diff": 2.5, "mean_mu_diff": 2.0,
                      "over_hit_rate": 0.42, "std_line_diff": 8.0},
            regular={"label": "Regular Season", "sample_size": 80, "mean_line_diff": 2.0,
                      "median_line_diff": 1.5, "mean_mu_diff": 1.5,
                      "over_hit_rate": 0.45, "std_line_diff": 7.0},
            playoff={"label": "Playoff", "sample_size": 20, "mean_line_diff": 6.0,
                      "median_line_diff": 5.0, "mean_mu_diff": 4.0,
                      "over_hit_rate": 0.30, "std_line_diff": 9.0},
            significance={"t_test": {"p_value": 0.03, "significant_at_05": True,
                                      "significant_at_10": True, "t_statistic": 2.2},
                           "chi_squared": None},
            adjustment=-3.5,
            seasons=[2024],
        )
        assert report["suggested_adjustment"]["value"] == -3.5
        assert "playoff" in report["suggested_adjustment"]["applies_to"]


# ---------------------------------------------------------------------------
# Integration: end-to-end with synthetic data
# ---------------------------------------------------------------------------


class TestEndToEndSynthetic:

    def test_overpriced_playoff_tes_detected(self):
        """Synthetic scenario: playoff TEs are consistently overpriced."""
        regular_weeks = list(range(1, 15))
        playoff_weeks = [19, 20, 21, 22]

        # Regular season: line ~= actual (no bias)
        reg_proj = _make_projections(40, regular_weeks, mu_base=45, line_base=45)
        reg_actuals = _make_actuals(40, regular_weeks, actual_base=45)

        # Playoff: line much higher than actual (overpriced)
        plo_proj = _make_projections(20, playoff_weeks, mu_base=50, line_base=55)
        plo_actuals = _make_actuals(20, playoff_weeks, actual_base=38)

        all_proj = pd.concat([reg_proj, plo_proj], ignore_index=True)
        all_actuals = pd.concat([reg_actuals, plo_actuals], ignore_index=True)

        merged = merge_projections_with_actuals(all_proj, all_actuals)
        assert not merged.empty

        playoff_df = merged[merged["game_type"] == "playoff"]
        regular_df = merged[merged["game_type"] == "regular"]

        playoff_metrics = compute_bias_metrics(playoff_df, "Playoff")
        overall_metrics = compute_bias_metrics(merged, "Overall")

        # Playoff lines should be overpriced (positive line_diff)
        assert playoff_metrics["mean_line_diff"] > 5.0

        # Regular season should be roughly neutral
        assert abs(regular_df["line_diff"].mean()) < 15.0

        # Adjustment should be suggested
        adj = compute_suggested_adjustment(playoff_metrics, overall_metrics)
        assert adj is not None
        assert adj < 0
