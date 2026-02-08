"""Tests for risk_manager: correlation detection, exposure caps, Monte Carlo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from risk_manager import (
    assess_risk,
    build_correlation_matrix,
    compute_exposure,
    detect_correlations,
    detect_team_stacks,
    monte_carlo_drawdown,
    risk_adjusted_kelly,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_value_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal value-bet DataFrame from a list of dicts."""
    defaults = {
        "season": 2025,
        "week": 22,
        "player_id": "P1",
        "event_id": "EVT1",
        "team": "KC",
        "market": "rushing_yards",
        "sportsbook": "DraftKings",
        "line": 60.5,
        "price": -110,
        "mu": 72.0,
        "sigma": 12.0,
        "p_win": 0.62,
        "edge_percentage": 0.10,
        "expected_roi": 0.08,
        "kelly_fraction": 0.04,
        "stake": 40.0,
        "generated_at": "2025-01-01T00:00:00",
    }
    filled = [{**defaults, **r} for r in rows]
    return pd.DataFrame(filled)


# ── Correlation detection ─────────────────────────────────────────────

class TestCorrelationDetection:
    """Verify correlated and uncorrelated props are classified correctly."""

    def test_positive_correlation_qb_wr(self):
        df = _make_value_df([
            {"player_id": "QB1", "market": "passing_yards", "team": "KC", "event_id": "G1"},
            {"player_id": "WR1", "market": "receiving_yards", "team": "KC", "event_id": "G1"},
        ])
        result = detect_correlations(df)
        assert "correlation_group" in result.columns
        groups = result["correlation_group"].dropna().unique()
        assert len(groups) == 1, "QB pass + WR receiving should form one group"
        assert "pos_corr" in groups[0]

    def test_negative_correlation(self):
        df = _make_value_df([
            {"player_id": "QB1", "market": "passing_yards", "team": "KC", "event_id": "G1"},
            {"player_id": "RB1", "market": "rushing_yards", "team": "KC", "event_id": "G1"},
        ])
        result = detect_correlations(df)
        groups = result["correlation_group"].dropna().unique()
        assert len(groups) == 1
        assert "neg_corr" in groups[0]

    def test_uncorrelated_different_games(self):
        df = _make_value_df([
            {"player_id": "QB1", "market": "passing_yards", "team": "KC", "event_id": "G1"},
            {"player_id": "WR2", "market": "receiving_yards", "team": "BUF", "event_id": "G2"},
        ])
        result = detect_correlations(df)
        n_grouped = result["correlation_group"].notna().sum()
        assert n_grouped == 0, "Different games should not be correlated"

    def test_same_team_stacking(self):
        df = _make_value_df([
            {"player_id": "RB1", "market": "rushing_yards", "team": "KC", "event_id": "G1"},
            {"player_id": "RB2", "market": "rushing_yards", "team": "KC", "event_id": "G1"},
        ])
        result = detect_correlations(df)
        groups = result["correlation_group"].dropna().unique()
        assert len(groups) == 1
        assert "same_team" in groups[0]

    def test_empty_dataframe(self):
        df = _make_value_df([])
        result = detect_correlations(df)
        assert "correlation_group" in result.columns
        assert len(result) == 0


# ── Team stack detection ──────────────────────────────────────────────

class TestTeamStacks:

    def test_detects_stack(self):
        df = _make_value_df([
            {"player_id": "P1", "team": "KC"},
            {"player_id": "P2", "team": "KC"},
            {"player_id": "P3", "team": "BUF"},
        ])
        stacks = detect_team_stacks(df)
        assert "KC" in stacks
        assert len(stacks["KC"]) == 2
        assert "BUF" not in stacks

    def test_no_stacks(self):
        df = _make_value_df([
            {"player_id": "P1", "team": "KC"},
            {"player_id": "P2", "team": "BUF"},
        ])
        stacks = detect_team_stacks(df)
        assert len(stacks) == 0


# ── Exposure caps ─────────────────────────────────────────────────────

class TestExposureCaps:

    def test_team_exposure_warning(self):
        df = _make_value_df([
            {"player_id": "P1", "team": "KC", "stake": 350.0},
            {"player_id": "P2", "team": "KC", "stake": 50.0},
        ])
        result = compute_exposure(df, bankroll=1000.0)
        assert "exposure_warning" in result.columns
        warnings = result["exposure_warning"].dropna()
        assert len(warnings) > 0
        assert any("team_exposure" in str(w) for w in warnings)

    def test_player_exposure_warning(self):
        df = _make_value_df([
            {"player_id": "P1", "team": "KC", "stake": 160.0},
        ])
        result = compute_exposure(df, bankroll=1000.0)
        warnings = result["exposure_warning"].dropna()
        assert len(warnings) > 0
        assert any("player_exposure" in str(w) for w in warnings)

    def test_no_exposure_warning_within_limits(self):
        df = _make_value_df([
            {"player_id": "P1", "team": "KC", "stake": 50.0},
            {"player_id": "P2", "team": "BUF", "stake": 50.0},
        ])
        result = compute_exposure(df, bankroll=1000.0)
        n_warn = result["exposure_warning"].notna().sum()
        assert n_warn == 0

    def test_empty_dataframe(self):
        df = _make_value_df([])
        result = compute_exposure(df, bankroll=1000.0)
        assert "exposure_warning" in result.columns
        assert len(result) == 0


# ── Monte Carlo simulation ───────────────────────────────────────────

class TestMonteCarloDrawdown:

    def test_basic_simulation(self):
        kelly = np.array([0.05, 0.04, 0.03])
        probs = np.array([0.55, 0.60, 0.50])
        odds = np.array([-110, -110, +100])

        stats = monte_carlo_drawdown(kelly, probs, odds, iterations=500)
        assert "mean_drawdown" in stats
        assert "max_drawdown" in stats
        assert "p95_drawdown" in stats
        assert stats["mean_drawdown"] >= 0
        assert stats["max_drawdown"] >= stats["mean_drawdown"]

    def test_empty_bets(self):
        stats = monte_carlo_drawdown(
            np.array([]), np.array([]), np.array([]), iterations=100,
        )
        assert stats["mean_drawdown"] == 0.0
        assert stats["max_drawdown"] == 0.0

    def test_high_kelly_produces_larger_drawdown(self):
        low_kelly = np.array([0.01, 0.01])
        high_kelly = np.array([0.10, 0.10])
        probs = np.array([0.55, 0.55])
        odds = np.array([-110, -110])

        low_stats = monte_carlo_drawdown(low_kelly, probs, odds, iterations=500)
        high_stats = monte_carlo_drawdown(high_kelly, probs, odds, iterations=500)
        assert high_stats["mean_drawdown"] > low_stats["mean_drawdown"]

    def test_deterministic_seed(self):
        kelly = np.array([0.05, 0.04])
        probs = np.array([0.55, 0.60])
        odds = np.array([-110, -110])

        stats1 = monte_carlo_drawdown(kelly, probs, odds, iterations=200)
        stats2 = monte_carlo_drawdown(kelly, probs, odds, iterations=200)
        assert stats1["mean_drawdown"] == stats2["mean_drawdown"]


# ── Risk-adjusted Kelly ──────────────────────────────────────────────

class TestRiskAdjustedKelly:

    def test_no_scaling_below_threshold(self):
        dd = {"mean_drawdown": 0.05, "max_drawdown": 0.10, "p95_drawdown": 0.15}
        result = risk_adjusted_kelly(0.05, dd)
        assert result == 0.05

    def test_scales_down_above_threshold(self):
        dd = {"mean_drawdown": 0.10, "max_drawdown": 0.40, "p95_drawdown": 0.30}
        result = risk_adjusted_kelly(0.05, dd)
        assert result < 0.05
        expected = 0.05 * (0.20 / 0.30)
        assert abs(result - expected) < 1e-10


# ── Integration: assess_risk ──────────────────────────────────────────

class TestAssessRisk:

    def test_full_pipeline(self):
        df = _make_value_df([
            {"player_id": "QB1", "market": "passing_yards", "team": "KC",
             "event_id": "G1", "stake": 40.0, "kelly_fraction": 0.04,
             "p_win": 0.60, "price": -110},
            {"player_id": "WR1", "market": "receiving_yards", "team": "KC",
             "event_id": "G1", "stake": 30.0, "kelly_fraction": 0.03,
             "p_win": 0.55, "price": -110},
        ])
        result = assess_risk(df, bankroll=1000.0)
        assert "correlation_group" in result.columns
        assert "exposure_warning" in result.columns
        assert "risk_adjusted_kelly" in result.columns
        assert len(result) == 2
        assert result["risk_adjusted_kelly"].notna().all()

    def test_empty_input(self):
        df = _make_value_df([])
        result = assess_risk(df)
        assert "correlation_group" in result.columns
        assert "exposure_warning" in result.columns
        assert "risk_adjusted_kelly" in result.columns
        assert len(result) == 0

    def test_does_not_mutate_input(self):
        df = _make_value_df([
            {"player_id": "P1", "stake": 40.0, "kelly_fraction": 0.04,
             "p_win": 0.60, "price": -110},
        ])
        original_cols = set(df.columns)
        assess_risk(df, bankroll=1000.0)
        assert set(df.columns) == original_cols, "assess_risk must not mutate input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
