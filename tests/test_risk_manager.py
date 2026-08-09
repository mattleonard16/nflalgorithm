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
    compute_exposure,
    detect_correlations,
    detect_team_stacks,
    monte_carlo_drawdown,
    normalize_portfolio_stakes,
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

    def test_transitive_pairs_collapse_into_one_group(self):
        """(QB,WR) forms the group; later pairs re-use the existing label via
        the already-grouped i side instead of minting new groups."""
        df = _make_value_df([
            {"player_id": "QB1", "market": "passing_yards", "team": "KC", "event_id": "G1"},
            {"player_id": "WR1", "market": "receiving_yards", "team": "KC", "event_id": "G1"},
            {"player_id": "RB1", "market": "rushing_yards", "team": "KC", "event_id": "G1"},
        ])
        result = detect_correlations(df)
        groups = result["correlation_group"]
        assert groups.notna().all()
        assert groups.nunique() == 1, "chained pairs must share one group label"

    def test_join_through_already_grouped_j_side(self):
        """Pair order is (0,1),(0,2),(0,3),(1,2),(1,3),(2,3): the group forms
        at (1,3); (2,3) then matches with only the j side grouped."""
        df = _make_value_df([
            {"player_id": "X0", "market": "rushing_yards", "team": "SEA", "event_id": "G0"},
            {"player_id": "QB1", "market": "passing_yards", "team": "KC", "event_id": "G2"},
            {"player_id": "WR9", "market": "receiving_tds", "team": "BUF", "event_id": "G2"},
            {"player_id": "QB2", "market": "passing_tds", "team": "KC", "event_id": "G2"},
        ])
        result = detect_correlations(df)
        groups = result["correlation_group"]
        assert pd.isna(groups.iloc[0])
        assert groups.iloc[1] == groups.iloc[2] == groups.iloc[3]

    def test_same_game_unrelated_markets_not_grouped(self):
        """Same event, different teams, no market pair → no correlation."""
        df = _make_value_df([
            {"player_id": "WR1", "market": "receiving_yards", "team": "KC", "event_id": "G1"},
            {"player_id": "WR2", "market": "receiving_yards", "team": "BUF", "event_id": "G1"},
        ])
        result = detect_correlations(df)
        assert result["correlation_group"].isna().all()


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

    def test_empty_dataframe_returns_empty_dict(self):
        assert detect_team_stacks(_make_value_df([])) == {}


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

    def test_game_exposure_warning(self):
        """Two teams in one game can breach the game cap while each team stays
        under the team cap; both rows must carry the game_exposure flag."""
        from config import config as _config

        bankroll = 1000.0
        # Two rows → total = 1.2x the game cap; per-team share is 0.6x the
        # game cap, below the (higher) team cap for the shipped config.
        per_stake = bankroll * _config.risk.max_game_exposure * 0.6
        assert per_stake / bankroll < _config.risk.max_team_exposure
        df = _make_value_df([
            {"player_id": "P1", "team": "KC", "event_id": "G1", "stake": per_stake},
            {"player_id": "P2", "team": "BUF", "event_id": "G1", "stake": per_stake},
        ])
        result = compute_exposure(df, bankroll=bankroll)
        warnings = result["exposure_warning"]
        assert warnings.notna().all()
        assert all("game_exposure" in str(w) for w in warnings)
        assert not any("team_exposure" in str(w) for w in warnings)


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


# ── Portfolio-level stake normalization ───────────────────────────────

class TestNormalizePortfolioStakes:

    def test_scales_proportionally_when_capped_bets_exceed_bankroll(self):
        """Per-bet capped stakes (10% each) can still sum past the bankroll."""
        df = _make_value_df([
            {"player_id": f"P{i}", "stake": 100.0, "kelly_fraction": 0.10}
            for i in range(15)
        ])
        assert df["stake"].sum() == pytest.approx(1500.0)

        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        assert result["stake"].sum() == pytest.approx(1000.0)
        # Uniform scaling: every stake shrinks by the same factor.
        assert np.allclose(result["stake"], 100.0 * (1000.0 / 1500.0))
        # Stake-derived kelly_fraction scales identically.
        assert np.allclose(result["kelly_fraction"], 0.10 * (1000.0 / 1500.0))

    def test_preserves_relative_sizing(self):
        df = _make_value_df([
            {"player_id": "P1", "stake": 600.0, "kelly_fraction": 0.60},
            {"player_id": "P2", "stake": 300.0, "kelly_fraction": 0.30},
            {"player_id": "P3", "stake": 900.0, "kelly_fraction": 0.90},
        ])
        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        assert result["stake"].sum() == pytest.approx(1000.0)
        ratios = result["stake"] / df["stake"]
        assert ratios.nunique() == 1 or np.allclose(ratios, ratios.iloc[0])
        assert result["stake"].iloc[2] == pytest.approx(3 * result["stake"].iloc[1])

    def test_under_limit_card_unchanged(self):
        df = _make_value_df([
            {"player_id": "P1", "stake": 100.0, "kelly_fraction": 0.10},
            {"player_id": "P2", "stake": 50.0, "kelly_fraction": 0.05},
        ])
        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        pd.testing.assert_frame_equal(result, df)

    def test_max_total_fraction_tightens_limit(self):
        df = _make_value_df([
            {"player_id": "P1", "stake": 400.0, "kelly_fraction": 0.40},
            {"player_id": "P2", "stake": 400.0, "kelly_fraction": 0.40},
        ])
        result = normalize_portfolio_stakes(df, bankroll=1000.0, max_total_fraction=0.5)
        assert result["stake"].sum() == pytest.approx(500.0)

    def test_w22_regression_shape(self):
        """Real 2025 W22 card shape: 9 bets, 3233u staked on a 1000u bankroll.

        All 9 kelly fractions exceeded the 0.10 per-bet cap (max 0.743).
        Post-normalization, the persisted card must never stake more than
        the bankroll.
        """
        stakes = [743.0, 520.0, 450.0, 400.0, 350.0, 300.0, 250.0, 150.0, 70.0]
        assert sum(stakes) == pytest.approx(3233.0)
        df = _make_value_df([
            {"player_id": f"P{i}", "stake": s, "kelly_fraction": s / 1000.0}
            for i, s in enumerate(stakes)
        ])
        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        assert result["stake"].sum() <= 1000.0 + 1e-9
        assert result["stake"].sum() == pytest.approx(1000.0)
        # Largest bet stays largest.
        assert result["stake"].idxmax() == df["stake"].idxmax()

    def test_empty_frame_passes_through(self):
        df = _make_value_df([])
        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        assert result.empty
        assert list(result.columns) == list(df.columns)

    def test_empty_but_columned_frame_passes_through(self):
        """The shape materialize_week produces when filters drop every row."""
        df = _make_value_df([{"player_id": "P1", "stake": 100.0}]).iloc[0:0]
        assert df.empty and len(df.columns) > 0
        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        assert result.empty
        assert list(result.columns) == list(df.columns)

    def test_nan_stake_raises(self):
        df = _make_value_df([
            {"player_id": "P1", "stake": float("nan")},
            {"player_id": "P2", "stake": 1200.0},
        ])
        with pytest.raises(ValueError, match="non-NaN"):
            normalize_portfolio_stakes(df, bankroll=1000.0)

    def test_negative_stake_raises(self):
        """Negative stakes would let one bet exceed the cap after scaling."""
        df = _make_value_df([
            {"player_id": "P1", "stake": -50.0},
            {"player_id": "P2", "stake": 1200.0},
        ])
        with pytest.raises(ValueError, match="non-negative"):
            normalize_portfolio_stakes(df, bankroll=1000.0)

    def test_zero_stake_frame_passes_through(self):
        df = _make_value_df([
            {"player_id": "P1", "stake": 0.0, "kelly_fraction": 0.0},
        ])
        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        pd.testing.assert_frame_equal(result, df)

    def test_does_not_mutate_input(self):
        df = _make_value_df([
            {"player_id": "P1", "stake": 2000.0, "kelly_fraction": 2.0},
        ])
        before = df.copy(deep=True)
        normalize_portfolio_stakes(df, bankroll=1000.0)
        pd.testing.assert_frame_equal(df, before)

    def test_invalid_bankroll_raises(self):
        df = _make_value_df([{"player_id": "P1", "stake": 100.0}])
        with pytest.raises(ValueError, match="bankroll"):
            normalize_portfolio_stakes(df, bankroll=0.0)

    def test_invalid_max_total_fraction_raises(self):
        df = _make_value_df([{"player_id": "P1", "stake": 100.0}])
        with pytest.raises(ValueError, match="max_total_fraction"):
            normalize_portfolio_stakes(df, bankroll=1000.0, max_total_fraction=0.0)

    def test_exactly_at_limit_is_not_scaled(self):
        """sum == bankroll * max_total_fraction is legal — must pass through."""
        df = _make_value_df([
            {"player_id": "P1", "stake": 600.0, "kelly_fraction": 0.60},
            {"player_id": "P2", "stake": 400.0, "kelly_fraction": 0.40},
        ])
        assert df["stake"].sum() == 1000.0
        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        pd.testing.assert_frame_equal(result, df)

    def test_exactly_at_fractional_limit_is_not_scaled(self):
        """Boundary holds for max_total_fraction != 1.0 too."""
        df = _make_value_df([
            {"player_id": "P1", "stake": 300.0, "kelly_fraction": 0.30},
            {"player_id": "P2", "stake": 200.0, "kelly_fraction": 0.20},
        ])
        result = normalize_portfolio_stakes(
            df, bankroll=1000.0, max_total_fraction=0.5
        )
        pd.testing.assert_frame_equal(result, df)

    def test_hair_over_limit_scales_down(self):
        df = _make_value_df([
            {"player_id": "P1", "stake": 600.0, "kelly_fraction": 0.60},
            {"player_id": "P2", "stake": 400.0 + 1e-6, "kelly_fraction": 0.40},
        ])
        result = normalize_portfolio_stakes(df, bankroll=1000.0)
        assert result["stake"].sum() == pytest.approx(1000.0)
        assert result["stake"].sum() < df["stake"].sum()

    def test_fractional_limit_scales_stake_and_kelly_to_exact_values(self):
        """max_total_fraction != 1.0: limit 500, total 1000 → scale exactly 0.5."""
        df = _make_value_df([
            {"player_id": "P1", "stake": 600.0, "kelly_fraction": 0.60},
            {"player_id": "P2", "stake": 400.0, "kelly_fraction": 0.40},
        ])
        result = normalize_portfolio_stakes(
            df, bankroll=1000.0, max_total_fraction=0.5
        )
        assert result["stake"].tolist() == pytest.approx([300.0, 200.0])
        assert result["kelly_fraction"].tolist() == pytest.approx([0.30, 0.20])

    def test_max_total_fraction_above_one_allows_oversubscription(self):
        """A leveraged limit (>1.0) is respected, not silently clamped to 1.0."""
        under = _make_value_df([
            {"player_id": "P1", "stake": 700.0, "kelly_fraction": 0.70},
            {"player_id": "P2", "stake": 700.0, "kelly_fraction": 0.70},
        ])
        result = normalize_portfolio_stakes(
            under, bankroll=1000.0, max_total_fraction=1.5
        )
        pd.testing.assert_frame_equal(result, under)

        over = _make_value_df([
            {"player_id": "P1", "stake": 800.0, "kelly_fraction": 0.80},
            {"player_id": "P2", "stake": 800.0, "kelly_fraction": 0.80},
        ])
        result = normalize_portfolio_stakes(
            over, bankroll=1000.0, max_total_fraction=1.5
        )
        assert result["stake"].sum() == pytest.approx(1500.0)
        assert result["stake"].tolist() == pytest.approx([750.0, 750.0])

    STAKE_VECTORS = [
        [0.0],
        [0.0, 0.0, 0.0],
        [100.0],
        [1e6, 1.0],
        [0.01] * 7,
        [500.0, 500.0],
        [750.0, 250.0, 0.0],
        [1234.5, 8.25, 0.0, 42.0],
    ]

    @pytest.mark.parametrize(
        "bankroll,fraction",
        [(1000.0, 1.0), (1000.0, 0.5), (250.0, 0.8), (1000.0, 1.5), (10.0, 0.05)],
    )
    @pytest.mark.parametrize("stakes", STAKE_VECTORS)
    def test_grid_invariants_sum_capped_and_uniform_unit_scale(
        self, stakes, bankroll, fraction
    ):
        """Property-style grid (hypothesis not installed): for any non-negative
        stake vector, the result sum never exceeds the limit, the applied
        factor is uniform, in (0, 1], and kelly scales by the same factor."""
        df = _make_value_df([
            {"player_id": f"P{i}", "stake": s, "kelly_fraction": s / 1000.0}
            for i, s in enumerate(stakes)
        ])
        result = normalize_portfolio_stakes(
            df, bankroll=bankroll, max_total_fraction=fraction
        )
        limit = bankroll * fraction
        total = float(df["stake"].sum())
        expected_factor = 1.0 if total <= limit else limit / total
        assert 0.0 < expected_factor <= 1.0
        assert result["stake"].sum() <= limit + 1e-9
        assert (result["stake"].to_numpy() >= 0).all()
        assert np.allclose(result["stake"], df["stake"] * expected_factor)
        assert np.allclose(
            result["kelly_fraction"], df["kelly_fraction"] * expected_factor
        )


# ── Kelly cap feature flag defaults (tracked config/runtime.py) ───────

class TestKellyCapFlagDefault:
    """CI-runnable flag checks against the tracked config/runtime.py.

    The default was deliberately flipped ON (user-approved change from the
    NFR-6 default-off gate) after real cards oversized without it.

    Mirrored in tests/test_kelly_cap.py (CI-skipped there via conftest, since
    that file imports value_betting_engine) — update both together.
    """

    @staticmethod
    def _load_runtime():
        import importlib.util

        cfg_path = Path(__file__).parent.parent / "config" / "runtime.py"
        spec = importlib.util.spec_from_file_location("_runtime_under_test", cfg_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv("NFL_FEATURE_KELLY_CAP", raising=False)
        module = self._load_runtime()
        assert module.config.features.kelly_cap_enabled is True

    def test_env_can_disable(self, monkeypatch):
        monkeypatch.setenv("NFL_FEATURE_KELLY_CAP", "0")
        module = self._load_runtime()
        assert module.config.features.kelly_cap_enabled is False

    def test_effective_config_default_is_on(self):
        """The *effective* config (private override included) must be ON.

        A gitignored config.py can shadow the tracked default via
        config/__init__._fill_missing_settings, silently reverting the
        user-approved flip on the deployment machine while the tracked-file
        tests above stay green. Import the config package in a clean
        subprocess so this catches that divergence wherever it runs (in CI
        there is no private config.py, so the tracked default is asserted).
        """
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "NFL_FEATURE_KELLY_CAP"}
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from config import config; print(config.features.kelly_cap_enabled)",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent),
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "True"


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


# ── run_risk_check report + CLI wiring (DB layer stubbed) ─────────────


class TestRunRiskCheckReport:
    def test_report_printed_and_assessed_frame_returned(self, monkeypatch, capsys):
        import risk_manager as rm

        df = _make_value_df([
            {"player_id": "QB1", "market": "passing_yards", "team": "KC",
             "event_id": "G1", "stake": 40.0, "kelly_fraction": 0.04,
             "p_win": 0.60, "price": -110},
            {"player_id": "WR1", "market": "receiving_yards", "team": "KC",
             "event_id": "G1", "stake": 30.0, "kelly_fraction": 0.03,
             "p_win": 0.55, "price": -110},
        ])
        monkeypatch.setattr(rm, "read_dataframe", lambda q, params=None: df)
        result = rm.run_risk_check(2025, 8)
        out = capsys.readouterr().out
        assert "Risk Assessment: 2 bets analyzed" in out
        assert "Correlated groups: 2" in out
        assert "Team stacks:" in out
        assert "KC: 2 bets" in out
        assert "Total Kelly:" in out
        assert "risk_adjusted_kelly" in result.columns
        assert len(result) == 2

    def test_empty_view_prints_and_returns_empty(self, monkeypatch, capsys):
        import risk_manager as rm

        monkeypatch.setattr(
            rm, "read_dataframe", lambda q, params=None: pd.DataFrame()
        )
        result = rm.run_risk_check(2025, 8)
        assert "No value bets for season=2025 week=8" in capsys.readouterr().out
        assert result.empty

    def test_db_error_prints_and_returns_empty(self, monkeypatch, capsys):
        import risk_manager as rm

        def boom(q, params=None):
            raise RuntimeError("db unavailable")

        monkeypatch.setattr(rm, "read_dataframe", boom)
        result = rm.run_risk_check(2025, 8)
        assert "No data for season=2025 week=8" in capsys.readouterr().out
        assert result.empty

    def test_main_parses_args_and_dispatches(self, monkeypatch):
        import risk_manager as rm

        calls = {}

        def fake_run(season, week):
            calls["season"] = season
            calls["week"] = week
            return pd.DataFrame()

        monkeypatch.setattr(rm, "run_risk_check", fake_run)
        monkeypatch.setattr(
            sys, "argv", ["risk_manager", "--season", "2025", "--week", "8"]
        )
        rm.main()
        assert calls == {"season": 2025, "week": 8}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
