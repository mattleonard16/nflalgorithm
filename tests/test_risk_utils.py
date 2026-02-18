"""Unit tests for sport-agnostic risk utility functions."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _patch_config(tmp_path, monkeypatch):
    """Provide a minimal config so risk_utils can read thresholds."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")


class TestMonteCarlDrawdown:
    def test_empty_arrays_returns_zero_drawdowns(self):
        from utils.risk_utils import monte_carlo_drawdown

        result = monte_carlo_drawdown(
            np.array([]),
            np.array([]),
            np.array([]),
        )
        assert result == {"mean_drawdown": 0.0, "max_drawdown": 0.0, "p95_drawdown": 0.0}

    def test_single_bet_returns_dict_with_expected_keys(self):
        from utils.risk_utils import monte_carlo_drawdown

        result = monte_carlo_drawdown(
            np.array([0.05]),
            np.array([0.6]),
            np.array([-110]),
            iterations=100,
        )
        assert "mean_drawdown" in result
        assert "max_drawdown" in result
        assert "p95_drawdown" in result

    def test_drawdown_values_are_non_negative(self):
        from utils.risk_utils import monte_carlo_drawdown

        result = monte_carlo_drawdown(
            np.array([0.05, 0.03]),
            np.array([0.6, 0.55]),
            np.array([-110, -120]),
            iterations=200,
        )
        assert result["mean_drawdown"] >= 0
        assert result["max_drawdown"] >= 0
        assert result["p95_drawdown"] >= 0

    def test_max_drawdown_gte_mean(self):
        from utils.risk_utils import monte_carlo_drawdown

        result = monte_carlo_drawdown(
            np.array([0.05, 0.04, 0.03]),
            np.array([0.6, 0.55, 0.7]),
            np.array([-110, -120, 150]),
            iterations=500,
        )
        assert result["max_drawdown"] >= result["mean_drawdown"]

    def test_p95_between_mean_and_max(self):
        from utils.risk_utils import monte_carlo_drawdown

        result = monte_carlo_drawdown(
            np.array([0.05, 0.04, 0.03]),
            np.array([0.6, 0.55, 0.7]),
            np.array([-110, -120, 150]),
            iterations=500,
        )
        assert result["p95_drawdown"] >= result["mean_drawdown"]
        assert result["p95_drawdown"] <= result["max_drawdown"]

    def test_deterministic_with_same_seed(self):
        from utils.risk_utils import monte_carlo_drawdown

        args = (np.array([0.05, 0.03]), np.array([0.6, 0.55]), np.array([-110, -120]))
        r1 = monte_carlo_drawdown(*args, iterations=100)
        r2 = monte_carlo_drawdown(*args, iterations=100)
        assert r1 == r2


class TestRiskAdjustedKelly:
    def test_no_drawdown_returns_original(self):
        from utils.risk_utils import risk_adjusted_kelly

        result = risk_adjusted_kelly(0.05, {"p95_drawdown": 0.0})
        assert result == 0.05

    def test_below_threshold_returns_original(self):
        from utils.risk_utils import risk_adjusted_kelly

        result = risk_adjusted_kelly(0.05, {"p95_drawdown": 0.01})
        assert result == 0.05

    def test_above_threshold_scales_down(self, monkeypatch):
        import config as cfg

        monkeypatch.setattr(cfg.config.risk, "max_drawdown_threshold", 0.10)
        from utils.risk_utils import risk_adjusted_kelly

        result = risk_adjusted_kelly(0.05, {"p95_drawdown": 0.20})
        # scale = 0.10 / 0.20 = 0.5, result = 0.05 * 0.5 = 0.025
        assert result == pytest.approx(0.025)

    def test_zero_threshold_returns_original(self, monkeypatch):
        import config as cfg

        monkeypatch.setattr(cfg.config.risk, "max_drawdown_threshold", 0.0)
        from utils.risk_utils import risk_adjusted_kelly

        result = risk_adjusted_kelly(0.05, {"p95_drawdown": 0.20})
        assert result == 0.05


class TestAppendWarning:
    def test_none_returns_new(self):
        from utils.risk_utils import append_warning

        assert append_warning(None, "hello") == "hello"

    def test_existing_appends_with_semicolon(self):
        from utils.risk_utils import append_warning

        assert append_warning("first", "second") == "first; second"

    def test_chain_multiple(self):
        from utils.risk_utils import append_warning

        result = append_warning(append_warning(None, "a"), "b")
        assert result == "a; b"
