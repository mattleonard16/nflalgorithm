"""Tests for _compute_market_mu WR receiving yards priors.

Issue 3: Ensure WRs produce non-zero mu using:
- EWMA over historical receiving yards
- rolling_targets * yards_per_target
- defense_vs_wr_multiplier (stubbed)
- role-based cluster priors (alpha/secondary/slot/fringe)
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from data_pipeline import DataPipeline


def test_market_mu_wr_uses_history_and_targets(monkeypatch):
    """High-usage WR (Tyreek-type) with strong rolling targets and history."""
    hist = pd.DataFrame(
        {"receiving_yards": [120, 95, 80, 105, 90], "targets": [12, 10, 9, 11, 8]}
    )
    monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: hist)

    dp = DataPipeline.__new__(DataPipeline)
    dp._load_projection_baseline = lambda: pd.DataFrame()

    row = SimpleNamespace(
        player_id="WR_ALPH",
        receiving_yards=0.0,
        rolling_targets=10.0,
        snap_percentage=88.0,
        breakout_percentile=0.8,
        usage_delta=0.02,
        opponent="NE",
    )
    mu = dp._compute_market_mu(row, "receiving_yards", season=2025, week=10)
    assert mu > 70.0, f"Expected mu > 70 for alpha WR with history, got {mu}"
    assert mu < 150.0, f"Expected mu < 150, got {mu}"


def test_market_mu_wr_role_prior_when_no_history(monkeypatch):
    """Slot WR with no history but high snap% -> should use role prior."""
    monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: pd.DataFrame())

    dp = DataPipeline.__new__(DataPipeline)
    dp._load_projection_baseline = lambda: pd.DataFrame()

    row = SimpleNamespace(
        player_id="WR_SLOT",
        receiving_yards=0.0,
        rolling_targets=0.0,  # force role-based prior
        snap_percentage=82.0,
        breakout_percentile=0.75,
        usage_delta=0.08,
        opponent="",
    )
    mu = dp._compute_market_mu(row, "receiving_yards", season=2025, week=10)
    # Alpha cluster (snap >= 80), should get role_prior ~75 as the main component
    assert mu >= 15.0, f"Expected mu >= 15 (minimum floor), got {mu}"
    assert mu <= 80.0, f"Expected mu <= 80 (pure role prior), got {mu}"


def test_market_mu_wr_secondary_cluster(monkeypatch):
    """Secondary WR with moderate targets, lower air yards."""
    hist = pd.DataFrame(
        {"receiving_yards": [55, 48, 62], "targets": [6, 5, 7]}
    )
    monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: hist)

    dp = DataPipeline.__new__(DataPipeline)
    dp._load_projection_baseline = lambda: pd.DataFrame()

    row = SimpleNamespace(
        player_id="WR_SEC",
        receiving_yards=0.0,
        rolling_targets=5.5,
        snap_percentage=68.0,
        breakout_percentile=0.55,
        usage_delta=0.01,
        opponent="DAL",
    )
    mu = dp._compute_market_mu(row, "receiving_yards", season=2025, week=10)
    assert mu > 40.0, f"Expected mu > 40 for secondary WR, got {mu}"
    assert mu < 100.0, f"Expected mu < 100, got {mu}"


def test_market_mu_wr_fringe_cluster_gets_minimum(monkeypatch):
    """Fringe WR with low snap% and no data should still get non-zero mu."""
    monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: pd.DataFrame())

    dp = DataPipeline.__new__(DataPipeline)
    dp._load_projection_baseline = lambda: pd.DataFrame()

    row = SimpleNamespace(
        player_id="WR_FRINGE",
        receiving_yards=0.0,
        rolling_targets=0.0,
        snap_percentage=35.0,
        breakout_percentile=0.2,
        usage_delta=0.0,
        opponent="",
    )
    mu = dp._compute_market_mu(row, "receiving_yards", season=2025, week=10)
    # Fringe cluster gives role_prior=30, but minimum floor is 15
    assert mu >= 15.0, f"Expected mu >= 15 (floor), got {mu}"
    assert mu <= 35.0, f"Expected mu <= 35 (fringe role prior), got {mu}"


def test_market_mu_wr_never_zero(monkeypatch):
    """Ensure mu_prior for receiving_yards is never exactly zero."""
    monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: pd.DataFrame())

    dp = DataPipeline.__new__(DataPipeline)
    dp._load_projection_baseline = lambda: pd.DataFrame()

    # Extreme case: all zeros
    row = SimpleNamespace(
        player_id="WR_ZERO",
        receiving_yards=0.0,
        rolling_targets=0.0,
        snap_percentage=0.0,
        breakout_percentile=0.0,
        usage_delta=0.0,
        opponent="",
    )
    mu = dp._compute_market_mu(row, "receiving_yards", season=2025, week=10)
    assert mu > 0, f"Expected mu > 0 even with all zeros, got {mu}"
    assert mu >= 15.0, f"Expected mu >= 15 (minimum floor), got {mu}"


def test_market_mu_wr_hist_targets_fallback(monkeypatch):
    """When row has no rolling_targets but history has targets, use hist_targets_mu."""
    hist = pd.DataFrame(
        {"receiving_yards": [70, 65, 58], "targets": [8, 7, 6]}
    )
    monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: hist)

    dp = DataPipeline.__new__(DataPipeline)
    dp._load_projection_baseline = lambda: pd.DataFrame()

    row = SimpleNamespace(
        player_id="WR_HISTTGT",
        receiving_yards=0.0,
        rolling_targets=0.0,  # no current targets
        snap_percentage=70.0,
        breakout_percentile=0.6,
        usage_delta=0.0,
        opponent="",
    )
    mu = dp._compute_market_mu(row, "receiving_yards", season=2025, week=10)
    # Should blend hist_mu + hist_targets_mu (via fallback) + role_prior
    assert mu > 45.0, f"Expected mu > 45 from history, got {mu}"
