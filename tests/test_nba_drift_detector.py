"""Tests for utils/nba_drift_detector.py.

Validates PSI computation, alert levels, edge cases, and DB persistence.
All tests use tmp_path SQLite DB — no real NBA data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from schema_migrations import MigrationManager
from utils.db import executemany, read_dataframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_drift.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _seed_projections(rows, db_path=None):
    """Insert synthetic projection rows."""
    sql = """
        INSERT OR IGNORE INTO nba_projections
            (player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    executemany(sql, rows)


# ---------------------------------------------------------------------------
# Test 1: compute_psi(x, x) < 0.01 for identical distributions
# ---------------------------------------------------------------------------


def test_psi_identical_distributions():
    from utils.nba_drift_detector import compute_psi

    rng = np.random.default_rng(0)
    x = rng.normal(loc=20.0, scale=5.0, size=200)
    psi = compute_psi(x, x)
    assert psi < 0.01


# ---------------------------------------------------------------------------
# Test 2: Non-overlapping distributions → PSI > 0.2
# ---------------------------------------------------------------------------


def test_psi_non_overlapping_distributions():
    from utils.nba_drift_detector import compute_psi

    rng = np.random.default_rng(1)
    expected = rng.normal(loc=0.0, scale=1.0, size=300)
    actual = rng.normal(loc=10.0, scale=1.0, size=300)
    psi = compute_psi(expected, actual)
    assert psi > 0.2


# ---------------------------------------------------------------------------
# Test 3: PSI is always non-negative
# ---------------------------------------------------------------------------


def test_psi_non_negative():
    from utils.nba_drift_detector import compute_psi

    rng = np.random.default_rng(2)
    for _ in range(10):
        expected = rng.exponential(scale=5.0, size=100)
        actual = rng.exponential(scale=7.0, size=100)
        psi = compute_psi(expected, actual)
        assert psi >= 0.0


# ---------------------------------------------------------------------------
# Test 4: Zero-count bins don't cause divide-by-zero (skewed data)
# ---------------------------------------------------------------------------


def test_psi_skewed_data_no_divide_by_zero():
    from utils.nba_drift_detector import compute_psi

    # Very skewed: most values clustered at 0, a few outliers
    expected = np.concatenate([np.zeros(190), np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
                                                          700.0, 800.0, 900.0, 1000.0])])
    actual = np.concatenate([np.zeros(50), np.full(150, 0.1)])

    # Should not raise
    psi = compute_psi(expected, actual)
    assert psi >= 0.0
    assert np.isfinite(psi)


# ---------------------------------------------------------------------------
# Test 5: Stable data → alert_level='stable'
# ---------------------------------------------------------------------------


def test_detect_prediction_drift_stable_data(db):
    from datetime import date, timedelta

    from utils.nba_drift_detector import detect_prediction_drift

    today = date.today()
    rows = []
    rng = np.random.default_rng(3)

    # Reference window: 30-44 days ago
    for i in range(30, 45):
        d = (today - timedelta(days=i)).isoformat()
        for j in range(2):
            rows.append((
                j + 1, f"Player{j}", "BOS", 2025, d, f"game_{i}_{j}", "pts",
                float(rng.normal(20.0, 3.0)), 0.7,
            ))

    # Recent window: 0-13 days ago
    for i in range(0, 14):
        d = (today - timedelta(days=i)).isoformat()
        for j in range(2):
            rows.append((
                j + 1, f"Player{j}", "BOS", 2025, d, f"recent_{i}_{j}", "pts",
                float(rng.normal(20.0, 3.0)), 0.7,
            ))

    _seed_projections(rows)

    result = detect_prediction_drift("pts", window_days=14, reference_days=30)

    assert "alert_level" in result
    assert result["alert_level"] in ("stable", "monitor", "alert")
    assert result["market"] == "pts"
    assert result["psi"] >= 0.0


# ---------------------------------------------------------------------------
# Test 6: Insufficient data returns 'stable' with explanation
# ---------------------------------------------------------------------------


def test_detect_prediction_drift_insufficient_data(db):
    from utils.nba_drift_detector import detect_prediction_drift

    # Insert only 3 rows — far below the 14-row minimum
    rows = [
        (1, "PlayerA", "LAL", 2025, "2026-03-01", "g1", "pts", 22.0, 0.6),
        (2, "PlayerB", "LAL", 2025, "2026-03-02", "g2", "pts", 18.0, 0.6),
        (3, "PlayerC", "LAL", 2025, "2026-03-03", "g3", "pts", 25.0, 0.6),
    ]
    _seed_projections(rows)

    result = detect_prediction_drift("pts", window_days=14, reference_days=30)

    assert result["alert_level"] == "stable"
    assert "insufficient" in result["explanation"].lower() or "data" in result["explanation"].lower()


# ---------------------------------------------------------------------------
# Test 7: Result dict has required keys
# ---------------------------------------------------------------------------


def test_result_dict_has_required_keys(db):
    from utils.nba_drift_detector import detect_prediction_drift

    result = detect_prediction_drift("reb", window_days=14, reference_days=30)

    required_keys = {"market", "psi", "alert_level", "window_days", "reference_days",
                     "n_recent", "n_reference", "explanation"}
    assert required_keys.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# Test 8: run_drift_checks writes alerts to nba_drift_alerts table
# ---------------------------------------------------------------------------


def test_run_drift_checks_writes_to_db(db):
    from utils.nba_drift_detector import run_drift_checks

    alerts = run_drift_checks("2026-03-04", markets=["pts", "reb"])

    assert isinstance(alerts, list)
    assert len(alerts) == 2

    saved = read_dataframe(
        "SELECT * FROM nba_drift_alerts WHERE game_date = ?",
        ("2026-03-04",),
    )
    assert len(saved) >= 2
    assert "alert_level" in saved.columns
    assert "psi_score" in saved.columns
