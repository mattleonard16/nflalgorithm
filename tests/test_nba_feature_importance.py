"""Tests for utils/nba_feature_importance.py.

Tests permutation importance, SHAP importance, feature count consistency,
SHAP fallback, and DB persistence — all on synthetic data / tmp SQLite DB.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

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
    db_path = str(tmp_path / "test_importance.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _make_small_model(n_features: int):
    """Train a GBR on synthetic data and return (model, X, y)."""
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, n_features))
    y = rng.standard_normal(60)
    model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=0)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Test 1: Permutation importance returns DataFrame with 'feature' column
# ---------------------------------------------------------------------------


def test_permutation_importance_has_feature_column():
    from utils.nba_feature_importance import compute_permutation_importance

    n_features = 6
    feature_names = [f"f{i}" for i in range(n_features)]
    model, X, y = _make_small_model(n_features)

    result = compute_permutation_importance(model, X, y, feature_names)

    assert isinstance(result, pd.DataFrame)
    assert "feature" in result.columns
    assert set(result["feature"]) == set(feature_names)


# ---------------------------------------------------------------------------
# Test 2: Features are ranked by importance descending
# ---------------------------------------------------------------------------


def test_permutation_importance_ranked_descending():
    from utils.nba_feature_importance import compute_permutation_importance

    n_features = 6
    feature_names = [f"f{i}" for i in range(n_features)]
    model, X, y = _make_small_model(n_features)

    result = compute_permutation_importance(model, X, y, feature_names)

    assert "rank" in result.columns
    assert "importance" in result.columns
    # rank 1 should have the highest importance
    top = result[result["rank"] == 1]["importance"].iloc[0]
    bottom = result[result["rank"] == len(feature_names)]["importance"].iloc[0]
    assert top >= bottom


# ---------------------------------------------------------------------------
# Test 3: Feature count matches get_feature_cols(market)
# ---------------------------------------------------------------------------


def test_feature_count_matches_stat_model():
    from models.nba.stat_model import get_feature_cols
    from utils.nba_feature_importance import compute_permutation_importance

    market = "pts"
    feature_names = get_feature_cols(market)
    n_features = len(feature_names)

    model, X, y = _make_small_model(n_features)
    result = compute_permutation_importance(model, X, y, feature_names)

    assert len(result) == n_features


# ---------------------------------------------------------------------------
# Test 4: SHAP importance works when shap is available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("shap") is None,
    reason="shap not installed",
)
def test_shap_importance_when_available():
    from utils.nba_feature_importance import compute_shap_importance

    n_features = 6
    feature_names = [f"f{i}" for i in range(n_features)]
    model, X, _ = _make_small_model(n_features)

    result = compute_shap_importance(model, X, feature_names, market="pts")

    assert isinstance(result, pd.DataFrame)
    assert "feature" in result.columns
    assert "rank" in result.columns
    assert len(result) == n_features


# ---------------------------------------------------------------------------
# Test 5: Falls back to permutation / feature_importances_ when shap missing
# ---------------------------------------------------------------------------


def test_shap_fallback_when_shap_not_installed():
    """Monkeypatch shap import to simulate it not being installed."""
    # Patch the _SHAP_AVAILABLE flag in the module under test
    import utils.nba_feature_importance as fim

    original_flag = fim._SHAP_AVAILABLE
    try:
        fim._SHAP_AVAILABLE = False

        n_features = 6
        feature_names = [f"f{i}" for i in range(n_features)]
        model, X, _ = _make_small_model(n_features)

        result = fim.compute_shap_importance(model, X, feature_names, market="pts")

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert len(result) == n_features
        # Should still have rank and mean_abs_shap columns
        assert "rank" in result.columns
        assert "mean_abs_shap" in result.columns
    finally:
        fim._SHAP_AVAILABLE = original_flag


# ---------------------------------------------------------------------------
# Test 6: save_importance_snapshot writes correct number of rows to DB
# ---------------------------------------------------------------------------


def test_save_importance_snapshot_writes_rows(db):
    from utils.nba_feature_importance import compute_permutation_importance, save_importance_snapshot

    n_features = 6
    feature_names = [f"f{i}" for i in range(n_features)]
    model, X, y = _make_small_model(n_features)

    importance_df = compute_permutation_importance(model, X, y, feature_names)
    rows_written = save_importance_snapshot(importance_df, market="pts", game_date="2026-03-04")

    assert rows_written == n_features

    saved = read_dataframe(
        "SELECT * FROM nba_feature_importance_history WHERE market = ? AND game_date = ?",
        ("pts", "2026-03-04"),
    )
    assert len(saved) == n_features
    assert set(saved["feature"]) == set(feature_names)
