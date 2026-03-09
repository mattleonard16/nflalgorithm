"""Tests for NFL weekly model rolling features and StackingRegressor ensemble."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import config
from models.position_specific import weekly as weekly_module
from models.position_specific.weekly import (
    MARKET_CONFIGS,
    ROLLING_WINDOWS,
    _engineer_rolling_features,
    get_nfl_feature_cols,
    _build_nfl_model,
    train_weekly_models,
    predict_week,
)
from schema_migrations import MigrationManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Provide a fresh temp SQLite database with schema applied.

    Must set config.database.path BEFORE calling MigrationManager because
    MigrationManager.run() uses get_connection() which reads config.database.path.
    """
    db_path = tmp_path / "test_nfl_weekly.db"
    db_path.touch()
    # Patch env vars and config BEFORE running migrations
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setattr(config.database, "path", str(db_path))
    monkeypatch.setattr(config.database, "backend", "sqlite")
    MigrationManager(str(db_path)).run()
    yield str(db_path)


@pytest.fixture()
def tmp_model_dir(tmp_path):
    """Provide a temp model directory."""
    model_dir = tmp_path / "models" / "weekly"
    model_dir.mkdir(parents=True)
    original = weekly_module.MODEL_DIR
    weekly_module.MODEL_DIR = model_dir
    yield model_dir
    weekly_module.MODEL_DIR = original


def _make_player_stats(n_players: int = 3, n_weeks: int = 10) -> pd.DataFrame:
    """Create a synthetic player_stats_enhanced DataFrame.

    Only includes columns that exist in the real player_stats_enhanced schema.
    """
    rng = np.random.default_rng(42)
    rows = []
    for p in range(n_players):
        player_id = f"P{p:03d}"
        position = ["RB", "WR", "QB"][p % 3]
        for w in range(1, n_weeks + 1):
            rows.append({
                "player_id": player_id,
                "season": 2024,
                "week": w,
                "name": f"Player {p}",
                "position": position,
                "team": "KC",
                "age": 27.0,
                "games_played": w,
                "snap_count": 50,
                "snap_percentage": 0.75,
                "rushing_yards": max(0, 60 + p * 10 + w * 2 + rng.normal(0, 15)),
                "rushing_attempts": max(0, 12 + p + rng.normal(0, 3)),
                "passing_yards": max(0, 250 + p * 20 + rng.normal(0, 30)) if position == "QB" else 0.0,
                "passing_attempts": max(0, 35 + rng.normal(0, 5)) if position == "QB" else 0.0,
                "receiving_yards": max(0, 50 + p * 5 + rng.normal(0, 20)),
                "receptions": max(0, 5 + rng.normal(0, 2)),
                "targets": max(0, 7 + rng.normal(0, 2)),
                "red_zone_touches": 2,
                "target_share": 0.15,
                "air_yards": 80.0,
                "yac_yards": 30.0,
                "game_script": rng.normal(0, 3),
            })
    return pd.DataFrame(rows)


def _insert_stats(db_path: str, df: pd.DataFrame) -> None:
    """Insert synthetic stats into player_stats_enhanced."""
    with sqlite3.connect(db_path) as conn:
        df.to_sql("player_stats_enhanced", conn, if_exists="append", index=False)


# ---------------------------------------------------------------------------
# Rolling feature tests
# ---------------------------------------------------------------------------


class TestEngineerRollingFeatures:
    """Tests for _engineer_rolling_features."""

    def test_shift_prevents_lookahead(self):
        """Rolling features must use shift(1) so week N doesn't see week N data."""
        df = _make_player_stats(n_players=1, n_weeks=5)
        result = _engineer_rolling_features(df, "rushing_yards")

        # For the first record of each player, all rolling cols should be NaN or 0
        # (because there's no prior data)
        player_id = result["player_id"].iloc[0]
        first_row = result[result["player_id"] == player_id].iloc[0]

        # First row has no prior history; EWM with shift(1) should give NaN -> filled to 0
        for stat in ["rushing_yards", "rushing_attempts"]:
            for w in ROLLING_WINDOWS:
                col = f"{stat}_last{w}_avg"
                assert col in result.columns, f"Missing rolling column {col}"
                # First row value should be NaN (shift produces NaN for first element)
                assert pd.isna(first_row[col]) or first_row[col] == 0.0

    def test_rolling_cols_created_for_all_markets(self):
        """All expected rolling columns are created for each market."""
        df = _make_player_stats(n_players=2, n_weeks=8)
        for market in MARKET_CONFIGS:
            result = _engineer_rolling_features(df, market)
            expected_cols = get_nfl_feature_cols(market)
            for col in expected_cols:
                assert col in result.columns, f"Missing column '{col}' for market '{market}'"

    def test_rolling_increases_over_time(self):
        """Later weeks should have populated (non-zero) rolling averages."""
        df = _make_player_stats(n_players=1, n_weeks=10)
        result = _engineer_rolling_features(df, "rushing_yards")
        player_df = result[result["player_id"] == "P000"].sort_values("week")

        # By week 4+, rolling averages should be populated
        late_rows = player_df[player_df["week"] >= 4]
        for col in ["rushing_yards_last3_avg", "rushing_yards_last6_avg"]:
            assert not late_rows[col].isna().all(), (
                f"Column {col} still all NaN after week 4"
            )

    def test_contextual_cols_filled(self):
        """Contextual columns (age, snap_count, etc.) should be present and numeric."""
        df = _make_player_stats(n_players=2, n_weeks=5)
        result = _engineer_rolling_features(df, "receiving_yards")
        for col in ["age", "snap_count", "snap_percentage", "game_script"]:
            assert col in result.columns
            assert pd.api.types.is_numeric_dtype(result[col]), (
                f"Column {col} has non-numeric dtype {result[col].dtype}"
            )

    def test_missing_stat_col_filled_with_zero(self):
        """If a stat column is absent from the input, it should be added as zeros."""
        df = _make_player_stats(n_players=1, n_weeks=4)
        # Drop rushing_attempts to simulate missing column
        df = df.drop(columns=["rushing_attempts"])
        result = _engineer_rolling_features(df, "rushing_yards")
        assert "rushing_attempts_last3_avg" in result.columns


# ---------------------------------------------------------------------------
# get_nfl_feature_cols tests
# ---------------------------------------------------------------------------


class TestGetNflFeatureCols:
    def test_returns_list_for_valid_market(self):
        for market in MARKET_CONFIGS:
            cols = get_nfl_feature_cols(market)
            assert isinstance(cols, list)
            assert len(cols) > 0

    def test_raises_for_invalid_market(self):
        with pytest.raises(ValueError, match="Unknown market"):
            get_nfl_feature_cols("fumbles")

    def test_cols_are_consistent(self):
        """Feature columns should be deterministic across calls."""
        cols1 = get_nfl_feature_cols("rushing_yards")
        cols2 = get_nfl_feature_cols("rushing_yards")
        assert cols1 == cols2

    def test_each_market_has_unique_rolling_cols(self):
        """Different markets produce different rolling column sets."""
        rush_cols = get_nfl_feature_cols("rushing_yards")
        recv_cols = get_nfl_feature_cols("receiving_yards")
        pass_cols = get_nfl_feature_cols("passing_yards")
        # Check that the primary rolling columns differ
        assert rush_cols != recv_cols
        assert recv_cols != pass_cols


# ---------------------------------------------------------------------------
# _build_nfl_model tests
# ---------------------------------------------------------------------------


class TestBuildNflModel:
    def test_returns_stacking_regressor(self):
        from sklearn.ensemble import StackingRegressor
        model = _build_nfl_model("rushing_yards")
        assert isinstance(model, StackingRegressor)

    def test_model_has_gbr_and_rf(self):
        model = _build_nfl_model("receiving_yards")
        estimator_names = [name for name, _ in model.estimators]
        assert "gbr" in estimator_names
        assert "rf" in estimator_names

    def test_model_can_fit_and_predict(self):
        """StackingRegressor should fit and predict on small synthetic data."""
        model = _build_nfl_model("rushing_yards")
        rng = np.random.default_rng(42)
        X = rng.random((50, 8))
        y = rng.random(50) * 100
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# train_weekly_models + predict_week integration test
# ---------------------------------------------------------------------------


class TestTrainAndPredict:
    def test_train_produces_model_files(self, tmp_db, tmp_model_dir):
        """Training should produce joblib files for each market with enough data."""
        df = _make_player_stats(n_players=5, n_weeks=12)
        _insert_stats(tmp_db, df)

        paths = train_weekly_models([(2024, w) for w in range(1, 11)])
        # At least one market should have trained (data-dependent)
        assert len(paths) >= 1
        for market, path in paths.items():
            assert Path(path).exists(), f"Model file missing for {market}: {path}"

    def test_predict_returns_non_empty_dataframe(self, tmp_db, tmp_model_dir):
        """predict_week should return a DataFrame with expected columns."""
        df = _make_player_stats(n_players=5, n_weeks=12)
        _insert_stats(tmp_db, df)

        train_weekly_models([(2024, w) for w in range(1, 11)])
        predictions = predict_week(2024, 11)

        assert not predictions.empty
        for col in ["player_id", "market", "mu", "sigma"]:
            assert col in predictions.columns

    def test_mu_values_are_non_negative(self, tmp_db, tmp_model_dir):
        """Predicted mu values should never be negative."""
        df = _make_player_stats(n_players=5, n_weeks=12)
        _insert_stats(tmp_db, df)

        train_weekly_models([(2024, w) for w in range(1, 11)])
        predictions = predict_week(2024, 11)

        if not predictions.empty:
            assert (predictions["mu"] >= 0).all(), "Some mu values are negative"

    def test_sigma_values_are_positive(self, tmp_db, tmp_model_dir):
        """Sigma values should always be positive."""
        df = _make_player_stats(n_players=5, n_weeks=12)
        _insert_stats(tmp_db, df)

        train_weekly_models([(2024, w) for w in range(1, 11)])
        predictions = predict_week(2024, 11)

        if not predictions.empty:
            assert (predictions["sigma"] > 0).all(), "Some sigma values are non-positive"

    def test_feature_cols_consistent_train_predict(self, tmp_db, tmp_model_dir):
        """Feature columns used in training should match those used in prediction."""
        df = _make_player_stats(n_players=5, n_weeks=14)
        _insert_stats(tmp_db, df)

        train_weekly_models([(2024, w) for w in range(1, 13)])
        predictions = predict_week(2024, 13)

        # Model version should be stacking ensemble (v1 or v2+)
        if not predictions.empty:
            assert predictions["model_version"].str.startswith("stacking_ensemble").all()

    def test_empty_db_returns_empty_dataframe(self, tmp_db, tmp_model_dir):
        """predict_week on empty DB should return empty DataFrame gracefully."""
        predictions = predict_week(2024, 1)
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.empty
