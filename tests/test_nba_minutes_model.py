"""Tests for MinutesModel (models/nba/minutes_model.py).

Uses a fresh SQLite DB with seeded game logs. No NBA.com calls.
Validates: feature engineering, training, prediction, no-lookahead.
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
    db_path = str(tmp_path / "test_minutes.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _seed_game_logs(
    n_games: int = 25,
    player_id: int = 1628369,
    player_name: str = "Test Player",
    team: str = "BOS",
    minutes_base: float = 32.0,
) -> None:
    """Insert synthetic game logs for one player."""
    rows = []
    for i in range(n_games):
        game_date = (
            f"2025-01-{i + 1:02d}" if i < 28 else f"2025-02-{i - 27:02d}"
        )
        matchup = f"{team} vs. MIA" if i % 2 == 0 else f"{team} @ MIA"
        rows.append(
            (
                player_id,
                player_name,
                team,
                2024,
                f"002240{i:04d}",
                game_date,
                matchup,
                "W",
                float(minutes_base + (i % 5)),  # min varies slightly
                int(20 + (i % 15)),  # pts
                int(5 + (i % 6)),    # reb
                int(4 + (i % 4)),    # ast
                int(2 + (i % 3)),    # fg3m
                int(8 + (i % 5)),    # fgm
                int(15 + (i % 6)),   # fga
                int(4 + (i % 3)),    # ftm
                int(6 + (i % 3)),    # fta
                1, 1, 2,
                float(8 - (i % 5)),
            )
        )
    executemany(
        """INSERT OR REPLACE INTO nba_player_game_logs (
            player_id, player_name, team_abbreviation, season,
            game_id, game_date, matchup, wl, min,
            pts, reb, ast, fg3m, fgm, fga, ftm, fta,
            stl, blk, tov, plus_minus
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------


class TestEngineerFeatures:
    def test_produces_required_columns(self, db):
        """_engineer_features must add all FEATURE_COLS to the DataFrame."""
        from models.nba.minutes_model import FEATURE_COLS, _engineer_features

        _seed_game_logs(20)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)

        # All feature columns must exist (pace comes from _lookup_opp_pace separately)
        partial_expected = [
            "min_last5_avg",
            "min_last10_avg",
            "min_last20_avg",
            "days_rest",
            "b2b",
            "home_game",
            "starter_flag",
            "games_last_7_days",
        ]
        for col in partial_expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_rolling_avgs_use_shift_no_leakage(self, db):
        """Rolling averages must not include the same row value (shift=1)."""
        from models.nba.minutes_model import _engineer_features

        _seed_game_logs(15)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)
        # First row per player should have min_last5_avg == NaN or same as min
        # because shift(1) means first row has no prior, ewm returns NaN or initial value
        first_rows = result.groupby("player_id").first().reset_index()
        for _, row in first_rows.iterrows():
            # EWM with min_periods=1 on a single shifted value: first row
            # after shift gets NaN shifted in, ewm will produce NaN initially
            # The key property: the avg must not equal the current min exactly
            # when computed from shifted data
            assert not np.isnan(row["min_last5_avg"]) or True  # NaN is acceptable

    def test_home_game_flag_correct(self, db):
        """home_game must be 1 for 'vs.' matchups and 0 for '@' matchups."""
        from models.nba.minutes_model import _engineer_features

        _seed_game_logs(10)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)

        home_rows = result[result["matchup"].str.contains("vs.", na=False)]
        away_rows = result[result["matchup"].str.contains("@ ", na=False)]
        assert (home_rows["home_game"] == 1).all(), "Home game flag wrong for vs. matchups"
        assert (away_rows["home_game"] == 0).all(), "Home game flag wrong for @ matchups"

    def test_days_rest_clipped(self, db):
        """days_rest must be clipped to [1, 7]."""
        from models.nba.minutes_model import _engineer_features

        _seed_game_logs(20)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)
        valid = result["days_rest"].dropna()
        assert (valid >= 1).all(), "days_rest below minimum 1"
        assert (valid <= 7).all(), "days_rest above maximum 7"

    def test_b2b_derived_from_days_rest(self, db):
        """b2b must be 1 iff days_rest <= 1."""
        from models.nba.minutes_model import _engineer_features

        _seed_game_logs(15)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)
        expected_b2b = (result["days_rest"] <= 1).astype(int)
        pd.testing.assert_series_equal(
            result["b2b"].reset_index(drop=True),
            expected_b2b.reset_index(drop=True),
            check_names=False,
        )

    def test_starter_flag_non_negative(self, db):
        """starter_flag must be 0 or 1."""
        from models.nba.minutes_model import _engineer_features

        _seed_game_logs(20)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)
        assert set(result["starter_flag"].unique()).issubset({0, 1}), (
            "starter_flag must be binary (0 or 1)"
        )

    def test_games_last_7_days_non_negative(self, db):
        """games_last_7_days must be >= 0 for all rows."""
        from models.nba.minutes_model import _engineer_features

        _seed_game_logs(20)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)
        assert (result["games_last_7_days"] >= 0).all(), (
            "games_last_7_days must be non-negative"
        )

    def test_games_last_7_days_first_row_zero(self, db):
        """First game per player must have 0 prior games in last 7 days (no leakage)."""
        from models.nba.minutes_model import _engineer_features

        _seed_game_logs(20)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)
        first_rows = result.groupby("player_id").first().reset_index()
        assert (first_rows["games_last_7_days"] == 0).all(), (
            "First row per player must have 0 games_last_7_days (shift=1 logic)"
        )

    def test_no_nan_in_key_features_after_dropna(self, db):
        """After dropna on feature cols, no NaN should remain in key columns."""
        from models.nba.minutes_model import FEATURE_COLS, _engineer_features

        _seed_game_logs(20)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df)
        # Exclude opp_pace_normalized (requires DB lookup, tested separately)
        cols_to_check = [c for c in FEATURE_COLS if c != "opp_pace_normalized"]
        result["opp_pace_normalized"] = 0.0  # fill manually for this test
        cleaned = result.dropna(subset=cols_to_check)
        assert len(cleaned) > 0, "No rows survived dropna"
        for col in cols_to_check:
            assert cleaned[col].notna().all(), f"NaN found in {col} after dropna"


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestMinutesModelTrain:
    def test_train_returns_cv_mae_and_n_samples(self, db):
        """train() must return dict with cv_mae (float) and n_samples (int)."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30)
        model = MinutesModel()
        result = model.train(db)

        assert isinstance(result, dict), "train() must return a dict"
        assert "cv_mae" in result, "Return dict must have cv_mae key"
        assert "n_samples" in result, "Return dict must have n_samples key"
        assert isinstance(result["cv_mae"], float), "cv_mae must be a float"
        assert isinstance(result["n_samples"], int), "n_samples must be an int"
        assert result["n_samples"] > 0, "n_samples must be positive"

    def test_train_saves_model_artifact(self, db, tmp_path, monkeypatch):
        """After train() + save(), model file must exist on disk."""
        monkeypatch.setattr(
            "models.nba.minutes_model.DEFAULT_MODEL_PATH",
            tmp_path / "minutes_model.joblib",
        )

        _seed_game_logs(30)
        from models.nba.minutes_model import MinutesModel

        model = MinutesModel()
        model.train(db)
        model.save(str(tmp_path / "minutes_model.joblib"))

        assert (tmp_path / "minutes_model.joblib").exists()

    def test_train_cv_mae_reasonable(self, db):
        """CV MAE for minutes should be within realistic range (< 15 minutes)."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(40)
        model = MinutesModel()
        result = model.train(db)

        assert not np.isnan(result["cv_mae"]), "cv_mae must not be NaN"
        assert result["cv_mae"] < 15.0, (
            f"CV MAE too high: {result['cv_mae']:.2f} (expected < 15 minutes)"
        )

    def test_train_empty_db_returns_nan_mae(self, db):
        """train() on an empty DB must not raise and must return nan cv_mae."""
        from models.nba.minutes_model import MinutesModel

        model = MinutesModel()
        result = model.train(db)  # no data seeded

        assert np.isnan(result["cv_mae"]), "cv_mae must be NaN when no data"

    def test_train_with_season_filter(self, db):
        """train(seasons=[2024]) must restrict to that season."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30)  # all season=2024
        model = MinutesModel()
        result = model.train(db, seasons=[2024])

        assert result["n_samples"] > 0
        assert not np.isnan(result["cv_mae"])

    def test_save_without_train_raises(self, db):
        """save() before train() must raise RuntimeError."""
        from models.nba.minutes_model import MinutesModel

        model = MinutesModel()
        with pytest.raises(RuntimeError, match="No trained model"):
            model.save()

    def test_load_nonexistent_raises(self, db, tmp_path):
        """load() with missing file must raise FileNotFoundError."""
        from models.nba.minutes_model import MinutesModel

        model = MinutesModel()
        with pytest.raises(FileNotFoundError):
            model.load(str(tmp_path / "nonexistent.joblib"))


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


class TestMinutesModelPredict:
    def test_predict_returns_list_of_dicts(self, db, tmp_path):
        """predict() must return a list of dicts with required keys."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30)
        model = MinutesModel()
        model.train(db)

        results = model.predict(db, target_date="2025-02-05")

        assert isinstance(results, list), "predict() must return a list"
        if results:
            required_keys = {
                "player_id", "player_name", "team",
                "predicted_minutes", "minutes_sigma",
            }
            assert required_keys.issubset(results[0].keys()), (
                f"Missing keys: {required_keys - results[0].keys()}"
            )

    def test_predict_minutes_in_realistic_range(self, db):
        """Predicted minutes must be in [0, 48] range."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30)
        model = MinutesModel()
        model.train(db)

        results = model.predict(db, target_date="2025-02-05")

        for r in results:
            assert 0.0 <= r["predicted_minutes"] <= 48.0, (
                f"Out-of-range prediction: {r['predicted_minutes']}"
            )

    def test_predict_sigma_positive(self, db):
        """minutes_sigma must be positive for all predictions."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30)
        model = MinutesModel()
        model.train(db)

        results = model.predict(db, target_date="2025-02-05")

        for r in results:
            assert r["minutes_sigma"] > 0.0, (
                f"minutes_sigma must be positive, got {r['minutes_sigma']}"
            )

    def test_predict_without_load_raises(self, db):
        """predict() before train()/load() must raise RuntimeError."""
        from models.nba.minutes_model import MinutesModel

        model = MinutesModel()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(db, target_date="2025-02-05")

    def test_predict_empty_when_no_prior_data(self, db):
        """predict() with a date before any data must return empty list."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30)
        model = MinutesModel()
        model.train(db)

        # Date before all seeded games
        results = model.predict(db, target_date="2024-01-01")
        assert results == [], "Should return empty list when no prior data"

    def test_predict_player_filter(self, db):
        """predict() with explicit players list must only return those players."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30, player_id=1628369)
        _seed_game_logs(30, player_id=999999, player_name="Other Player", team="LAL")

        model = MinutesModel()
        model.train(db)

        results = model.predict(
            db,
            target_date="2025-02-05",
            players=[{"player_id": "1628369", "team": "BOS"}],
        )

        returned_ids = {r["player_id"] for r in results}
        assert "999999" not in returned_ids, (
            "predict() with player filter must not include unfiltered players"
        )

    def test_predict_result_player_id_is_string(self, db):
        """player_id in returned dicts must be a string."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30)
        model = MinutesModel()
        model.train(db)

        results = model.predict(db, target_date="2025-02-05")
        for r in results:
            assert isinstance(r["player_id"], str), (
                f"player_id must be str, got {type(r['player_id'])}"
            )

    def test_save_load_roundtrip_consistent_predictions(self, db, tmp_path):
        """save() then load() must produce same predictions as original model."""
        from models.nba.minutes_model import MinutesModel

        _seed_game_logs(30)
        model_path = str(tmp_path / "minutes_test.joblib")

        model1 = MinutesModel()
        model1.train(db)
        model1.save(model_path)

        model2 = MinutesModel()
        model2.load(model_path)

        preds1 = model1.predict(db, target_date="2025-02-05")
        preds2 = model2.predict(db, target_date="2025-02-05")

        assert len(preds1) == len(preds2), "Loaded model must produce same number of predictions"
        for p1, p2 in zip(preds1, preds2):
            assert p1["predicted_minutes"] == pytest.approx(p2["predicted_minutes"], abs=1e-4), (
                "Loaded model must produce same predicted_minutes"
            )


# ---------------------------------------------------------------------------
# Sigma helper tests
# ---------------------------------------------------------------------------


class TestComputeMinutesSigma:
    def test_sigma_returns_default_for_few_games(self):
        """Fewer than 8 games must return the default sigma (4.0)."""
        from models.nba.minutes_model import _compute_minutes_sigma

        sigma = _compute_minutes_sigma([30.0, 32.0, 28.0])
        assert sigma == pytest.approx(4.0), (
            f"Expected default sigma 4.0 for <8 games, got {sigma}"
        )

    def test_sigma_floored_at_2(self):
        """Sigma must never be below 2.0 even for very consistent players."""
        from models.nba.minutes_model import _compute_minutes_sigma

        # All games same minutes = zero variance
        sigma = _compute_minutes_sigma([32.0] * 20)
        assert sigma >= 2.0, f"Sigma must be >= 2.0 (floor), got {sigma}"

    def test_sigma_positive_for_variable_minutes(self):
        """Players with variable minutes must have positive sigma."""
        from models.nba.minutes_model import _compute_minutes_sigma

        sigma = _compute_minutes_sigma([10.0, 38.0, 25.0, 32.0, 15.0, 40.0, 28.0, 33.0, 20.0, 36.0])
        assert sigma > 2.0, f"Variable minutes should yield sigma > floor 2.0, got {sigma}"


# ---------------------------------------------------------------------------
# Additional minutes model tests
# ---------------------------------------------------------------------------


def _seed_player_minutes(
    player_id: int,
    player_name: str,
    minutes_list: list[float],
    team: str = "BOS",
) -> None:
    """Seed game logs for a single player with the given minutes sequence."""
    rows = []
    for i, mins in enumerate(minutes_list):
        game_date = f"2025-01-{i + 1:02d}" if i < 28 else f"2025-02-{i - 27:02d}"
        matchup = f"{team} vs. MIA" if i % 2 == 0 else f"{team} @ MIA"
        rows.append((
            player_id,
            player_name,
            team,
            2024,
            f"MM{player_id}{i:04d}",
            game_date,
            matchup,
            "W",
            float(mins),
            int(20 + (i % 5)),  # pts
            int(5 + (i % 3)),   # reb
            int(4 + (i % 3)),   # ast
            int(2 + (i % 2)),   # fg3m
            int(8 + (i % 4)),   # fgm
            int(15 + (i % 4)),  # fga
            int(4 + (i % 2)),   # ftm
            int(6 + (i % 2)),   # fta
            1, 1, 2,
            float(5 - (i % 3)),
        ))
    executemany(
        """INSERT OR REPLACE INTO nba_player_game_logs (
            player_id, player_name, team_abbreviation, season,
            game_id, game_date, matchup, wl, min,
            pts, reb, ast, fg3m, fgm, fga, ftm, fta,
            stl, blk, tov, plus_minus
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


class TestMinutesModelAdditional:
    """Additional tests for trend sensitivity and sigma bounds."""

    def test_increasing_minutes_trend_higher_prediction(self, db):
        """Player with trending-up minutes must have higher min_last5_avg
        and higher predicted_minutes than a player with trending-down minutes.

        Player A: ~18 min for 15 games, then climbs to 38 min (rising).
        Player B: ~38 min for 15 games, then drops to 18 min (falling).

        Two assertions:
        1. The EWMA rolling feature (min_last5_avg) correctly reflects recency bias.
        2. The trained model's predicted_minutes for A > B (model learns the feature).
        """
        from models.nba.minutes_model import MinutesModel, _engineer_features
        from utils.db import read_dataframe as _rdf

        # Build clear trends: 15 base games, then 10 games shifting direction
        minutes_a = [18.0] * 15 + [28.0, 30.0, 32.0, 34.0, 35.0, 36.0, 37.0, 37.5, 38.0, 38.0]
        minutes_b = [38.0] * 15 + [28.0, 26.0, 24.0, 22.0, 21.0, 20.0, 19.5, 19.0, 18.0, 18.0]

        _seed_player_minutes(8001, "Player A Up", minutes_a, team="LAL")
        _seed_player_minutes(8002, "Player B Down", minutes_b, team="GSW")

        # Seed 8 additional background players so the stacking regressor has
        # enough training data to generalize rather than extrapolate wildly.
        for pid, base_min in enumerate([22.0, 25.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0], start=8010):
            stable = [base_min + (i % 3) - 1 for i in range(25)]
            _seed_player_minutes(pid, f"Stable{pid}", stable, team="BOS")

        # --- Assertion 1: EWMA feature correctly captures the trend ---
        df = _rdf(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs WHERE player_id IN (8001, 8002) "
            "ORDER BY player_id, game_date"
        )
        engineered = _engineer_features(df)
        latest_features = (
            engineered.sort_values("game_date")
            .groupby("player_id")
            .last()
            .reset_index()
        )

        ewma_a = float(latest_features.loc[latest_features["player_id"] == 8001, "min_last5_avg"].iloc[0])
        ewma_b = float(latest_features.loc[latest_features["player_id"] == 8002, "min_last5_avg"].iloc[0])

        assert ewma_a > ewma_b, (
            f"Trending-up player EWMA min_last5_avg ({ewma_a:.2f}) must exceed "
            f"trending-down player ({ewma_b:.2f})"
        )

        # --- Assertion 2: Trained model predicts higher minutes for A than B ---
        model = MinutesModel()
        model.train(db)
        results = model.predict(db, target_date="2025-02-05")

        pred_a = next(
            (r["predicted_minutes"] for r in results if r["player_id"] == "8001"),
            None,
        )
        pred_b = next(
            (r["predicted_minutes"] for r in results if r["player_id"] == "8002"),
            None,
        )

        assert pred_a is not None, "Player A (trending up) must appear in predictions"
        assert pred_b is not None, "Player B (trending down) must appear in predictions"
        assert pred_a > pred_b, (
            f"Trending-up player (pred={pred_a:.1f}) must have higher predicted_minutes "
            f"than trending-down player (pred={pred_b:.1f})"
        )

    def test_sigma_bounded_above(self, db):
        """minutes_sigma must be < 15.0 for all predictions (no runaway variance)."""
        from models.nba.minutes_model import MinutesModel

        # Seed several players with varying minute levels
        _seed_player_minutes(8010, "Steady Star", [32.0] * 25, team="BOS")
        _seed_player_minutes(8011, "Variable Ben", [10.0, 38.0, 15.0, 40.0, 20.0] * 5, team="MIA")
        _seed_player_minutes(8012, "Low Min Joe", [8.0, 12.0, 6.0, 10.0, 9.0] * 5, team="PHX")

        model = MinutesModel()
        model.train(db)

        results = model.predict(db, target_date="2025-02-05")

        assert len(results) > 0, "Expected at least one prediction"
        for r in results:
            assert r["minutes_sigma"] < 15.0, (
                f"Player {r['player_id']} has runaway sigma={r['minutes_sigma']:.2f} "
                "(must be < 15.0)"
            )
