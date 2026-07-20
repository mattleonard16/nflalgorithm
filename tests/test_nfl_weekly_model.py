"""Tests for NFL weekly model rolling features and StackingRegressor ensemble."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from config import config
from models.position_specific import weekly as weekly_module
from models.position_specific.weekly import (
    MARKET_CONFIGS,
    ROLLING_WINDOWS,
    _build_nfl_model,
    _build_roster_week_data,
    _eligible_role_mask,
    _engineer_rolling_features,
    _iter_chronological_week_folds,
    _load_player_history_for_rolling,
    _load_training_data,
    get_nfl_feature_cols,
    predict_week,
    train_weekly_models,
)
from schema_migrations import MigrationManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Provide a fresh temp SQLite database with schema applied.

    Production code under test still reads config.database.path after the
    explicit migration target is initialized.
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
            rows.append(
                {
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
                    "passing_yards": (
                        max(0, 250 + p * 20 + rng.normal(0, 30)) if position == "QB" else 0.0
                    ),
                    "passing_attempts": max(0, 35 + rng.normal(0, 5)) if position == "QB" else 0.0,
                    "receiving_yards": max(0, 50 + p * 5 + rng.normal(0, 20)),
                    "receptions": max(0, 5 + rng.normal(0, 2)),
                    "targets": max(0, 7 + rng.normal(0, 2)),
                    "red_zone_touches": 2,
                    "target_share": 0.15,
                    "air_yards": 80.0,
                    "yac_yards": 30.0,
                    "game_script": rng.normal(0, 3),
                }
            )
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
            assert not late_rows[col].isna().all(), f"Column {col} still all NaN after week 4"

    def test_contextual_cols_filled(self):
        """Pregame contextual estimates should be present and numeric."""
        df = _make_player_stats(n_players=2, n_weeks=5)
        result = _engineer_rolling_features(df, "receiving_yards")
        for col in ["age", "expected_snap_count", "expected_snap_percentage"]:
            assert col in result.columns
            assert pd.api.types.is_numeric_dtype(
                result[col]
            ), f"Column {col} has non-numeric dtype {result[col].dtype}"

    def test_outcome_derived_context_is_lagged(self):
        """Week N features must not contain Week N usage or outcome context."""
        df = _make_player_stats(n_players=1, n_weeks=2)
        df.loc[
            df["week"] == 1,
            [
                "snap_count",
                "snap_percentage",
                "target_share",
                "air_yards",
                "yac_yards",
                "red_zone_touches",
                "game_script",
            ],
        ] = [10.0, 20.0, 0.10, 30.0, 12.0, 1.0, -4.0]
        df.loc[
            df["week"] == 2,
            [
                "snap_count",
                "snap_percentage",
                "target_share",
                "air_yards",
                "yac_yards",
                "red_zone_touches",
                "game_script",
            ],
        ] = [90.0, 95.0, 0.90, 200.0, 80.0, 8.0, 11.0]

        result = _engineer_rolling_features(df, "receiving_yards").set_index("week")

        assert result.loc[2, "expected_snap_count"] == pytest.approx(10.0)
        assert result.loc[2, "expected_snap_percentage"] == pytest.approx(20.0)
        assert result.loc[2, "expected_target_share"] == pytest.approx(0.10)
        assert result.loc[2, "expected_air_yards"] == pytest.approx(30.0)
        assert result.loc[2, "expected_yac_yards"] == pytest.approx(12.0)
        assert result.loc[2, "expected_red_zone_touches"] == pytest.approx(1.0)
        assert result.loc[2, "expected_game_script"] == pytest.approx(-4.0)

    def test_explicit_pregame_estimate_overrides_history(self):
        """Roster/injury role estimates should override a historical fallback."""
        df = _make_player_stats(n_players=1, n_weeks=2)
        df["expected_target_share"] = np.nan
        df.loc[df["week"] == 2, "expected_target_share"] = 0.42

        result = _engineer_rolling_features(df, "receiving_yards").set_index("week")

        assert result.loc[2, "expected_target_share"] == pytest.approx(0.42)

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

    def test_raw_outcome_context_is_not_a_model_feature(self):
        outcome_context = {
            "snap_count",
            "snap_percentage",
            "game_script",
            "target_share",
            "air_yards",
            "yac_yards",
            "red_zone_touches",
        }

        for market in MARKET_CONFIGS:
            assert outcome_context.isdisjoint(get_nfl_feature_cols(market))


class TestCausalTrainingWindows:
    def test_role_eligibility_uses_expected_not_realized_activity(self):
        df = pd.DataFrame(
            {
                "targets": [12.0, 0.0],
                "expected_targets": [0.0, 6.0],
            }
        )

        assert _eligible_role_mask(df, "receiving_yards").tolist() == [False, True]

    def test_chronological_folds_keep_future_weeks_out_of_training(self):
        rows = []
        for week in range(1, 7):
            for player_id in ("B", "A"):
                rows.append({"player_id": player_id, "season": 2024, "week": week})
        df = pd.DataFrame(rows).sort_values(["player_id", "week"]).reset_index(drop=True)

        folds = list(_iter_chronological_week_folds(df, max_splits=3))

        assert len(folds) == 3
        for train_idx, validation_idx in folds:
            train_periods = set(
                zip(df.loc[train_idx, "season"], df.loc[train_idx, "week"], strict=True)
            )
            validation_periods = set(
                zip(
                    df.loc[validation_idx, "season"],
                    df.loc[validation_idx, "week"],
                    strict=True,
                )
            )
            assert max(train_periods) < min(validation_periods)

    def test_training_loader_bounds_history_and_marks_requested_weeks(self, tmp_db):
        df = _make_player_stats(n_players=2, n_weeks=6)
        _insert_stats(tmp_db, df)

        loaded = _load_training_data([(2024, 3), (2024, 4)])

        assert loaded["week"].max() == 4
        marked = loaded.loc[loaded["_is_training_target"], ["season", "week"]]
        assert set(marked.itertuples(index=False, name=None)) == {(2024, 3), (2024, 4)}


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


def test_qb_decomposition_history_is_bounded_before_target_week(monkeypatch):
    captured: dict[str, object] = {}

    def capture_history(query, params=None):
        captured["query"] = query
        captured["params"] = params
        return pd.DataFrame({"passing_yards": [250.0], "passing_attempts": [35.0]})

    monkeypatch.setattr(weekly_module, "read_dataframe", capture_history)
    monkeypatch.setattr(weekly_module, "_lookup_pass_defense_rank", lambda *args: 16.0)
    monkeypatch.setattr(
        weekly_module,
        "decompose_qb_passing",
        lambda **kwargs: {
            "context_sensitivity": 0.1,
            "pass_attempts_predicted": 35.0,
            "yards_per_attempt_predicted": 7.0,
        },
    )

    weekly_module._enrich_with_decomposition(
        {},
        pd.Series(
            {
                "player_id": "NYJ_current_qb",
                "gsis_id": "00-0012345",
                "opponent": "BUF",
            }
        ),
        2026,
        1,
    )

    assert "season < ? OR (season = ? AND week < ?)" in str(captured["query"])
    assert captured["params"] == (
        "NYJ_current_qb",
        "00-0012345",
        "00-0012345",
        2026,
        2026,
        1,
    )


# ---------------------------------------------------------------------------
# train_weekly_models + predict_week integration test
# ---------------------------------------------------------------------------


class TestTrainAndPredict:
    def test_week_one_frame_uses_current_roster_and_stable_gsis_history(self, tmp_db, monkeypatch):
        history = _make_player_stats(n_players=1, n_weeks=2)
        history["player_id"] = "LV_season_ready"
        history["gsis_id"] = "00-0039999"
        history["name"] = "Season Ready"
        history["team"] = "LV"
        history["position"] = "WR"
        history["season"] = 2025
        history["week"] = [17, 18]
        history.loc[history["week"] == 18, ["targets", "receptions", "receiving_yards"]] = 0
        reserve_history = history.copy()
        reserve_history["player_id"] = "LV_reserve_player"
        reserve_history["gsis_id"] = "00-0038888"
        reserve_history["name"] = "Reserve Player"
        history = pd.concat([history, reserve_history], ignore_index=True)
        _insert_stats(tmp_db, history)

        with sqlite3.connect(tmp_db) as conn:
            conn.execute("""
                INSERT INTO nfl_roster_players
                    (season, gsis_id, player_id, player_name, team, position, roster_status,
                     updated_at)
                VALUES (2026, '00-0039999', 'NYJ_season_ready', 'Season Ready', 'NYJ', 'WR',
                        'ACT', 'now')
                """)
            conn.execute("""
                INSERT INTO nfl_roster_players
                    (season, gsis_id, player_id, player_name, team, position, roster_status,
                     updated_at)
                VALUES (2026, '00-0038888', 'NYJ_reserve_player', 'Reserve Player', 'NYJ',
                        'WR', 'RES', 'now')
                """)
            conn.execute("""
                INSERT INTO games
                    (game_id, season, week, home_team, away_team, game_date)
                VALUES ('2026_W1_BUF_at_NYJ', 2026, 1, 'NYJ', 'BUF', '2026-09-10')
                """)
            conn.execute("""
                INSERT INTO weekly_projections
                    (season, week, player_id, team, opponent, market, mu, sigma,
                     model_version, featureset_hash, generated_at)
                VALUES (2026, 1, 'NYJ_stale_player', 'NYJ', 'BUF', 'receiving_yards',
                        10, 5, 'old', 'old', 'before')
                """)

        frame = _build_roster_week_data(2026, 1)

        assert frame[["player_id", "gsis_id", "team", "opponent", "season", "week"]].to_dict(
            "records"
        ) == [
            {
                "player_id": "NYJ_season_ready",
                "gsis_id": "00-0039999",
                "team": "NYJ",
                "opponent": "BUF",
                "season": 2026,
                "week": 1,
            }
        ]
        assert frame.iloc[0]["targets"] == 0
        assert frame.iloc[0]["expected_targets"] > 0
        with sqlite3.connect(tmp_db) as conn:
            target_rows = conn.execute(
                "SELECT COUNT(*) FROM player_stats_enhanced WHERE season = 2026 AND week = 1"
            ).fetchone()[0]
        assert target_rows == 0

        aligned_history = _load_player_history_for_rolling(frame, 2026, 1)
        assert aligned_history["player_id"].unique().tolist() == ["NYJ_season_ready"]

        class FixedModel:
            def predict(self, values):
                return np.full(len(values), 72.5)

        monkeypatch.setattr(
            weekly_module, "_load_or_train_models", lambda: {"receiving_yards": FixedModel()}
        )
        monkeypatch.setattr(weekly_module, "get_defense_multiplier", lambda **kwargs: 1.0)

        predictions = predict_week(2026, 1)

        assert predictions[["player_id", "market", "mu"]].to_dict("records") == [
            {
                "player_id": "NYJ_season_ready",
                "market": "receiving_yards",
                "mu": 72.5,
            }
        ]
        with sqlite3.connect(tmp_db) as conn:
            written = conn.execute("""
                SELECT player_id, team, opponent, market
                FROM weekly_projections WHERE season = 2026 AND week = 1
                """).fetchall()
        assert written == [("NYJ_season_ready", "NYJ", "BUF", "receiving_yards")]

    def test_week_one_frame_includes_rookie_without_player_history(self, tmp_db):
        with sqlite3.connect(tmp_db) as conn:
            conn.execute("""
                INSERT INTO nfl_roster_players
                    (season, gsis_id, player_id, player_name, team, position, roster_status,
                     roster_week, updated_at)
                VALUES (2026, 'rookie', 'BUF_rookie_receiver', 'Rookie Receiver', 'BUF', 'WR',
                        'ACT', 1, '2026-09-01T00:00:00Z')
                """)
            conn.execute("""
                INSERT INTO games
                    (game_id, season, week, home_team, away_team, game_date)
                VALUES ('2026_W1_MIA_at_BUF', 2026, 1, 'BUF', 'MIA', '2026-09-10')
                """)
            conn.execute("""
                INSERT INTO nfl_player_context_snapshots
                    (season, week, gsis_id, player_id, team, position, roster_status,
                     depth_position, depth_rank, is_starter, injury_status, practice_status,
                     primary_injury, expected_snap_count, expected_snap_percentage,
                     expected_rushing_attempts, expected_targets, expected_passing_attempts,
                     expected_target_share, expected_air_yards, expected_yac_yards,
                     expected_red_zone_touches, expected_game_script, is_rookie, is_new_team,
                     uncertainty_multiplier, prior_source, source_updated_at, captured_at)
                VALUES
                    (2026, 1, 'rookie', 'BUF_rookie_receiver', 'BUF', 'WR', 'ACT',
                     'WR', 1, 1, NULL, NULL, NULL, 38, 58, 0, 5, 0, 0.15, 55, 18, 1, 0,
                     1, 0, 1.5, 'rookie_prior', '2026-09-01T00:00:00Z',
                     '2026-09-01T00:00:00Z')
                """)

        frame = _build_roster_week_data(2026, 1)

        assert frame[["player_id", "team", "opponent"]].to_dict("records") == [
            {"player_id": "BUF_rookie_receiver", "team": "BUF", "opponent": "MIA"}
        ]
        assert frame.iloc[0]["expected_targets"] == pytest.approx(5.0)
        assert frame.iloc[0]["uncertainty_multiplier"] == pytest.approx(1.5)
        assert frame.iloc[0]["is_rookie"] == 1

    def test_no_history_role_prior_generates_prediction_with_wider_uncertainty(self, monkeypatch):
        frame = pd.DataFrame(
            {
                "player_id": ["BUF_rookie_receiver"],
                "gsis_id": ["rookie"],
                "name": ["Rookie Receiver"],
                "team": ["BUF"],
                "opponent": ["MIA"],
                "position": ["WR"],
                "season": [2026],
                "week": [1],
                "age": [22],
                "expected_targets": [5.0],
                "expected_snap_percentage": [58.0],
                "expected_target_share": [0.15],
                "uncertainty_multiplier": [1.5],
            }
        )

        class FixedModel:
            def predict(self, values):
                return np.full(len(values), 55.0)

        monkeypatch.setattr(weekly_module, "_load_week_data", lambda season, week: frame)
        monkeypatch.setattr(
            weekly_module, "_load_or_train_models", lambda: {"receiving_yards": FixedModel()}
        )
        monkeypatch.setattr(
            weekly_module,
            "_load_player_history_for_rolling",
            lambda current, season, week: pd.DataFrame(),
        )
        monkeypatch.setattr(weekly_module, "compute_player_sigma", lambda history, market: 10.0)
        monkeypatch.setattr(weekly_module, "get_defense_multiplier", lambda **kwargs: 1.0)
        monkeypatch.setattr(weekly_module, "_write_predictions", lambda *args: None)

        predictions = predict_week(2026, 1)

        assert predictions[["player_id", "market", "mu"]].to_dict("records") == [
            {
                "player_id": "BUF_rookie_receiver",
                "market": "receiving_yards",
                "mu": 55.0,
            }
        ]
        assert predictions.iloc[0]["sigma"] == pytest.approx(15.0)

    def test_train_produces_model_files(self, tmp_db, tmp_model_dir):
        """Training should produce joblib files for each market with enough data."""
        df = _make_player_stats(n_players=5, n_weeks=12)
        _insert_stats(tmp_db, df)

        paths = train_weekly_models([(2024, w) for w in range(1, 11)])
        # At least one market should have trained (data-dependent)
        assert len(paths) >= 1
        for market, path in paths.items():
            assert Path(path).exists(), f"Model file missing for {market}: {path}"

    def test_saved_models_record_the_causal_feature_contract(self, tmp_db, tmp_model_dir):
        df = _make_player_stats(n_players=5, n_weeks=12)
        _insert_stats(tmp_db, df)

        paths = train_weekly_models([(2024, w) for w in range(1, 11)])

        assert paths
        for market, path in paths.items():
            bundle = joblib.load(path)
            assert bundle["model_version"] == "causal_asof_v1"
            assert bundle["feature_cols"] == get_nfl_feature_cols(market)
            assert hasattr(bundle["model"], "predict")

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

        # Model version records the causal as-of feature contract.
        if not predictions.empty:
            assert (predictions["model_version"] == "causal_asof_v1").all()

    def test_roster_backed_prediction_rejects_missing_market_models(
        self, monkeypatch, tmp_model_dir
    ):
        frame = pd.DataFrame(
            {
                "player_id": ["BUF_receiver"],
                "position": ["WR"],
                "season": [2026],
                "week": [1],
                "team": ["BUF"],
                "opponent": ["BAL"],
                "targets": [7.0],
                "expected_targets": [7.0],
                "receptions": [5.0],
                "receiving_yards": [60.0],
            }
        )

        class FixedModel:
            def predict(self, values):
                return np.full(len(values), 60.0)

        monkeypatch.setattr(weekly_module, "_build_roster_week_data", lambda season, week: frame)
        monkeypatch.setattr(
            weekly_module, "_load_or_train_models", lambda: {"receiving_yards": FixedModel()}
        )
        monkeypatch.setattr(
            weekly_module,
            "_load_player_history_for_rolling",
            lambda current, season, week: pd.DataFrame(),
        )
        monkeypatch.setattr(weekly_module, "get_defense_multiplier", lambda **kwargs: 1.0)

        with pytest.raises(RuntimeError, match="missing required markets"):
            predict_week(2026, 1, roster_backed=True)

    def test_empty_db_returns_empty_dataframe(self, tmp_db, tmp_model_dir):
        """predict_week on empty DB should return empty DataFrame gracefully."""
        predictions = predict_week(2024, 1)
        assert isinstance(predictions, pd.DataFrame)
        assert predictions.empty
