"""Tests for NBA projection models.

- stat_model.py    (multi-market: pts/reb/ast/fg3m)

Uses a fresh SQLite DB with seeded game logs. No NBA.com calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from schema_migrations import MigrationManager
from utils.db import execute, executemany, read_dataframe


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_model.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _seed_game_logs(n_games: int = 20, player_id: int = 1628369) -> None:
    """Insert synthetic game logs for one player."""
    rows = []
    for i in range(n_games):
        rows.append(
            (
                player_id,
                "Test Player",
                "BOS",
                2024,
                f"002240{i:04d}",
                f"2025-01-{i + 1:02d}" if i < 28 else f"2025-02-{i - 27:02d}",
                "BOS vs. MIA" if i % 2 == 0 else "BOS @ MIA",
                "W",
                float(34 + (i % 5)),
                int(20 + (i % 15)),  # pts varies 20-34
                int(5 + (i % 6)),
                int(4 + (i % 4)),
                int(2 + (i % 3)),
                int(8 + (i % 5)),
                int(15 + (i % 6)),
                int(4 + (i % 3)),
                int(6 + (i % 3)),
                1,
                1,
                2,
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
# Multi-market model tests (models/nba/stat_model.py)
# ---------------------------------------------------------------------------

MARKET_RANGES = {
    "pts": (5, 60),
    "reb": (0, 25),
    "ast": (0, 20),
    "fg3m": (0, 12),
}

ALL_MARKETS = list(MARKET_RANGES.keys())


class TestMultiMarketModel:
    """Tests for the generalised stat_model.py (pts/reb/ast/fg3m)."""

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def test_engineer_features_adds_stat_specific_rolling_cols(self, db):
        """_engineer_features(df, market) must produce <stat>_last5_avg columns."""
        from models.nba.stat_model import _engineer_features

        _seed_game_logs(20)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs"
        )
        for market in ALL_MARKETS:
            result = _engineer_features(df, market=market)
            assert f"{market}_last5_avg" in result.columns, (
                f"Missing {market}_last5_avg for market={market}"
            )
            assert f"{market}_last10_avg" in result.columns, (
                f"Missing {market}_last10_avg for market={market}"
            )
            # Contextual features must always be present
            assert "home_game" in result.columns
            assert "b2b" in result.columns
            assert "days_rest" in result.columns

    def test_rolling_avg_uses_shift_no_leakage(self, db):
        """Rolling averages must not include the same row's value (shift=1)."""
        from models.nba.stat_model import _engineer_features

        _seed_game_logs(15)
        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs"
        )
        result = _engineer_features(df, market="pts").dropna(
            subset=["pts_last5_avg", "pts"]
        )
        for _, row in result.head(5).iterrows():
            assert not np.isnan(row["pts_last5_avg"])

    # ------------------------------------------------------------------
    # Training — individual markets
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("market", ALL_MARKETS)
    def test_train_saves_model_file(self, db, tmp_path, monkeypatch, market):
        """train(market) must write a .joblib model file and a shared encoder file."""
        import joblib

        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)
        monkeypatch.setattr(
            "models.nba.stat_model.ENCODER_PATH", model_dir / "team_encoder.joblib"
        )

        _seed_game_logs(30)
        from models.nba.stat_model import train

        train(market=market)

        model_path = model_dir / f"{market}_model.joblib"
        encoder_path = model_dir / "team_encoder.joblib"
        assert model_path.exists(), f"Model file not found for market={market}"
        assert encoder_path.exists(), f"Encoder file not found for market={market}"

        model = joblib.load(model_path)
        assert hasattr(model, "predict"), "Saved object must have a predict() method"

    @pytest.mark.parametrize("market", ALL_MARKETS)
    def test_train_no_data_does_not_raise(self, db, tmp_path, monkeypatch, market):
        """train(market) with empty DB must log an error but not raise."""
        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)
        monkeypatch.setattr(
            "models.nba.stat_model.ENCODER_PATH", model_dir / "team_encoder.joblib"
        )

        from models.nba.stat_model import train

        train(market=market)  # no data seeded — must not raise
        model_path = model_dir / f"{market}_model.joblib"
        assert not model_path.exists(), "No model should be written when data is absent"

    # ------------------------------------------------------------------
    # Training — all markets at once (via CLI main())
    # ------------------------------------------------------------------

    def test_train_all_markets_via_cli(self, db, tmp_path, monkeypatch):
        """CLI with --market all must train and save models for every market."""
        import sys

        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)
        monkeypatch.setattr(
            "models.nba.stat_model.ENCODER_PATH", model_dir / "team_encoder.joblib"
        )

        _seed_game_logs(30)

        # Simulate CLI invocation with --train --market all
        monkeypatch.setattr(sys, "argv", ["stat_model.py", "--train", "--market", "all"])
        from models.nba.stat_model import main

        main()

        for market in ALL_MARKETS:
            model_path = model_dir / f"{market}_model.joblib"
            assert model_path.exists(), (
                f"Model file missing after --market all: {market}"
            )

    # ------------------------------------------------------------------
    # Invalid market
    # ------------------------------------------------------------------

    def test_invalid_market_raises_value_error(self, db, tmp_path, monkeypatch):
        """Passing an unknown market string must raise ValueError from train()."""
        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)
        monkeypatch.setattr(
            "models.nba.stat_model.ENCODER_PATH", model_dir / "team_encoder.joblib"
        )

        _seed_game_logs(30)
        from models.nba.stat_model import train

        with pytest.raises(ValueError, match="(?i)(market|unknown)"):
            train(market="touchdowns")

    # ------------------------------------------------------------------
    # Prediction ranges
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("market,expected_range", list(MARKET_RANGES.items()))
    def test_model_predicts_in_reasonable_range(
        self, db, tmp_path, monkeypatch, market, expected_range
    ):
        """After training, in-sample predictions should be within realistic NBA bounds."""
        import joblib

        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        encoder_path = model_dir / "team_encoder.joblib"
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)
        monkeypatch.setattr("models.nba.stat_model.ENCODER_PATH", encoder_path)

        _seed_game_logs(30)
        from models.nba.stat_model import train, _engineer_features, _encode_opponents, get_feature_cols

        train(market=market)

        model = joblib.load(model_dir / f"{market}_model.joblib")
        encoder = joblib.load(encoder_path)

        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs"
        )
        df = _engineer_features(df, market=market)
        df, _ = _encode_opponents(df, encoder=encoder)
        feature_cols = get_feature_cols(market)
        df = df.dropna(subset=feature_cols)

        preds = model.predict(df[feature_cols].values)
        lo, hi = expected_range
        assert all(lo <= p <= hi for p in preds), (
            f"[{market}] Out-of-range prediction: {preds.min():.2f} - {preds.max():.2f}"
        )

    # ------------------------------------------------------------------
    # nba_projections writes correct market column
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("market", ALL_MARKETS)
    def test_projections_written_with_correct_market(
        self, db, tmp_path, monkeypatch, market
    ):
        """predict(market) must insert rows into nba_projections with the correct market."""
        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        encoder_path = model_dir / "team_encoder.joblib"
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)
        monkeypatch.setattr("models.nba.stat_model.ENCODER_PATH", encoder_path)

        _seed_game_logs(30)
        from models.nba.stat_model import train, predict

        train(market=market)

        # Patch _get_todays_games so predict() doesn't hit NBA.com.
        # Return a fake game that matches the seeded team (BOS).
        fake_games = [
            {
                "game_id": "fake001",
                "game_date": "2025-01-16",  # date after last seeded game
                "home_team": "BOS",
                "away_team": "MIA",
            }
        ]
        monkeypatch.setattr("models.nba.stat_model._get_todays_games", lambda: fake_games)

        predict(market=market)

        rows = read_dataframe(
            "SELECT market FROM nba_projections WHERE market = ?",
            [market],
        )
        assert len(rows) > 0, f"No rows in nba_projections for market={market}"
        assert (rows["market"] == market).all(), "market column has wrong value"
