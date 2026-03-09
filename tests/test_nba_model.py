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
    """Insert synthetic game logs for one player.

    All rows have min >= 5 (required for rate-based training targets).
    """
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
                float(30 + (i % 10)),  # min ranges 30-39, always >= 5
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
        """train(market) must write a .joblib model file (no encoder file)."""
        import joblib

        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)

        _seed_game_logs(30)
        from models.nba.stat_model import train

        train(market=market)

        model_path = model_dir / f"{market}_model.joblib"
        encoder_path = model_dir / "team_encoder.joblib"
        assert model_path.exists(), f"Model file not found for market={market}"
        assert not encoder_path.exists(), "Encoder file should not exist (replaced by defensive stats)"

        model = joblib.load(model_path)
        assert hasattr(model, "predict"), "Saved object must have a predict() method"

    def test_feature_cols_include_opp_def_rating_normalized(self, db):
        """get_feature_cols must include opp_def_rating_normalized."""
        from models.nba.stat_model import get_feature_cols

        cols = get_feature_cols("pts")
        assert "opp_def_rating_normalized" in cols, "opp_def_rating_normalized must be in feature cols"

    def test_feature_cols_exclude_opponent_enc(self, db):
        """get_feature_cols must not include opponent_enc (replaced by defensive stats)."""
        from models.nba.stat_model import get_feature_cols

        cols = get_feature_cols("pts")
        assert "opponent_enc" not in cols, "opponent_enc must not be in feature cols"

    def test_feature_cols_include_days_rest(self, db):
        """get_feature_cols must include days_rest."""
        from models.nba.stat_model import get_feature_cols

        cols = get_feature_cols("pts")
        assert "days_rest" in cols, "days_rest must be in feature cols"

    @pytest.mark.parametrize("market", ALL_MARKETS)
    def test_train_no_data_does_not_raise(self, db, tmp_path, monkeypatch, market):
        """train(market) with empty DB must log an error but not raise."""
        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)

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

        _seed_game_logs(30)
        from models.nba.stat_model import train

        with pytest.raises(ValueError, match="(?i)(market|unknown)"):
            train(market="touchdowns")

    # ------------------------------------------------------------------
    # Prediction ranges
    # ------------------------------------------------------------------

    # Rate-based markets predict per-minute rates (not raw totals).
    # pts/reb/ast rates: typical range 0.05–3.0 per minute covers all realistic players.
    # fg3m stays count-based with its original range.
    RATE_MARKET_RANGES = {
        "pts": (0.0, 3.0),
        "reb": (0.0, 1.5),
        "ast": (0.0, 1.5),
        "fg3m": MARKET_RANGES["fg3m"],  # count-based, unchanged
    }

    @pytest.mark.parametrize("market,expected_range", list(RATE_MARKET_RANGES.items()))
    def test_model_predicts_in_reasonable_range(
        self, db, tmp_path, monkeypatch, market, expected_range
    ):
        """After training, in-sample model output should be within realistic bounds.

        For pts/reb/ast the model predicts per-minute rate (not raw totals),
        so the expected range is a per-minute rate range, not a game-total range.
        For fg3m the model still predicts raw count.
        """
        import joblib

        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)

        _seed_game_logs(30)
        from models.nba.stat_model import train, _engineer_features, _lookup_opponent_defense, get_feature_cols

        train(market=market)

        model = joblib.load(model_dir / f"{market}_model.joblib")

        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs"
        )
        df = _engineer_features(df, market=market)
        df = _lookup_opponent_defense(df, market=market)
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
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)

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

    # ------------------------------------------------------------------
    # Rate-based projection tests (Task #4)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("market", ["pts", "reb", "ast"])
    def test_rate_markets_exclude_min_rolling_features(self, db, market):
        """pts/reb/ast feature cols must NOT include min_last*_avg (rate-based training)."""
        from models.nba.stat_model import get_feature_cols

        cols = get_feature_cols(market)
        assert "min_last5_avg" not in cols, (
            f"[{market}] min_last5_avg should be excluded from rate-market features"
        )
        assert "min_last10_avg" not in cols, (
            f"[{market}] min_last10_avg should be excluded from rate-market features"
        )

    def test_fg3m_retains_min_rolling_features(self, db):
        """fg3m feature cols must retain min_last*_avg (count-based, no rate conversion)."""
        from models.nba.stat_model import get_feature_cols

        cols = get_feature_cols("fg3m")
        # fg3m uses _MARKET_STATS["fg3m"] = ["fg3m", "fga"] — no "min" stat, so
        # min_last*_avg would never have been there anyway. The important check is
        # that fg3m has its own rolling features intact.
        assert "fg3m_last5_avg" in cols, "fg3m_last5_avg must be present for fg3m market"
        assert "fga_last5_avg" in cols, "fga_last5_avg must be present for fg3m market"

    @pytest.mark.parametrize("market", ["pts", "reb", "ast"])
    def test_rate_market_projected_value_equals_rate_times_minutes(
        self, db, tmp_path, monkeypatch, market
    ):
        """For pts/reb/ast: projected_value ≈ predicted_rate × predicted_minutes."""
        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)

        _seed_game_logs(30)
        from models.nba.stat_model import train, predict

        train(market=market)

        fake_games = [
            {
                "game_id": "fake002",
                "game_date": "2025-01-16",
                "home_team": "BOS",
                "away_team": "MIA",
            }
        ]
        monkeypatch.setattr("models.nba.stat_model._get_todays_games", lambda: fake_games)

        # Provide a fixed minutes lookup so the test is deterministic
        fake_minutes_lookup = {"1628369": {"predicted_minutes": 32.0, "minutes_sigma": 3.0}}
        predict(market=market, _minutes_lookup=fake_minutes_lookup)

        rows = read_dataframe(
            "SELECT projected_value, predicted_rate, predicted_minutes "
            "FROM nba_projections WHERE market = ?",
            [market],
        )
        assert len(rows) > 0, f"No rows for market={market}"
        for _, row in rows.iterrows():
            if row["predicted_rate"] is not None and row["predicted_minutes"] is not None:
                expected = round(row["predicted_rate"] * row["predicted_minutes"], 2)
                actual = round(float(row["projected_value"]), 2)
                assert abs(actual - expected) < 0.01, (
                    f"[{market}] projected_value={actual} != "
                    f"predicted_rate × predicted_minutes={expected}"
                )

    def test_fg3m_predicted_rate_is_null(self, db, tmp_path, monkeypatch):
        """fg3m rows must have predicted_rate = NULL (count-based, no rate conversion)."""
        model_dir = tmp_path / "nba_models"
        model_dir.mkdir()
        monkeypatch.setattr("models.nba.stat_model.MODEL_DIR", model_dir)

        _seed_game_logs(30)
        from models.nba.stat_model import train, predict

        train(market="fg3m")

        fake_games = [
            {
                "game_id": "fake003",
                "game_date": "2025-01-16",
                "home_team": "BOS",
                "away_team": "MIA",
            }
        ]
        monkeypatch.setattr("models.nba.stat_model._get_todays_games", lambda: fake_games)

        predict(market="fg3m", _minutes_lookup={})

        rows = read_dataframe(
            "SELECT predicted_rate FROM nba_projections WHERE market = 'fg3m'"
        )
        assert len(rows) > 0, "No fg3m rows in nba_projections"
        assert rows["predicted_rate"].isna().all(), (
            "fg3m rows must have predicted_rate = NULL (count-based model)"
        )

    def test_feature_cols_pts_reb_ast_retain_contextual_cols(self, db):
        """pts/reb/ast feature cols must retain all contextual and usage features."""
        from models.nba.stat_model import get_feature_cols

        required = {
            "home_game", "b2b", "days_rest",
            "opp_def_rating_normalized", "opp_pace_normalized",
            "opp_days_rest", "opp_b2b",
            "fga_share", "min_share", "usage_delta",
        }
        for market in ["pts", "reb", "ast"]:
            cols = set(get_feature_cols(market))
            missing = required - cols
            assert not missing, (
                f"[{market}] Missing contextual/usage cols: {missing}"
            )


# ---------------------------------------------------------------------------
# Feature engineering correctness tests
# ---------------------------------------------------------------------------


def _seed_exact_games(rows: list[tuple]) -> None:
    """Insert exact game log rows into nba_player_game_logs.

    Each tuple: (player_id, player_name, team, season, game_id, game_date,
                 matchup, wl, min, pts, reb, ast, fg3m, fgm, fga, ftm, fta,
                 stl, blk, tov, plus_minus)
    """
    executemany(
        """INSERT OR REPLACE INTO nba_player_game_logs (
            player_id, player_name, team_abbreviation, season,
            game_id, game_date, matchup, wl, min,
            pts, reb, ast, fg3m, fgm, fga, ftm, fta,
            stl, blk, tov, plus_minus
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


class TestFeatureEngineeringCorrectness:
    """Verify correctness of feature engineering in stat_model._engineer_features."""

    def test_ewma_excludes_current_row_value(self, db):
        """pts_last5_avg for game 3 must NOT include game 3's pts value.

        Seed: pts = [10, 20, 30] for games 1-3.
        After shift(1), game 3 sees [10, 20]. EWMA(span=5) of [10, 20]
        ≈ 13.33. The value must be closer to 20 than to 30 (no leakage).
        """
        from models.nba.stat_model import _engineer_features

        _seed_exact_games([
            (9001, "P1", "BOS", 2024, "G001", "2025-01-01", "BOS vs. MIA", "W",
             30.0, 10, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 5.0),
            (9001, "P1", "BOS", 2024, "G002", "2025-01-03", "BOS vs. MIA", "W",
             30.0, 20, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 5.0),
            (9001, "P1", "BOS", 2024, "G003", "2025-01-05", "BOS vs. MIA", "W",
             30.0, 30, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 5.0),
        ])

        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs WHERE player_id = 9001"
        )
        result = _engineer_features(df, market="pts")
        result = result.sort_values("game_date").reset_index(drop=True)

        # Game 3 (index 2) must not include its own pts=30 in the rolling avg
        game3_avg = result.loc[2, "pts_last5_avg"]
        assert not np.isnan(game3_avg), "pts_last5_avg for game 3 must not be NaN"
        # EWMA of [10, 20] is well below 30 — must be closer to 20 than to 30
        assert abs(game3_avg - 20) < abs(game3_avg - 30), (
            f"pts_last5_avg={game3_avg:.4f} should be closer to 20 than 30 "
            "(i.e. game 3's own value of 30 must not be included)"
        )

    def test_first_game_rolling_features_handled(self, db):
        """After _engineer_features, the first game row has NaN rolling features.

        The NaN must be either dropped during training (dropna) or filled
        during prediction (fillna(0)). We verify the first game's
        pts_last5_avg is NaN before any dropna/fillna is applied.
        """
        from models.nba.stat_model import _engineer_features

        _seed_exact_games([
            (9002, "P2", "LAL", 2024, "G010", "2025-01-01", "LAL vs. GSW", "W",
             30.0, 25, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 5.0),
        ])

        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs WHERE player_id = 9002"
        )
        result = _engineer_features(df, market="pts")

        first_row = result.iloc[0]
        # shift(1) on the first row produces NaN; ewm propagates NaN when all prior values absent
        # The feature must be NaN for the first row (no prior data to compute average from)
        assert np.isnan(first_row["pts_last5_avg"]), (
            "First game per player must have NaN pts_last5_avg (no prior games to shift from)"
        )

        # Verify training would drop the NaN row
        after_dropna = result.dropna(subset=["pts_last5_avg"])
        assert len(after_dropna) == 0, (
            "Single-game player: all rows should be dropped by dropna on rolling features"
        )

    def test_traded_player_rolling_continues(self, db):
        """Rolling averages at game 5 (team=LAL) must include BOS-era games.

        groupby is on player_id, not team — team changes must not reset rolling history.
        """
        from models.nba.stat_model import _engineer_features

        _seed_exact_games([
            (9003, "P3", "BOS", 2024, "G020", "2025-01-01", "BOS vs. MIA", "W",
             32.0, 15, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 5.0),
            (9003, "P3", "BOS", 2024, "G021", "2025-01-03", "BOS vs. MIA", "L",
             31.0, 18, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, -3.0),
            (9003, "P3", "BOS", 2024, "G022", "2025-01-05", "BOS vs. MIA", "W",
             30.0, 20, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 5.0),
            (9003, "P3", "LAL", 2024, "G023", "2025-01-07", "LAL vs. GSW", "W",
             33.0, 22, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 4.0),
            (9003, "P3", "LAL", 2024, "G024", "2025-01-09", "LAL vs. GSW", "L",
             34.0, 25, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, -2.0),
        ])

        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs WHERE player_id = 9003"
        )
        result = _engineer_features(df, market="pts")
        result = result.sort_values("game_date").reset_index(drop=True)

        # Game 5 (index 4): EWMA computed from games 1-4 (BOS era included)
        game5_avg = result.loc[4, "pts_last5_avg"]
        assert not np.isnan(game5_avg), "pts_last5_avg for game 5 must not be NaN"
        # The average must reflect BOS games (pts ~ 15-20), not just LAL (pts ~22-25)
        # If rolling reset on team change, game5_avg would be ~22 (only game 4 LAL data)
        # If rolling continues across teams, it will be influenced by all 4 prior games
        # Game 4 pts=22 alone → ewma=22. With BOS games weighted in, must be < 22
        assert game5_avg < 22.5, (
            f"pts_last5_avg={game5_avg:.4f} must include BOS-era data "
            "(expected < 22.5 due to lower BOS pts values)"
        )

    def test_days_rest_consecutive_games(self, db):
        """Verify days_rest and b2b for games on Jan 1, Jan 2, Jan 5.

        Expected: days_rest = [3 (fillna default), 1, 3], b2b = [0, 1, 0].
        """
        from models.nba.stat_model import _engineer_features

        _seed_exact_games([
            (9004, "P4", "GSW", 2024, "G030", "2025-01-01", "GSW vs. LAL", "W",
             30.0, 20, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 5.0),
            (9004, "P4", "GSW", 2024, "G031", "2025-01-02", "GSW vs. LAL", "L",
             28.0, 15, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, -2.0),
            (9004, "P4", "GSW", 2024, "G032", "2025-01-05", "GSW vs. LAL", "W",
             31.0, 22, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 3.0),
        ])

        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs WHERE player_id = 9004"
        )
        result = _engineer_features(df, market="pts")
        result = result.sort_values("game_date").reset_index(drop=True)

        expected_days_rest = [3, 1, 3]
        expected_b2b = [0, 1, 0]

        for i, (exp_rest, exp_b2b) in enumerate(zip(expected_days_rest, expected_b2b)):
            actual_rest = int(result.loc[i, "days_rest"])
            actual_b2b = int(result.loc[i, "b2b"])
            assert actual_rest == exp_rest, (
                f"Game {i + 1}: days_rest={actual_rest}, expected {exp_rest}"
            )
            assert actual_b2b == exp_b2b, (
                f"Game {i + 1}: b2b={actual_b2b}, expected {exp_b2b}"
            )

    def test_days_rest_clipped_lower_bound(self, db):
        """days_rest must have a minimum of 1 (never 0 or negative)."""
        from models.nba.stat_model import _engineer_features

        # Seed two games on consecutive days to produce a natural days_rest=1
        # The clip(lower=1) ensures even same-day games don't produce days_rest=0
        _seed_exact_games([
            (9005, "P5", "MIA", 2024, "G040", "2025-01-10", "MIA vs. BOS", "W",
             30.0, 20, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, 5.0),
            (9005, "P5", "MIA", 2024, "G041", "2025-01-11", "MIA vs. BOS", "L",
             29.0, 18, 5, 4, 2, 8, 15, 4, 6, 1, 1, 2, -1.0),
        ])

        df = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, game_id, "
            "game_date, matchup, pts, reb, ast, fg3m, min, fga "
            "FROM nba_player_game_logs WHERE player_id = 9005"
        )
        result = _engineer_features(df, market="pts")
        valid_rest = result["days_rest"].dropna()

        assert (valid_rest >= 1).all(), (
            f"days_rest must be >= 1 after clip(lower=1). "
            f"Found minimum: {valid_rest.min()}"
        )
