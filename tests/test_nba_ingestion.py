"""Tests for NBA data ingestion (scripts/ingest_nba_data.py).

Uses a fresh SQLite DB with migrations applied, monkeypatching the DB env.
All NBA.com calls are mocked — no network required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from schema_migrations import MigrationManager
from utils.db import execute, read_dataframe


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _fake_game_log_df() -> pd.DataFrame:
    """Return a minimal DataFrame matching nba_api PlayerGameLogs output."""
    return pd.DataFrame(
        [
            {
                "PLAYER_ID": 1628369,
                "PLAYER_NAME": "Jayson Tatum",
                "TEAM_ABBREVIATION": "BOS",
                "GAME_ID": "0022401001",
                "GAME_DATE": "2025-01-10",
                "MATCHUP": "BOS vs. MIA",
                "WL": "W",
                "MIN": 36.5,
                "PTS": 31,
                "REB": 8,
                "AST": 5,
                "FG3M": 4,
                "FGM": 11,
                "FGA": 20,
                "FTM": 5,
                "FTA": 6,
                "STL": 1,
                "BLK": 0,
                "TOV": 2,
                "PLUS_MINUS": 12.0,
            },
            {
                "PLAYER_ID": 1628384,
                "PLAYER_NAME": "Jaylen Brown",
                "TEAM_ABBREVIATION": "BOS",
                "GAME_ID": "0022401001",
                "GAME_DATE": "2025-01-10",
                "MATCHUP": "BOS vs. MIA",
                "WL": "W",
                "MIN": 34.0,
                "PTS": 22,
                "REB": 5,
                "AST": 3,
                "FG3M": 2,
                "FGM": 9,
                "FGA": 18,
                "FTM": 2,
                "FTA": 3,
                "STL": 2,
                "BLK": 1,
                "TOV": 1,
                "PLUS_MINUS": 8.0,
            },
        ]
    )


class TestTransform:
    def test_transform_maps_columns(self):
        from scripts.ingest_nba_data import _transform

        df = _fake_game_log_df()
        rows = _transform(df, season_year=2024)

        assert len(rows) == 2
        row = rows[0]
        assert row["player_id"] == 1628369
        assert row["player_name"] == "Jayson Tatum"
        assert row["team_abbreviation"] == "BOS"
        assert row["season"] == 2024
        assert row["pts"] == 31
        assert row["reb"] == 8
        assert row["ast"] == 5
        assert row["fg3m"] == 4

    def test_transform_normalizes_date(self):
        from scripts.ingest_nba_data import _transform

        df = _fake_game_log_df()
        rows = _transform(df, season_year=2024)
        # All dates should be ISO format YYYY-MM-DD
        for r in rows:
            assert len(r["game_date"]) == 10
            assert r["game_date"][4] == "-"

    def test_transform_empty_df_returns_empty(self):
        from scripts.ingest_nba_data import _transform

        rows = _transform(pd.DataFrame(), season_year=2024)
        assert rows == []

    def test_season_str(self):
        from scripts.ingest_nba_data import _season_str

        assert _season_str(2024) == "2024-25"
        assert _season_str(2025) == "2025-26"
        assert _season_str(2023) == "2023-24"


class TestUpsert:
    def test_upsert_inserts_rows(self, db):
        from scripts.ingest_nba_data import _transform, _upsert_rows

        df = _fake_game_log_df()
        rows = _transform(df, season_year=2024)
        n = _upsert_rows(rows)

        assert n == 2
        result = read_dataframe("SELECT COUNT(*) as cnt FROM nba_player_game_logs")
        assert result.iloc[0]["cnt"] == 2

    def test_upsert_is_idempotent(self, db):
        """Inserting the same rows twice should not duplicate."""
        from scripts.ingest_nba_data import _transform, _upsert_rows

        df = _fake_game_log_df()
        rows = _transform(df, season_year=2024)

        _upsert_rows(rows)
        _upsert_rows(rows)  # second insert — same player+game keys

        result = read_dataframe("SELECT COUNT(*) as cnt FROM nba_player_game_logs")
        assert result.iloc[0]["cnt"] == 2

    def test_upsert_updates_on_conflict(self, db):
        """Updated stats for same player+game should overwrite."""
        from scripts.ingest_nba_data import _transform, _upsert_rows

        df = _fake_game_log_df()
        rows = _transform(df, season_year=2024)
        _upsert_rows(rows)

        # Mutate pts for Tatum
        rows[0] = {**rows[0], "pts": 99}
        _upsert_rows(rows)

        result = read_dataframe(
            "SELECT pts FROM nba_player_game_logs WHERE player_id = ? AND game_id = ?",
            [1628369, "0022401001"],
        )
        assert result.iloc[0]["pts"] == 99

    def test_upsert_empty_list_is_noop(self, db):
        from scripts.ingest_nba_data import _upsert_rows

        n = _upsert_rows([])
        assert n == 0

    def test_schema_has_required_columns(self, db):
        """Verify nba_player_game_logs table has expected columns."""
        cols_df = read_dataframe("PRAGMA table_info(nba_player_game_logs)")
        col_names = set(cols_df["name"].tolist())
        required = {"player_id", "player_name", "team_abbreviation", "season", "game_id",
                    "game_date", "pts", "reb", "ast", "fg3m", "plus_minus"}
        assert required.issubset(col_names)
