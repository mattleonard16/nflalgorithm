"""Tests for NBA data ingestion (scripts/ingest_nba_data.py).

Uses a fresh SQLite DB with migrations applied, monkeypatching the DB env.
All NBA.com calls are mocked — no network required.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from unittest.mock import MagicMock, call, patch

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


# ---------------------------------------------------------------------------
# _fetch_with_retry — exponential backoff
# ---------------------------------------------------------------------------


class TestFetchWithRetry:
    """Tests for the _fetch_with_retry() helper."""

    @patch("scripts.ingest_nba_data.time")
    def test_succeeds_on_second_attempt(self, mock_time):
        """If first call raises Timeout but second succeeds, returns the DataFrame."""
        import requests.exceptions

        from scripts.ingest_nba_data import _fetch_with_retry

        expected_df = _fake_game_log_df()
        call_count = 0

        def flaky_fetch():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.Timeout("simulated timeout")
            return expected_df

        result = _fetch_with_retry(flaky_fetch, "test-season")

        assert call_count == 2
        assert len(result) == len(expected_df)
        # Should have slept once (2^1 = 2 seconds)
        mock_time.sleep.assert_called_once_with(2)

    @patch("scripts.ingest_nba_data.time")
    def test_three_failures_returns_empty_dataframe(self, mock_time):
        """When all MAX_RETRIES attempts fail, returns empty DataFrame without raising."""
        import requests.exceptions

        from scripts.ingest_nba_data import MAX_RETRIES, _fetch_with_retry

        call_count = 0

        def always_timeout():
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.Timeout("always fails")

        result = _fetch_with_retry(always_timeout, "bad-season")

        assert result.empty
        assert call_count == MAX_RETRIES + 1

    @patch("scripts.ingest_nba_data.time")
    def test_connection_error_triggers_retry(self, mock_time):
        """ConnectionError is also retried (not just Timeout)."""
        import requests.exceptions

        from scripts.ingest_nba_data import _fetch_with_retry

        expected_df = _fake_game_log_df()
        call_count = 0

        def conn_error_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.ConnectionError("simulated connection reset")
            return expected_df

        result = _fetch_with_retry(conn_error_then_ok, "test-season")

        assert call_count == 2
        assert len(result) == len(expected_df)

    @patch("scripts.ingest_nba_data.time")
    def test_non_transient_exception_propagates(self, mock_time):
        """ValueError and other non-transient errors must propagate immediately."""
        from scripts.ingest_nba_data import _fetch_with_retry

        call_count = 0

        def bad_fetch():
            nonlocal call_count
            call_count += 1
            raise ValueError("unexpected schema change")

        with pytest.raises(ValueError, match="unexpected schema change"):
            _fetch_with_retry(bad_fetch, "bad-season")

        assert call_count == 1  # no retries
        mock_time.sleep.assert_not_called()

    @patch("scripts.ingest_nba_data.time")
    def test_exponential_backoff_delays(self, mock_time):
        """Sleep durations must follow 2^(attempt+1) pattern."""
        import requests.exceptions

        from scripts.ingest_nba_data import MAX_RETRIES, _fetch_with_retry

        def always_timeout():
            raise requests.exceptions.Timeout("always fails")

        _fetch_with_retry(always_timeout, "test")

        sleep_calls = [c.args[0] for c in mock_time.sleep.call_args_list]
        expected_delays = [2 ** (i + 1) for i in range(MAX_RETRIES)]
        assert sleep_calls == expected_delays


# ---------------------------------------------------------------------------
# data_freshness_check
# ---------------------------------------------------------------------------


class TestDataFreshnessCheck:
    """Tests for the data_freshness_check() function."""

    def test_warns_when_data_is_stale(self, db, caplog):
        """Should log a WARNING when latest game_date exceeds max_stale_days."""
        import logging

        from scripts.ingest_nba_data import _transform, _upsert_rows, data_freshness_check

        # Insert a row with a game_date well in the past (10 days ago)
        stale_date = (date.today() - timedelta(days=10)).isoformat()
        df = pd.DataFrame([{
            "PLAYER_ID": 1,
            "PLAYER_NAME": "Test Player",
            "TEAM_ABBREVIATION": "TST",
            "GAME_ID": "g001",
            "GAME_DATE": stale_date,
            "MATCHUP": "TST vs. OPP",
            "WL": "W",
            "MIN": 30.0,
            "PTS": 20,
            "REB": 5,
            "AST": 3,
            "FG3M": 2,
            "FGM": 8,
            "FGA": 15,
            "FTM": 2,
            "FTA": 3,
            "STL": 1,
            "BLK": 0,
            "TOV": 1,
            "PLUS_MINUS": 5.0,
        }])
        rows = _transform(df, season_year=2024)
        _upsert_rows(rows)

        with caplog.at_level(logging.WARNING, logger="scripts.ingest_nba_data"):
            data_freshness_check(db, max_stale_days=3)

        assert any("stale" in r.message.lower() or "days ago" in r.message.lower()
                   for r in caplog.records
                   if r.levelno == logging.WARNING)

    def test_no_warning_when_data_is_fresh(self, db, caplog):
        """Should NOT log WARNING when latest game_date is within max_stale_days."""
        import logging

        from scripts.ingest_nba_data import _transform, _upsert_rows, data_freshness_check

        # Insert a row with yesterday's date (fresh)
        fresh_date = (date.today() - timedelta(days=1)).isoformat()
        df = pd.DataFrame([{
            "PLAYER_ID": 2,
            "PLAYER_NAME": "Fresh Player",
            "TEAM_ABBREVIATION": "TST",
            "GAME_ID": "g002",
            "GAME_DATE": fresh_date,
            "MATCHUP": "TST vs. OPP",
            "WL": "W",
            "MIN": 30.0,
            "PTS": 20,
            "REB": 5,
            "AST": 3,
            "FG3M": 2,
            "FGM": 8,
            "FGA": 15,
            "FTM": 2,
            "FTA": 3,
            "STL": 1,
            "BLK": 0,
            "TOV": 1,
            "PLUS_MINUS": 5.0,
        }])
        rows = _transform(df, season_year=2024)
        _upsert_rows(rows)

        with caplog.at_level(logging.WARNING, logger="scripts.ingest_nba_data"):
            data_freshness_check(db, max_stale_days=3)

        stale_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and ("stale" in r.message.lower() or "days ago" in r.message.lower())
        ]
        assert len(stale_warnings) == 0

    def test_warns_when_table_is_empty(self, db, caplog):
        """Should log WARNING when no rows exist in the table."""
        import logging

        from scripts.ingest_nba_data import data_freshness_check

        with caplog.at_level(logging.WARNING, logger="scripts.ingest_nba_data"):
            data_freshness_check(db, max_stale_days=3)

        assert any("no data" in r.message.lower() or "nba_player_game_logs" in r.message
                   for r in caplog.records
                   if r.levelno == logging.WARNING)
