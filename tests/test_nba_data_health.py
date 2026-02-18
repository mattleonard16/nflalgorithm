"""NBA data health checks.

Verifies that all NBA tables and indexes defined in schema_migrations.py
exist after migrations run, and that column schemas and date formats are correct.
"""

from __future__ import annotations

import re

import pytest

from schema_migrations import MigrationManager
from utils.db import execute, fetchall


GAME_DATE = "2026-02-17"
SEASON = 2025
PLAYER_ID = 1234
PLAYER_NAME = "Health Test Player"
TEAM = "GSW"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    import config as cfg
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    from schema_migrations import MigrationManager
    MigrationManager(db_path).run()
    return db_path


class TestNbaDataHealth:
    def test_nba_tables_exist(self, db):
        """All 6 NBA tables must exist after migrations."""
        expected_tables = {
            "nba_player_game_logs",
            "nba_projections",
            "nba_odds",
            "nba_materialized_value_view",
            "nba_bet_outcomes",
            "nba_daily_performance",
        }

        rows = fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in rows}

        missing = expected_tables - existing_tables
        assert not missing, f"Missing NBA tables: {missing}"

    def test_nba_bet_outcomes_columns(self, db):
        """nba_bet_outcomes must contain all required columns."""
        required_columns = {
            "bet_id",
            "season",
            "game_date",
            "player_id",
            "player_name",
            "market",
            "sportsbook",
            "side",
            "line",
            "price",
            "actual_result",
            "result",
            "profit_units",
            "confidence_tier",
            "edge_at_placement",
            "recorded_at",
        }

        rows = fetchall("PRAGMA table_info(nba_bet_outcomes)")
        existing_columns = {row[1] for row in rows}

        missing = required_columns - existing_columns
        assert not missing, (
            f"nba_bet_outcomes is missing columns: {missing}"
        )

    def test_nba_daily_performance_columns(self, db):
        """nba_daily_performance must contain all required columns."""
        required_columns = {
            "season",
            "game_date",
            "total_bets",
            "wins",
            "losses",
            "pushes",
            "profit_units",
            "roi_pct",
            "avg_edge",
            "best_bet",
            "worst_bet",
            "updated_at",
        }

        rows = fetchall("PRAGMA table_info(nba_daily_performance)")
        existing_columns = {row[1] for row in rows}

        missing = required_columns - existing_columns
        assert not missing, (
            f"nba_daily_performance is missing columns: {missing}"
        )

    def test_nba_indexes_exist(self, db):
        """All required NBA indexes must exist after migrations."""
        required_indexes = {
            "idx_nba_bet_outcomes_date",
            "idx_nba_daily_performance_lookup",
            "idx_nba_game_logs_player",
            "idx_nba_projections_lookup",
            "idx_nba_odds_lookup",
            "idx_nba_value_lookup",
        }

        rows = fetchall("SELECT name FROM sqlite_master WHERE type='index'")
        existing_indexes = {row[0] for row in rows}

        missing = required_indexes - existing_indexes
        assert not missing, f"Missing NBA indexes: {missing}"

    def test_date_format_validation(self, db):
        """game_date values in nba_player_game_logs must be in YYYY-MM-DD format."""
        execute(
            "INSERT INTO nba_player_game_logs "
            "(player_id, player_name, team_abbreviation, season, game_id, game_date, matchup, wl, min, pts, reb, ast, fg3m, fgm, fga, ftm, fta, stl, blk, tov, plus_minus) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=(PLAYER_ID, PLAYER_NAME, TEAM, SEASON, "0022500999", GAME_DATE, "GSW vs LAL", "W", 32.0, 22, 6, 4, 2, 9, 18, 4, 5, 2, 1, 3, 5.0),
        )

        rows = fetchall(
            "SELECT game_date FROM nba_player_game_logs WHERE player_id = ?",
            params=(PLAYER_ID,),
        )

        assert len(rows) >= 1, "No rows returned after insert"

        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for row in rows:
            game_date = row[0]
            assert date_pattern.match(game_date), (
                f"game_date '{game_date}' does not match YYYY-MM-DD format"
            )
