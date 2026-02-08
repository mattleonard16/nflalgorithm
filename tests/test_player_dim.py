"""Tests for the player_dim canonical dimension table."""

import sqlite3
from datetime import datetime, timezone

import pytest

from schema_migrations import MigrationManager
from scripts.populate_player_dim import populate_player_dim
from utils.db import execute, fetchone, read_dataframe


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Provision a temporary SQLite database with schema."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    # Patch config.database.path so get_connection uses our temp db
    import config as cfg
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _seed_player(db, player_id, name, position, team, season, week):
    execute(
        """
        INSERT OR REPLACE INTO player_stats_enhanced
            (player_id, season, week, name, team, position, age, games_played,
             snap_count, snap_percentage, rushing_yards, rushing_attempts,
             passing_yards, passing_attempts, receiving_yards, receptions,
             targets, red_zone_touches, target_share, air_yards, yac_yards,
             game_script)
        VALUES (?, ?, ?, ?, ?, ?, 25, 1, 50, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        """,
        params=(player_id, season, week, name, team, position),
    )


class TestPopulatePlayerDim:
    def test_populates_from_stats(self, db):
        _seed_player(db, "P001", "Josh Allen", "QB", "BUF", 2025, 10)
        count = populate_player_dim()
        assert count == 1

        row = fetchone("SELECT * FROM player_dim WHERE player_id = ?", params=("P001",))
        assert row is not None
        assert row[1] == "Josh Allen"  # player_name
        assert row[2] == "QB"  # position
        assert row[3] == "BUF"  # team

    def test_latest_season_week_wins(self, db):
        _seed_player(db, "P002", "Tyreek Hill", "WR", "MIA", 2024, 18)
        _seed_player(db, "P002", "Tyreek Hill", "WR", "MIA", 2025, 5)
        _seed_player(db, "P002", "Tyreek Hill", "WR", "MIA", 2025, 10)

        populate_player_dim()

        row = fetchone("SELECT last_season, last_week FROM player_dim WHERE player_id = ?", params=("P002",))
        assert row == (2025, 10)

    def test_idempotent(self, db):
        _seed_player(db, "P003", "Derrick Henry", "RB", "BAL", 2025, 8)

        count1 = populate_player_dim()
        count2 = populate_player_dim()
        assert count1 == count2 == 1

        df = read_dataframe("SELECT COUNT(*) as n FROM player_dim WHERE player_id = ?", params=("P003",))
        assert int(df.iloc[0]["n"]) == 1

    def test_empty_table(self, db):
        count = populate_player_dim()
        assert count == 0

    def test_team_trade_update(self, db):
        """When a player changes teams, player_dim should reflect latest."""
        _seed_player(db, "P004", "Davante Adams", "WR", "LV", 2024, 15)
        _seed_player(db, "P004", "Davante Adams", "WR", "NYJ", 2025, 1)

        populate_player_dim()

        row = fetchone("SELECT team FROM player_dim WHERE player_id = ?", params=("P004",))
        assert row[0] == "NYJ"

    def test_multiple_players(self, db):
        _seed_player(db, "P010", "Patrick Mahomes", "QB", "KC", 2025, 12)
        _seed_player(db, "P011", "Travis Kelce", "TE", "KC", 2025, 12)
        _seed_player(db, "P012", "Saquon Barkley", "RB", "PHI", 2025, 12)

        count = populate_player_dim()
        assert count == 3

        df = read_dataframe("SELECT * FROM player_dim ORDER BY player_id")
        assert len(df) == 3
        assert set(df["position"]) == {"QB", "TE", "RB"}
