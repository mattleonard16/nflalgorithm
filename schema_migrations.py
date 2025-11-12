"""Database schema migrations for weekly NFL pipeline."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class MigrationManager:
    """Run deterministic migrations for the NFL database."""

    db_path: Path | str

    def run(self) -> None:
        with sqlite3.connect(self._db_path_str()) as conn:
            cursor = conn.cursor()
            for ddl in self._ddl_statements():
                cursor.execute(ddl)
            self._ensure_columns(cursor)
            self._ensure_indexes(cursor)
            conn.commit()

    def _db_path_str(self) -> str:
        return str(self.db_path)

    def _ddl_statements(self) -> Iterable[str]:
        return (
            """
            CREATE TABLE IF NOT EXISTS weekly_projections (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                team TEXT NOT NULL,
                opponent TEXT NOT NULL,
                market TEXT NOT NULL,
                mu REAL NOT NULL,
                sigma REAL NOT NULL,
                model_version TEXT NOT NULL,
                featureset_hash TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (season, week, player_id, market)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS weekly_odds (
                event_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                line REAL NOT NULL,
                price INTEGER NOT NULL,
                as_of TEXT NOT NULL,
                PRIMARY KEY (event_id, player_id, market, sportsbook, as_of)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bets_weekly (
                bet_id TEXT PRIMARY KEY,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                event_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                side TEXT NOT NULL,
                line REAL NOT NULL,
                price INTEGER NOT NULL,
                p_win REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                stake REAL NOT NULL,
                bankroll_before REAL NOT NULL,
                placed_at TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS clv_weekly (
                bet_id TEXT PRIMARY KEY,
                close_line REAL,
                close_price INTEGER,
                clv_bp REAL,
                closed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feed_freshness (
                feed TEXT PRIMARY KEY,
                season INTEGER,
                week INTEGER,
                as_of TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS materialized_value_view (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                line REAL NOT NULL,
                price INTEGER NOT NULL,
                mu REAL NOT NULL,
                sigma REAL NOT NULL,
                p_win REAL NOT NULL,
                edge_percentage REAL NOT NULL,
                expected_roi REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                stake REAL NOT NULL,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (season, week, player_id, market, sportsbook, event_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS player_mappings (
                player_id_canonical TEXT NOT NULL,
                player_id_odds TEXT NOT NULL,
                player_id_projections TEXT,
                player_name TEXT,
                team_projections TEXT,
                team_odds TEXT,
                match_type TEXT,
                confidence_score REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id_canonical, player_id_odds)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                kickoff_utc TEXT,
                game_date DATE NOT NULL,
                venue TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS player_stats_enhanced (
                player_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                age INTEGER NOT NULL,
                games_played INTEGER NOT NULL DEFAULT 0,
                snap_count INTEGER NOT NULL DEFAULT 0,
                snap_percentage REAL NOT NULL DEFAULT 0,
                rushing_yards REAL NOT NULL DEFAULT 0,
                rushing_attempts REAL NOT NULL DEFAULT 0,
                receiving_yards REAL NOT NULL DEFAULT 0,
                receptions REAL NOT NULL DEFAULT 0,
                targets REAL NOT NULL DEFAULT 0,
                red_zone_touches REAL NOT NULL DEFAULT 0,
                target_share REAL NOT NULL DEFAULT 0,
                air_yards REAL NOT NULL DEFAULT 0,
                yac_yards REAL NOT NULL DEFAULT 0,
                game_script REAL NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season, week)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS weather_data (
                game_id TEXT PRIMARY KEY,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                game_date DATE NOT NULL,
                temperature REAL NOT NULL,
                wind_speed REAL NOT NULL,
                precipitation REAL NOT NULL,
                humidity REAL NOT NULL,
                is_dome INTEGER NOT NULL,
                weather_description TEXT,
                last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS injury_data (
                player_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                status TEXT NOT NULL,
                practice_participation TEXT NOT NULL,
                injury_type TEXT,
                days_since_injury INTEGER NOT NULL DEFAULT 0,
                last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season, week)
            )
            """,
        )

    def _ensure_columns(self, cursor: sqlite3.Cursor) -> None:
        # Check if games table exists before altering it
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
        if cursor.fetchone() and not self._column_exists(cursor, "games", "kickoff_utc"):
            cursor.execute("ALTER TABLE games ADD COLUMN kickoff_utc TEXT")

    def _ensure_indexes(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_weekly_odds_lookup ON weekly_odds(season, week, player_id, market)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_weekly_projections_lookup ON weekly_projections(season, week, player_id, market)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_feed_freshness_week ON feed_freshness(season, week)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_materialized_value_view_lookup ON materialized_value_view(season, week)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_player_mappings_odds ON player_mappings(player_id_odds)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_player_mappings_canonical ON player_mappings(player_id_canonical)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_player_stats_enhanced_lookup ON player_stats_enhanced(season, week, player_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_injury_data_lookup ON injury_data(season, week, player_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_games_lookup ON games(season, week)"
        )

    @staticmethod
    def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
        cursor.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cursor.fetchall())

