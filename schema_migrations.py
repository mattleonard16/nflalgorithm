"""Database schema migrations for weekly NFL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from utils.db import column_exists, get_connection, table_exists


@dataclass
class MigrationManager:
    """Run deterministic migrations for the NFL database."""

    db_path: Path | str

    def run(self) -> None:
        with get_connection() as conn:
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
                context_sensitivity REAL DEFAULT 0,
                pass_attempts_predicted REAL DEFAULT 0,
                yards_per_attempt_predicted REAL DEFAULT 0,
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
                team TEXT,
                team_odds TEXT,
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
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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
                age INTEGER NOT NULL DEFAULT 26,
                games_played INTEGER NOT NULL DEFAULT 0,
                snap_count INTEGER NOT NULL DEFAULT 0,
                snap_percentage REAL NOT NULL DEFAULT 0,
                rushing_yards REAL NOT NULL DEFAULT 0,
                rushing_attempts REAL NOT NULL DEFAULT 0,
                passing_yards REAL NOT NULL DEFAULT 0,
                passing_attempts REAL NOT NULL DEFAULT 0,
                receiving_yards REAL NOT NULL DEFAULT 0,
                receptions REAL NOT NULL DEFAULT 0,
                targets REAL NOT NULL DEFAULT 0,
                red_zone_touches REAL NOT NULL DEFAULT 0,
                target_share REAL NOT NULL DEFAULT 0,
                air_yards REAL NOT NULL DEFAULT 0,
                yac_yards REAL NOT NULL DEFAULT 0,
                game_script REAL NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
                last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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
                last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, season, week)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bet_outcomes (
                bet_id TEXT PRIMARY KEY,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                player_name TEXT,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                side TEXT NOT NULL,
                line REAL NOT NULL,
                price INTEGER NOT NULL,
                actual_result REAL,
                result TEXT,
                profit_units REAL,
                confidence_tier TEXT,
                edge_at_placement REAL,
                recorded_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS weekly_performance (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                total_bets INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                pushes INTEGER NOT NULL DEFAULT 0,
                profit_units REAL NOT NULL DEFAULT 0,
                roi_pct REAL NOT NULL DEFAULT 0,
                avg_edge REAL NOT NULL DEFAULT 0,
                clv_avg REAL NOT NULL DEFAULT 0,
                best_bet TEXT,
                worst_bet TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (season, week)
            )
            """,
            # ============================================
            # User Authentication Tables
            # ============================================
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                subscription_tier TEXT NOT NULL DEFAULT 'free',
                bankroll REAL NOT NULL DEFAULT 1000.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                default_min_edge REAL NOT NULL DEFAULT 0.05,
                default_kelly_fraction REAL NOT NULL DEFAULT 0.25,
                default_max_stake REAL NOT NULL DEFAULT 0.02,
                best_line_only INTEGER NOT NULL DEFAULT 1,
                show_synthetic_odds INTEGER NOT NULL DEFAULT 0,
                defense_multipliers INTEGER NOT NULL DEFAULT 1,
                weather_adjustments INTEGER NOT NULL DEFAULT 1,
                injury_weighting INTEGER NOT NULL DEFAULT 1,
                preferred_sportsbooks TEXT,
                preferred_markets TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_bets (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                player_name TEXT,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                side TEXT NOT NULL,
                line REAL NOT NULL,
                price INTEGER NOT NULL,
                stake_units REAL NOT NULL,
                stake_dollars REAL,
                model_edge REAL,
                confidence_tier TEXT,
                actual_result REAL,
                outcome TEXT,
                profit_units REAL,
                profit_dollars REAL,
                placed_at TEXT NOT NULL,
                graded_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS line_accuracy_history (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                line REAL NOT NULL,
                actual REAL NOT NULL,
                mu REAL,
                sigma REAL,
                open_line REAL,
                delta_vs_close REAL,
                actual_vs_line REAL,
                model_abs_error REAL,
                line_abs_error REAL,
                model_beats_line INTEGER,
                is_over_hit INTEGER,
                computed_at TEXT NOT NULL,
                PRIMARY KEY (season, week, player_id, market, sportsbook)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_assessments (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                event_id TEXT NOT NULL,
                correlation_group TEXT,
                exposure_warning TEXT,
                risk_adjusted_kelly REAL NOT NULL,
                mean_drawdown REAL,
                max_drawdown REAL,
                p95_drawdown REAL,
                assessed_at TEXT NOT NULL,
                PRIMARY KEY (season, week, player_id, market, sportsbook, event_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_decisions (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                market TEXT NOT NULL,
                decision TEXT NOT NULL,
                merged_confidence REAL NOT NULL,
                votes TEXT NOT NULL,
                rationale TEXT,
                coordinator_override INTEGER NOT NULL DEFAULT 0,
                agent_reports TEXT,
                decided_at TEXT NOT NULL,
                PRIMARY KEY (season, week, player_id, market)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS player_dim (
                player_id TEXT PRIMARY KEY,
                player_name TEXT NOT NULL,
                position TEXT NOT NULL,
                team TEXT NOT NULL,
                last_season INTEGER NOT NULL,
                last_week INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                stages_requested INTEGER NOT NULL DEFAULT 7,
                stages_completed INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                report_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_performance (
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                agent_name TEXT NOT NULL,
                player_id TEXT NOT NULL,
                market TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                confidence REAL NOT NULL,
                final_decision TEXT NOT NULL,
                outcome TEXT NOT NULL,
                profit_units REAL NOT NULL DEFAULT 0,
                correct INTEGER NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (season, week, agent_name, player_id, market)
            )
            """,
            # ============================================
            # NBA Tables
            # ============================================
            """
            CREATE TABLE IF NOT EXISTS nba_player_game_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team_abbreviation TEXT NOT NULL,
                season INTEGER NOT NULL,
                game_id TEXT NOT NULL,
                game_date TEXT NOT NULL,
                matchup TEXT,
                wl TEXT,
                min REAL,
                pts INTEGER,
                reb INTEGER,
                ast INTEGER,
                fg3m INTEGER,
                fgm INTEGER,
                fga INTEGER,
                ftm INTEGER,
                fta INTEGER,
                stl INTEGER,
                blk INTEGER,
                tov INTEGER,
                plus_minus REAL,
                UNIQUE(player_id, game_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nba_projections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                season INTEGER NOT NULL,
                game_date TEXT NOT NULL,
                game_id TEXT NOT NULL,
                market TEXT NOT NULL,
                projected_value REAL NOT NULL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, game_id, market)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nba_odds (
                event_id    TEXT NOT NULL,
                season      INTEGER NOT NULL,
                game_date   TEXT NOT NULL,
                player_id   INTEGER,
                player_name TEXT NOT NULL,
                team        TEXT,
                market      TEXT NOT NULL,
                sportsbook  TEXT NOT NULL,
                line        REAL NOT NULL,
                over_price  INTEGER,
                under_price INTEGER,
                as_of       TEXT NOT NULL,
                PRIMARY KEY (event_id, player_name, market, sportsbook, as_of)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nba_materialized_value_view (
                season          INTEGER NOT NULL,
                game_date       TEXT NOT NULL,
                player_id       INTEGER,
                player_name     TEXT NOT NULL,
                team            TEXT,
                event_id        TEXT NOT NULL,
                market          TEXT NOT NULL,
                sportsbook      TEXT NOT NULL,
                line            REAL NOT NULL,
                over_price      INTEGER NOT NULL,
                under_price     INTEGER,
                mu              REAL NOT NULL,
                sigma           REAL NOT NULL,
                p_win           REAL NOT NULL,
                edge_percentage REAL NOT NULL,
                expected_roi    REAL NOT NULL,
                kelly_fraction  REAL NOT NULL,
                confidence      REAL,
                generated_at    TEXT NOT NULL,
                PRIMARY KEY (game_date, player_id, market, sportsbook, event_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nba_bet_outcomes (
                bet_id TEXT PRIMARY KEY,
                season INTEGER NOT NULL,
                game_date TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                player_name TEXT,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                side TEXT NOT NULL,
                line REAL NOT NULL,
                price INTEGER NOT NULL,
                actual_result REAL,
                result TEXT,
                profit_units REAL,
                confidence_tier TEXT,
                edge_at_placement REAL,
                recorded_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nba_daily_performance (
                season INTEGER NOT NULL,
                game_date TEXT NOT NULL,
                total_bets INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                pushes INTEGER NOT NULL DEFAULT 0,
                profit_units REAL NOT NULL DEFAULT 0,
                roi_pct REAL NOT NULL DEFAULT 0,
                avg_edge REAL NOT NULL DEFAULT 0,
                best_bet TEXT,
                worst_bet TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (season, game_date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nba_risk_assessments (
                game_date           TEXT NOT NULL,
                player_id           INTEGER NOT NULL,
                market              TEXT NOT NULL,
                sportsbook          TEXT NOT NULL,
                event_id            TEXT NOT NULL,
                correlation_group   TEXT,
                exposure_warning    TEXT,
                risk_adjusted_kelly REAL NOT NULL,
                mean_drawdown       REAL,
                max_drawdown        REAL,
                p95_drawdown        REAL,
                assessed_at         TEXT NOT NULL,
                PRIMARY KEY (game_date, player_id, market, sportsbook, event_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nba_agent_decisions (
                game_date            TEXT NOT NULL,
                player_id            INTEGER NOT NULL,
                market               TEXT NOT NULL,
                decision             TEXT NOT NULL,
                merged_confidence    REAL NOT NULL,
                votes                TEXT NOT NULL,
                rationale            TEXT,
                coordinator_override INTEGER NOT NULL DEFAULT 0,
                agent_reports        TEXT,
                decided_at           TEXT NOT NULL,
                PRIMARY KEY (game_date, player_id, market)
            )
            """,
            # ============================================
            # NBA Phase 6 — Injury Tables
            # ============================================
            """
            CREATE TABLE IF NOT EXISTS nba_injuries (
                player_id INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                team TEXT,
                game_date TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                source TEXT,
                scraped_at TEXT NOT NULL,
                PRIMARY KEY (player_id, game_date)
            )
            """,
            # ============================================
            # NBA Phase 7 — CLV & Agent Performance
            # ============================================
            """
            CREATE TABLE IF NOT EXISTS nba_clv (
                bet_id TEXT PRIMARY KEY,
                player_id INTEGER,
                market TEXT NOT NULL,
                sportsbook TEXT NOT NULL,
                game_date TEXT NOT NULL,
                open_line REAL,
                close_line REAL,
                clv_points REAL,
                clv_pct REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS nba_agent_performance (
                game_date TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                market TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                confidence REAL NOT NULL,
                final_decision TEXT NOT NULL,
                outcome TEXT NOT NULL,
                profit_units REAL NOT NULL DEFAULT 0,
                correct INTEGER NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (game_date, agent_name, player_id, market)
            )
            """,
        )

    def _ensure_columns(self, cursor) -> None:
        # Check if games table exists before altering it
        if table_exists("games", conn=cursor.connection) and not column_exists("games", "kickoff_utc", conn=cursor.connection):
            cursor.execute("ALTER TABLE games ADD COLUMN kickoff_utc TEXT")

        # Backfill new columns on existing materialized_value_view tables
        if table_exists("materialized_value_view", conn=cursor.connection):
            if not column_exists("materialized_value_view", "team", conn=cursor.connection):
                cursor.execute("ALTER TABLE materialized_value_view ADD COLUMN team TEXT")
            if not column_exists("materialized_value_view", "team_odds", conn=cursor.connection):
                cursor.execute("ALTER TABLE materialized_value_view ADD COLUMN team_odds TEXT")

        # Add confidence_score and confidence_tier to materialized_value_view
        if table_exists("materialized_value_view", conn=cursor.connection):
            if not column_exists("materialized_value_view", "confidence_score", conn=cursor.connection):
                cursor.execute("ALTER TABLE materialized_value_view ADD COLUMN confidence_score REAL")
            if not column_exists("materialized_value_view", "confidence_tier", conn=cursor.connection):
                cursor.execute("ALTER TABLE materialized_value_view ADD COLUMN confidence_tier TEXT")

        # Add volatility_score and target_share to weekly_projections
        if table_exists("weekly_projections", conn=cursor.connection):
            if not column_exists("weekly_projections", "volatility_score", conn=cursor.connection):
                cursor.execute("ALTER TABLE weekly_projections ADD COLUMN volatility_score REAL")
            if not column_exists("weekly_projections", "target_share", conn=cursor.connection):
                cursor.execute("ALTER TABLE weekly_projections ADD COLUMN target_share REAL")
            if not column_exists("weekly_projections", "context_sensitivity", conn=cursor.connection):
                cursor.execute("ALTER TABLE weekly_projections ADD COLUMN context_sensitivity REAL DEFAULT 0")
            if not column_exists("weekly_projections", "pass_attempts_predicted", conn=cursor.connection):
                cursor.execute("ALTER TABLE weekly_projections ADD COLUMN pass_attempts_predicted REAL DEFAULT 0")
            if not column_exists("weekly_projections", "yards_per_attempt_predicted", conn=cursor.connection):
                cursor.execute("ALTER TABLE weekly_projections ADD COLUMN yards_per_attempt_predicted REAL DEFAULT 0")

        # Add data_health_json to pipeline_runs
        if table_exists("pipeline_runs", conn=cursor.connection):
            if not column_exists("pipeline_runs", "data_health_json", conn=cursor.connection):
                cursor.execute("ALTER TABLE pipeline_runs ADD COLUMN data_health_json TEXT")

        # Phase 1+2: Add sigma, usage_rate, volatility_score to nba_projections
        if table_exists("nba_projections", conn=cursor.connection):
            if not column_exists("nba_projections", "sigma", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_projections ADD COLUMN sigma REAL")
            if not column_exists("nba_projections", "usage_rate", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_projections ADD COLUMN usage_rate REAL")
            if not column_exists("nba_projections", "volatility_score", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_projections ADD COLUMN volatility_score REAL")

        # Phase 4+6: Add confidence and injury columns to nba_materialized_value_view
        if table_exists("nba_materialized_value_view", conn=cursor.connection):
            if not column_exists("nba_materialized_value_view", "confidence_score", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_materialized_value_view ADD COLUMN confidence_score REAL")
            if not column_exists("nba_materialized_value_view", "confidence_tier", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_materialized_value_view ADD COLUMN confidence_tier TEXT")
            if not column_exists("nba_materialized_value_view", "injury_boost_multiplier", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_materialized_value_view ADD COLUMN injury_boost_multiplier REAL")
            if not column_exists("nba_materialized_value_view", "injury_boost_players", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_materialized_value_view ADD COLUMN injury_boost_players TEXT")
            if not column_exists("nba_materialized_value_view", "base_mu", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_materialized_value_view ADD COLUMN base_mu REAL")
            if not column_exists("nba_materialized_value_view", "injury_adjusted_mu", conn=cursor.connection):
                cursor.execute("ALTER TABLE nba_materialized_value_view ADD COLUMN injury_adjusted_mu REAL")

        # Phase 5: Migrate nba_odds PK to include as_of
        self._migrate_nba_odds_pk(cursor)

        # Add passing columns to player_stats_enhanced
        if table_exists("player_stats_enhanced", conn=cursor.connection):
            if not column_exists("player_stats_enhanced", "passing_yards", conn=cursor.connection):
                cursor.execute("ALTER TABLE player_stats_enhanced ADD COLUMN passing_yards REAL NOT NULL DEFAULT 0")
            if not column_exists("player_stats_enhanced", "passing_attempts", conn=cursor.connection):
                cursor.execute("ALTER TABLE player_stats_enhanced ADD COLUMN passing_attempts REAL NOT NULL DEFAULT 0")

    def _migrate_nba_odds_pk(self, cursor: Any) -> None:
        """Recreate nba_odds with as_of in PK if it has the old 4-column PK."""
        if not table_exists("nba_odds", conn=cursor.connection):
            return
        # Check current CREATE TABLE sql for the old PK pattern
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='nba_odds'"
        )
        row = cursor.fetchone()
        if row is None:
            return
        create_sql = row[0] if isinstance(row, (tuple, list)) else row["sql"]
        # If as_of is already in the PK, nothing to do
        if "as_of)" in create_sql and "sportsbook, as_of)" in create_sql:
            return
        # Recreate with new PK
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nba_odds_new (
                event_id    TEXT NOT NULL,
                season      INTEGER NOT NULL,
                game_date   TEXT NOT NULL,
                player_id   INTEGER,
                player_name TEXT NOT NULL,
                team        TEXT,
                market      TEXT NOT NULL,
                sportsbook  TEXT NOT NULL,
                line        REAL NOT NULL,
                over_price  INTEGER,
                under_price INTEGER,
                as_of       TEXT NOT NULL,
                PRIMARY KEY (event_id, player_name, market, sportsbook, as_of)
            )
        """)
        cursor.execute("""
            INSERT OR IGNORE INTO nba_odds_new
            SELECT event_id, season, game_date, player_id, player_name,
                   team, market, sportsbook, line, over_price, under_price, as_of
            FROM nba_odds
        """)
        cursor.execute("DROP TABLE nba_odds")
        cursor.execute("ALTER TABLE nba_odds_new RENAME TO nba_odds")

    def _ensure_indexes(self, cursor: Any) -> None:
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
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_bet_outcomes_week ON bet_outcomes(season, week)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_weekly_performance_lookup ON weekly_performance(season, week)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_line_accuracy_history_lookup ON line_accuracy_history(season, week, market)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_risk_assessments_lookup ON risk_assessments(season, week)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_decisions_lookup ON agent_decisions(season, week)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_performance_lookup ON agent_performance(season, week, agent_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_player_dim_team ON player_dim(team)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_lookup ON pipeline_runs(season, week, status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_game_logs_player ON nba_player_game_logs(player_id, game_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_game_logs_season ON nba_player_game_logs(season, game_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_projections_lookup ON nba_projections(game_date, market)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_game_logs_team ON nba_player_game_logs(team_abbreviation, season)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_odds_lookup ON nba_odds(game_date, market, sportsbook)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_value_lookup ON nba_materialized_value_view(season, game_date, market)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_bet_outcomes_date ON nba_bet_outcomes(season, game_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_daily_performance_lookup ON nba_daily_performance(season, game_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_risk_game_date ON nba_risk_assessments(game_date, market)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_agent_game_date ON nba_agent_decisions(game_date, market)"
        )
        # Phase 5: nba_odds snapshot index
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_odds_snapshot ON nba_odds(game_date, player_id, market, as_of)"
        )
        # Phase 6: nba_injuries indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_injuries_date ON nba_injuries(game_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_injuries_player ON nba_injuries(player_id, game_date)"
        )
        # Phase 7: nba_clv indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_clv_date ON nba_clv(game_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_clv_player ON nba_clv(player_id, game_date)"
        )
        # Phase 7: nba_agent_performance indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_agent_perf_date ON nba_agent_performance(game_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nba_agent_perf_agent ON nba_agent_performance(agent_name, game_date)"
        )


