"""Unit tests for season/week constraint handling."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from config import config
from materialized_value_view import materialize_week
from schema_migrations import MigrationManager
from value_betting_engine import rank_weekly_value


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])

    # Setup schema
    MigrationManager(tmp).run()
    MigrationManager(tmp_path).run()

    # Insert test data
    with sqlite3.connect(tmp) as conn:
        # Create required tables for join_odds_projections
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_stats_enhanced (
                player_id TEXT,
                season INTEGER,
                week INTEGER,
                name TEXT,
                position TEXT,
                team TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS injury_data (
                player_id TEXT,
                season INTEGER,
                week INTEGER,
                status TEXT,
                practice_participation TEXT
            )
        """)
        
        # Add test player data
        conn.execute(
            "INSERT INTO player_stats_enhanced (player_id, season, week, name, position, team) VALUES (?, ?, ?, ?, ?, ?)",
            ("test_player_1_team1", 2023, 1, "Test Player", "RB", "TEAM1")
        )
        
        # Add test projections
        conn.execute(
            """
            INSERT INTO weekly_projections 
            (season, week, player_id, team, opponent, market, mu, sigma, model_version, featureset_hash, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (2023, 1, "test_player_1_team1", "TEAM1", "TEAM2", "rushing_yards", 75.0, 10.0, "v1", "hash1", "2023-09-01T00:00:00")
        )
        
        # Add test odds
        conn.execute(
            """
            INSERT INTO weekly_odds
            (event_id, season, week, player_id, market, sportsbook, line, price, as_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("event1", 2023, 1, "test_player_1_team1", "rushing_yards", "TestBook", 70.5, -110, "2023-09-01T00:00:00")
        )
        
        conn.commit()
    
    yield str(tmp)


def test_rank_weekly_value_with_valid_data(temp_db):
    """Test that rank_weekly_value correctly handles valid data with season/week."""
    result = rank_weekly_value(2023, 1, min_edge=0.0, place=False)
    
    assert not result.empty
    assert 'season' in result.columns
    assert 'week' in result.columns
    assert (result['season'] == 2023).all()
    assert (result['week'] == 1).all()


def test_materialize_week_with_valid_data(temp_db):
    """Test that materialize_week correctly handles valid data."""
    result = materialize_week(2023, 1, min_edge=0.0)
    
    # Check that data was materialized
    with sqlite3.connect(temp_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM materialized_value_view WHERE season=? AND week=?",
            (2023, 1)
        ).fetchone()[0]
    
    assert count > 0


def test_materialize_week_with_no_data(temp_db):
    """Test that materialize_week handles empty data gracefully."""
    # Try to materialize for a week with no data
    result = materialize_week(2024, 10, min_edge=0.0)
    
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    
    # Ensure no rows were inserted
    with sqlite3.connect(temp_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM materialized_value_view WHERE season=? AND week=?",
            (2024, 10)
        ).fetchone()[0]
    
    assert count == 0


def test_season_week_constraint_not_null(temp_db):
    """Test that season and week constraints are enforced in the database."""
    with sqlite3.connect(temp_db) as conn:
        # Try to insert a row with NULL season (should fail)
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            conn.execute(
                """
                INSERT INTO materialized_value_view
                (season, week, player_id, event_id, market, sportsbook, line, price, 
                 mu, sigma, p_win, edge_percentage, expected_roi, kelly_fraction, stake, generated_at)
                VALUES (NULL, 1, 'test', 'event1', 'rushing_yards', 'TestBook', 70.5, -110,
                        75.0, 10.0, 0.65, 0.05, 0.03, 0.02, 20.0, '2023-09-01T00:00:00')
                """
            )
        
        # Try to insert a row with NULL week (should fail)
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            conn.execute(
                """
                INSERT INTO materialized_value_view
                (season, week, player_id, event_id, market, sportsbook, line, price,
                 mu, sigma, p_win, edge_percentage, expected_roi, kelly_fraction, stake, generated_at)
                VALUES (2023, NULL, 'test', 'event1', 'rushing_yards', 'TestBook', 70.5, -110,
                        75.0, 10.0, 0.65, 0.05, 0.03, 0.02, 20.0, '2023-09-01T00:00:00')
                """
            )
