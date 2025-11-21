"""Test season/week handling with merge suffixes in join_odds_projections."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
import os

import pandas as pd
import pytest

from config import config
from prop_integration import join_odds_projections
from schema_migrations import MigrationManager


@pytest.fixture
def temp_db_with_data():
    """Create a temporary database with test data."""
    # Use isolated temporary database file
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    original_path = config.database.path
    original_backend = config.database.backend
    env_backend = os.environ.get("DB_BACKEND")
    env_sqlite_path = os.environ.get("SQLITE_DB_PATH")
    os.environ["DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(tmp)
    config.database.backend = "sqlite"
    config.database.path = str(tmp)

    # Setup schema
    MigrationManager(tmp).run()
    
    # Insert test data that would trigger merge suffixes
    with sqlite3.connect(tmp) as conn:
        # Create player_stats_enhanced table (required by join_odds_projections)
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
        
        # Create injury_data table (required by join_odds_projections)
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
            ("test_player_team1", 2023, 9, "Test Player", "RB", "TEAM1")
        )
        
        # Add test projections
        conn.execute(
            """
            INSERT INTO weekly_projections 
            (season, week, player_id, team, opponent, market, mu, sigma, model_version, featureset_hash, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (2023, 9, "test_player_team1", "TEAM1", "TEAM2", "rushing_yards", 75.0, 10.0, "v1", "hash1", "2023-11-01T00:00:00")
        )
        
        # Add test odds with same season/week
        conn.execute(
            """
            INSERT INTO weekly_odds
            (event_id, season, week, player_id, market, sportsbook, line, price, as_of)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("event1", 2023, 9, "test_player_team1", "rushing_yards", "TestBook", 70.5, -110, "2023-11-01T00:00:00")
        )
        
        conn.commit()
    
    yield str(tmp)

    config.database.path = original_path
    config.database.backend = original_backend
    if env_backend is not None:
        os.environ["DB_BACKEND"] = env_backend
    else:
        os.environ.pop("DB_BACKEND", None)
    if env_sqlite_path is not None:
        os.environ["SQLITE_DB_PATH"] = env_sqlite_path
    else:
        os.environ.pop("SQLITE_DB_PATH", None)
    tmp.unlink(missing_ok=True)


def test_join_odds_projections_handles_merge_suffixes(temp_db_with_data):
    """Test that join_odds_projections properly handles merge suffixes for season/week."""
    result = join_odds_projections(2023, 9)
    
    # Should have data
    assert not result.empty
    
    # Should have season and week columns (not season_proj/season_odds)
    assert 'season' in result.columns
    assert 'week' in result.columns
    
    # Should NOT have merge suffix columns
    assert 'season_proj' not in result.columns
    assert 'season_odds' not in result.columns
    assert 'week_proj' not in result.columns
    assert 'week_odds' not in result.columns
    
    # All season and week values should be non-null
    assert result['season'].notna().all()
    assert result['week'].notna().all()
    
    # All season values should be 2023
    assert (result['season'] == 2023).all()
    
    # All week values should be 9
    assert (result['week'] == 9).all()
    
    print(f"✅ join_odds_projections returned {len(result)} rows with proper season/week handling")


def test_join_odds_projections_no_null_season_week(temp_db_with_data):
    """Verify that join_odds_projections never returns null season/week values."""
    result = join_odds_projections(2023, 9)
    
    if not result.empty:
        null_seasons = result['season'].isna().sum()
        null_weeks = result['week'].isna().sum()
        
        assert null_seasons == 0, f"Found {null_seasons} null season values"
        assert null_weeks == 0, f"Found {null_weeks} null week values"
        
        print(f"✅ No null season/week values in {len(result)} rows")
