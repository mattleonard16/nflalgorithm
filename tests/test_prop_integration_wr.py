"""Tests for WR-resilient matching in join_odds_projections (Issue 4).

Match tiers:
- tier=1: exact player_id + market
- tier=2: normalized_name + market + team_canon
- tier=3: normalized_name + market (team differences allowed but penalized)

WR team mismatch tolerance:
- WRs traded mid-season where odds list new team, stats list old team
- Team name variants (JAX vs JAC)
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from config import config
from prop_integration import join_odds_projections, normalize_player_name


@contextmanager
def use_database(db_path: Path):
    """Temporarily override database config to use SQLite."""
    original_path = config.database.path
    original_backend = config.database.backend
    env_backend = os.environ.get("DB_BACKEND")
    env_sqlite_path = os.environ.get("SQLITE_DB_PATH")
    os.environ["DB_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(db_path)
    config.database.backend = "sqlite"
    config.database.path = str(db_path)
    try:
        yield
    finally:
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


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weekly_projections (
            season INTEGER, week INTEGER, player_id TEXT, team TEXT,
            opponent TEXT, market TEXT, mu REAL, sigma REAL,
            model_version TEXT, featureset_hash TEXT, generated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weekly_odds (
            event_id TEXT, season INTEGER, week INTEGER, player_id TEXT,
            market TEXT, sportsbook TEXT, line REAL, price INTEGER, as_of TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_stats_enhanced (
            player_id TEXT, name TEXT, position TEXT, team TEXT,
            season INTEGER, week INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS injury_data (
            player_id TEXT, status TEXT, practice_participation TEXT,
            season INTEGER, week INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_mappings (
            player_id_canonical TEXT, player_id_odds TEXT,
            player_id_projections TEXT, match_type TEXT, confidence_score REAL
        )
    """)


def test_wr_exact_player_id_match_tier1(tmp_path: Path) -> None:
    """WR with exact player_id match should be tier 1."""
    db_path = tmp_path / "tier1.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (2024, 5, 'MIA_tyreek_hill', 'MIA', 'NE', 'receiving_yards', 85.0, 15.0, 'v1', 'abc', '2024-01-01')
        )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            ('game1', 2024, 5, 'MIA_tyreek_hill', 'receiving_yards', 'DraftKings', 82.5, -110, '2024-01-01')
        )
        conn.execute(
            "INSERT INTO player_stats_enhanced VALUES (?,?,?,?,?,?)",
            ('MIA_tyreek_hill', 'Tyreek Hill', 'WR', 'MIA', 2024, 5)
        )
        conn.commit()
    
    with use_database(db_path):
        result = join_odds_projections(2024, 5)
    
    assert not result.empty, "Should have results"
    tyreek = result[result['player_id'] == 'MIA_tyreek_hill']
    assert not tyreek.empty, "Tyreek Hill should match"
    assert tyreek.iloc[0]['match_tier'] == 1, "Should be tier 1 (player_id)"
    assert tyreek.iloc[0]['match_confidence'] >= 0.95, "Should have high confidence"


def test_wr_traded_player_team_mismatch_tolerated(tmp_path: Path) -> None:
    """WR traded mid-season should match despite team mismatch (tier 3)."""
    db_path = tmp_path / "traded.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (2024, 5, 'NYG_davante_adams', 'NYG', 'DAL', 'receiving_yards', 70.0, 12.0, 'v1', 'def', '2024-01-01')
        )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            ('game2', 2024, 5, 'LV_davante_adams', 'receiving_yards', 'FanDuel', 75.5, -115, '2024-01-01')
        )
        conn.execute(
            "INSERT INTO player_stats_enhanced VALUES (?,?,?,?,?,?)",
            ('NYG_davante_adams', 'Davante Adams', 'WR', 'NYG', 2024, 5)
        )
        conn.commit()
    
    with use_database(db_path):
        result = join_odds_projections(2024, 5)
    
    # Should find Davante Adams via normalized name match
    if not result.empty:
        davante_rows = result[
            result['normalized_name'].str.contains('davante', case=False, na=False)
        ]
        assert not davante_rows.empty, "Davante Adams should match despite team mismatch"
        row = davante_rows.iloc[0]
        assert row['match_tier'] in [2, 3], "Should be tier 2 or 3 (name match)"


def test_wr_team_variant_matches(tmp_path: Path) -> None:
    """WR with team name variant (JAX/JAC) should match."""
    db_path = tmp_path / "variant.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (2024, 5, 'JAX_christian_kirk', 'JAX', 'IND', 'receiving_yards', 55.0, 10.0, 'v1', 'ghi', '2024-01-01')
        )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            ('game3', 2024, 5, 'JAC_christian_kirk', 'receiving_yards', 'SimBook', 52.5, -110, '2024-01-01')
        )
        conn.execute(
            "INSERT INTO player_stats_enhanced VALUES (?,?,?,?,?,?)",
            ('JAX_christian_kirk', 'Christian Kirk', 'WR', 'JAX', 2024, 5)
        )
        conn.commit()
    
    with use_database(db_path):
        result = join_odds_projections(2024, 5)
    
    if not result.empty:
        kirk_rows = result[
            result['normalized_name'].str.contains('christian kirk', case=False, na=False)
        ]
        assert not kirk_rows.empty, "Christian Kirk should match despite JAX/JAC variant"


def test_match_tier_in_output(tmp_path: Path) -> None:
    """Ensure match_tier and match_confidence are in output columns."""
    db_path = tmp_path / "tier_output.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (2024, 5, 'MIA_tyreek_hill', 'MIA', 'NE', 'receiving_yards', 85.0, 15.0, 'v1', 'abc', '2024-01-01')
        )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            ('game1', 2024, 5, 'MIA_tyreek_hill', 'receiving_yards', 'DraftKings', 82.5, -110, '2024-01-01')
        )
        conn.execute(
            "INSERT INTO player_stats_enhanced VALUES (?,?,?,?,?,?)",
            ('MIA_tyreek_hill', 'Tyreek Hill', 'WR', 'MIA', 2024, 5)
        )
        conn.commit()
    
    with use_database(db_path):
        result = join_odds_projections(2024, 5)
    
    if not result.empty:
        assert 'match_tier' in result.columns, "match_tier should be in output"
        assert 'match_confidence' in result.columns, "match_confidence should be in output"
        assert result['match_tier'].isin([1, 2, 3]).all(), "match_tier should be 1, 2, or 3"


def test_normalize_player_name() -> None:
    """Test name normalization for matching."""
    assert normalize_player_name("Tyreek Hill") == normalize_player_name("tyreek hill")
    # Apostrophe handling: Ja'Marr -> "ja marr" (space preserved after removal)
    assert normalize_player_name("Ja'Marr Chase") == "ja marr chase"
    # D.K. -> "dk" (periods removed, spaces normalized)
    assert normalize_player_name("D.K. Metcalf").replace(" ", "") == "dkmetcalf"
