import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from config import config
from prop_integration import join_odds_projections, normalize_player_name


@contextmanager
def use_database(db_path: Path):
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weekly_projections (
            season INTEGER,
            week INTEGER,
            player_id TEXT,
            team TEXT,
            opponent TEXT,
            market TEXT,
            mu REAL,
            sigma REAL,
            model_version TEXT,
            featureset_hash TEXT,
            generated_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weekly_odds (
            event_id TEXT,
            season INTEGER,
            week INTEGER,
            player_id TEXT,
            market TEXT,
            sportsbook TEXT,
            line REAL,
            price INTEGER,
            as_of TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_stats_enhanced (
            player_id TEXT,
            name TEXT,
            position TEXT,
            team TEXT,
            season INTEGER,
            week INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS injury_data (
            player_id TEXT,
            status TEXT,
            practice_participation TEXT,
            season INTEGER,
            week INTEGER
        )
        """
    )


def test_normalize_player_name_handles_suffixes_and_accents() -> None:
    assert normalize_player_name("C.J. Stroud Jr.") == "cj stroud"
    assert normalize_player_name("Élías Núñez III") == "elias nunez"


def test_join_matches_by_normalized_name(tmp_path: Path) -> None:
    db_path = tmp_path / "matching.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                2024,
                1,
                "HOU_cj_stroud",
                "HOU",
                "IND",
                "passing_yards",
                270.0,
                15.0,
                "v1",
                "hash",
                "2024-09-01T00:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "evt1",
                2024,
                1,
                "HOU_c.j._stroud",
                "passing_yards",
                "Book",
                255.5,
                -110,
                "2024-09-01T12:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO player_stats_enhanced VALUES (?,?,?,?,?,?)",
            (
                "HOU_cj_stroud",
                "C.J. Stroud",
                "QB",
                "HOU",
                2024,
                1,
            ),
        )
        conn.execute(
            "INSERT INTO injury_data VALUES (?,?,?,?,?)",
            (
                "HOU_cj_stroud",
                "QUESTIONABLE",
                "LIMITED",
                2024,
                1,
            ),
        )
        conn.commit()

    with use_database(db_path):
        result = join_odds_projections(2024, 1)

    assert not result.empty
    row = result.loc[result['player_id'] == "HOU_cj_stroud"].iloc[0]
    # Match type can be 'normalized_name_team' (tier 2) or 'normalized_name' (tier 3)
    assert row['match_type'] in ('normalized_name', 'normalized_name_team')
    assert row['player_id_odds'] == "HOU_c.j._stroud"
    assert row['match_score'] >= 0.85  # tier 3 gets 0.85, tier 2 gets 0.95
    assert row['match_confidence'] >= 0.85
    assert bool(row['team_match_flag'])
    assert row['status'] == "QUESTIONABLE"
    assert row['practice_participation'] == "LIMITED"


def test_join_matches_by_fuzzy_name_with_team_mismatch(tmp_path: Path) -> None:
    """Test fuzzy matching with team mismatch - uses SimBook for tier 3 allowance."""
    db_path = tmp_path / "fuzzy.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        # Use WR position and SimBook (tier 3 allowed for synthetic)
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                2024,
                1,
                "KC_mecole_hardman",
                "KC",
                "LAC",
                "receiving_yards",
                48.0,
                12.0,
                "v1",
                "hash",
                "2024-09-01T00:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "evt2",
                2024,
                1,
                "DEN_mecole_hardmn",  # Typo to trigger fuzzy match
                "receiving_yards",
                "SimBook",  # Synthetic allows tier 3
                52.5,
                -105,
                "2024-09-01T12:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO player_stats_enhanced VALUES (?,?,?,?,?,?)",
            (
                "KC_mecole_hardman",
                "Mecole Hardman",
                "WR",
                "KC",
                2024,
                1,
            ),
        )
        conn.execute(
            "INSERT INTO injury_data VALUES (?,?,?,?,?)",
            (
                "KC_mecole_hardman",
                "ACTIVE",
                "FULL",
                2024,
                1,
            ),
        )
        conn.commit()

    with use_database(db_path):
        result = join_odds_projections(2024, 1)

    # WR with team mismatch + SimBook (tier 3 allowed) should survive
    assert not result.empty
    row = result.loc[result['player_id'] == "KC_mecole_hardman"].iloc[0]
    assert row['match_type'] == 'fuzzy_name'
    assert row['player_id_odds'] == "DEN_mecole_hardmn"
    assert row['match_tier'] == 3  # Fuzzy is tier 3
    assert row['match_score'] >= 0.82  # Fuzzy + team penalty
    assert row['match_confidence'] >= 0.82


def test_join_handles_team_aliases(tmp_path: Path) -> None:
    """Test that team aliases (KAN -> KC) are properly canonicalized."""
    db_path = tmp_path / "alias.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        # Use WR to ensure team mismatch tolerance applies
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    2024,
                    2,
                    "KC_skyy_moore",
                    "KC",
                    "BUF",
                    "receiving_yards",
                    40.0,
                    11.0,
                    "v1",
                    "hash",
                    "2024-09-08T00:00:00Z",
                ),
            )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    "evt3",
                    2024,
                    2,
                    "KAN_skyy_moore",  # Team alias KAN -> KC
                    "receiving_yards",
                    "Book",
                    35.0,
                    -108,
                    "2024-09-08T12:00:00Z",
                ),
            )
        conn.execute(
            "INSERT INTO player_stats_enhanced VALUES (?,?,?,?,?,?)",
                (
                    "KC_skyy_moore",
                    "Skyy Moore",
                    "WR",
                    "KC",
                    2024,
                    2,
                ),
            )
        conn.execute(
            "INSERT INTO injury_data VALUES (?,?,?,?,?)",
                (
                    "KC_skyy_moore",
                    "ACTIVE",
                    "FULL",
                    2024,
                    2,
                ),
            )
        conn.commit()

    with use_database(db_path):
        result = join_odds_projections(2024, 2)

    assert not result.empty
    row = result.loc[result['player_id'] == "KC_skyy_moore"].iloc[0]
    assert row['player_id_odds'] == "KAN_skyy_moore"
    # KAN should be canonicalized to KC
    assert row['team_odds'] == "KC"
    assert row['match_confidence'] >= 0.85


def test_team_priority_stats_over_projections_and_odds(tmp_path: Path) -> None:
    """Test that team from stats takes priority over projections and odds."""
    db_path = tmp_path / "team_priority.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        # Projections think player is on DEN
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                2024,
                3,
                "BUF_test_player",
                "DEN",
                "KC",
                "receiving_yards",
                60.0,
                10.0,
                "v1",
                "hash",
                "2024-09-15T00:00:00Z",
            ),
        )
        # Odds use alias team in player_id (e.g., KAN for KC), SimBook for tier 3
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "evt4",
                2024,
                3,
                "KAN_test_player",
                "receiving_yards",
                "SimBook",  # Allow tier 3 with team mismatch
                55.5,
                -110,
                "2024-09-15T12:00:00Z",
            ),
        )
        # Stats indicate true team is BUF
        conn.execute(
            "INSERT INTO player_stats_enhanced VALUES (?,?,?,?,?,?)",
            (
                "BUF_test_player",
                "Test Player",
                "WR",
                "BUF",
                2024,
                3,
            ),
        )
        conn.execute(
            "INSERT INTO injury_data VALUES (?,?,?,?,?)",
            (
                "BUF_test_player",
                "ACTIVE",
                "FULL",
                2024,
                3,
            ),
        )
        conn.commit()

    with use_database(db_path):
        result = join_odds_projections(2024, 3)

    assert not result.empty
    row = result.iloc[0]
    # Team should follow stats (BUF), not projections (DEN) or odds (KC)
    assert row['team'] == "BUF"
    assert row['team_odds'] == "KC"
    # Team mismatch - BUF != KC
    assert not bool(row['team_match_flag'])
