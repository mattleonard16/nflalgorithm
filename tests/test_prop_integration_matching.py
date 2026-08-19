import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nfl_roster_players (
            season INTEGER NOT NULL,
            gsis_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            team TEXT NOT NULL,
            position TEXT NOT NULL,
            roster_status TEXT,
            roster_week INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (season, gsis_id)
        )
        """
    )


def _insert_roster(
    conn: sqlite3.Connection, *, season, gsis_id, player_id, player_name, team, position
) -> None:
    conn.execute(
        "INSERT INTO nfl_roster_players "
        "(season, gsis_id, player_id, player_name, team, position, roster_status, roster_week, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (season, gsis_id, player_id, player_name, team, position, "ACTIVE", 1, "2024-09-01T00:00:00Z"),
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


def test_join_blocks_cross_position_name_collision(tmp_path: Path) -> None:
    """A QB projection and a same-named DB's odds row must not merge.

    Mirrors the Lamar Jackson collision documented in utils/matching.py:
    the projection side is a real QB passing_yards prop, the odds side is a
    same-name defensive back with no passing prop of their own. Without the
    positions_compatible guard at tier 3, these would merge on
    (norm_name, market) alone.
    """
    db_path = tmp_path / "cross_position.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                2024,
                5,
                "BAL_jordan_smith",
                "BAL",
                "PIT",
                "passing_yards",
                250.0,
                14.0,
                "v1",
                "hash",
                "2024-10-06T00:00:00Z",
            ),
        )
        # Same normalized name, different team, no player_id overlap so tier
        # 1/2 cannot fire; only tier 3's (norm_name, market) merge reaches it.
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "evt5",
                2024,
                5,
                "ATL_jordan_smith",
                "passing_yards",
                "SimBook",
                240.5,
                -110,
                "2024-10-06T12:00:00Z",
            ),
        )
        _insert_roster(
            conn,
            season=2024,
            gsis_id="g-qb-1",
            player_id="BAL_jordan_smith",
            player_name="Jordan Smith",
            team="BAL",
            position="QB",
        )
        _insert_roster(
            conn,
            season=2024,
            gsis_id="g-db-1",
            player_id="ATL_jordan_smith",
            player_name="Jordan Smith",
            team="ATL",
            position="DB",
        )
        conn.commit()

    with use_database(db_path):
        result = join_odds_projections(2024, 5)

    assert result.loc[result["player_id"] == "BAL_jordan_smith"].empty


def test_join_blocks_suffix_conflict_but_matches_dropped_suffix(tmp_path: Path) -> None:
    """David Long Jr. vs David Long must not merge; Pittman Jr. vs Pittman (same
    player, book dropped the suffix) must.
    """
    db_path = tmp_path / "suffix.db"
    with sqlite3.connect(db_path) as conn:
        _init_tables(conn)

        # -- David Long Jr. (TEN LB) vs David Long (IND DB): distinct players,
        # roster raw names carry different suffixes -> suffix_conflict blocks.
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                2024,
                6,
                "TEN_david_long",
                "TEN",
                "HOU",
                "receiving_yards",
                20.0,
                8.0,
                "v1",
                "hash",
                "2024-10-13T00:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "evt6",
                2024,
                6,
                "IND_david_long",
                "receiving_yards",
                "SimBook",
                18.5,
                -110,
                "2024-10-13T12:00:00Z",
            ),
        )
        _insert_roster(
            conn,
            season=2024,
            gsis_id="g-long-ten",
            player_id="TEN_david_long",
            player_name="David Long Jr.",
            team="TEN",
            position="LB",
        )
        _insert_roster(
            conn,
            season=2024,
            gsis_id="g-long-ind",
            player_id="IND_david_long",
            player_name="David Long",
            team="IND",
            position="DB",
        )

        # -- Michael Pittman Jr. (projection) vs Michael Pittman (odds, book
        # dropped the suffix, SAME player) -> should match. The odds pid has
        # no roster entry of its own, so norm_name_full_odds resolves to ""
        # (unknown) rather than a conflicting suffix, and the stripped
        # norm_name still ties them together at tier 3.
        conn.execute(
            "INSERT INTO weekly_projections VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                2024,
                6,
                "IND_michael_pittman",
                "IND",
                "HOU",
                "receiving_yards",
                65.0,
                15.0,
                "v1",
                "hash",
                "2024-10-13T00:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO weekly_odds VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "evt7",
                2024,
                6,
                "HOU_michael_pittman",
                "receiving_yards",
                "SimBook",
                62.5,
                -110,
                "2024-10-13T12:00:00Z",
            ),
        )
        _insert_roster(
            conn,
            season=2024,
            gsis_id="g-pittman",
            player_id="IND_michael_pittman",
            player_name="Michael Pittman Jr.",
            team="IND",
            position="WR",
        )
        conn.commit()

    with use_database(db_path):
        result = join_odds_projections(2024, 6)

    # David Long Jr. / David Long: suffix conflict blocks the merge.
    assert result.loc[result["player_id"] == "TEN_david_long"].empty

    # Michael Pittman Jr. / Michael Pittman: same player, book dropped the
    # suffix on their side; the odds pid has no roster entry of its own, so
    # there is no conflicting suffix evidence and the match proceeds.
    pittman = result.loc[result["player_id"] == "IND_michael_pittman"]
    assert not pittman.empty
    assert pittman.iloc[0]["player_id_odds"] == "HOU_michael_pittman"
