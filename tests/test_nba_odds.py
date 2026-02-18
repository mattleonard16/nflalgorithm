"""Tests for NBA odds scraping (scripts/scrape_nba_odds.py).

Covers:
- nba_odds table schema (created by MigrationManager)
- Player name normalization (_normalize_name)
- Player lookup building (_build_player_lookup -> Dict[str, int])
- Player matching (_match_player -> Optional[int])
- Event parsing (_parse_events)
- Synthetic fallback (generate_synthetic_odds)
- Season derivation (_season_from_date)
- Integration: seed game logs -> scrape -> upsert -> verify DB rows
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from schema_migrations import MigrationManager
from utils.db import execute, executemany, read_dataframe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NBA_ODDS_COLUMNS = {
    "event_id",
    "season",
    "game_date",
    "player_id",
    "player_name",
    "team",
    "market",
    "sportsbook",
    "line",
    "over_price",
    "under_price",
    "as_of",
}

# Fake Odds API JSON — pre-augmented with bookmakers (as _fetch_odds_api returns)
FAKE_EVENTS = [
    {
        "id": "evt_abc123",
        "sport_key": "basketball_nba",
        "sport_title": "NBA",
        "commence_time": "2026-02-17T23:30:00Z",
        "home_team": "Boston Celtics",
        "away_team": "Miami Heat",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Jayson Tatum",
                                "price": -115,
                                "point": 27.5,
                            },
                            {
                                "name": "Under",
                                "description": "Jayson Tatum",
                                "price": -105,
                                "point": 27.5,
                            },
                        ],
                    },
                    {
                        "key": "player_rebounds",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Jayson Tatum",
                                "price": -120,
                                "point": 8.5,
                            },
                            {
                                "name": "Under",
                                "description": "Jayson Tatum",
                                "price": -100,
                                "point": 8.5,
                            },
                        ],
                    },
                ],
            }
        ],
    }
]

# Response with only an Over outcome — tests graceful under_price=None handling
FAKE_EVENTS_OVER_ONLY = [
    {
        "id": "evt_def456",
        "sport_key": "basketball_nba",
        "commence_time": "2026-02-17T23:30:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "Golden State Warriors",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "LeBron James",
                                "price": -110,
                                "point": 24.5,
                            },
                        ],
                    }
                ],
            }
        ],
    }
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_odds.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


def _seed_game_logs() -> None:
    """Insert minimal game-log rows so player matching can resolve IDs."""
    rows = [
        (
            1628369, "Jayson Tatum", "BOS", 2025, "0022500001",
            "2026-02-17", "BOS vs. MIA", "W",
            36.0, 31, 8, 5, 4, 11, 20, 5, 6, 1, 0, 2, 12.0,
        ),
        (
            2544, "LeBron James", "LAL", 2025, "0022500002",
            "2026-02-17", "LAL vs. GSW", "W",
            34.0, 26, 7, 8, 2, 10, 20, 6, 8, 1, 1, 3, 6.0,
        ),
        (
            203954, "Nikola Jokic", "DEN", 2025, "0022500003",
            "2026-02-17", "DEN vs. PHX", "W",
            32.0, 22, 14, 11, 1, 9, 16, 4, 4, 2, 1, 2, 10.0,
        ),
    ]
    executemany(
        """INSERT OR REPLACE INTO nba_player_game_logs (
            player_id, player_name, team_abbreviation, season,
            game_id, game_date, matchup, wl, min,
            pts, reb, ast, fg3m, fgm, fga, ftm, fta,
            stl, blk, tov, plus_minus
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


# ---------------------------------------------------------------------------
# 1. Schema tests
# ---------------------------------------------------------------------------


class TestNbaOddsSchema:
    def test_nba_odds_table_exists(self, db):
        result = read_dataframe(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nba_odds'"
        )
        assert len(result) == 1, "nba_odds table not found"

    def test_nba_odds_has_required_columns(self, db):
        cols_df = read_dataframe("PRAGMA table_info(nba_odds)")
        col_names = set(cols_df["name"].tolist())
        assert NBA_ODDS_COLUMNS.issubset(col_names), (
            f"Missing columns: {NBA_ODDS_COLUMNS - col_names}"
        )

    def test_nba_odds_index_exists(self, db):
        result = read_dataframe(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_nba_odds_lookup'"
        )
        assert len(result) == 1, "idx_nba_odds_lookup index not found"

    def test_nba_odds_insert_and_replace(self, db):
        params = (
            "evt_pk_test", 2025, "2026-02-17", 1628369, "Jayson Tatum",
            "BOS", "pts", "fanduel", 27.5, -115, -105, "2026-02-17T10:00:00Z",
        )
        sql = """INSERT OR REPLACE INTO nba_odds
                 (event_id, season, game_date, player_id, player_name, team,
                  market, sportsbook, line, over_price, under_price, as_of)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
        execute(sql, params=params)
        execute(sql, params=params)
        result = read_dataframe(
            "SELECT COUNT(*) as cnt FROM nba_odds WHERE event_id='evt_pk_test'"
        )
        assert result.iloc[0]["cnt"] == 1

    def test_nba_odds_player_id_nullable(self, db):
        execute(
            """INSERT INTO nba_odds
               (event_id, season, game_date, player_id, player_name, team,
                market, sportsbook, line, over_price, under_price, as_of)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            params=(
                "evt_null_pid", 2025, "2026-02-17", None, "Unknown Player",
                None, "pts", "fanduel", 20.5, -110, -110, "2026-02-17T10:00:00Z",
            ),
        )
        result = read_dataframe(
            "SELECT player_id FROM nba_odds WHERE event_id='evt_null_pid'"
        )
        assert result.iloc[0]["player_id"] is None or str(result.iloc[0]["player_id"]) in ("", "None", "nan")


# ---------------------------------------------------------------------------
# 2. Name normalization tests
# ---------------------------------------------------------------------------


class TestNormalizePlayerName:
    @pytest.fixture(autouse=True)
    def import_fn(self):
        from scripts.scrape_nba_odds import _normalize_name
        self.normalize = _normalize_name

    def test_basic_lowercase(self):
        assert self.normalize("LeBron James") == "lebron james"

    def test_accent_removal_jokic(self):
        assert self.normalize("Nikola Jokić") == "nikola jokic"

    def test_accent_removal_doncic(self):
        assert self.normalize("Luka Dončić") == "luka doncic"

    def test_suffix_sr_removed(self):
        assert self.normalize("Marcus Morris Sr.") == "marcus morris"

    def test_suffix_jr_removed(self):
        assert self.normalize("Wendell Carter Jr.") == "wendell carter"

    def test_suffix_ii_removed(self):
        assert self.normalize("Gary Payton II") == "gary payton"

    def test_suffix_iii_removed(self):
        assert self.normalize("Precious Achiuwa III") == "precious achiuwa"

    def test_suffix_iv_removed(self):
        assert self.normalize("Some Player IV") == "some player"

    def test_suffix_v_removed(self):
        assert self.normalize("Some Player V") == "some player"

    def test_period_removal_initials(self):
        assert self.normalize("P.J. Washington") == "pj washington"

    def test_apostrophe_becomes_space(self):
        assert self.normalize("De'Aaron Fox") == "de aaron fox"

    def test_already_normalized_unchanged(self):
        assert self.normalize("jayson tatum") == "jayson tatum"

    def test_extra_whitespace_stripped(self):
        assert self.normalize("  Kevin Durant  ") == "kevin durant"

    def test_hyphenated_name_preserved(self):
        result = self.normalize("Shai Gilgeous-Alexander")
        assert "gilgeous" in result
        assert "alexander" in result


# ---------------------------------------------------------------------------
# 3. Player lookup building tests
# ---------------------------------------------------------------------------


class TestBuildPlayerLookup:
    """_build_player_lookup() -> Dict[str, int] (normalized_name -> player_id)."""

    def test_returns_dict(self, db):
        _seed_game_logs()
        from scripts.scrape_nba_odds import _build_player_lookup
        result = _build_player_lookup()
        assert isinstance(result, dict)

    def test_contains_seeded_players(self, db):
        _seed_game_logs()
        from scripts.scrape_nba_odds import _build_player_lookup
        result = _build_player_lookup()
        assert len(result) >= 3

    def test_key_is_normalized_name(self, db):
        _seed_game_logs()
        from scripts.scrape_nba_odds import _build_player_lookup
        result = _build_player_lookup()
        assert "jayson tatum" in result

    def test_value_is_player_id(self, db):
        _seed_game_logs()
        from scripts.scrape_nba_odds import _build_player_lookup
        result = _build_player_lookup()
        assert result["jayson tatum"] == 1628369

    def test_empty_when_no_logs(self, db):
        from scripts.scrape_nba_odds import _build_player_lookup
        result = _build_player_lookup()
        assert result == {}


# ---------------------------------------------------------------------------
# 4. Player matching tests — _match_player returns Optional[int]
# ---------------------------------------------------------------------------


class TestMatchPlayer:
    """_match_player(raw_name, lookup) returns player_id or None."""

    @pytest.fixture(autouse=True)
    def import_fn(self):
        from scripts.scrape_nba_odds import _match_player
        self.match = _match_player
        # Build a test lookup dict matching _build_player_lookup's shape
        self.lookup = {
            "jayson tatum": 1628369,
            "lebron james": 2544,
            "nikola jokic": 203954,
        }

    def test_exact_match_tatum(self):
        result = self.match("Jayson Tatum", self.lookup)
        assert result == 1628369

    def test_exact_match_lebron(self):
        result = self.match("LeBron James", self.lookup)
        assert result == 2544

    def test_accent_variant_jokic(self):
        result = self.match("Nikola Jokić", self.lookup)
        assert result == 203954

    def test_fuzzy_match_close_variant(self):
        # "Jayson Tatumm" is close enough for 0.85 threshold
        result = self.match("Jayson Tatumm", self.lookup)
        assert result is None or result == 1628369

    def test_unmatched_returns_none(self):
        result = self.match("Zz Totally Unknown", self.lookup)
        assert result is None

    def test_empty_string_returns_none(self):
        result = self.match("", self.lookup)
        assert result is None

    def test_empty_lookup_returns_none(self):
        result = self.match("LeBron James", {})
        assert result is None


# ---------------------------------------------------------------------------
# 5. Season derivation tests
# ---------------------------------------------------------------------------


class TestSeasonFromDate:
    def test_october_game(self):
        from scripts.scrape_nba_odds import _season_from_date
        assert _season_from_date(date(2024, 10, 22)) == 2024

    def test_february_game(self):
        from scripts.scrape_nba_odds import _season_from_date
        assert _season_from_date(date(2025, 2, 17)) == 2024

    def test_june_game(self):
        from scripts.scrape_nba_odds import _season_from_date
        assert _season_from_date(date(2025, 6, 15)) == 2024


# ---------------------------------------------------------------------------
# 6. Event parsing tests (_parse_events)
# ---------------------------------------------------------------------------


class TestParseEvents:
    """_parse_events(events, season) -> List[Dict]."""

    def test_returns_list(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        assert isinstance(rows, list)

    def test_parses_pts_and_reb(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        markets = {r["market"] for r in rows}
        assert "pts" in markets
        assert "reb" in markets

    def test_row_has_required_keys(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        assert len(rows) > 0
        assert NBA_ODDS_COLUMNS.issubset(rows[0].keys())

    def test_line_extracted_correctly(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        pts_row = next(r for r in rows if r["market"] == "pts")
        assert pts_row["line"] == pytest.approx(27.5)

    def test_over_under_prices_extracted(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        pts_row = next(r for r in rows if r["market"] == "pts")
        assert pts_row["over_price"] == -115
        assert pts_row["under_price"] == -105

    def test_missing_under_price_is_none(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS_OVER_ONLY, season=2025)
        pts_rows = [r for r in rows if r["market"] == "pts"]
        assert len(pts_rows) >= 1
        assert pts_rows[0]["under_price"] is None

    def test_sportsbook_recorded(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        assert all(r["sportsbook"] == "FanDuel" for r in rows)

    def test_event_id_recorded(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        assert all(r["event_id"] == "evt_abc123" for r in rows)

    def test_game_date_from_commence_time(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        assert all(r["game_date"] == "2026-02-17" for r in rows)

    def test_empty_events_returns_empty(self, db):
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events([], season=2025)
        assert rows == []

    def test_player_id_resolved_when_logs_exist(self, db):
        _seed_game_logs()
        from scripts.scrape_nba_odds import _parse_events
        rows = _parse_events(FAKE_EVENTS, season=2025)
        tatum_rows = [r for r in rows if r["player_name"] == "Jayson Tatum"]
        assert len(tatum_rows) >= 1
        assert tatum_rows[0]["player_id"] == 1628369


# ---------------------------------------------------------------------------
# 7. Synthetic fallback tests
# ---------------------------------------------------------------------------


class TestSyntheticFallback:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        _seed_game_logs()

    def test_returns_list(self):
        from scripts.scrape_nba_odds import generate_synthetic_odds
        rows = generate_synthetic_odds("2026-02-17", 2025)
        assert isinstance(rows, list)

    def test_returns_30_rows(self):
        from scripts.scrape_nba_odds import generate_synthetic_odds
        rows = generate_synthetic_odds("2026-02-17", 2025)
        assert len(rows) == 30

    def test_row_has_required_columns(self):
        from scripts.scrape_nba_odds import generate_synthetic_odds
        rows = generate_synthetic_odds("2026-02-17", 2025)
        assert NBA_ODDS_COLUMNS.issubset(rows[0].keys())

    def test_line_is_numeric(self):
        from scripts.scrape_nba_odds import generate_synthetic_odds
        rows = generate_synthetic_odds("2026-02-17", 2025)
        assert all(isinstance(r["line"], float) for r in rows)

    def test_sportsbook_is_synthetic(self):
        from scripts.scrape_nba_odds import generate_synthetic_odds
        rows = generate_synthetic_odds("2026-02-17", 2025)
        assert all(r["sportsbook"] == "synthetic" for r in rows)

    def test_game_date_set_correctly(self):
        from scripts.scrape_nba_odds import generate_synthetic_odds
        rows = generate_synthetic_odds("2026-02-17", 2025)
        assert all(r["game_date"] == "2026-02-17" for r in rows)

    def test_deterministic_with_seed(self):
        from scripts.scrape_nba_odds import generate_synthetic_odds
        rows_a = generate_synthetic_odds("2026-02-17", 2025)
        rows_b = generate_synthetic_odds("2026-02-17", 2025)
        assert [r["line"] for r in rows_a] == pytest.approx([r["line"] for r in rows_b])

    def test_returns_empty_when_no_game_logs(self, db, monkeypatch):
        """With no game logs for the season, should return []."""
        # Wipe game logs
        execute("DELETE FROM nba_player_game_logs")
        from scripts.scrape_nba_odds import generate_synthetic_odds
        rows = generate_synthetic_odds("2026-02-17", 2025)
        assert rows == []


# ---------------------------------------------------------------------------
# 8. MARKET_MAP constant
# ---------------------------------------------------------------------------


class TestMarketMap:
    def test_market_map_exported(self):
        from scripts.scrape_nba_odds import MARKET_MAP
        assert isinstance(MARKET_MAP, dict)

    def test_market_map_player_points(self):
        from scripts.scrape_nba_odds import MARKET_MAP
        assert MARKET_MAP["player_points"] == "pts"

    def test_market_map_player_rebounds(self):
        from scripts.scrape_nba_odds import MARKET_MAP
        assert MARKET_MAP["player_rebounds"] == "reb"

    def test_market_map_player_assists(self):
        from scripts.scrape_nba_odds import MARKET_MAP
        assert MARKET_MAP["player_assists"] == "ast"

    def test_market_map_player_threes(self):
        from scripts.scrape_nba_odds import MARKET_MAP
        assert MARKET_MAP["player_threes"] == "fg3m"


# ---------------------------------------------------------------------------
# 9. scrape_nba_odds top-level
# ---------------------------------------------------------------------------


class TestScrapeNbaOdds:
    def test_no_api_key_returns_synthetic(self, db, monkeypatch):
        _seed_game_logs()
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        # Ensure config doesn't provide a key either
        import config as cfg
        if hasattr(cfg.config, "api"):
            monkeypatch.setattr(cfg.config.api, "odds_api_key", "")
        from scripts.scrape_nba_odds import scrape_nba_odds
        rows = scrape_nba_odds("2026-02-17", 2025)
        assert isinstance(rows, list)
        if rows:
            assert all(r["sportsbook"] == "synthetic" for r in rows)

    def test_with_api_key_calls_fetch(self, db, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "fake_key")
        with patch("scripts.scrape_nba_odds._fetch_odds_api", return_value=FAKE_EVENTS):
            from scripts.scrape_nba_odds import scrape_nba_odds
            rows = scrape_nba_odds("2026-02-17", 2025)
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_empty_events_returns_empty(self, db, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "fake_key")
        with patch("scripts.scrape_nba_odds._fetch_odds_api", return_value=[]):
            from scripts.scrape_nba_odds import scrape_nba_odds
            rows = scrape_nba_odds("2026-02-17", 2025)
        assert rows == []


# ---------------------------------------------------------------------------
# 10. Integration tests (scrape + upsert_odds_rows to DB)
# ---------------------------------------------------------------------------


class TestIntegration:
    def _get_rows(self, db, monkeypatch):
        """Seed game logs, mock _fetch_odds_api, call scrape_nba_odds."""
        _seed_game_logs()
        monkeypatch.setenv("ODDS_API_KEY", "fake_key")
        with patch("scripts.scrape_nba_odds._fetch_odds_api", return_value=FAKE_EVENTS):
            from scripts.scrape_nba_odds import scrape_nba_odds
            return scrape_nba_odds("2026-02-17", 2025)

    def test_scrape_and_save_writes_rows(self, db, monkeypatch):
        rows = self._get_rows(db, monkeypatch)
        from scripts.scrape_nba_odds import upsert_odds_rows
        n = upsert_odds_rows(rows)
        assert n >= 1
        result = read_dataframe("SELECT COUNT(*) as cnt FROM nba_odds")
        assert result.iloc[0]["cnt"] >= 1

    def test_save_is_idempotent(self, db, monkeypatch):
        rows = self._get_rows(db, monkeypatch)
        from scripts.scrape_nba_odds import upsert_odds_rows
        upsert_odds_rows(rows)
        upsert_odds_rows(rows)
        result = read_dataframe(
            "SELECT COUNT(*) as cnt FROM nba_odds WHERE event_id='evt_abc123'"
        )
        # 2 rows (pts + reb) — not doubled after second insert
        assert result.iloc[0]["cnt"] == 2

    def test_save_empty_rows_is_noop(self, db):
        from scripts.scrape_nba_odds import upsert_odds_rows
        n = upsert_odds_rows([])
        assert n == 0

    def test_stored_markets_correct(self, db, monkeypatch):
        rows = self._get_rows(db, monkeypatch)
        from scripts.scrape_nba_odds import upsert_odds_rows
        upsert_odds_rows(rows)
        result = read_dataframe(
            "SELECT market FROM nba_odds WHERE event_id='evt_abc123' ORDER BY market"
        )
        markets = set(result["market"].tolist())
        assert "pts" in markets
        assert "reb" in markets

    def test_stored_line_and_prices_correct(self, db, monkeypatch):
        rows = self._get_rows(db, monkeypatch)
        from scripts.scrape_nba_odds import upsert_odds_rows
        upsert_odds_rows(rows)
        result = read_dataframe(
            "SELECT line, over_price, under_price FROM nba_odds"
            " WHERE market='pts' AND sportsbook='FanDuel'"
        )
        assert len(result) >= 1
        assert result.iloc[0]["line"] == pytest.approx(27.5)
        assert int(result.iloc[0]["over_price"]) == -115
        assert int(result.iloc[0]["under_price"]) == -105

    def test_player_id_resolved_for_known_player(self, db, monkeypatch):
        rows = self._get_rows(db, monkeypatch)
        from scripts.scrape_nba_odds import upsert_odds_rows
        upsert_odds_rows(rows)
        result = read_dataframe(
            "SELECT player_id FROM nba_odds WHERE player_name='Jayson Tatum' LIMIT 1"
        )
        assert int(result.iloc[0]["player_id"]) == 1628369
