"""Live-scrape coverage, alternate-line dedup and atomic snapshot writes.

Complements tests/test_odds_event_coverage.py (kickoff-only schedules). These
cases use schedules that carry the matchup, which is what production reads.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pipelines.odds_validation import OddsRequirements, validate_odds_snapshot
from scripts import prop_line_scraper
from scripts.prop_line_scraper import NFLPropScraper

# Week 1: Thursday opener, two Sunday 1pm games sharing a kickoff, Monday night.
SCHEDULE = pd.DataFrame(
    [
        {
            "game_id": "2026_01_KC_BAL",
            "home_team": "BAL",
            "away_team": "KC",
            "kickoff_utc": "2026-09-11T00:20:00Z",
        },
        {
            "game_id": "2026_01_DEN_BUF",
            "home_team": "BUF",
            "away_team": "DEN",
            "kickoff_utc": "2026-09-13T17:00:00Z",
        },
        {
            "game_id": "2026_01_DAL_PHI",
            "home_team": "PHI",
            "away_team": "DAL",
            "kickoff_utc": "2026-09-13T17:00:00Z",
        },
        {
            "game_id": "2026_01_SF_LA",
            "home_team": "LA",
            "away_team": "SF",
            "kickoff_utc": "2026-09-15T00:15:00Z",
        },
    ]
)
SATURDAY = pd.Timestamp("2026-09-12T15:00:00Z")
EVENTS = [
    {
        "id": "e-kc-bal",
        "home_team": "Baltimore Ravens",
        "away_team": "Kansas City Chiefs",
        "commence_time": "2026-09-11T00:20:00Z",
    },
    {
        "id": "e-den-buf",
        "home_team": "Buffalo Bills",
        "away_team": "Denver Broncos",
        "commence_time": "2026-09-13T17:00:00Z",
    },
    {
        "id": "e-dal-phi",
        "home_team": "Philadelphia Eagles",
        "away_team": "Dallas Cowboys",
        "commence_time": "2026-09-13T17:00:00Z",
    },
    {
        "id": "e-sf-lar",
        "home_team": "Los Angeles Rams",
        "away_team": "San Francisco 49ers",
        "commence_time": "2026-09-15T00:15:00Z",
    },
]


def _ids(events: list[dict]) -> list[str]:
    return [event["id"] for event in events]


# --------------------------------------------------------------- selection


def test_started_game_is_excluded_from_requirement_and_selection() -> None:
    # The Thursday game is over but the feed still lists it. It has no pregame
    # line to capture, so it is neither required nor scraped.
    selected = NFLPropScraper._select_scheduled_events(EVENTS, SCHEDULE, now=SATURDAY)
    assert _ids(selected) == ["e-den-buf", "e-dal-phi", "e-sf-lar"]


def test_games_are_matched_by_identity_not_count() -> None:
    # DAL@PHI is missing. A duplicate DEN@BUF listing shares its kickoff, so a
    # count (or kickoff-slot) check would pass; identity must not.
    duplicate = dict(EVENTS[1], id="e-den-buf-dup")
    events = [EVENTS[0], EVENTS[1], duplicate, EVENTS[3]]
    with pytest.raises(RuntimeError, match="2026_01_DAL_PHI"):
        NFLPropScraper._select_scheduled_events(events, SCHEDULE, now=SATURDAY)


def test_in_progress_event_cannot_mask_a_missing_upcoming_game() -> None:
    # Sunday 18:00: both 1pm games are under way and still listed; MNF is
    # missing. Before the fix, the two in-progress events covered the count.
    sunday = pd.Timestamp("2026-09-13T18:00:00Z")
    events = [EVENTS[1], EVENTS[2], {**EVENTS[3], "home_team": "Seattle Seahawks"}]
    with pytest.raises(RuntimeError, match="2026_01_SF_LA"):
        NFLPropScraper._select_scheduled_events(events, SCHEDULE, now=sunday)


def test_flexed_kickoff_still_matches_its_game() -> None:
    flexed = [dict(EVENTS[1], commence_time="2026-09-13T20:25:00Z"), EVENTS[2], EVENTS[3]]
    selected = NFLPropScraper._select_scheduled_events(flexed, SCHEDULE, now=SATURDAY)
    assert _ids(selected) == ["e-den-buf", "e-dal-phi", "e-sf-lar"]


def test_rams_schedule_code_la_matches_feed_name() -> None:
    selected = NFLPropScraper._select_scheduled_events(
        [EVENTS[3]], SCHEDULE.iloc[[3]], now=SATURDAY
    )
    assert _ids(selected) == ["e-sf-lar"]


# ------------------------------------------------- audit denominator (end to end)


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.headers = {"X-Cache": "MISS", "X-Cache-Age-Seconds": "1.0"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def _market_payload(market: str) -> dict:
    outcomes = [
        {"name": "Over", "description": "Some Player", "point": 50.5, "price": -110},
        {"name": "Under", "description": "Some Player", "point": 50.5, "price": -110},
    ]
    return {
        "bookmakers": [
            {"title": book, "markets": [{"key": market, "outcomes": outcomes}]}
            for book in ("BookA", "BookB")
        ]
    }


class _Client:
    def get(self, url: str, params: dict, api_type: str) -> _Response:
        if url.endswith("/events"):
            return _Response(EVENTS)
        return _Response(_market_payload(params["markets"]))


def test_audit_denominator_is_upcoming_games_after_first_kickoff(monkeypatch) -> None:
    monkeypatch.setattr(prop_line_scraper.time, "sleep", lambda _s: None)

    def fake_read(sql: str, params: tuple = ()) -> pd.DataFrame:
        if "FROM games" in sql:
            return SCHEDULE.copy()
        return pd.DataFrame({"player_id": []})

    monkeypatch.setattr(prop_line_scraper, "read_dataframe", fake_read)
    scraper = NFLPropScraper(odds_api_key="test-key")
    scraper.client = _Client()

    rows = scraper.get_upcoming_week_props(1, 2026, now=SATURDAY)

    audit = scraper.last_weekly_audit
    assert audit["scheduled_events"] == 3
    assert audit["started_events"] == 1
    assert audit["week_events"] == 4
    assert audit["covered_events"] == 3
    assert {row["event_id"] for row in rows} == {
        "2026_01_DEN_BUF",
        "2026_01_DAL_PHI",
        "2026_01_SF_LA",
    }

    markets = tuple(prop_line_scraper.config.pipeline.odds_required_markets)
    result = validate_odds_snapshot(
        audit,
        requirements=OddsRequirements(
            max_age_seconds=300,
            min_event_coverage=1.0,
            min_market_coverage=1.0,
            min_sportsbooks_per_event_market=2,
            required_markets=markets,
        ),
    )
    assert result["valid"] is True, result["reason"]
    assert result["event_coverage"] == 1.0
    assert result["started_events"] == 1


# ---------------------------------------------------------- alternate lines


def _row(line: float, over: int, under: int) -> dict:
    return {
        "event_id": "2026_01_DEN_BUF",
        "player_id": "BUF_some_player",
        "stat": "receiving_yards",
        "book": "BookA",
        "line": line,
        "over_odds": over,
        "under_odds": under,
    }


def test_main_line_survives_alternates_in_any_order() -> None:
    alt_low = _row(39.5, -250, +190)
    main = _row(52.5, -112, -108)
    alt_high = _row(64.5, +200, -260)
    for ordering in (
        [alt_low, main, alt_high],
        [main, alt_high, alt_low],
        [alt_high, alt_low, main],
    ):
        kept = NFLPropScraper._keep_main_lines(ordering)
        assert len(kept) == 1
        assert kept[0]["line"] == 52.5


def test_main_line_tie_keeps_first_and_other_keys_are_independent() -> None:
    first = _row(50.5, -110, -110)
    second = _row(51.5, -110, -110)
    other_book = dict(_row(70.5, +150, -180), book="BookB")
    kept = NFLPropScraper._keep_main_lines([first, second, other_book])
    assert [row["line"] for row in kept] == [50.5, 70.5]


# ------------------------------------------------------- atomic snapshot


class _Conn:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    def __enter__(self) -> "_Conn":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def _save_rows() -> list[dict]:
    return [
        {
            **_row(50.5, -110, -110),
            "player_id": f"BUF_player_{i}",
            "player": f"P{i}",
            "home_team": "BUF",
            "away_team": "DEN",
        }
        for i in range(3)
    ]


def test_row_failure_rolls_back_the_whole_snapshot(monkeypatch) -> None:
    conn = _Conn()
    calls: list[tuple] = []

    def failing_execute(sql: str, params: tuple, conn: Any) -> int:
        calls.append(params)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return 1

    monkeypatch.setattr(prop_line_scraper, "get_connection", lambda: conn)
    monkeypatch.setattr(prop_line_scraper, "execute", failing_execute)
    scraper = NFLPropScraper.__new__(NFLPropScraper)

    saved = scraper.save_weekly_odds(_save_rows(), week=1, season=2026)

    assert saved == 0
    assert conn.rolled_back is True
    assert conn.committed is False


def test_clean_batch_commits_once(monkeypatch) -> None:
    conn = _Conn()
    monkeypatch.setattr(prop_line_scraper, "get_connection", lambda: conn)
    monkeypatch.setattr(prop_line_scraper, "execute", lambda sql, params, conn: 1)
    scraper = NFLPropScraper.__new__(NFLPropScraper)

    assert scraper.save_weekly_odds(_save_rows(), week=1, season=2026) == 3
    assert conn.committed is True
    assert conn.rolled_back is False


# ------------------------------------------------------------- anytime TD


def _anytime_td_payload() -> dict:
    outcomes = [
        {"name": "Yes", "description": "Some Player", "price": 150},
        {"name": "No", "description": "Some Player", "price": -200},
        # Yes-only quote: one-sided, cannot be de-vigged, never stored.
        {"name": "Yes", "description": "Other Player", "price": 300},
    ]
    return {
        "bookmakers": [
            {"title": "BookA", "markets": [{"key": "player_anytime_td", "outcomes": outcomes}]}
        ]
    }


class _AnytimeClient:
    def __init__(self) -> None:
        self.markets: list[str] = []

    def get(self, url: str, params: dict, api_type: str) -> _Response:
        if url.endswith("/events"):
            return _Response(EVENTS)
        self.markets.append(params["markets"])
        if params["markets"] == "player_anytime_td":
            return _Response(_anytime_td_payload())
        return _Response(_market_payload(params["markets"]))


def _scrape_with_optional(monkeypatch, optional: tuple[str, ...]):
    monkeypatch.setattr(prop_line_scraper.time, "sleep", lambda _s: None)

    def fake_read(sql: str, params: tuple = ()) -> pd.DataFrame:
        if "FROM games" in sql:
            return SCHEDULE.copy()
        return pd.DataFrame({"player_id": []})

    monkeypatch.setattr(prop_line_scraper, "read_dataframe", fake_read)
    monkeypatch.setattr(prop_line_scraper.config.pipeline, "odds_optional_markets", optional)
    scraper = NFLPropScraper(odds_api_key="test-key")
    client = _AnytimeClient()
    scraper.client = client
    rows = scraper.get_upcoming_week_props(1, 2026, now=SATURDAY)
    return scraper, client, rows


def test_anytime_td_yes_no_is_stored_as_a_paired_half_point_line(monkeypatch) -> None:
    scraper, client, rows = _scrape_with_optional(monkeypatch, ("player_anytime_td",))

    assert "player_anytime_td" in client.markets
    td_rows = [row for row in rows if row["stat"] == "anytime_touchdown"]
    # One per upcoming game; the Yes-only "Other Player" quote never pairs.
    assert len(td_rows) == 3
    assert {row["player"] for row in td_rows} == {"Some Player"}
    assert {(row["line"], row["over_odds"], row["under_odds"]) for row in td_rows} == {
        (0.5, 150, -200)
    }
    assert not any(row["stat"] == "player_anytime_td" for row in rows)

    # The optional market neither pads nor fails the required coverage: one
    # book on anytime TD would fail breadth if it were counted.
    audit = scraper.last_weekly_audit
    assert not any(
        key.endswith(":player_anytime_td") for key in audit["sportsbooks_per_event_market"]
    )
    required = tuple(prop_line_scraper.config.pipeline.odds_required_markets)
    assert audit["covered_event_markets"] == 3 * len(required)
    result = validate_odds_snapshot(
        audit,
        requirements=OddsRequirements(
            max_age_seconds=300,
            min_event_coverage=1.0,
            min_market_coverage=1.0,
            min_sportsbooks_per_event_market=2,
            required_markets=required,
        ),
    )
    assert result["valid"] is True, result["reason"]


def test_unmapped_market_is_not_fetched_or_stored_under_its_provider_key(monkeypatch) -> None:
    _scraper, client, rows = _scrape_with_optional(monkeypatch, ("player_kicking_points",))

    assert "player_kicking_points" not in client.markets
    assert rows and all(not row["stat"].startswith("player_") for row in rows)


def test_default_markets_fetch_anytime_td_without_requiring_it() -> None:
    from config.runtime import pipeline

    assert "player_anytime_td" in pipeline.odds_optional_markets
    assert "player_anytime_td" not in pipeline.odds_required_markets
    assert prop_line_scraper.ODDS_API_MARKETS["player_anytime_td"] == "anytime_touchdown"
