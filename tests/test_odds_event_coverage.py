"""Which scheduled games a live odds scrape is required to cover.

`scripts.prop_line_scraper` is tracked, but its only existing coverage lives in
`tests/test_synthetic_odds_wr.py`, which imports gitignored `data_pipeline` and
is therefore skipped in CI. This rule decides whether a card gets built from a
partial slate, so it is tested here where CI can see it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines.odds_validation import validate_odds_snapshot
from scripts import prop_line_scraper
from scripts.prop_line_scraper import NFLPropScraper

# A Thursday opener, two Sunday windows, and a Monday night game.
KICKOFFS = [
    "2026-09-10T00:15:00Z",
    "2026-09-13T17:00:00Z",
    "2026-09-13T20:25:00Z",
    "2026-09-15T00:15:00Z",
]


def _schedule(kickoffs: list[str] = KICKOFFS) -> pd.DataFrame:
    return pd.DataFrame({"kickoff_utc": kickoffs})


def _events(kickoffs: list[str]) -> list[dict]:
    return [{"id": f"event-{i}", "commence_time": k} for i, k in enumerate(kickoffs)]


def test_full_coverage_before_the_first_kickoff() -> None:
    selected = NFLPropScraper._select_scheduled_events(
        _events(KICKOFFS), _schedule(), now=pd.Timestamp("2026-09-09T12:00:00Z")
    )
    assert [event["id"] for event in selected] == ["event-0", "event-1", "event-2", "event-3"]


def test_events_outside_the_requested_week_are_ignored() -> None:
    other_week = [{"id": "next-week", "commence_time": "2026-09-20T17:00:00Z"}]
    selected = NFLPropScraper._select_scheduled_events(
        other_week + _events(KICKOFFS), _schedule(), now=pd.Timestamp("2026-09-09T12:00:00Z")
    )
    assert "next-week" not in [event["id"] for event in selected]


def test_a_game_already_kicked_off_is_not_required() -> None:
    # Saturday of week 1. The Thursday opener is over and the API no longer
    # lists it. Demanding the whole week here would fail every scrape after the
    # week's first kickoff, which is most of them.
    selected = NFLPropScraper._select_scheduled_events(
        _events(KICKOFFS[1:]), _schedule(), now=pd.Timestamp("2026-09-12T22:00:00Z")
    )
    assert [event["id"] for event in selected] == ["event-0", "event-1", "event-2"]


def test_a_game_under_way_is_not_scraped() -> None:
    # Sunday afternoon of week 1. The early game is still listed while it is
    # played, but its quotes are in-game lines the card drops, so fetching
    # them only spends credits.
    selected = NFLPropScraper._select_scheduled_events(
        _events(KICKOFFS[1:]), _schedule(), now=pd.Timestamp("2026-09-13T18:00:00Z")
    )
    assert [event["id"] for event in selected] == ["event-1", "event-2"]


def test_a_missing_upcoming_game_still_fails_loud() -> None:
    # The case the guard exists for: a game that has not started is absent from
    # the feed, so the card would be priced off a partial slate.
    with pytest.raises(RuntimeError, match="still to kick off"):
        NFLPropScraper._select_scheduled_events(
            _events(KICKOFFS[:2]), _schedule(), now=pd.Timestamp("2026-09-09T12:00:00Z")
        )


def test_every_game_started_requires_nothing() -> None:
    # Scraping a week that is fully played returns nothing rather than raising.
    # There is no pregame line left to capture, so this is not an error.
    assert (
        NFLPropScraper._select_scheduled_events(
            [], _schedule(), now=pd.Timestamp("2026-09-16T12:00:00Z")
        )
        == []
    )


def test_a_missing_kickoff_in_the_schedule_is_an_error() -> None:
    # A null kickoff makes "has this started" unanswerable, so the scrape must
    # not silently treat the game as already played.
    with pytest.raises(RuntimeError, match="missing kickoff timestamps"):
        NFLPropScraper._select_scheduled_events(
            _events(KICKOFFS),
            _schedule(KICKOFFS[:3] + [None]),  # type: ignore[list-item]
            now=pd.Timestamp("2026-09-09T12:00:00Z"),
        )


def test_an_empty_schedule_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="no kickoff timestamps"):
        NFLPropScraper._select_scheduled_events(
            _events(KICKOFFS), pd.DataFrame(), now=pd.Timestamp("2026-09-09T12:00:00Z")
        )


class _Response:
    headers = {"X-Cache": "MISS", "X-Cache-Age-Seconds": "1"}

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> object:
        return self._payload


class _FakeOddsApi:
    """Lists ``events`` and quotes one player at two books in every market."""

    def __init__(self, events: list[dict]) -> None:
        self.events = events

    def get(self, url: str, params: dict, api_type: str) -> _Response:
        if url.endswith("/events"):
            return _Response(self.events)
        outcomes = [
            {"name": side, "description": "Sam Passer", "point": 50.5, "price": -110}
            for side in ("Over", "Under")
        ]
        market = {"key": params["markets"], "outcomes": outcomes}
        return _Response(
            {"bookmakers": [{"title": book, "markets": [market]} for book in ("BookA", "BookB")]}
        )


def test_a_scrape_after_the_first_kickoff_that_covers_every_remaining_game_is_valid(
    monkeypatch,
) -> None:
    # Sunday of week 3: the Thursday game is over and gone from the API, and the
    # scrape prices the one game still to come in full. That is a complete card.
    now = pd.Timestamp.now(tz="UTC")
    thursday = (now - pd.Timedelta(days=3)).isoformat()
    sunday = (now + pd.Timedelta(hours=3)).isoformat()
    schedule = pd.DataFrame({"game_id": ["thursday", "sunday"], "kickoff_utc": [thursday, sunday]})
    monkeypatch.setattr(
        prop_line_scraper,
        "read_dataframe",
        lambda query, **kwargs: (
            schedule if "FROM games" in query else pd.DataFrame(columns=["player_id"])
        ),
    )
    monkeypatch.setattr(prop_line_scraper.time, "sleep", lambda seconds: None)
    scraper = NFLPropScraper(odds_api_key="test-key")
    scraper.client = _FakeOddsApi(
        [
            {
                "id": "sunday-event",
                "commence_time": sunday,
                "home_team": "Seattle Seahawks",
                "away_team": "Arizona Cardinals",
            }
        ]
    )

    scraper.get_upcoming_week_props(3, 2026, allow_synthetic=False)
    result = validate_odds_snapshot(scraper.last_weekly_audit)

    assert result["valid"], result["reason"]
