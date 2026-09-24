"""Player-to-club resolution in the weekly odds scraper."""

from __future__ import annotations

from unittest.mock import Mock

import requests

from scripts.prop_line_scraper import NFLPropScraper

HOME, AWAY = "Kansas City Chiefs", "Denver Broncos"
ROSTER = frozenset({"KC_patrick_mahomes", "DEN_bo_nix"})


def _scraper() -> NFLPropScraper:
    return NFLPropScraper(odds_api_key="test-key")


def test_away_player_is_stored_under_the_away_club():
    scraper = _scraper()

    info = scraper._extract_player_info("Bo Nix", HOME, AWAY, ROSTER)

    assert info == {"name": "Bo Nix", "team": "DEN", "position": "UNKNOWN"}
    assert scraper.last_weekly_audit.get("team_unresolved", 0) == 0


def test_unrostered_player_falls_back_to_home_and_is_counted():
    scraper = _scraper()

    info = scraper._extract_player_info("Practice Squad Guy", HOME, AWAY, ROSTER)

    assert info["team"] == "KC"
    assert scraper.last_weekly_audit["team_unresolved"] == 1


def test_failed_events_request_log_never_carries_the_api_key(caplog):
    # requests puts the full URL, apiKey included, into the HTTPError message.
    scraper = NFLPropScraper(odds_api_key="sk-secret-123")
    scraper.client = Mock()
    scraper.client.get.side_effect = requests.HTTPError(
        "401 Client Error for url: https://api.test/v4/sports/americanfootball_nfl/events"
        "?apiKey=sk-secret-123"
    )

    with caplog.at_level("WARNING", logger="scripts.prop_line_scraper"):
        rows = scraper.get_upcoming_week_props(week=1, season=2026, allow_synthetic=False)

    assert rows == []
    assert "sk-secret-123" not in caplog.text
    assert "apiKey=***" in caplog.text
