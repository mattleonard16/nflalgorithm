"""Player-to-club resolution in the weekly odds scraper."""

from __future__ import annotations

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
