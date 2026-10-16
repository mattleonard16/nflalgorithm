"""Player-to-club resolution in the weekly odds scraper."""

from __future__ import annotations

import pytest

from scripts.prop_line_scraper import NFLPropScraper

HOME, AWAY = "Kansas City Chiefs", "Denver Broncos"
ROSTER = frozenset({"KC_patrick_mahomes", "DEN_bo_nix"})


def _scraper() -> NFLPropScraper:
    return NFLPropScraper(odds_api_key="test-key")


def test_away_player_is_stored_under_the_away_club():
    scraper = _scraper()

    info = scraper._extract_player_info("Bo Nix", HOME, AWAY, ROSTER)

    assert info == {
        "name": "Bo Nix",
        "team": "DEN",
        "position": "UNKNOWN",
        "player_id": "DEN_bo_nix",
        "team_guessed": False,
    }
    assert scraper.last_weekly_audit.get("team_unresolved", 0) == 0


def test_unrostered_player_falls_back_to_home_and_is_counted():
    scraper = _scraper()

    info = scraper._extract_player_info("Practice Squad Guy", HOME, AWAY, ROSTER)

    assert info["team"] == "KC"
    assert info["team_guessed"] is True
    assert info["player_id"] == "KC_practice_squad_guy"
    assert scraper.last_weekly_audit["team_unresolved"] == 1


def test_club_named_by_the_book_is_not_a_guess():
    scraper = _scraper()

    info = scraper._extract_player_info("Practice Squad Guy - Denver Broncos", HOME, AWAY, ROSTER)

    assert info["team"] == "DEN"
    assert info["team_guessed"] is False


# Book spelling -> roster player_id, as minted from the nflverse roster name
# (rows checked against nfl_roster_players 2025/2026).
BOOK_VARIANTS = [
    ("JaMarr Chase", "CIN", "CIN_ja_marr_chase"),
    ("DAndre Swift", "CHI", "CHI_d_andre_swift"),
    ("Hollywood Brown", "KC", "KC_marquise_brown"),
    ("Zonovan Knight", "ARI", "ARI_bam_knight"),
    ("Nathaniel Dell", "HOU", "HOU_tank_dell"),
    ("Joshua Palmer", "BUF", "BUF_josh_palmer"),
    ("Gabriel Davis", "BUF", "BUF_gabe_davis"),
    ("Cameron Skattebo", "NYG", "NYG_cam_skattebo"),
    ("Chigoziem Okonkwo", "TEN", "TEN_chig_okonkwo"),
    ("Ken Walker III", "SEA", "SEA_kenneth_walker"),
    # Spellings that already matched exactly must keep matching.
    ("Marvin Harrison Jr.", "ARI", "ARI_marvin_harrison"),
    ("A.J. Brown", "PHI", "PHI_aj_brown"),
    ("DK Metcalf", "PIT", "PIT_dk_metcalf"),
    ("Amon-Ra St. Brown", "DET", "DET_amon_ra_st_brown"),
    ("Jaxon Smith-Njigba", "SEA", "SEA_jaxon_smith_njigba"),
]
FEED_NAMES = {
    "ARI": "Arizona Cardinals",
    "BUF": "Buffalo Bills",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "DET": "Detroit Lions",
    "HOU": "Houston Texans",
    "KC": "Kansas City Chiefs",
    "NYG": "New York Giants",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "TEN": "Tennessee Titans",
}
OPPONENT = "Las Vegas Raiders"


@pytest.mark.parametrize("home_side", [True, False], ids=["home", "away"])
@pytest.mark.parametrize(("book_name", "team", "roster_pid"), BOOK_VARIANTS)
def test_book_name_variants_resolve_to_the_roster_player(
    book_name: str, team: str, roster_pid: str, home_side: bool
) -> None:
    roster = frozenset({roster_pid, "LV_brock_bowers", "LV_ashton_jeanty"})
    club = FEED_NAMES[team]
    home, away = (club, OPPONENT) if home_side else (OPPONENT, club)
    scraper = _scraper()

    info = scraper._extract_player_info(book_name, home, away, roster)

    assert info["team"] == team
    assert info["player_id"] == roster_pid
    assert info["team_guessed"] is False
    assert scraper.last_weekly_audit.get("team_unresolved", 0) == 0


def test_same_name_on_both_rosters_stays_unresolved() -> None:
    # Michael Carter (ARI RB) and Michael Carter II (PHI DB) both mint
    # "<club>_michael_carter": neither club may be picked from the name.
    roster = frozenset({"ARI_michael_carter", "PHI_michael_carter"})
    scraper = _scraper()

    info = scraper._extract_player_info(
        "Michael Carter", "Philadelphia Eagles", "Arizona Cardinals", roster
    )

    assert info["team_guessed"] is True
    assert scraper.last_weekly_audit["team_unresolved"] == 1
