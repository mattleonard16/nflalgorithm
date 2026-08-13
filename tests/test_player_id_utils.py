import pytest

from utils.player_id_utils import VALID_NFL_TEAMS, canonicalize_team, name_from_player_id

# Full club names exactly as The Odds API sends them.
ODDS_API_TEAM_NAMES = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


@pytest.mark.parametrize("full_name,expected", sorted(ODDS_API_TEAM_NAMES.items()))
def test_odds_api_full_club_names_resolve(full_name: str, expected: str) -> None:
    # An unresolved club silently becomes "" and poisons every downstream
    # player_id and event_id built from it.
    assert canonicalize_team(full_name) == expected


def test_every_club_is_covered() -> None:
    assert set(ODDS_API_TEAM_NAMES.values()) == VALID_NFL_TEAMS


def test_relocated_clubs_still_resolve_from_historical_feeds() -> None:
    assert canonicalize_team("Oakland Raiders") == "LV"
    assert canonicalize_team("San Diego Chargers") == "LAC"
    assert canonicalize_team("Washington Football Team") == "WAS"


def test_unknown_club_returns_empty_rather_than_guessing() -> None:
    assert canonicalize_team("Toronto Argonauts") == ""


def test_name_from_team_scoped_player_id() -> None:
    assert name_from_player_id("MIA_tyreek_hill") == "tyreek hill"


def test_name_from_unscoped_player_id() -> None:
    assert name_from_player_id("tyreek") == "tyreek"


def test_name_from_empty_player_id() -> None:
    assert name_from_player_id(None) == ""
