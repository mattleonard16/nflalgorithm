from utils.player_id_utils import name_from_player_id


def test_name_from_team_scoped_player_id() -> None:
    assert name_from_player_id("MIA_tyreek_hill") == "tyreek hill"


def test_name_from_unscoped_player_id() -> None:
    assert name_from_player_id("tyreek") == "tyreek"


def test_name_from_empty_player_id() -> None:
    assert name_from_player_id(None) == ""
