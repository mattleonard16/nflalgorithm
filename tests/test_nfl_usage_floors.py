from sports.nfl import MARKET_MIN_EXPECTED_VOLUME


def test_usage_floors_keep_rotation_players_not_depth_chart_leftovers() -> None:
    assert MARKET_MIN_EXPECTED_VOLUME == {
        "rushing_yards": 3.0,
        "receiving_yards": 2.0,
        "passing_yards": 12.0,
        "receptions": 1.5,
        "anytime_touchdown": 0.5,
    }
