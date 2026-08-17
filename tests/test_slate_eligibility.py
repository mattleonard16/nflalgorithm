from sports.nfl import MARKET_MIN_EXPECTED_VOLUME
from utils.slate_eligibility import likely_to_play


def test_starting_qb_is_on_the_pass_slate() -> None:
    row = {
        "position": "QB",
        "roster_status": "ACT",
        "depth_rank": 1,
        "is_starter": 1,
        "expected_passing_attempts": 32,
    }
    assert likely_to_play(row, "passing_yards") is True


def test_backup_qb_is_off_the_pass_slate() -> None:
    siemian = {
        "position": "QB",
        "roster_status": "ACT",
        "depth_rank": 3,
        "is_starter": 0,
        "expected_passing_attempts": 11.6,
    }
    shedeur = {
        "position": "QB",
        "roster_status": "ACT",
        "depth_rank": 2,
        "is_starter": 0,
        "expected_passing_attempts": 17.2,
    }
    assert likely_to_play(siemian, "passing_yards") is False
    assert likely_to_play(shedeur, "passing_yards") is False


def test_inactive_and_out_players_are_excluded() -> None:
    cut = {"position": "WR", "roster_status": "CUT", "depth_rank": 1, "is_starter": 1}
    injured = {
        "position": "RB",
        "roster_status": "ACT",
        "depth_rank": 1,
        "is_starter": 1,
        "injury_status": "Out",
    }
    assert likely_to_play(cut, "receiving_yards") is False
    assert likely_to_play(injured, "rushing_yards") is False


def test_rotation_skill_players_stay_on_the_slate() -> None:
    wr3 = {
        "position": "WR",
        "roster_status": "ACT",
        "depth_rank": 3,
        "is_starter": 0,
        "expected_targets": 1.5,
    }
    wr4 = {
        "position": "WR",
        "roster_status": "ACT",
        "depth_rank": 4,
        "is_starter": 0,
        "expected_targets": 0.7,
    }
    rb2 = {
        "position": "RB",
        "roster_status": "ACT",
        "depth_rank": 2,
        "is_starter": 0,
        "expected_rushing_attempts": 5.1,
    }
    assert likely_to_play(wr3, "receiving_yards") is True
    assert likely_to_play(wr4, "receiving_yards") is False
    assert likely_to_play(rb2, "rushing_yards") is True


def test_missing_depth_falls_back_to_usage_floor() -> None:
    unknown = {
        "position": "RB",
        "roster_status": "ACT",
        "expected_rushing_attempts": MARKET_MIN_EXPECTED_VOLUME["rushing_yards"],
    }
    fringe = {
        "position": "RB",
        "roster_status": "ACT",
        "expected_rushing_attempts": 1.0,
    }
    assert likely_to_play(unknown, "rushing_yards") is True
    assert likely_to_play(fringe, "rushing_yards") is False
