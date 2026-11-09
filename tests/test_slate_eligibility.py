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


def test_receptions_follow_the_receiving_rule() -> None:
    wr3 = {"position": "WR", "roster_status": "ACT", "depth_rank": 3, "is_starter": 0}
    wr4 = {"position": "WR", "roster_status": "ACT", "depth_rank": 4, "is_starter": 0}
    assert likely_to_play(wr3, "receptions") is True
    assert likely_to_play(wr4, "receptions") is False


def test_anytime_touchdown_keeps_anyone_who_rushes_or_catches() -> None:
    rb2 = {"position": "RB", "roster_status": "ACT", "depth_rank": 2, "is_starter": 0}
    te1 = {"position": "TE", "roster_status": "ACT", "depth_rank": 1, "is_starter": 1}
    starting_qb = {"position": "QB", "roster_status": "ACT", "depth_rank": 1, "is_starter": 1}
    backup_qb = {"position": "QB", "roster_status": "ACT", "depth_rank": 2, "is_starter": 0}
    assert likely_to_play(rb2, "anytime_touchdown") is True
    assert likely_to_play(te1, "anytime_touchdown") is True
    assert likely_to_play(starting_qb, "anytime_touchdown") is True
    assert likely_to_play(backup_qb, "anytime_touchdown") is False


def test_a_player_ruled_out_on_this_weeks_report_is_ruled_out() -> None:
    from utils.slate_eligibility import ruled_out

    for status in ("Out", "OUT", "IR", "Injured Reserve"):
        assert ruled_out({"injury_status": status, "injury_report_week": 4}, 4) is True


def test_doubtful_and_questionable_players_are_not_ruled_out() -> None:
    # Props void when a player sits, so an if-he-plays projection stays valid.
    from utils.slate_eligibility import ruled_out

    for status in ("Doubtful", "Questionable", None, ""):
        assert ruled_out({"injury_status": status, "injury_report_week": 4}, 4) is False


def test_last_weeks_out_status_does_not_rule_a_player_out() -> None:
    # Wednesday's run comes before clubs file, so last week's status carries
    # forward. A player back at practice would vanish until Saturday.
    from utils.slate_eligibility import ruled_out

    assert ruled_out({"injury_status": "Out", "injury_report_week": 3}, 4) is False


def test_an_out_status_with_no_report_week_is_not_ruled_out() -> None:
    from utils.slate_eligibility import ruled_out

    assert ruled_out({"injury_status": "Out", "injury_report_week": None}, 4) is False


def test_a_qb_listed_behind_the_starter_is_a_backup() -> None:
    from utils.slate_eligibility import backup_qb

    assert backup_qb({"position": "QB", "depth_rank": 2, "is_starter": 0}) is True


def test_starters_unlisted_qbs_and_other_positions_are_not_backups() -> None:
    from utils.slate_eligibility import backup_qb

    # A QB with no depth chart entry keeps his projection: missing data is not
    # evidence that he sits.
    for row in (
        {"position": "QB", "depth_rank": 1, "is_starter": 1},
        {"position": "QB", "depth_rank": 2, "is_starter": 1},
        {"position": "QB", "depth_rank": None, "is_starter": 0},
        {"position": "WR", "depth_rank": 3, "is_starter": 0},
    ):
        assert backup_qb(row) is False
