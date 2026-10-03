import pandas as pd
import pytest

from utils.nfl_availability import (
    MAX_VOLUME_MULTIPLIER,
    masked_lagged_ewm,
    promote_backup_qbs,
    redistribute_out_volume,
    shortened_game_mask,
)


def _games(shares: list[float], player: str = "qb") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [player] * len(shares),
            "week": range(1, len(shares) + 1),
            "snap_percentage": shares,
        }
    )


def test_a_starter_who_leaves_early_is_flagged() -> None:
    mask = shortened_game_mask(_games([100.0] * 6 + [10.0]))

    assert mask.tolist() == [False] * 6 + [True]


def test_a_part_time_player_is_never_flagged() -> None:
    mask = shortened_game_mask(_games([55.0] * 6 + [10.0]))

    assert not mask.any()


def test_a_zero_snap_share_is_unknown_not_shortened() -> None:
    mask = shortened_game_mask(_games([100.0] * 6 + [0.0]))

    assert not mask.any()


def test_a_game_is_flagged_only_from_its_own_share_and_earlier_games() -> None:
    full = _games([100.0] * 6 + [10.0, 100.0, 5.0])
    through_week_seven = full.iloc[:7]

    assert shortened_game_mask(full).iloc[:7].tolist() == (
        shortened_game_mask(through_week_seven).tolist()
    )


def test_one_players_history_does_not_set_anothers_median() -> None:
    frame = pd.concat([_games([100.0] * 6, "starter"), _games([10.0], "backup")], ignore_index=True)

    assert not shortened_game_mask(frame).any()


def test_a_flagged_game_carries_no_weight_in_the_next_games_average() -> None:
    attempts = pd.Series([30.0, 30.0, 2.0, 0.0])
    players = pd.Series(["qb"] * 4)
    mask = pd.Series([False, False, True, False])

    masked = masked_lagged_ewm(attempts, players, mask, span=3)

    assert masked.iloc[3] == pytest.approx(30.0)


def test_with_nothing_flagged_the_average_matches_the_plain_lagged_ewm() -> None:
    values = pd.Series([12.0, 4.0, 9.0, 15.0, 7.0])
    players = pd.Series(["wr"] * 5)

    masked = masked_lagged_ewm(values, players, pd.Series([False] * 5), span=3)
    plain = values.shift(1).ewm(span=3, min_periods=1).mean()

    pd.testing.assert_series_equal(masked, plain, check_names=False)


def _receivers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["BUF", "BUF", "BUF", "BUF", "MIA"],
            "position": ["WR", "WR", "WR", "TE", "WR"],
            "expected_targets": [4.0, 6.0, 2.0, 4.0, 7.0],
        }
    )


def test_an_out_receivers_targets_go_to_teammates_by_share() -> None:
    out = pd.Series([True, False, False, False, False])

    result = redistribute_out_volume(_receivers(), out)

    # 4 freed x 0.6 damping = 2.4, split 6:2 between the two active WRs.
    assert result["expected_targets"].tolist()[1:3] == pytest.approx([6.0 + 1.8, 2.0 + 0.6])


def test_no_teammate_rises_past_the_cap() -> None:
    frame = pd.DataFrame(
        {"team": ["BUF", "BUF"], "position": ["WR", "WR"], "expected_targets": [10.0, 2.0]}
    )

    result = redistribute_out_volume(frame, pd.Series([True, False]))

    assert result["expected_targets"].iloc[1] == pytest.approx(2.0 * MAX_VOLUME_MULTIPLIER)


def test_a_team_with_nobody_out_is_unchanged() -> None:
    frame = _receivers()

    result = redistribute_out_volume(frame, pd.Series([False] * len(frame)))

    pd.testing.assert_frame_equal(result, frame)


def test_volume_stays_within_the_team_and_position() -> None:
    out = pd.Series([True, False, False, False, False])

    result = redistribute_out_volume(_receivers(), out)

    assert result["expected_targets"].tolist()[3:] == [4.0, 7.0]


def _quarterbacks(starter_status: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["BUF", "BUF"],
            "position": ["QB", "QB"],
            "depth_rank": [1, 2],
            "injury_status": [starter_status, None],
            "expected_passing_attempts": [33.0, 4.0],
        }
    )


def test_the_backup_qb_starts_when_the_starter_is_out() -> None:
    frame = _quarterbacks("Out")

    result = promote_backup_qbs(frame, pd.Series([True, False]))

    assert result["p_start"].tolist() == [0.0, 1.0]
    assert result["expected_passing_attempts"].iloc[1] == pytest.approx(33.0)


def test_the_backup_qb_stays_a_backup_when_the_starter_plays() -> None:
    result = promote_backup_qbs(_quarterbacks("Questionable"), pd.Series([False, False]))

    assert result["p_start"].tolist() == [0.75, 0.02]
    assert result["expected_passing_attempts"].iloc[1] == pytest.approx(4.0)
