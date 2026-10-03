import pandas as pd
import pytest

from utils.nfl_availability import masked_lagged_ewm, shortened_game_mask


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
