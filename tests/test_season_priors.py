"""Early-season 70/30 volume priors and the two-season train window."""

from __future__ import annotations

import pandas as pd

from utils.season_priors import (
    CRASH_LAST_WEIGHT,
    CRASH_PRIOR_WEIGHT,
    HEALTHY_LAST_WEIGHT,
    HEALTHY_PRIOR_WEIGHT,
    apply_early_season_role_prior,
    blend_per_game_volume,
    is_volume_crash,
    regular_season_training_weeks,
    season_prior_weights,
)


def _season_rows(player_id: str, season: int, weeks: int, targets: float) -> list[dict]:
    return [
        {
            "player_id": player_id,
            "season": season,
            "week": week,
            "targets": targets,
            "expected_targets": 1.5,
        }
        for week in range(1, weeks + 1)
    ]


def test_healthy_blend_is_seventy_thirty() -> None:
    last_w, prior_w = season_prior_weights(last_games=16, last_volume=96.0, prior_volume=128.0)
    assert (last_w, prior_w) == (HEALTHY_LAST_WEIGHT, HEALTHY_PRIOR_WEIGHT)
    assert blend_per_game_volume(
        6.0, 8.0, last_games=16, last_volume=96.0, prior_volume=128.0
    ) == 0.70 * 6.0 + 0.30 * 8.0


def test_short_season_or_halved_volume_is_a_crash() -> None:
    assert is_volume_crash(4, 12.0, 128.0) is True
    assert is_volume_crash(16, 50.0, 128.0) is True
    assert is_volume_crash(16, 96.0, 128.0) is False
    last_w, prior_w = season_prior_weights(last_games=4, last_volume=12.0, prior_volume=128.0)
    assert (last_w, prior_w) == (CRASH_LAST_WEIGHT, CRASH_PRIOR_WEIGHT)


def test_missing_prior_season_uses_last_season_only() -> None:
    assert season_prior_weights(last_games=16, last_volume=96.0, prior_volume=0.0) == (1.0, 0.0)
    assert blend_per_game_volume(
        6.0, 0.0, last_games=16, last_volume=96.0, prior_volume=0.0
    ) == 6.0


def test_week1_healthy_role_is_70_30_per_game() -> None:
    rows = (
        _season_rows("KC_xavier_worthy", 2024, 10, 8.0)
        + _season_rows("KC_xavier_worthy", 2025, 10, 6.0)
        + [
            {
                "player_id": "KC_xavier_worthy",
                "season": 2026,
                "week": 1,
                "targets": 0.0,
                "expected_targets": 1.5,
            }
        ]
    )
    out = apply_early_season_role_prior(pd.DataFrame(rows), "targets")
    week_one = out[(out["season"] == 2026) & (out["week"] == 1)].iloc[0]
    assert week_one["last_season_targets_pg"] == 6.0
    assert week_one["prior_season_targets_pg"] == 8.0
    assert week_one["expected_targets"] == 0.70 * 6.0 + 0.30 * 8.0


def test_week1_crash_lifts_prior_season() -> None:
    rows = (
        _season_rows("CLE_d_njoku", 2024, 16, 8.0)
        + _season_rows("CLE_d_njoku", 2025, 4, 3.0)
        + [
            {
                "player_id": "CLE_d_njoku",
                "season": 2026,
                "week": 1,
                "targets": 0.0,
                "expected_targets": 1.5,
            }
        ]
    )
    out = apply_early_season_role_prior(pd.DataFrame(rows), "targets")
    week_one = out[(out["season"] == 2026) & (out["week"] == 1)].iloc[0]
    assert week_one["expected_targets"] == 0.45 * 3.0 + 0.55 * 8.0


def test_week5_keeps_existing_expected_role() -> None:
    rows = (
        _season_rows("KC_xavier_worthy", 2024, 10, 8.0)
        + _season_rows("KC_xavier_worthy", 2025, 10, 6.0)
        + _season_rows("KC_xavier_worthy", 2026, 5, 2.0)
    )
    frame = pd.DataFrame(rows)
    frame.loc[(frame["season"] == 2026) & (frame["week"] == 5), "expected_targets"] = 4.2
    out = apply_early_season_role_prior(frame, "targets")
    week_five = out[(out["season"] == 2026) & (out["week"] == 5)].iloc[0]
    assert week_five["expected_targets"] == 4.2


def test_rookie_without_last_season_keeps_expected() -> None:
    rows = [
        {
            "player_id": "DAL_rookie",
            "season": 2026,
            "week": 1,
            "targets": 0.0,
            "expected_targets": 5.0,
        }
    ]
    out = apply_early_season_role_prior(pd.DataFrame(rows), "targets")
    assert out.iloc[0]["expected_targets"] == 5.0


def test_regular_season_training_weeks_are_last_two_seasons_without_playoffs() -> None:
    available = pd.DataFrame(
        {
            "season": [2023, 2024, 2024, 2025, 2025, 2025],
            "week": [18, 1, 22, 1, 18, 22],
        }
    )
    assert regular_season_training_weeks(available) == [
        (2024, 1),
        (2025, 1),
        (2025, 18),
    ]
