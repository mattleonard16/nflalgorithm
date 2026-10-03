from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest_real_nfl_data import build_player_context_snapshots
from utils.nfl_backtest import compare_walk_forward
from utils.nfl_replay import (
    first_kickoff_cutoffs,
    players_without_stats,
    rekey_to_stat_ids,
    require_scratch_database,
    roster_for_week,
)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [3, 3, 4],
            "home_team": ["BUF", "KC", "BUF"],
            "away_team": ["MIA", "LV", "NE"],
            "kickoff_utc": [
                "2025-09-21T17:00:00+00:00",
                "2025-09-19T00:15:00+00:00",
                "2025-09-28T17:00:00+00:00",
            ],
        }
    )


def _roster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025],
            "week": [3],
            "gsis_id": ["wr1"],
            "full_name": ["Receiver One"],
            "team": ["BUF"],
            "position": ["WR"],
            "status": ["ACT"],
        }
    )


def test_the_cutoff_is_the_weeks_first_kickoff() -> None:
    assert first_kickoff_cutoffs(_games(), season=2025, week=3) == {
        2025: "2025-09-19T00:15:00+00:00"
    }


def test_a_depth_chart_dated_after_the_first_kickoff_is_ignored() -> None:
    depth = pd.DataFrame(
        {
            "season": [2025, 2025],
            "gsis_id": ["wr1", "wr1"],
            "dt": ["2025-09-17T10:00:00Z", "2025-09-20T10:00:00Z"],
            "pos_abb": ["WR", "WR"],
            "pos_rank": [1, 3],
        }
    )

    snapshots = build_player_context_snapshots(
        _roster(),
        depth,
        pd.DataFrame(),
        pd.DataFrame(),
        target_week=3,
        target_cutoffs=first_kickoff_cutoffs(_games(), season=2025, week=3),
    )

    assert snapshots["depth_rank"].tolist() == [1]


def test_the_roster_is_that_weeks_and_game_day_inactives_count_as_active() -> None:
    rosters = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [2, 3, 3],
            "gsis_id": ["cut_in_week_2", "inactive", "on_ir"],
            "status": ["ACT", "INA", "RES"],
        }
    )

    roster = roster_for_week(rosters, season=2025, week=3)

    assert roster[["gsis_id", "status"]].to_dict("records") == [
        {"gsis_id": "inactive", "status": "ACT"},
        {"gsis_id": "on_ir", "status": "RES"},
    ]


def test_the_replay_refuses_the_production_database(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("nfl_data.db").touch()

    with pytest.raises(ValueError, match="production database"):
        require_scratch_database(tmp_path / "nfl_data.db", Path("nfl_data.db"))


def test_the_replay_accepts_a_scratch_copy(tmp_path) -> None:
    scratch = tmp_path / "replay.db"
    scratch.touch()

    assert require_scratch_database(scratch, tmp_path / "nfl_data.db") == scratch.resolve()


def test_players_projected_without_a_stat_line_are_counted_once() -> None:
    predictions = pd.DataFrame(
        {
            "player_id": ["played", "sat", "sat"],
            "market": ["receiving_yards", "receiving_yards", "receptions"],
        }
    )
    actuals = pd.DataFrame({"season": [2025], "week": [3], "player_id": ["played"]})

    assert players_without_stats(predictions, actuals, season=2025, week=3) == 1


def test_a_replay_does_not_compare_against_a_walk_forward() -> None:
    walk_forward = {"season": 2025, "weeks_evaluated": [1], "by_market": {}}
    replay = {**walk_forward, "mode": "replay"}

    comparison = compare_walk_forward(walk_forward, replay)

    assert comparison["passed"] is False
    assert "mode differs between baseline and candidate" in comparison["blockers"]


def test_predictions_take_the_stat_rows_ids_through_gsis() -> None:
    predictions = pd.DataFrame({"player_id": ["ARI_greg_dortch", "ARI_no_stats"], "mu": [40, 9]})
    id_map = pd.DataFrame({"roster_id": ["ARI_greg_dortch"], "stat_id": ["ARI_g_dortch"]})

    result = rekey_to_stat_ids(predictions, id_map)

    assert result["player_id"].tolist() == ["ARI_g_dortch", "ARI_no_stats"]
