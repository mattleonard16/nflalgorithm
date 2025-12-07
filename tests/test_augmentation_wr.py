import pandas as pd

from data_pipeline import DataPipeline
from utils.player_id_utils import make_player_id


def test_wr_historical_augments_with_single_game(monkeypatch):
    """WRs with meaningful usage and one game should be added to baseline and keep player_id."""

    # Fake historical data: one WR with 1 game and solid receiving, one RB with only 1 game (ignored)
    hist_df = pd.DataFrame([
        {
            "player_id": "WR1_TEAM",
            "name": "Alpha Receiver",
            "team": "AAA",
            "position": "WR",
            "avg_rush": 5.0,
            "avg_rec": 60.0,
            "avg_targets": 7.0,
            "age": 25,
            "games_played": 1,
        },
        {
            "player_id": "RB1_TEAM",
            "name": "Depth Back",
            "team": "BBB",
            "position": "RB",
            "avg_rush": 40.0,
            "avg_rec": 5.0,
            "avg_targets": 1.0,
            "age": 26,
            "games_played": 1,  # should be filtered out for non-WR
        },
    ])

    monkeypatch.setattr("data_pipeline.read_dataframe", lambda *_, **__: hist_df)

    baseline = pd.DataFrame([
        {"name": "Existing Guy", "team": "CCC", "position": "WR", "age_2024": 28},
    ])

    dp = DataPipeline.__new__(DataPipeline)
    augmented = dp._augment_baseline_with_historical_players(baseline, season=2025, week=5)

    # WR with 1 game and strong receiving is added
    assert (augmented["name"] == "Alpha Receiver").any()
    # Non-WR with only 1 game is ignored
    assert not (augmented["name"] == "Depth Back").any()

    # player_id is present and unique
    assert "player_id" in augmented.columns
    assert augmented["player_id"].is_unique
    # player_id generated for existing baseline rows as well
    assert make_player_id("Existing Guy", "CCC") in set(augmented["player_id"])
