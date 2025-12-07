import pandas as pd

from data_pipeline import DataPipeline
from utils.player_id_utils import make_player_id


def _sample_wr_stats():
    return pd.DataFrame([
        {
            "player_id": make_player_id("Alpha Receiver", "AAA"),
            "name": "Alpha Receiver",
            "team": "AAA",
            "position": "WR",
            "games_played": 1,
            "receiving_yards": 50.0,  # Higher to pass min_wr_baseline (line * 0.8 >= 35)
            "rolling_targets": 7.0,   # 7 * 8 = 56 yards baseline
            "rushing_yards": 2.0,
            "rolling_air_yards": 40.0,
            "game_id": "2025_W10_BBB_at_AAA",
        },
        {
            "player_id": make_player_id("Inactive Guy", "BBB"),
            "name": "Inactive Guy",
            "team": "BBB",
            "position": "WR",
            "games_played": 0,
            "receiving_yards": 0.0,
            "rolling_targets": 0.0,
            "rushing_yards": 0.0,
            "rolling_air_yards": 0.0,
            "game_id": "2025_W10_BBB_at_AAA",
        },
    ])


def test_synthesize_weekly_odds_wr_generates_receiving_lines():
    stats = _sample_wr_stats()
    dp = DataPipeline.__new__(DataPipeline)
    synthetic = dp._synthesize_weekly_odds(stats, season=2025, week=10)

    assert not synthetic.empty
    wr_lines = synthetic[
        (synthetic["player_id"] == stats.iloc[0]["player_id"])
        & (synthetic["market"] == "receiving_yards")
    ]
    assert not wr_lines.empty
    # Inactive player should not get synthetic odds
    assert synthetic[synthetic["player_id"] == stats.iloc[1]["player_id"]].empty


def test_fetch_real_weekly_odds_adds_synthetic_receiving_when_missing(monkeypatch):
    stats = _sample_wr_stats()

    class FakeScraper:
        def get_upcoming_week_props(self, week, season):
            # Only rushing odds provided; receiving should be synthesized
            return [
                {
                    "player": "Alpha Receiver",
                    "team": "AAA",
                    "stat": "rushing_yards",
                    "line": 10.5,
                    "over_odds": -110,
                    "book": "RealBook",
                }
            ]

    monkeypatch.setattr("scripts.prop_line_scraper.NFLPropScraper", FakeScraper)

    dp = DataPipeline.__new__(DataPipeline)
    odds = dp._fetch_real_weekly_odds(season=2025, week=10, player_stats=stats)

    receiving_rows = odds[
        (odds["player_id"] == stats.iloc[0]["player_id"])
        & (odds["market"] == "receiving_yards")
        & (odds["sportsbook"] == "SimBook")
    ]
    assert not receiving_rows.empty
