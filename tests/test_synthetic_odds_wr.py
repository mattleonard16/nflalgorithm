import pandas as pd
import pytest

from data_pipeline import DataPipeline
from utils.player_id_utils import make_player_id


def test_live_weekly_odds_mode_rejects_synthetic_fallback():
    from scripts.prop_line_scraper import NFLPropScraper

    scraper = NFLPropScraper.__new__(NFLPropScraper)
    scraper.odds_api_key = None

    with pytest.raises(RuntimeError, match="Live odds"):
        scraper.get_upcoming_week_props(1, 2026, allow_synthetic=False)


def test_weekly_scraper_persists_canonical_timestamped_odds(monkeypatch):
    from scripts import prop_line_scraper

    statements: list[tuple[str, tuple[object, ...]]] = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def commit(self):
            return None

    monkeypatch.setattr(prop_line_scraper, "get_connection", lambda: FakeConnection())
    monkeypatch.setattr(
        prop_line_scraper,
        "execute",
        lambda query, params, conn: statements.append((query, params)),
    )

    scraper = prop_line_scraper.NFLPropScraper.__new__(prop_line_scraper.NFLPropScraper)
    saved = scraper.save_weekly_odds(
        [
            {
                "event_id": "event-1",
                "player_id": "BUF_alpha_receiver",
                "player": "Alpha Receiver",
                "team": "BUF",
                "position": "WR",
                "book": "Book",
                "stat": "receiving_yards",
                "line": 55.5,
                "over_odds": -110,
                "under_odds": -105,
            }
        ],
        week=1,
        season=2026,
    )

    assert saved == 1
    assert len(statements) == 1
    query, params = statements[0]
    assert "INSERT INTO weekly_odds" in query
    assert "weekly_prop_lines" not in query
    assert params[:8] == (
        "event-1",
        2026,
        1,
        "BUF_alpha_receiver",
        "receiving_yards",
        "Book",
        55.5,
        -110,
    )
    assert params[8] == -105
    assert isinstance(params[9], str)


def test_weekly_event_selection_uses_entire_requested_schedule() -> None:
    from scripts.prop_line_scraper import NFLPropScraper

    schedule = pd.DataFrame(
        {"kickoff_utc": [f"2026-09-{10 + day:02d}T17:00:00Z" for day in range(12)]}
    )
    requested = [
        {"id": f"week-event-{day}", "commence_time": kickoff}
        for day, kickoff in enumerate(schedule["kickoff_utc"])
    ]
    unrelated = [{"id": "other-week", "commence_time": "2026-10-10T17:00:00Z"}]

    selected = NFLPropScraper._select_scheduled_events(unrelated + requested, schedule)

    assert [event["id"] for event in selected] == [f"week-event-{day}" for day in range(12)]


def _sample_wr_stats():
    return pd.DataFrame(
        [
            {
                "player_id": make_player_id("Alpha Receiver", "AAA"),
                "name": "Alpha Receiver",
                "team": "AAA",
                "position": "WR",
                "games_played": 1,
                "receiving_yards": 50.0,  # Higher to pass min_wr_baseline (line * 0.8 >= 35)
                "rolling_targets": 7.0,  # 7 * 8 = 56 yards baseline
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
        ]
    )


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
