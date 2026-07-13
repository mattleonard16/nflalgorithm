"""Coverage for the NFL ingestion entrypoint used by the production runner."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from scripts import ingest_real_nfl_data
from scripts.production_runner import stage_prepare_week


class _PandasResult:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame


@pytest.fixture(autouse=True)
def _stub_optional_pregame_context(monkeypatch):
    """Keep ingestion unit tests offline unless a test opts into context feeds."""
    monkeypatch.setattr(
        ingest_real_nfl_data, "fetch_weekly_rosters", lambda seasons: pd.DataFrame()
    )
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_depth_charts", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_injuries", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "refresh_player_context_snapshots",
        lambda rosters, depth_charts, injuries, through_week: 0,
    )


def test_cli_defaults_include_current_season(monkeypatch) -> None:
    monkeypatch.setattr(ingest_real_nfl_data.nfl, "get_current_season", lambda roster: 2026)

    args = ingest_real_nfl_data.parse_args([])

    seasons = [int(value) for value in args.seasons.split(",")]
    assert seasons == [*ingest_real_nfl_data.default_history_seasons(2026), 2026]


def test_cli_defaults_roll_forward_previous_seasons(monkeypatch) -> None:
    monkeypatch.setattr(ingest_real_nfl_data.nfl, "get_current_season", lambda roster: 2027)

    args = ingest_real_nfl_data.parse_args([])

    assert args.seasons == "2025,2026,2027"


def test_weekly_fetch_keeps_available_history_when_current_feed_is_missing(monkeypatch) -> None:
    calls: list[list[int]] = []

    def fake_load_player_stats(seasons: list[int]) -> _PandasResult:
        calls.append(seasons)
        season = seasons[0]
        if season == 2026:
            raise FileNotFoundError("2026 player stats are not published yet")
        return _PandasResult(pd.DataFrame({"season": [season], "week": [1]}))

    monkeypatch.setattr(ingest_real_nfl_data.nfl, "get_current_season", lambda roster: 2026)
    monkeypatch.setattr(ingest_real_nfl_data.nfl, "load_player_stats", fake_load_player_stats)

    weekly = ingest_real_nfl_data.fetch_weekly_stats([2024, 2025, 2026])

    assert calls == [[2024], [2025], [2026]]
    assert weekly["season"].tolist() == [2024, 2025]


def test_weekly_fetch_propagates_unexpected_current_season_failure(monkeypatch) -> None:
    def fail_current_feed(seasons: list[int]) -> _PandasResult:
        raise ConnectionError("nflverse timed out")

    monkeypatch.setattr(ingest_real_nfl_data.nfl, "get_current_season", lambda roster: 2026)
    monkeypatch.setattr(ingest_real_nfl_data.nfl, "load_player_stats", fail_current_feed)

    with pytest.raises(ConnectionError, match="timed out"):
        ingest_real_nfl_data.fetch_weekly_stats([2026])


def test_weekly_fetch_propagates_missing_historical_feed(monkeypatch) -> None:
    def fail_historical_feed(seasons: list[int]) -> _PandasResult:
        raise FileNotFoundError("historical feed missing")

    monkeypatch.setattr(ingest_real_nfl_data.nfl, "get_current_season", lambda roster: 2026)
    monkeypatch.setattr(ingest_real_nfl_data.nfl, "load_player_stats", fail_historical_feed)

    with pytest.raises(FileNotFoundError, match="historical feed missing"):
        ingest_real_nfl_data.fetch_weekly_stats([2025])


def test_schedule_fetch_propagates_current_season_outage(monkeypatch) -> None:
    def fail_schedule_fetch(seasons: list[int]) -> _PandasResult:
        raise ConnectionError("schedule service unavailable")

    monkeypatch.setattr(ingest_real_nfl_data.nfl, "get_current_season", lambda roster: 2026)
    monkeypatch.setattr(ingest_real_nfl_data.nfl, "load_schedules", fail_schedule_fetch)

    with pytest.raises(ConnectionError, match="schedule service unavailable"):
        ingest_real_nfl_data.fetch_schedules([2026])


def test_pbp_fetch_aggregates_available_red_zone_touches_by_season(monkeypatch) -> None:
    calls: list[list[int]] = []

    def fake_load_pbp(seasons: list[int]) -> pl.DataFrame:
        calls.append(seasons)
        if seasons == [2026]:
            raise FileNotFoundError("2026 play-by-play is not published yet")
        return pl.DataFrame(
            {
                "season": [2025, 2025, 2025],
                "week": [1, 1, 1],
                "yardline_100": [10.0, 15.0, 40.0],
                "rusher_player_id": ["rush-1", "rush-1", "outside-rz"],
                "receiver_player_id": [None, "recv-1", None],
                "unused_large_column": ["x", "y", "z"],
            }
        )

    monkeypatch.setattr(ingest_real_nfl_data.nfl, "load_pbp", fake_load_pbp)

    touches = ingest_real_nfl_data.fetch_pbp_red_zone([2025, 2026])

    assert calls == [[2025], [2026]]
    assert touches.set_index("player_gsis_id")["red_zone_touches"].to_dict() == {
        "recv-1": 1,
        "rush-1": 2,
    }


def test_ingest_seasons_filters_week_and_returns_upsert_count(monkeypatch) -> None:
    weekly = pd.DataFrame({"week": [1, 2], "marker": ["keep", "drop"]})
    captured: dict[str, object] = {}

    def fake_weekly_fetch(seasons: list[int]) -> pd.DataFrame:
        captured["seasons"] = seasons
        return weekly

    monkeypatch.setattr(ingest_real_nfl_data, "fetch_weekly_stats", fake_weekly_fetch)
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_snap_counts", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_rosters", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_schedules", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_pbp_red_zone", lambda seasons: pd.DataFrame())

    def fake_transform(
        df: pd.DataFrame,
        snaps: pd.DataFrame,
        *,
        rosters: pd.DataFrame,
        schedule: pd.DataFrame,
        pbp_rz: pd.DataFrame,
    ) -> pd.DataFrame:
        captured["filtered"] = df.copy()
        return pd.DataFrame(
            {
                "player_id": ["p1"],
                "season": [2026],
                "week": [1],
                "position": ["RB"],
            }
        )

    monkeypatch.setattr(ingest_real_nfl_data, "transform_to_enhanced_stats", fake_transform)
    monkeypatch.setattr(ingest_real_nfl_data, "upsert_player_stats", lambda df: len(df))

    count = ingest_real_nfl_data.ingest_seasons([2026], through_week=1)

    assert count == 1
    assert captured["seasons"] == [2026]
    filtered = captured["filtered"]
    assert isinstance(filtered, pd.DataFrame)
    assert filtered["marker"].tolist() == ["keep"]


def test_ingest_seasons_persists_preseason_context_without_weekly_rows(monkeypatch) -> None:
    schedule = pd.DataFrame(
        {
            "game_id": ["2026_01_BAL_BUF"],
            "season": [2026],
            "week": [1],
            "home_team": ["BUF"],
            "away_team": ["BAL"],
            "gameday": ["2026-09-10"],
            "gametime": ["20:20"],
            "stadium": ["Highmark Stadium"],
        }
    )
    rosters = pd.DataFrame(
        {
            "gsis_id": ["00-0039999"],
            "full_name": ["Season Ready"],
            "position": ["WR"],
            "team": ["BUF"],
            "season": [2026],
        }
    )
    persisted: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(
        ingest_real_nfl_data,
        "fetch_weekly_stats",
        lambda seasons: pd.DataFrame(columns=["week"]),
    )
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_rosters", lambda seasons: rosters)
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_schedules", lambda seasons: schedule)

    def capture_games(frame: pd.DataFrame) -> int:
        persisted["games"] = frame.copy()
        return len(frame)

    def capture_roster(frame: pd.DataFrame) -> int:
        persisted["roster"] = frame.copy()
        return len(frame)

    monkeypatch.setattr(ingest_real_nfl_data, "upsert_games", capture_games)
    monkeypatch.setattr(ingest_real_nfl_data, "upsert_roster_players", capture_roster)

    assert ingest_real_nfl_data.ingest_seasons([2026], through_week=1) == 0
    assert persisted["games"].iloc[0].to_dict() == {
        "game_id": "2026_01_BAL_BUF",
        "season": 2026,
        "week": 1,
        "home_team": "BUF",
        "away_team": "BAL",
        "kickoff_utc": "2026-09-11T00:20:00+00:00",
        "game_date": "2026-09-10",
        "venue": "Highmark Stadium",
    }
    assert persisted["roster"].iloc[0]["gsis_id"] == "00-0039999"


def test_ingest_seasons_refreshes_week_versioned_player_context(monkeypatch) -> None:
    roster = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "gsis_id": ["rookie"],
            "full_name": ["Rookie Receiver"],
            "team": ["BUF"],
            "position": ["WR"],
            "status": ["ACT"],
        }
    )
    depth = pd.DataFrame({"season": [2026], "gsis_id": ["rookie"], "pos_rank": [1]})
    injuries = pd.DataFrame()
    captured: dict[str, object] = {}

    monkeypatch.setattr(ingest_real_nfl_data, "fetch_rosters", lambda seasons: roster)
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_weekly_rosters", lambda seasons: roster)
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_depth_charts", lambda seasons: depth)
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_injuries", lambda seasons: injuries)
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_schedules", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(
        ingest_real_nfl_data, "fetch_weekly_stats", lambda seasons: pd.DataFrame(columns=["week"])
    )
    monkeypatch.setattr(ingest_real_nfl_data, "upsert_roster_players", lambda frame: len(frame))

    def capture_refresh(rosters, depth_charts, injury_rows, *, through_week):
        captured["rosters"] = rosters
        captured["depth"] = depth_charts
        captured["injuries"] = injury_rows
        captured["through_week"] = through_week
        return 1

    monkeypatch.setattr(ingest_real_nfl_data, "refresh_player_context_snapshots", capture_refresh)

    assert ingest_real_nfl_data.ingest_seasons([2026], through_week=1) == 0
    assert captured["through_week"] == 1
    assert isinstance(captured["rosters"], pd.DataFrame)
    assert captured["depth"] is depth
    assert captured["injuries"] is injuries


def test_ingest_seasons_can_refresh_week_one_context_without_loading_outcomes(monkeypatch) -> None:
    weekly_fetches: list[list[int]] = []
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "fetch_weekly_stats",
        lambda seasons: weekly_fetches.append(seasons) or pd.DataFrame(),
    )
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_rosters", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_schedules", lambda seasons: pd.DataFrame())

    assert ingest_real_nfl_data.ingest_seasons([2026], through_week=1, stats_through_week=0) == 0
    assert weekly_fetches == []


def test_schedule_context_filters_to_requested_week() -> None:
    schedule = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 2],
            "home_team": ["BUF", "KC"],
            "away_team": ["BAL", "DEN"],
            "gameday": ["2026-09-10", "2026-09-17"],
        }
    )

    games = ingest_real_nfl_data.create_games_from_schedule(schedule, through_week=1)

    assert games[["season", "week"]].to_dict("records") == [{"season": 2026, "week": 1}]
    assert games.iloc[0]["game_id"] == "2026_W1_BAL_at_BUF"


def test_schedule_context_excludes_preseason_week_numbers() -> None:
    schedule = pd.DataFrame(
        {
            "game_type": ["PRE", "REG"],
            "season": [2026, 2026],
            "week": [1, 1],
            "home_team": ["BUF", "KC"],
            "away_team": ["BAL", "DEN"],
            "gameday": ["2026-08-13", "2026-09-10"],
        }
    )

    games = ingest_real_nfl_data.create_games_from_schedule(schedule, through_week=1)

    assert games[["home_team", "away_team"]].to_dict("records") == [
        {"home_team": "KC", "away_team": "DEN"}
    ]


def test_schedule_context_leaves_tba_kickoff_empty() -> None:
    schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "home_team": ["BUF"],
            "away_team": ["BAL"],
            "gameday": ["2026-09-10"],
            "gametime": [None],
        }
    )

    games = ingest_real_nfl_data.create_games_from_schedule(schedule, through_week=1)

    assert games.iloc[0]["kickoff_utc"] is None


def test_nonempty_schedule_with_missing_contract_fails_closed() -> None:
    malformed = pd.DataFrame({"season": [2026], "week": [1], "home_team": ["BUF"]})

    with pytest.raises(ValueError, match="Schedule data is missing required columns"):
        ingest_real_nfl_data.create_games_from_schedule(malformed, through_week=1)


def test_nonempty_roster_with_missing_contract_fails_closed() -> None:
    malformed = pd.DataFrame({"full_name": ["Season Ready"], "team": ["BUF"]})

    with pytest.raises(ValueError, match="Roster data is missing required columns"):
        ingest_real_nfl_data.upsert_roster_players(malformed)


def test_roster_context_does_not_regress_an_existing_week(tmp_path, monkeypatch) -> None:
    import config as cfg
    from schema_migrations import MigrationManager
    from utils.db import execute, fetchone

    db_path = str(tmp_path / "preseason-context.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(db_path).run()

    execute(
        """
        INSERT INTO player_dim
            (player_id, player_name, position, team, last_season, last_week, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        params=("BUF_season_ready", "Season Ready", "WR", "BUF", 2026, 6, "before"),
    )
    execute(
        """
        INSERT INTO nfl_roster_players
            (season, gsis_id, player_id, player_name, team, position, roster_status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        params=(2026, "00-stale", "BUF_stale_player", "Stale Player", "BUF", "WR", "ACT", "before"),
    )
    roster = pd.DataFrame(
        {
            "gsis_id": ["00-0039999", "00-0039999"],
            "full_name": ["Season Ready", "Season Ready"],
            "position": ["WR", "WR"],
            "team": ["LV", "BUF"],
            "season": [2026, 2026],
            "week": [1, 2],
            "status": ["ACT", "ACT"],
        }
    )

    assert ingest_real_nfl_data.upsert_roster_players(roster) == 1
    row = fetchone(
        "SELECT gsis_id, last_season, last_week FROM player_dim WHERE player_id = ?",
        params=("BUF_season_ready",),
    )
    assert row == ("00-0039999", 2026, 6)
    assert fetchone("SELECT COUNT(*) FROM player_dim") == (1,)
    assert (
        fetchone(
            """
        SELECT gsis_id, player_id, team, position, roster_week
        FROM nfl_roster_players WHERE season = ? AND gsis_id = ?
        """,
            params=(2026, "00-0039999"),
        )
        == ("00-0039999", "BUF_season_ready", "BUF", "WR", 2)
    )
    assert fetchone("SELECT COUNT(*) FROM nfl_roster_players WHERE season = ?", params=(2026,)) == (
        2,
    )


def test_roster_context_preserves_each_requested_season(tmp_path, monkeypatch) -> None:
    import config as cfg
    from schema_migrations import MigrationManager
    from utils.db import read_dataframe

    db_path = str(tmp_path / "multi-season-roster.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(db_path).run()
    roster = pd.DataFrame(
        {
            "gsis_id": ["00-0039999", "00-0039999"],
            "full_name": ["Season Ready", "Season Ready"],
            "position": ["WR", "WR"],
            "team": ["BUF", "BUF"],
            "season": [2025, 2026],
            "week": [18, 1],
            "status": ["ACT", "ACT"],
        }
    )

    ingest_real_nfl_data.upsert_roster_players(roster)

    snapshots = read_dataframe("SELECT season, gsis_id FROM nfl_roster_players ORDER BY season")
    assert snapshots.to_dict("records") == [
        {"season": 2025, "gsis_id": "00-0039999"},
        {"season": 2026, "gsis_id": "00-0039999"},
    ]
    current_identity = read_dataframe(
        "SELECT gsis_id, last_season, last_week FROM player_dim WHERE player_id = ?",
        params=("BUF_season_ready",),
    )
    assert current_identity.to_dict("records") == [
        {"gsis_id": "00-0039999", "last_season": 2026, "last_week": 0}
    ]


def test_player_context_snapshot_combines_role_availability_and_priors() -> None:
    rosters = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 1],
            "gsis_id": ["rookie", "veteran"],
            "full_name": ["Rookie Receiver", "Veteran Receiver"],
            "team": ["BUF", "BUF"],
            "position": ["WR", "WR"],
            "status": ["ACT", "ACT"],
            "years_exp": [0, 5],
            "rookie_year": [2026, 2021],
            "birth_date": ["2003-05-01", "1998-05-01"],
        }
    )
    depth = pd.DataFrame(
        {
            "season": [2026, 2026],
            "gsis_id": ["rookie", "veteran"],
            "pos_abb": ["WR", "WR"],
            "pos_rank": [1, 2],
            "pos_slot": [1, 2],
            "dt": ["2026-08-30T00:00:00Z", "2026-08-30T00:00:00Z"],
        }
    )
    injuries = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "gsis_id": ["veteran"],
            "report_status": ["Questionable"],
            "practice_status": ["Limited Participation in Practice"],
            "report_primary_injury": ["Hamstring"],
        }
    )
    history = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [17, 18],
            "gsis_id": ["veteran", "veteran"],
            "team": ["MIA", "MIA"],
            "snap_count": [52.0, 56.0],
            "snap_percentage": [78.0, 82.0],
            "rushing_attempts": [0.0, 0.0],
            "targets": [8.0, 10.0],
            "passing_attempts": [0.0, 0.0],
            "target_share": [0.22, 0.25],
            "air_yards": [85.0, 100.0],
            "yac_yards": [25.0, 30.0],
            "red_zone_touches": [1.0, 2.0],
        }
    )

    snapshots = ingest_real_nfl_data.build_player_context_snapshots(
        rosters,
        depth,
        injuries,
        history,
        target_week=1,
        captured_at="2026-09-01T00:00:00+00:00",
    ).set_index("gsis_id")

    rookie = snapshots.loc["rookie"]
    assert rookie["is_rookie"] == 1
    assert rookie["is_starter"] == 1
    assert rookie["expected_targets"] > 0
    assert rookie["expected_snap_percentage"] > 0
    assert rookie["uncertainty_multiplier"] > 1.0

    veteran = snapshots.loc["veteran"]
    assert veteran["is_new_team"] == 1
    assert veteran["depth_rank"] == 2
    assert veteran["injury_status"] == "Questionable"
    assert veteran["practice_status"] == "Limited Participation in Practice"
    assert veteran["expected_targets"] > 0
    assert veteran["uncertainty_multiplier"] > 1.0


def test_player_context_excludes_depth_rows_after_target_week_cutoff() -> None:
    rosters = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "gsis_id": ["player"],
            "full_name": ["Depth Player"],
            "team": ["BUF"],
            "position": ["WR"],
            "status": ["ACT"],
        }
    )
    depth = pd.DataFrame(
        {
            "season": [2026, 2026],
            "gsis_id": ["player", "player"],
            "pos_abb": ["WR", "WR"],
            "pos_rank": [1, 3],
            "dt": ["2026-09-01T00:00:00Z", "2026-09-20T00:00:00Z"],
        }
    )

    snapshot = ingest_real_nfl_data.build_player_context_snapshots(
        rosters,
        depth,
        pd.DataFrame(),
        pd.DataFrame(),
        target_week=1,
        target_cutoffs={2026: "2026-09-10T17:00:00Z"},
        captured_at="2026-09-08T00:00:00Z",
    )

    assert snapshot.iloc[0]["depth_rank"] == 1


def test_player_context_snapshots_preserve_week_history(tmp_path, monkeypatch) -> None:
    import config as cfg
    from schema_migrations import MigrationManager
    from utils.db import read_dataframe

    db_path = str(tmp_path / "player-context.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(db_path).run()

    roster = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "gsis_id": ["rookie"],
            "full_name": ["Rookie Receiver"],
            "team": ["BUF"],
            "position": ["WR"],
            "status": ["ACT"],
            "years_exp": [0],
            "rookie_year": [2026],
        }
    )
    week_one = ingest_real_nfl_data.build_player_context_snapshots(
        roster,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        target_week=1,
        captured_at="2026-09-01T00:00:00Z",
    )
    week_two = ingest_real_nfl_data.build_player_context_snapshots(
        roster.assign(week=2),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        target_week=2,
        captured_at="2026-09-08T00:00:00Z",
    )

    assert ingest_real_nfl_data.upsert_player_context_snapshots(week_one) == 1
    assert ingest_real_nfl_data.upsert_player_context_snapshots(week_two) == 1

    stored = read_dataframe("""
        SELECT week, is_rookie, expected_targets, uncertainty_multiplier, captured_at
        FROM nfl_player_context_snapshots
        ORDER BY week
        """)
    assert stored["week"].tolist() == [1, 2]
    assert stored["is_rookie"].tolist() == [1, 1]
    assert (stored["expected_targets"] > 0).all()
    assert (stored["uncertainty_multiplier"] > 1.0).all()
    assert stored["captured_at"].tolist() == [
        "2026-09-01T00:00:00Z",
        "2026-09-08T00:00:00Z",
    ]


@pytest.mark.parametrize(
    ("seasons", "through_week", "message"),
    [
        ([], 18, "At least one NFL season is required"),
        ([2026], 0, "through_week must be between 1 and 22"),
        ([2026], 23, "through_week must be between 1 and 22"),
    ],
)
def test_ingest_seasons_rejects_invalid_scope(
    seasons: list[int], through_week: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ingest_real_nfl_data.ingest_seasons(seasons, through_week=through_week)


def test_ingest_seasons_rejects_weekly_feed_without_week_column(monkeypatch) -> None:
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_rosters", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_schedules", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "fetch_weekly_stats",
        lambda seasons: pd.DataFrame({"season": [2026]}),
    )

    with pytest.raises(ValueError, match="required 'week' column"):
        ingest_real_nfl_data.ingest_seasons([2026], through_week=1)


def test_ingest_seasons_handles_no_rows_before_cutoff(monkeypatch) -> None:
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_rosters", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_schedules", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "fetch_weekly_stats",
        lambda seasons: pd.DataFrame({"week": [2]}),
    )

    assert ingest_real_nfl_data.ingest_seasons([2026], through_week=1) == 0


def test_ingest_seasons_handles_empty_transformation(monkeypatch) -> None:
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "fetch_weekly_stats",
        lambda seasons: pd.DataFrame({"week": [1]}),
    )
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_snap_counts", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_rosters", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_schedules", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest_real_nfl_data, "fetch_pbp_red_zone", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "transform_to_enhanced_stats",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    assert ingest_real_nfl_data.ingest_seasons([2026], through_week=1) == 0


def test_production_stage_reports_canonical_prepare_summary(monkeypatch) -> None:
    from scripts import prepare_nfl_week

    monkeypatch.setattr(
        prepare_nfl_week,
        "prepare_week",
        lambda season, week, refresh_history: {
            "season": season,
            "week": week,
            "predictions": 42,
        },
    )

    result = stage_prepare_week(2026, 1, refresh_history=False)

    assert result == {
        "status": "ok",
        "stage": "prepare_week",
        "season": 2026,
        "week": 1,
        "predictions": 42,
    }


def test_production_stage_fails_closed_on_missing_pregame_context(monkeypatch) -> None:
    from scripts import prepare_nfl_week

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("No roster players are available")

    monkeypatch.setattr(prepare_nfl_week, "prepare_week", fail_prepare)
    result = stage_prepare_week(2026, 1)

    assert result == {
        "status": "error",
        "stage": "prepare_week",
        "error": "No roster players are available",
    }


def test_production_stage_reports_unexpected_ingestion_failure(monkeypatch) -> None:
    from scripts import prepare_nfl_week

    def fail_ingestion(*args, **kwargs):
        raise ConnectionError("nflverse timed out")

    monkeypatch.setattr(prepare_nfl_week, "prepare_week", fail_ingestion)

    result = stage_prepare_week(2026, 1)

    assert result == {
        "status": "error",
        "stage": "prepare_week",
        "error": "nflverse timed out",
    }
