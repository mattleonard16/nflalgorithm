"""Core-player defects in the NFL ingest/context layer.

Each test pins one verified defect from the 2025-2026 real-data audit:
stale injury carry-forward, week-18 rest and postseason volume drag, KR/PR
depth rows demoting starters, Out starters keeping rank 1, team-scoped
games_played, history seasons truncated by --through-week, nickname snap
misses, and offseason depth rows leaking into a season with no cutoff.
"""

from __future__ import annotations

import pandas as pd

from scripts import ingest_real_nfl_data as ingest


def _roster(*players: tuple[str, str, str], season: int = 2025, week: int = 1) -> pd.DataFrame:
    """Build a roster frame from (gsis_id, position, team) tuples."""
    return pd.DataFrame(
        {
            "season": [season] * len(players),
            "week": [week] * len(players),
            "gsis_id": [gsis for gsis, _, _ in players],
            "full_name": [f"Player {gsis}" for gsis, _, _ in players],
            "team": [team for _, _, team in players],
            "position": [position for _, position, _ in players],
            "status": ["ACT"] * len(players),
        }
    )


def _history(gsis_id: str, rows: list[tuple[int, int, float, float]], **const) -> pd.DataFrame:
    """History rows from (season, week, snap_percentage, passing_attempts)."""
    frame = pd.DataFrame(
        rows, columns=["season", "week", "snap_percentage", "passing_attempts"]
    )
    frame["gsis_id"] = gsis_id
    frame["team"] = const.get("team", "DAL")
    for column in (
        "snap_count",
        "rushing_attempts",
        "targets",
        "target_share",
        "air_yards",
        "yac_yards",
        "red_zone_touches",
    ):
        frame[column] = 0.0
    return frame


def _snapshot(rosters, depth=None, injuries=None, history=None, **kwargs) -> pd.DataFrame:
    return ingest.build_player_context_snapshots(
        rosters,
        depth if depth is not None else pd.DataFrame(),
        injuries if injuries is not None else pd.DataFrame(),
        history if history is not None else pd.DataFrame(),
        captured_at="2025-09-01T00:00:00+00:00",
        **kwargs,
    ).set_index("gsis_id")


# 1. Injury status is the target week's report, not the last one ever filed.


def test_injury_status_from_an_earlier_week_does_not_carry_forward() -> None:
    rosters = _roster(("wr", "WR", "BUF"), week=12)
    injuries = pd.DataFrame(
        {
            "season": [2025],
            "week": [3],
            "gsis_id": ["wr"],
            "report_status": ["Questionable"],
            "practice_status": ["Limited Participation in Practice"],
            "report_primary_injury": ["Hamstring"],
        }
    )
    history = _history("wr", [(2025, week, 80.0, 0.0) for week in range(1, 12)], team="BUF")
    history["targets"] = 8.0

    week_three = _snapshot(rosters.assign(week=3), injuries=injuries, history=history, target_week=3)
    week_twelve = _snapshot(rosters, injuries=injuries, history=history, target_week=12)

    assert week_three.loc["wr", "injury_status"] == "Questionable"
    healthy = week_twelve.loc["wr"]
    assert pd.isna(healthy["injury_status"])
    assert pd.isna(healthy["practice_status"])
    assert pd.isna(healthy["primary_injury"])
    assert healthy["uncertainty_multiplier"] == 1.0
    assert healthy["expected_targets"] == 8.0


def test_out_report_for_the_target_week_still_zeroes_volume() -> None:
    rosters = _roster(("rb", "RB", "BUF"), week=5)
    injuries = pd.DataFrame(
        {"season": [2025], "week": [5], "gsis_id": ["rb"], "report_status": ["Out"]}
    )
    history = _history("rb", [(2025, week, 70.0, 0.0) for week in range(1, 5)], team="BUF")
    history["rushing_attempts"] = 15.0

    snapshot = _snapshot(rosters, injuries=injuries, history=history, target_week=5)

    assert snapshot.loc["rb", "injury_status"] == "Out"
    assert snapshot.loc["rb", "expected_rushing_attempts"] == 0.0


# 2. Week-18 rest, postseason and partial games do not drive volume priors.


def test_week_one_prior_ignores_week_eighteen_rest_and_postseason() -> None:
    rosters = _roster(("qb", "QB", "DAL"), season=2026, week=1)
    regular = [(2025, week, 100.0, 36.0) for week in range(10, 18)]
    rest = [(2025, 18, 5.0, 2.0)]
    playoffs = [(2025, 19, 100.0, 20.0), (2025, 20, 100.0, 18.0)]
    history = _history("qb", regular + rest + playoffs)

    snapshot = _snapshot(rosters, history=history, target_week=1)

    assert snapshot.loc["qb", "expected_passing_attempts"] == 36.0
    assert snapshot.loc["qb", "expected_snap_percentage"] == 100.0
    assert snapshot.loc["qb", "prior_source"] == "history"


def test_partial_game_threshold_is_relative_to_the_players_own_median() -> None:
    # Dak Prescott's 2025 week-18 rest: 51% of a 100% median, 11 attempts.
    rosters = _roster(("qb", "QB", "DAL"), season=2026, week=1)
    history = _history(
        "qb", [(2025, week, 100.0, 36.0) for week in range(12, 18)] + [(2025, 18, 51.0, 11.0)]
    )

    snapshot = _snapshot(rosters, history=history, target_week=1)

    assert snapshot.loc["qb", "expected_passing_attempts"] == 36.0


def test_rotational_players_keep_their_low_snap_weeks() -> None:
    # A 35%-snap back's 12% week is ordinary variance, not a partial game.
    rosters = _roster(("rb2", "RB", "DAL"), season=2025, week=6)
    history = _history("rb2", [(2025, 1, 35.0, 0.0), (2025, 2, 35.0, 0.0), (2025, 3, 12.0, 0.0)])
    history["rushing_attempts"] = [8.0, 8.0, 2.0]

    snapshot = _snapshot(rosters, history=history, target_week=6)

    expected = history["rushing_attempts"].ewm(span=3, min_periods=1).mean().iloc[-1]
    assert snapshot.loc["rb2", "expected_rushing_attempts"] == expected


def test_rows_without_measured_snaps_are_kept() -> None:
    # An unmatched snap merge stores 0; those games still count.
    rosters = _roster(("te", "TE", "TEN"), season=2025, week=4)
    history = _history("te", [(2025, week, 0.0, 0.0) for week in range(1, 4)], team="TEN")
    history["targets"] = 5.0

    snapshot = _snapshot(rosters, history=history, target_week=4)

    assert snapshot.loc["te", "expected_targets"] == 5.0
    assert snapshot.loc["te", "prior_source"] == "history"


def test_postseason_only_history_falls_back_to_that_history() -> None:
    rosters = _roster(("qb", "QB", "DAL"), season=2026, week=1)
    history = _history("qb", [(2025, 19, 100.0, 30.0)])

    snapshot = _snapshot(rosters, history=history, target_week=1)

    assert snapshot.loc["qb", "expected_passing_attempts"] == 30.0


def test_regular_season_length_follows_the_schedule_era() -> None:
    assert ingest.regular_season_last_week(2020) == 17
    assert ingest.regular_season_last_week(2021) == 18
    flags = ingest._is_regular_season_week(
        pd.Series([2020, 2020, 2021, 2021]), pd.Series([17, 18, 18, 19])
    )
    assert flags.tolist() == [True, False, True, False]


# 4. Depth chart: offensive skill slots only, best rank on the latest chart.


def _depth(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["season"] = 2025
    return frame


def test_kick_return_slot_does_not_demote_a_starting_back() -> None:
    rosters = _roster(("rb", "RB", "LV"))
    depth = _depth(
        [
            {"gsis_id": "rb", "pos_abb": "RB", "pos_slot": 11, "pos_rank": 1,
             "dt": "2025-09-01T00:00:00Z"},
            {"gsis_id": "rb", "pos_abb": "KR", "pos_slot": 5, "pos_rank": 2,
             "dt": "2025-09-01T00:00:00Z"},
            {"gsis_id": "rb", "pos_abb": "PR", "pos_slot": 6, "pos_rank": 3,
             "dt": "2025-09-01T00:00:00Z"},
        ]
    )

    for ordered in (depth, depth.iloc[::-1]):
        snapshot = _snapshot(rosters, depth=ordered, target_week=1)
        assert snapshot.loc["rb", "depth_rank"] == 1
        assert snapshot.loc["rb", "is_starter"] == 1
        assert snapshot.loc["rb", "depth_position"] == "RB"


def test_depth_uses_only_the_players_latest_chart() -> None:
    rosters = _roster(("wr", "WR", "MIA"))
    depth = _depth(
        [
            {"gsis_id": "wr", "pos_abb": "WR", "pos_slot": 1, "pos_rank": 1,
             "dt": "2025-08-01T00:00:00Z"},
            {"gsis_id": "wr", "pos_abb": "WR", "pos_slot": 2, "pos_rank": 3,
             "dt": "2025-09-01T00:00:00Z"},
        ]
    )

    snapshot = _snapshot(rosters, depth=depth, target_week=1)

    assert snapshot.loc["wr", "depth_rank"] == 3
    assert snapshot.loc["wr", "is_starter"] == 0


def test_season_without_a_kickoff_cutoff_keeps_no_depth_rows() -> None:
    # 2025's chart runs into March 2026; with no cutoff it would leak the
    # offseason move (a PIT KR/RB2 row) into the season's snapshot.
    rosters = pd.concat(
        [_roster(("a", "RB", "CAR"), season=2025), _roster(("b", "RB", "BUF"), season=2026)],
        ignore_index=True,
    )
    depth = pd.DataFrame(
        {
            "season": [2025, 2026],
            "gsis_id": ["a", "b"],
            "pos_abb": ["RB", "RB"],
            "pos_rank": [2, 1],
            "dt": ["2026-03-14T07:32:09Z", "2026-09-01T00:00:00Z"],
        }
    )

    snapshot = _snapshot(
        rosters, depth=depth, target_week=1, target_cutoffs={2026: "2026-09-10T00:00:00Z"}
    )

    assert pd.isna(snapshot.loc["a", "depth_rank"])
    assert snapshot.loc["b", "depth_rank"] == 1


# Effective depth rank: a starter ruled out promotes the next man up.


def _qb_room(qb1_report=None, qb1_status="ACT") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rosters = _roster(("qb1", "QB", "SF"), ("qb2", "QB", "SF"), ("qb3", "QB", "SF"), week=8)
    rosters.loc[rosters["gsis_id"] == "qb1", "status"] = qb1_status
    depth = _depth(
        [
            {"gsis_id": gsis, "team": "SF", "pos_abb": "QB", "pos_slot": 9, "pos_rank": rank,
             "dt": "2025-10-20T00:00:00Z"}
            for gsis, rank in (("qb1", 1), ("qb2", 2), ("qb3", 3))
        ]
    )
    injuries = pd.DataFrame(
        {"season": [2025], "week": [8], "gsis_id": ["qb1"], "report_status": [qb1_report]}
    )
    return rosters, depth, injuries


def test_out_starter_promotes_the_backup_to_starter() -> None:
    rosters, depth, injuries = _qb_room("Out")

    snapshot = _snapshot(rosters, depth=depth, injuries=injuries, target_week=8)

    assert snapshot.loc["qb2", "depth_rank"] == 1
    assert snapshot.loc["qb2", "is_starter"] == 1
    assert snapshot.loc["qb3", "depth_rank"] == 2
    assert snapshot.loc["qb1", "depth_rank"] == 1
    assert snapshot.loc["qb1", "expected_passing_attempts"] == 0.0


def test_reserve_list_starter_promotes_the_backup() -> None:
    rosters, depth, injuries = _qb_room(None, qb1_status="RES")

    snapshot = _snapshot(rosters, depth=depth, injuries=injuries, target_week=8)

    assert snapshot.loc["qb2", "is_starter"] == 1


def test_questionable_starter_keeps_the_job() -> None:
    rosters, depth, injuries = _qb_room("Questionable")

    snapshot = _snapshot(rosters, depth=depth, injuries=injuries, target_week=8)

    assert snapshot.loc["qb1", "is_starter"] == 1
    assert snapshot.loc["qb2", "depth_rank"] == 2
    assert snapshot.loc["qb2", "is_starter"] == 0


def test_last_weeks_out_report_does_not_promote_the_backup() -> None:
    rosters, depth, injuries = _qb_room("Out")
    injuries["week"] = 7

    snapshot = _snapshot(rosters, depth=depth, injuries=injuries, target_week=8)

    assert snapshot.loc["qb1", "is_starter"] == 1
    assert snapshot.loc["qb2", "depth_rank"] == 2


def test_rank_gap_without_an_absent_player_is_not_promoted() -> None:
    # Rank 2 with no rank-1 row in the feeds: no evidence anyone is out.
    rosters = _roster(("wr", "WR", "BUF"))
    depth = _depth(
        [{"gsis_id": "wr", "pos_abb": "WR", "pos_slot": 2, "pos_rank": 2,
          "dt": "2025-09-01T00:00:00Z"}]
    )

    snapshot = _snapshot(rosters, depth=depth, target_week=1)

    assert snapshot.loc["wr", "depth_rank"] == 2


def test_wide_receiver_ranks_count_across_slots() -> None:
    # nflverse pos_rank runs 1..n over every WR slot; WR1 out lifts WR2 to 1.
    rosters = _roster(("wr1", "WR", "MIA"), ("wr2", "WR", "MIA"), ("wr4", "WR", "MIA"))
    depth = _depth(
        [
            {"gsis_id": "wr1", "team": "MIA", "pos_abb": "WR", "pos_slot": 1, "pos_rank": 1,
             "dt": "2025-09-01T00:00:00Z"},
            {"gsis_id": "wr2", "team": "MIA", "pos_abb": "WR", "pos_slot": 2, "pos_rank": 2,
             "dt": "2025-09-01T00:00:00Z"},
            {"gsis_id": "wr4", "team": "MIA", "pos_abb": "WR", "pos_slot": 1, "pos_rank": 4,
             "dt": "2025-09-01T00:00:00Z"},
        ]
    )
    injuries = pd.DataFrame(
        {"season": [2025], "week": [1], "gsis_id": ["wr1"], "report_status": ["Out"]}
    )

    snapshot = _snapshot(rosters, depth=depth, injuries=injuries, target_week=1)

    assert snapshot.loc["wr2", "depth_rank"] == 1
    assert snapshot.loc["wr4", "depth_rank"] == 3


# 5. games_played follows the player across a trade and counts REG games only.


def test_games_played_survives_a_trade_and_skips_postseason() -> None:
    weekly = pd.DataFrame(
        {
            "position": ["WR"] * 4,
            "player_id": ["00-0031381"] * 4,
            "player_name": ["D.Adams"] * 4,
            "player_display_name": ["Davante Adams"] * 4,
            "team": ["LV", "LV", "NYJ", "NYJ"],
            "opponent_team": ["MIA", "KC", "NE", "BUF"],
            "season": [2024] * 4,
            "week": [5, 6, 7, 19],
            "season_type": ["REG", "REG", "REG", "POST"],
            "rushing_yards": [0.0] * 4,
            "receiving_yards": [50.0] * 4,
            "receptions": [4.0] * 4,
            "targets": [7.0] * 4,
            "passing_yards": [0.0] * 4,
            "attempts": [0.0] * 4,
            "carries": [0.0] * 4,
        }
    )

    result = ingest.transform_to_enhanced_stats(weekly, pd.DataFrame())

    by_week = result.set_index("week")["games_played"]
    assert by_week.to_dict() == {5: 1, 6: 2, 7: 3, 19: 3}


# 6. --through-week scopes the newest season only.


def test_through_week_caps_only_the_named_season() -> None:
    schedule = pd.DataFrame(
        {
            "season": [2025, 2025, 2026, 2026],
            "week": [7, 15, 6, 7],
            "game_type": ["REG", "REG", "REG", "REG"],
            "home_team": ["BUF", "DEN", "DAL", "PHI"],
            "away_team": ["MIA", "NE", "NYG", "WAS"],
            "gameday": ["2025-10-19", "2025-12-14", "2026-10-11", "2026-10-18"],
        }
    )

    games = ingest.create_games_from_schedule(
        schedule, through_week=6, through_week_season=2026
    )

    assert sorted(zip(games["season"], games["week"])) == [(2025, 7), (2025, 15), (2026, 6)]


def test_ingest_keeps_history_weeks_past_through_week(monkeypatch) -> None:
    weekly = pd.DataFrame(
        {"season": [2025, 2025, 2026, 2026], "week": [6, 12, 6, 7], "marker": list("abcd")}
    )
    captured: dict[str, object] = {}
    for name in (
        "fetch_rosters",
        "fetch_weekly_rosters",
        "fetch_depth_charts",
        "fetch_injuries",
        "fetch_schedules",
        "fetch_snap_counts",
        "fetch_pbp_red_zone",
    ):
        monkeypatch.setattr(ingest, name, lambda seasons: pd.DataFrame())
    monkeypatch.setattr(ingest, "fetch_weekly_stats", lambda seasons: weekly)
    monkeypatch.setattr(ingest, "upsert_games", lambda games: 0)
    monkeypatch.setattr(ingest, "upsert_roster_players", lambda rosters: 0)
    monkeypatch.setattr(
        ingest, "refresh_player_context_snapshots", lambda *args, **kwargs: 0
    )

    def fake_transform(df, snaps, **kwargs):
        captured["markers"] = df["marker"].tolist()
        return pd.DataFrame({"player_id": ["p"], "season": [2026], "week": [6], "position": ["WR"]})

    monkeypatch.setattr(ingest, "transform_to_enhanced_stats", fake_transform)
    monkeypatch.setattr(ingest, "upsert_player_stats", lambda df: len(df))
    monkeypatch.setattr(ingest, "purge_unscoped_player_stats", lambda seasons: 0)

    ingest.ingest_seasons([2025, 2026], through_week=6)

    assert captured["markers"] == ["a", "b", "c"]


# 7. Snap counts join through PFR ids before falling back to names.


def test_snap_merge_matches_nickname_through_pfr_id() -> None:
    stats = pd.DataFrame(
        {
            "player_name": ["C.Okonkwo", "T.Kelce"],
            "player_display_name": ["Chigoziem Okonkwo", "Travis Kelce"],
            "gsis_id": ["00-0037809", "00-0030506"],
            "season": [2025, 2025],
            "week": [3, 3],
            "team": ["TEN", "KC"],
        }
    )
    snaps = pd.DataFrame(
        {
            "player": ["Chig Okonkwo", "Travis Kelce"],
            "pfr_player_id": ["OkonCh00", None],
            "season": [2025, 2025],
            "week": [3, 3],
            "team": ["TEN", "KC"],
            "offense_snaps": [48.0, 60.0],
            "offense_pct": [0.72, 0.9],
        }
    )
    rosters = pd.DataFrame(
        {"gsis_id": ["00-0037809", "00-0030506"], "pfr_id": ["OkonCh00", "KelcTr00"]}
    )

    merged = ingest._merge_snap_counts(stats, snaps, rosters).set_index("player_display_name")

    assert merged.loc["Chigoziem Okonkwo", "snap_count"] == 48.0
    assert merged.loc["Chigoziem Okonkwo", "snap_percentage"] == 72.0
    # No PFR id on the snap row: the name merge still matches.
    assert merged.loc["Travis Kelce", "snap_percentage"] == 90.0


def test_snap_merge_without_rosters_keeps_name_behavior() -> None:
    stats = pd.DataFrame(
        {
            "player_name": ["C.Okonkwo"],
            "player_display_name": ["Chigoziem Okonkwo"],
            "gsis_id": ["00-0037809"],
            "season": [2025],
            "week": [3],
            "team": ["TEN"],
        }
    )
    snaps = pd.DataFrame(
        {
            "player": ["Chig Okonkwo"],
            "pfr_player_id": ["OkonCh00"],
            "season": [2025],
            "week": [3],
            "team": ["TEN"],
            "offense_snaps": [48.0],
            "offense_pct": [0.72],
        }
    )

    merged = ingest._merge_snap_counts(stats, snaps)

    assert merged.iloc[0]["snap_percentage"] == 0.0
