"""Causal persistence behavior for NFL roster context refreshes."""

from __future__ import annotations

import pandas as pd

from scripts import ingest_real_nfl_data


def test_context_refresh_only_writes_requested_week(monkeypatch) -> None:
    built_weeks: list[int] = []

    monkeypatch.setattr(
        ingest_real_nfl_data, "read_dataframe", lambda query, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "build_player_context_snapshots",
        lambda *args, target_week, captured_at, **kwargs: built_weeks.append(target_week)
        or pd.DataFrame({"season": [2026]}),
    )
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "upsert_player_context_snapshots",
        lambda snapshots: len(snapshots),
    )

    count = ingest_real_nfl_data.refresh_player_context_snapshots(
        pd.DataFrame({"season": [2026]}),
        pd.DataFrame(),
        pd.DataFrame(),
        through_week=7,
    )

    assert count == 1
    assert built_weeks == [7]


def test_context_refresh_does_not_overwrite_snapshot_after_kickoff(monkeypatch) -> None:
    def read_frame(query, **kwargs):
        if "FROM games" in query:
            return pd.DataFrame(
                {
                    "season": [2020],
                    "home_team": ["KC"],
                    "away_team": ["HOU"],
                    "kickoff_utc": ["2020-09-10T17:00:00Z"],
                }
            )
        return pd.DataFrame()

    monkeypatch.setattr(ingest_real_nfl_data, "read_dataframe", read_frame)
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "build_player_context_snapshots",
        lambda *args, **kwargs: pd.DataFrame({"season": [2020], "team": ["KC"]}),
    )
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "upsert_player_context_snapshots",
        lambda snapshots: (_ for _ in ()).throw(AssertionError("post-kickoff overwrite")),
    )

    count = ingest_real_nfl_data.refresh_player_context_snapshots(
        pd.DataFrame({"season": [2020]}),
        pd.DataFrame(),
        pd.DataFrame(),
        through_week=1,
    )

    assert count == 0


def _late_week_refresh(monkeypatch, depth: pd.DataFrame) -> pd.DataFrame:
    """Refresh week 3 on Friday: ATL and GB played Thursday, SEA and ARI play Sunday."""
    now = pd.Timestamp.now(tz="UTC")
    games = pd.DataFrame(
        {
            "season": [2026, 2026],
            "home_team": ["ATL", "SEA"],
            "away_team": ["GB", "ARI"],
            "spread_line": [None, None],
            "kickoff_utc": [
                (now - pd.Timedelta(days=1)).isoformat(),
                (now + pd.Timedelta(days=2)).isoformat(),
            ],
        }
    )
    written: list[pd.DataFrame] = []
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "read_dataframe",
        lambda query, **kwargs: games if "FROM games" in query else pd.DataFrame(),
    )
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "upsert_player_context_snapshots",
        lambda snapshots: written.append(snapshots) or len(snapshots),
    )
    rosters = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [3, 3],
            "gsis_id": ["atl_qb", "sea_qb"],
            "full_name": ["Falcons Passer", "Seahawks Passer"],
            "team": ["ATL", "SEA"],
            "position": ["QB", "QB"],
            "status": ["ACT", "ACT"],
        }
    )
    ingest_real_nfl_data.refresh_player_context_snapshots(
        rosters, depth, pd.DataFrame(), through_week=3
    )
    return pd.concat(written) if written else pd.DataFrame(columns=["team"])


def test_context_refresh_after_first_kickoff_rewrites_only_teams_yet_to_play(
    monkeypatch,
) -> None:
    written = _late_week_refresh(monkeypatch, pd.DataFrame())

    assert written["team"].tolist() == ["SEA"]


def test_context_refresh_after_first_kickoff_reads_depth_charts_published_since(
    monkeypatch,
) -> None:
    """Sunday's teams should see Friday's depth chart, not Thursday's."""
    now = pd.Timestamp.now(tz="UTC")
    depth = pd.DataFrame(
        {
            "season": [2026, 2026],
            "gsis_id": ["sea_qb", "sea_qb"],
            "pos_abb": ["QB", "QB"],
            "pos_rank": [2, 1],
            "dt": [
                (now - pd.Timedelta(days=2)).isoformat(),
                (now - pd.Timedelta(hours=1)).isoformat(),
            ],
        }
    )

    written = _late_week_refresh(monkeypatch, depth)

    assert written.set_index("team").loc["SEA", "depth_rank"] == 1
