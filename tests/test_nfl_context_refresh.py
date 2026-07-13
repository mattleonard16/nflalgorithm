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
            return pd.DataFrame({"season": [2020], "kickoff_utc": ["2020-09-10T17:00:00Z"]})
        return pd.DataFrame()

    monkeypatch.setattr(ingest_real_nfl_data, "read_dataframe", read_frame)
    monkeypatch.setattr(
        ingest_real_nfl_data,
        "build_player_context_snapshots",
        lambda *args, **kwargs: pd.DataFrame({"season": [2020]}),
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
