"""Tests for the snap-counts merge in scripts/ingest_real_nfl_data.py (T0 #1)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.ingest_real_nfl_data import _merge_snap_counts


def _stats_row(player_name: str, team: str, season: int, week: int) -> dict:
    return {
        "player_name": player_name,
        "team": team,
        "season": season,
        "week": week,
        "position": "WR",
    }


def test_merge_assigns_snap_fields_when_match_exists():
    df = pd.DataFrame([_stats_row("J. Jefferson", "MIN", 2024, 5)])
    snaps = pd.DataFrame(
        [
            {
                "player": "J. Jefferson",
                "team": "MIN",
                "season": 2024,
                "week": 5,
                "offense_snaps": 60,
                "offense_pct": 0.92,
            }
        ]
    )

    merged = _merge_snap_counts(df, snaps)

    assert merged.loc[0, "snap_count"] == 60
    assert merged.loc[0, "snap_percentage"] == pytest.approx(92.0)


def test_merge_defaults_zero_when_no_match():
    """Unmatched rows default to 0 — schema is NOT NULL DEFAULT 0."""
    df = pd.DataFrame([_stats_row("Rookie X", "DAL", 2024, 1)])
    snaps = pd.DataFrame(
        [
            {
                "player": "Veteran Y",
                "team": "DAL",
                "season": 2024,
                "week": 1,
                "offense_snaps": 50,
                "offense_pct": 0.85,
            }
        ]
    )

    merged = _merge_snap_counts(df, snaps)

    assert merged.loc[0, "snap_count"] == 0.0
    assert merged.loc[0, "snap_percentage"] == 0.0


def test_merge_handles_empty_snap_frame():
    df = pd.DataFrame([_stats_row("Any Player", "KC", 2024, 9)])
    merged = _merge_snap_counts(df, pd.DataFrame())

    assert merged.loc[0, "snap_percentage"] == 0.0
    assert "snap_count" in merged.columns


def test_merge_normalizes_punctuation_difference():
    """Stats `player_name` (A.Rodgers) merges to snap `player` (Aaron Rodgers) via display_name."""
    df = pd.DataFrame([{
        "player_name": "A.Rodgers",
        "player_display_name": "Aaron Rodgers",
        "team": "NYJ",
        "season": 2024,
        "week": 2,
        "position": "QB",
    }])
    snaps = pd.DataFrame([{
        "player": "Aaron Rodgers",
        "team": "NYJ",
        "season": 2024,
        "week": 2,
        "offense_snaps": 70,
        "offense_pct": 1.0,
    }])

    merged = _merge_snap_counts(df, snaps)

    assert merged.loc[0, "snap_count"] == 70
    assert merged.loc[0, "snap_percentage"] == pytest.approx(100.0)


def test_merge_aggregates_duplicate_snap_rows():
    df = pd.DataFrame([_stats_row("Some RB", "PHI", 2024, 10)])
    snaps = pd.DataFrame(
        [
            {
                "player": "Some RB",
                "team": "PHI",
                "season": 2024,
                "week": 10,
                "offense_snaps": 25,
                "offense_pct": 0.40,
            },
            {
                "player": "Some RB",
                "team": "PHI",
                "season": 2024,
                "week": 10,
                "offense_snaps": 15,
                "offense_pct": 0.30,
            },
        ]
    )

    merged = _merge_snap_counts(df, snaps)

    assert merged.loc[0, "snap_count"] == 40
    assert merged.loc[0, "snap_percentage"] == pytest.approx(40.0)


def test_merge_passes_through_already_scaled_pct():
    df = pd.DataFrame([_stats_row("Star QB", "BUF", 2024, 3)])
    snaps = pd.DataFrame(
        [
            {
                "player": "Star QB",
                "team": "BUF",
                "season": 2024,
                "week": 3,
                "offense_snaps": 72,
                "offense_pct": 99.0,
            }
        ]
    )

    merged = _merge_snap_counts(df, snaps)

    assert merged.loc[0, "snap_percentage"] == pytest.approx(99.0)


def test_merge_skips_when_snap_frame_missing_columns():
    df = pd.DataFrame([_stats_row("Player Z", "NYJ", 2024, 4)])
    snaps = pd.DataFrame([{"player": "Player Z", "season": 2024, "week": 4}])

    merged = _merge_snap_counts(df, snaps)

    assert merged.loc[0, "snap_percentage"] == 0.0
