"""Tests for scripts/ingest_nba_injuries.py — DNP detection, injury ingestion.

Covers:
- Retrospective DNP detection from game logs
- Injury record format and required fields
- Edge cases: no DNPs, all active, game not found
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock config module since config.py is gitignored in this worktree.
# This must happen before any project imports that transitively import config.
if "config" not in sys.modules:
    _mock_config = MagicMock()
    _mock_config.config.database.backend = "sqlite"
    _mock_config.config.database.path = ":memory:"
    sys.modules["config"] = _mock_config

from unittest.mock import patch, call

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_teams_df(team: str = "BOS") -> pd.DataFrame:
    """DataFrame returned by the first read_dataframe call (teams playing)."""
    return pd.DataFrame({"team_abbreviation": [team]})


def _make_played_df(active_ids: list[int], team: str = "BOS") -> pd.DataFrame:
    """DataFrame returned by the second read_dataframe call (game log rows)."""
    rows = []
    for pid in active_ids:
        rows.append(
            {
                "player_id": pid,
                "player_name": f"Player {pid}",
                "team_abbreviation": team,
                "season": 2025,
                "game_id": "0022500100",
                "game_date": "2026-02-17",
                "min": 30.0,
                "pts": 20,
                "reb": 5,
                "ast": 3,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_id", "player_name", "team_abbreviation", "min"]
    )


def _make_roster_df(all_ids: list[int], team: str = "BOS") -> pd.DataFrame:
    """DataFrame returned by the third read_dataframe call (season roster)."""
    name_map = {
        1628369: "Jayson Tatum",
        1628384: "Jaylen Brown",
        203935: "Marcus Smart",
        1627759: "Derrick White",
    }
    rows = []
    for pid in all_ids:
        rows.append(
            {
                "player_id": pid,
                "player_name": name_map.get(pid, f"Player {pid}"),
                "team_abbreviation": team,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# detect_dnps
# ---------------------------------------------------------------------------


class TestDetectDNPs:
    """Test detection of players who were expected to play but didn't."""

    def _patch_read_dataframe(self, mock_read, active_ids, all_ids, team="BOS"):
        """Configure read_dataframe mock for 3 sequential calls."""
        mock_read.side_effect = [
            _make_teams_df(team),
            _make_played_df(active_ids, team),
            _make_roster_df(all_ids, team),
        ]

    @patch("scripts.ingest_nba_injuries.read_dataframe")
    def test_detects_missing_player(self, mock_read):
        from scripts.ingest_nba_injuries import detect_dnps

        all_ids = [1628369, 1628384, 203935, 1627759]
        active_ids = [1628369, 1628384, 203935]  # Derrick White absent
        self._patch_read_dataframe(mock_read, active_ids, all_ids)

        dnps = detect_dnps("2026-02-17")
        assert len(dnps) >= 1
        missing_ids = [d["player_id"] for d in dnps]
        assert 1627759 in missing_ids  # Derrick White was DNP

    @patch("scripts.ingest_nba_injuries.read_dataframe")
    def test_no_dnps_when_all_active(self, mock_read):
        from scripts.ingest_nba_injuries import detect_dnps

        all_ids = [1628369, 1628384, 203935, 1627759]
        active_ids = all_ids  # All 4 played
        self._patch_read_dataframe(mock_read, active_ids, all_ids)

        dnps = detect_dnps("2026-02-17")
        assert dnps == []

    @patch("scripts.ingest_nba_injuries.read_dataframe")
    def test_dnp_record_has_required_fields(self, mock_read):
        from scripts.ingest_nba_injuries import detect_dnps

        all_ids = [1628369, 1628384, 203935, 1627759]
        active_ids = [1628369, 1628384, 203935]
        self._patch_read_dataframe(mock_read, active_ids, all_ids)

        dnps = detect_dnps("2026-02-17")
        assert len(dnps) >= 1
        required_keys = {"player_id", "player_name", "team", "game_date", "status"}
        for dnp in dnps:
            assert required_keys.issubset(set(dnp.keys())), (
                f"Missing keys: {required_keys - set(dnp.keys())}"
            )

    @patch("scripts.ingest_nba_injuries.read_dataframe")
    def test_dnp_status_is_out(self, mock_read):
        from scripts.ingest_nba_injuries import detect_dnps

        all_ids = [1628369, 1628384, 203935, 1627759]
        active_ids = [1628369, 1628384, 203935]
        self._patch_read_dataframe(mock_read, active_ids, all_ids)

        dnps = detect_dnps("2026-02-17")
        assert all(d["status"] == "OUT" for d in dnps)

    @patch("scripts.ingest_nba_injuries.read_dataframe")
    def test_empty_roster_returns_empty(self, mock_read):
        from scripts.ingest_nba_injuries import detect_dnps

        # No teams playing on this date
        mock_read.side_effect = [
            pd.DataFrame(columns=["team_abbreviation"]),  # empty teams
        ]

        dnps = detect_dnps("2026-02-17")
        assert dnps == []

    @patch("scripts.ingest_nba_injuries.read_dataframe")
    def test_multiple_dnps_detected(self, mock_read):
        from scripts.ingest_nba_injuries import detect_dnps

        all_ids = [1628369, 1628384, 203935, 1627759]
        active_ids = [1628369, 1628384]  # 2 of 4 played
        self._patch_read_dataframe(mock_read, active_ids, all_ids)

        dnps = detect_dnps("2026-02-17")
        assert len(dnps) == 2

    @patch("scripts.ingest_nba_injuries.read_dataframe")
    def test_game_date_set_correctly(self, mock_read):
        from scripts.ingest_nba_injuries import detect_dnps

        all_ids = [1628369, 1628384, 203935, 1627759]
        active_ids = [1628369, 1628384, 203935]
        self._patch_read_dataframe(mock_read, active_ids, all_ids)

        dnps = detect_dnps("2026-02-17")
        assert all(d["game_date"] == "2026-02-17" for d in dnps)

    @patch("scripts.ingest_nba_injuries.read_dataframe")
    def test_team_abbreviation_preserved(self, mock_read):
        from scripts.ingest_nba_injuries import detect_dnps

        all_ids = [1628369, 1628384, 203935, 1627759]
        active_ids = [1628369, 1628384, 203935]
        self._patch_read_dataframe(mock_read, active_ids, all_ids)

        dnps = detect_dnps("2026-02-17")
        assert all(d["team"] == "BOS" for d in dnps)
