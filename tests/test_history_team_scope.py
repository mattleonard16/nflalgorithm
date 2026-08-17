"""History ingest must keep every franchise team-scoped.

A prior ingest wrote Rams skill players with empty ``team`` because nflverse
uses ``LA`` and that alias was missing. GSIS coverage still cleared 95%, so
week-refresh skipped the repair. These tests lock the fail-closed checks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from schema_migrations import MigrationManager
from scripts import ingest_real_nfl_data, prepare_nfl_week
from scripts.preflight import collect_nfl_week_counts, evaluate_nfl_week_readiness
from utils.db import execute
from utils.player_id_utils import canonicalize_team, make_player_id


def test_nflverse_la_canonicalizes_to_lar() -> None:
    assert canonicalize_team("LA") == "LAR"
    assert make_player_id("M.Stafford", "LA") == "LAR_m_stafford"


def test_transform_scopes_nflverse_la_rows() -> None:
    weekly = pd.DataFrame(
        {
            "position": ["QB", "WR"],
            "player_id": ["00-0026498", "00-0039067"],
            "player_name": ["M.Stafford", "P.Nacua"],
            "player_display_name": ["Matthew Stafford", "Puka Nacua"],
            "team": ["LA", "2TM"],
            "opponent_team": ["SEA", "SEA"],
            "season": [2025, 2025],
            "week": [1, 1],
            "rushing_yards": [0.0, 0.0],
            "receiving_yards": [0.0, 80.0],
            "receptions": [0.0, 6.0],
            "targets": [0.0, 8.0],
            "passing_yards": [250.0, 0.0],
            "attempts": [30.0, 0.0],
            "carries": [0.0, 0.0],
        }
    )

    result = ingest_real_nfl_data.transform_to_enhanced_stats(weekly, pd.DataFrame())

    assert result["team"].tolist() == ["LAR"]
    assert result["player_id"].tolist() == ["LAR_m_stafford"]
    assert result["gsis_id"].tolist() == ["00-0026498"]


def test_purge_unscoped_player_stats_removes_empty_team_orphans(tmp_path, monkeypatch) -> None:
    database = tmp_path / "history.db"
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", str(database))

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", str(database))
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(database).run()
    execute("""
        INSERT INTO player_stats_enhanced (
            player_id, gsis_id, season, week, name, team, position
        ) VALUES
            ('m_stafford', NULL, 2025, 1, 'Matthew Stafford', '', 'QB'),
            ('LAR_m_stafford', '00-0026498', 2025, 1, 'Matthew Stafford', 'LAR', 'QB')
        """)

    deleted = ingest_real_nfl_data.purge_unscoped_player_stats([2025])

    assert deleted == 1
    counts = collect_nfl_week_counts(2026, 1)
    assert counts["history_empty_team_rows"] == 0


def test_preflight_fails_closed_on_unscoped_history_rows() -> None:
    diagnostics = evaluate_nfl_week_readiness(
        2026,
        1,
        phase="pre-run",
        counts={
            "games": 16,
            "games_with_kickoff": 16,
            "roster_players": 1800,
            "history_rows": 10000,
            "history_empty_team_rows": 224,
            "history_incomplete_franchise_seasons": 1,
        },
    )
    failed = {item.name for item in diagnostics if item.status == "fail"}
    assert "nfl_history_team_scope" in failed
    assert "nfl_history_franchises" in failed


def test_history_is_unusable_when_a_franchise_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        prepare_nfl_week,
        "fetchall",
        lambda query, params: [(2024, 100, 98), (2025, 100, 96)],
    )
    monkeypatch.setattr(prepare_nfl_week, "_count_rows", lambda query, params: 224)
    monkeypatch.setattr(
        prepare_nfl_week,
        "_history_covers_all_franchises",
        lambda seasons: False,
    )

    assert prepare_nfl_week._has_gsis_history([2024, 2025]) is True
    assert prepare_nfl_week._has_unscoped_team_history([2024, 2025]) is True
    assert prepare_nfl_week._history_is_usable([2024, 2025]) is False


def test_prepare_week_refreshes_history_when_unscoped_rows_remain(monkeypatch) -> None:
    ingest_calls: list[tuple[list[int], int, int | None]] = []
    usable_calls = {"n": 0}

    def usable(seasons: list[int]) -> bool:
        usable_calls["n"] += 1
        return usable_calls["n"] > 1

    monkeypatch.setattr(prepare_nfl_week, "run_migrations", lambda: None)
    monkeypatch.setattr(prepare_nfl_week, "_history_is_usable", usable)
    monkeypatch.setattr(
        prepare_nfl_week,
        "ingest_seasons",
        lambda seasons, through_week, stats_through_week=None: ingest_calls.append(
            (seasons, through_week, stats_through_week)
        )
        or 25,
    )
    monkeypatch.setattr(prepare_nfl_week, "populate_player_dim", lambda: 0)
    monkeypatch.setattr(
        prepare_nfl_week,
        "predict_week",
        lambda season, week, roster_backed: pd.DataFrame({"player_id": ["LAR_m_stafford"]}),
    )
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_players", lambda season: 53)
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_teams", lambda season: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_prediction_eligible_roster", lambda season: 48)
    monkeypatch.setattr(prepare_nfl_week, "_count_games", lambda season, week: 16)
    monkeypatch.setattr(prepare_nfl_week, "_count_scheduled_teams", lambda season, week: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_players_with_history", lambda season: 45)

    result = prepare_nfl_week.prepare_week(2026, 1, history_seasons=[2024, 2025])

    assert ingest_calls == [([2024, 2025], 22, None), ([2026], 1, 0)]
    assert result["history_refreshed"] is True


def test_prepare_week_fails_closed_if_history_stays_unscoped(monkeypatch) -> None:
    monkeypatch.setattr(prepare_nfl_week, "run_migrations", lambda: None)
    monkeypatch.setattr(prepare_nfl_week, "_history_is_usable", lambda seasons: False)
    monkeypatch.setattr(
        prepare_nfl_week,
        "ingest_seasons",
        lambda seasons, through_week, stats_through_week=None: 0,
    )

    with pytest.raises(RuntimeError, match="team scope or franchise coverage"):
        prepare_nfl_week.prepare_week(2026, 1, history_seasons=[2024, 2025])
