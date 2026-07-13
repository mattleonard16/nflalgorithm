"""Tests for the one-command NFL weekly prediction refresh."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts import prepare_nfl_week


def test_gsis_history_requires_high_coverage_in_every_requested_season(monkeypatch) -> None:
    monkeypatch.setattr(
        prepare_nfl_week,
        "fetchall",
        lambda query, params: [(2024, 100, 100), (2025, 100, 94)],
    )

    assert prepare_nfl_week._has_gsis_history([2024, 2025]) is False


def test_gsis_history_accepts_complete_requested_season_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        prepare_nfl_week,
        "fetchall",
        lambda query, params: [(2024, 100, 98), (2025, 100, 95)],
    )

    assert prepare_nfl_week._has_gsis_history([2024, 2025]) is True


def test_default_history_seasons_roll_forward_with_target_season(monkeypatch) -> None:
    monkeypatch.setattr(prepare_nfl_week.config.pipeline, "default_seasons", [2024, 2025])

    assert prepare_nfl_week.default_history_seasons(2026) == [2024, 2025]
    assert prepare_nfl_week.default_history_seasons(2027) == [2025, 2026]


def test_history_count_applies_active_roster_filter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def capture_count(query: str, params: tuple[object, ...]) -> int:
        captured["query"] = query
        captured["params"] = params
        return 12

    monkeypatch.setattr(prepare_nfl_week, "_count_rows", capture_count)

    assert prepare_nfl_week._count_players_with_history(2026) == 12
    assert "UPPER(roster.roster_status) NOT IN" in str(captured["query"])
    assert "{" not in str(captured["query"])
    assert captured["params"] == (2026, 2026)


def test_prepare_week_refreshes_history_roster_schedule_and_predictions(monkeypatch) -> None:
    ingest_calls: list[tuple[list[int], int, int | None]] = []
    prediction_calls: list[tuple[int, int, bool]] = []

    monkeypatch.setattr(
        prepare_nfl_week,
        "ingest_seasons",
        lambda seasons, through_week, stats_through_week=None: ingest_calls.append(
            (seasons, through_week, stats_through_week)
        )
        or 25,
    )
    monkeypatch.setattr(prepare_nfl_week, "populate_player_dim", lambda: 40)
    monkeypatch.setattr(
        prepare_nfl_week,
        "predict_week",
        lambda season, week, roster_backed: prediction_calls.append((season, week, roster_backed))
        or pd.DataFrame({"player_id": ["BUF_season_ready"], "market": ["receiving_yards"]}),
    )

    monkeypatch.setattr(prepare_nfl_week, "_count_roster_players", lambda season: 53)
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_teams", lambda season: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_prediction_eligible_roster", lambda season: 48)
    monkeypatch.setattr(prepare_nfl_week, "_count_games", lambda season, week: 16)
    monkeypatch.setattr(prepare_nfl_week, "_count_scheduled_teams", lambda season, week: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_players_with_history", lambda season: 45)
    monkeypatch.setattr(prepare_nfl_week, "run_migrations", lambda: None)

    result = prepare_nfl_week.prepare_week(
        season=2026,
        week=1,
        history_seasons=[2024, 2025],
        refresh_history=True,
    )

    assert ingest_calls == [([2024, 2025], 22, None), ([2026], 1, 0)]
    assert prediction_calls == [(2026, 1, True)]
    assert result == {
        "season": 2026,
        "week": 1,
        "history_seasons": [2024, 2025],
        "history_refreshed": True,
        "historical_player_weeks": 25,
        "current_player_weeks": 25,
        "roster_players": 53,
        "roster_teams": 32,
        "prediction_eligible_roster_players": 48,
        "games": 16,
        "scheduled_teams": 32,
        "players_with_history": 45,
        "history_coverage": 0.9375,
        "player_dim_updates": 40,
        "predictions": 1,
        "predicted_players": 1,
    }


def test_prepare_week_fails_when_roster_context_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(prepare_nfl_week, "run_migrations", lambda: None)
    monkeypatch.setattr(
        prepare_nfl_week,
        "ingest_seasons",
        lambda seasons, through_week, stats_through_week=None: 0,
    )
    monkeypatch.setattr(prepare_nfl_week, "populate_player_dim", lambda: 0)
    monkeypatch.setattr(
        prepare_nfl_week,
        "predict_week",
        lambda season, week, roster_backed: pd.DataFrame(),
    )
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_players", lambda season: 0)
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_teams", lambda season: 0)
    monkeypatch.setattr(prepare_nfl_week, "_count_prediction_eligible_roster", lambda season: 0)
    monkeypatch.setattr(prepare_nfl_week, "_count_games", lambda season, week: 0)
    monkeypatch.setattr(prepare_nfl_week, "_count_scheduled_teams", lambda season, week: 0)
    monkeypatch.setattr(prepare_nfl_week, "_count_players_with_history", lambda season: 0)

    with pytest.raises(RuntimeError, match="No roster players"):
        prepare_nfl_week.prepare_week(2026, 1, history_seasons=[], refresh_history=False)


def test_prepare_week_reuses_existing_gsis_history_by_default(monkeypatch) -> None:
    ingest_calls: list[tuple[list[int], int, int | None]] = []

    monkeypatch.setattr(prepare_nfl_week, "run_migrations", lambda: None)
    monkeypatch.setattr(
        prepare_nfl_week,
        "ingest_seasons",
        lambda seasons, through_week, stats_through_week=None: ingest_calls.append(
            (seasons, through_week, stats_through_week)
        )
        or 0,
    )
    monkeypatch.setattr(prepare_nfl_week, "populate_player_dim", lambda: 0)
    monkeypatch.setattr(
        prepare_nfl_week,
        "predict_week",
        lambda season, week, roster_backed: pd.DataFrame({"player_id": ["BUF_season_ready"]}),
    )

    monkeypatch.setattr(prepare_nfl_week, "_has_gsis_history", lambda seasons: True)
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_players", lambda season: 53)
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_teams", lambda season: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_prediction_eligible_roster", lambda season: 48)
    monkeypatch.setattr(prepare_nfl_week, "_count_games", lambda season, week: 16)
    monkeypatch.setattr(prepare_nfl_week, "_count_scheduled_teams", lambda season, week: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_players_with_history", lambda season: 45)

    result = prepare_nfl_week.prepare_week(2026, 1, history_seasons=[2024, 2025])

    assert ingest_calls == [([2026], 1, 0)]
    assert result["history_refreshed"] is False


def test_prepare_week_fails_closed_on_incomplete_history_coverage(monkeypatch) -> None:
    monkeypatch.setattr(prepare_nfl_week, "run_migrations", lambda: None)
    monkeypatch.setattr(
        prepare_nfl_week,
        "ingest_seasons",
        lambda seasons, through_week, stats_through_week=None: 0,
    )
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_players", lambda season: 53)
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_teams", lambda season: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_prediction_eligible_roster", lambda season: 48)
    monkeypatch.setattr(prepare_nfl_week, "_count_games", lambda season, week: 16)
    monkeypatch.setattr(prepare_nfl_week, "_count_scheduled_teams", lambda season, week: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_players_with_history", lambda season: 1)

    with pytest.raises(RuntimeError, match="history coverage"):
        prepare_nfl_week.prepare_week(2026, 1, history_seasons=[], refresh_history=False)


def test_prepare_week_rejects_incomplete_week_one_schedule(monkeypatch) -> None:
    monkeypatch.setattr(prepare_nfl_week, "run_migrations", lambda: None)
    monkeypatch.setattr(
        prepare_nfl_week,
        "ingest_seasons",
        lambda seasons, through_week, stats_through_week=None: 0,
    )
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_players", lambda season: 1600)
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_teams", lambda season: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_prediction_eligible_roster", lambda season: 300)
    monkeypatch.setattr(prepare_nfl_week, "_count_games", lambda season, week: 15)
    monkeypatch.setattr(prepare_nfl_week, "_count_scheduled_teams", lambda season, week: 30)
    monkeypatch.setattr(prepare_nfl_week, "_count_players_with_history", lambda season: 250)

    with pytest.raises(RuntimeError, match="Week 1 schedule is incomplete"):
        prepare_nfl_week.prepare_week(2026, 1, history_seasons=[], refresh_history=False)


def test_prepare_week_refuses_to_overwrite_predictions_after_kickoff(monkeypatch) -> None:
    monkeypatch.setattr(prepare_nfl_week, "run_migrations", lambda: None)
    monkeypatch.setattr(
        prepare_nfl_week,
        "ingest_seasons",
        lambda seasons, through_week, stats_through_week=None: 0,
    )
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_players", lambda season: 53)
    monkeypatch.setattr(prepare_nfl_week, "_count_roster_teams", lambda season: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_prediction_eligible_roster", lambda season: 48)
    monkeypatch.setattr(prepare_nfl_week, "_count_games", lambda season, week: 16)
    monkeypatch.setattr(prepare_nfl_week, "_count_scheduled_teams", lambda season, week: 32)
    monkeypatch.setattr(prepare_nfl_week, "_count_players_with_history", lambda season: 45)
    monkeypatch.setattr(
        prepare_nfl_week,
        "_earliest_kickoff",
        lambda season, week: datetime(2020, 9, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        prepare_nfl_week,
        "predict_week",
        lambda *args, **kwargs: pytest.fail("post-kickoff prediction overwrite"),
    )

    with pytest.raises(RuntimeError, match="already kicked off"):
        prepare_nfl_week.prepare_week(2020, 1, history_seasons=[], refresh_history=False)
