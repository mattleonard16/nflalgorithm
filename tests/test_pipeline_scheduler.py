"""Scheduler coverage for the canonical NFL pregame workflow."""

from __future__ import annotations

from scripts import pipeline_scheduler


def test_scheduler_routes_current_week_through_production_runner(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    scheduler = pipeline_scheduler.PipelineScheduler.__new__(pipeline_scheduler.PipelineScheduler)

    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_season", lambda roster=True: 2026)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_week", lambda: 1)
    monkeypatch.setattr(
        pipeline_scheduler,
        "run_production_pipeline",
        lambda season, week: calls.append((season, week)) or {"success": True, "errors": []},
    )

    result = scheduler.run_pregame_pipeline()

    assert calls == [(2026, 1)]
    assert result["success"] is True


def test_scheduler_surfaces_failed_canonical_run(monkeypatch) -> None:
    scheduler = pipeline_scheduler.PipelineScheduler.__new__(pipeline_scheduler.PipelineScheduler)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_season", lambda roster=True: 2026)
    monkeypatch.setattr(pipeline_scheduler.nfl, "get_current_week", lambda: 1)
    monkeypatch.setattr(
        pipeline_scheduler,
        "run_production_pipeline",
        lambda season, week: {
            "success": False,
            "errors": [{"stage": "prepare_week", "error": "incomplete roster"}],
        },
    )

    result = scheduler.run_pregame_pipeline()

    assert result["success"] is False
