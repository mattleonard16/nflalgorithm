"""Tests for the NBA adapter to the shared pipeline orchestrator."""

from __future__ import annotations

from scripts import nba_production_runner


def test_nba_pipeline_keeps_its_nonblocking_error_policy(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        nba_production_runner,
        "STAGES",
        [
            (
                "ingest",
                lambda game_date, season: calls.append("ingest")
                or {"stage": "ingest", "status": "error", "detail": "rate limit"},
            ),
            (
                "predict",
                lambda game_date, season: calls.append("predict")
                or {"stage": "predict", "status": "ok", "detail": "done"},
            ),
        ],
    )
    monkeypatch.setattr(nba_production_runner, "_write_run_report", lambda *args: None)

    results = nba_production_runner.run_nba_pipeline("2026-02-21", season=2025)

    assert calls == ["ingest", "predict"]
    assert [result["status"] for result in results] == ["error", "ok"]


def test_nba_value_stage_receives_model_options(monkeypatch) -> None:
    calls: list[tuple[bool, bool]] = []

    def value_stage(
        game_date: str,
        season: int,
        *,
        use_monte_carlo: bool,
        calibrated: bool,
    ) -> dict[str, str]:
        calls.append((use_monte_carlo, calibrated))
        return {"stage": "value", "status": "ok", "detail": "done"}

    monkeypatch.setattr(nba_production_runner, "STAGES", [("value", value_stage)])
    monkeypatch.setattr(nba_production_runner, "_write_run_report", lambda *args: None)

    nba_production_runner.run_nba_pipeline(
        "2026-02-21",
        season=2025,
        use_monte_carlo=True,
        calibrated=True,
    )

    assert calls == [(True, True)]


def test_explicit_empty_stage_selection_runs_nothing(monkeypatch) -> None:
    monkeypatch.setattr(
        nba_production_runner,
        "STAGES",
        [("ingest", lambda game_date, season: {"stage": "ingest", "status": "ok"})],
    )
    monkeypatch.setattr(nba_production_runner, "_write_run_report", lambda *args: None)

    results = nba_production_runner.run_nba_pipeline(
        "2026-02-21",
        season=2025,
        stages=[],
    )

    assert results == []
