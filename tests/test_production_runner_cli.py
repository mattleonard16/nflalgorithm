"""CLI status coverage for the NFL production runner."""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import pytest

from scripts import activate_betting, production_runner, run_prop_update


def test_pipeline_always_runs_canonical_prepare_even_when_reusing_history(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def prepare(season: int, week: int, refresh_history=None):
        calls.append(("prepare", refresh_history))
        return {"status": "ok", "stage": "prepare_week", "predictions": 10}

    def odds(season: int, week: int):
        calls.append(("odds", None))
        return {"status": "ok", "stage": "odds", "odds_count": 10}

    monkeypatch.setattr(production_runner, "stage_prepare_week", prepare)
    monkeypatch.setattr(production_runner, "POST_PREPARE_STAGES", [("odds", odds)])
    monkeypatch.setattr(production_runner, "_persist_run_report", lambda report: None)

    report = production_runner.run_production_pipeline(2026, 1, skip_ingest=True)

    assert calls == [("prepare", False), ("odds", None)]
    assert report["success"] is True


def test_pipeline_stops_when_canonical_prepare_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        production_runner,
        "stage_prepare_week",
        lambda season, week, refresh_history=None: {
            "status": "error",
            "stage": "prepare_week",
            "error": "incomplete roster",
        },
    )
    monkeypatch.setattr(
        production_runner,
        "POST_PREPARE_STAGES",
        [("odds", lambda season, week: pytest.fail("odds must not run"))],
    )
    monkeypatch.setattr(production_runner, "_persist_run_report", lambda report: None)

    report = production_runner.run_production_pipeline(2026, 1)

    assert report["success"] is False
    assert [stage["stage"] for stage in report["stages"]] == ["prepare_week"]


def test_skip_odds_never_builds_card_from_cached_lines(monkeypatch) -> None:
    monkeypatch.setattr(
        production_runner,
        "stage_prepare_week",
        lambda season, week, refresh_history=None: {"status": "ok", "stage": "prepare_week"},
    )
    monkeypatch.setattr(
        production_runner,
        "POST_PREPARE_STAGES",
        [
            ("odds", lambda season, week: pytest.fail("odds must be skipped")),
            ("value_ranking", lambda season, week: pytest.fail("ranking must not use stale odds")),
        ],
    )
    monkeypatch.setattr(production_runner, "_persist_run_report", lambda report: None)

    report = production_runner.run_production_pipeline(2026, 1, skip_odds=True)

    assert report["success"] is True
    assert report["stages"][-1] == {
        "status": "skipped",
        "stage": "odds",
        "reason": "skip_odds",
    }


def test_odds_stage_uses_live_only_weekly_scraper(monkeypatch) -> None:
    from scripts import prop_line_scraper

    calls: list[tuple[int, int, bool]] = []

    class FakeScraper:
        def run_weekly_update(self, week, season, allow_synthetic=True):
            calls.append((week, season, allow_synthetic))
            return pd.DataFrame({"line": [55.5]})

    monkeypatch.setattr(prop_line_scraper, "NFLPropScraper", FakeScraper)

    result = production_runner.stage_odds(2026, 1)

    assert calls == [(1, 2026, False)]
    assert result == {"status": "ok", "stage": "odds", "odds_count": 1}


def test_weekly_report_refresh_uses_prepare_then_live_odds(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        run_prop_update,
        "prepare_week",
        lambda season, week: calls.append(("prepare", (season, week))) or {"predictions": 12},
    )

    class FakeScraper:
        def run_weekly_update(self, week, season, allow_synthetic=True):
            calls.append(("odds", allow_synthetic))
            return pd.DataFrame({"line": [55.5]})

    monkeypatch.setattr(run_prop_update, "NFLPropScraper", FakeScraper)

    summary, odds = run_prop_update.refresh_pregame_inputs(2026, 1)

    assert calls == [("prepare", (2026, 1)), ("odds", False)]
    assert summary["predictions"] == 12
    assert len(odds) == 1


def test_activation_routes_through_canonical_pregame_refresh(monkeypatch, tmp_path) -> None:
    calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        activate_betting,
        "refresh_pregame_inputs",
        lambda season, week: calls.append((season, week))
        or ({"predictions": 1}, pd.DataFrame({"line": [55.5]})),
    )

    class StopAfterRefresh:
        def __init__(self):
            raise RuntimeError("stop after canonical refresh")

    monkeypatch.setattr(activate_betting, "PropIntegration", StopAfterRefresh)
    monkeypatch.setattr(
        sys,
        "argv",
        ["activate_betting", "--season", "2026", "--week", "1"],
    )

    with pytest.raises(RuntimeError, match="stop after canonical refresh"):
        activate_betting.main()
    assert calls == [(2026, 1)]


def test_makefile_week_entrypoints_do_not_call_legacy_prediction_paths() -> None:
    makefile = (production_runner.config.project_root / "Makefile").read_text()

    assert "from models.position_specific import predict_week" not in makefile
    assert "from data_pipeline import update_week" not in makefile


def test_main_exits_nonzero_when_pipeline_report_has_errors(monkeypatch) -> None:
    report: dict[str, Any] = {
        "success": False,
        "stages": [
            {
                "status": "error",
                "stage": "ingest",
                "error": "nflverse timed out",
            }
        ],
        "errors": [{"stage": "ingest", "error": "nflverse timed out"}],
    }

    def failed_pipeline(*args: object, **kwargs: object) -> dict[str, Any]:
        return report

    monkeypatch.setattr(sys, "argv", ["production_runner", "--season", "2026", "--week", "1"])
    monkeypatch.setattr(production_runner, "run_production_pipeline", failed_pipeline)

    with pytest.raises(SystemExit) as exc_info:
        production_runner.main()

    assert exc_info.value.code == 1


def test_main_returns_normally_for_successful_pipeline_report(monkeypatch) -> None:
    report: dict[str, Any] = {
        "success": True,
        "stages": [{"status": "ok", "stage": "ingest"}],
        "errors": [],
    }

    def successful_pipeline(*args: object, **kwargs: object) -> dict[str, Any]:
        return report

    monkeypatch.setattr(sys, "argv", ["production_runner", "--season", "2026", "--week", "1"])
    monkeypatch.setattr(production_runner, "run_production_pipeline", successful_pipeline)

    production_runner.main()
