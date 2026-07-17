"""CLI status coverage for the NFL production runner."""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import pytest

from scripts import production_runner


def valid_odds_audit() -> dict[str, object]:
    return {
        "source_statuses": ["MISS"],
        "response_ages_seconds": [1.0],
        "responses_observed": 1,
        "snapshot_at": "2026-09-01T12:00:00+00:00",
        "scheduled_events": 1,
        "covered_events": 1,
        "covered_event_markets": 3,
        "sportsbooks_per_event_market": {
            "event-1:player_pass_yds": 2,
            "event-1:player_rush_yds": 2,
            "event-1:player_rec_yds": 2,
        },
        "odds_rows": 1,
    }


def test_run_reports_are_unique_and_written_atomically(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(production_runner.config, "logs_dir", tmp_path)
    report = {
        "season": 2026,
        "week": 1,
        "started_at": "2026-07-16T12:00:00+00:00",
        "success": True,
    }

    first = production_runner._persist_run_report(report)
    second = production_runner._persist_run_report(report)

    assert first is not None and second is not None
    assert first != second
    assert first.exists() and second.exists()
    assert list((tmp_path / "production_runs").glob("*.tmp")) == []


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

    assert report["success"] is False
    assert report["incomplete"] is True
    assert report["stages"][-1] == {
        "status": "skipped",
        "stage": "odds",
        "reason": "skip_odds",
    }


def test_live_odds_failure_never_reaches_card_materialization(monkeypatch) -> None:
    materialize_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        production_runner,
        "stage_prepare_week",
        lambda season, week, refresh_history=None: {"status": "ok", "stage": "prepare_week"},
    )
    monkeypatch.setattr(
        production_runner,
        "POST_PREPARE_STAGES",
        [
            (
                "odds",
                lambda season, week: {
                    "status": "error",
                    "stage": "odds",
                    "error": "live provider unavailable",
                },
            ),
            (
                "materialize",
                lambda season, week: materialize_calls.append((season, week))
                or {"status": "ok", "stage": "materialize", "card_size": 1},
            ),
        ],
    )
    monkeypatch.setattr(production_runner, "_persist_run_report", lambda report: None)

    report = production_runner.run_production_pipeline(2026, 1)

    assert materialize_calls == []
    assert report["success"] is False
    assert report["errors"] == [
        {"status": "error", "stage": "odds", "error": "live provider unavailable"}
    ]


def test_odds_stage_uses_live_only_weekly_scraper(monkeypatch) -> None:
    from scripts import prop_line_scraper

    calls: list[tuple[int, int, bool]] = []

    class FakeScraper:
        def run_weekly_update(self, week, season, allow_synthetic=True):
            calls.append((week, season, allow_synthetic))
            odds = pd.DataFrame({"line": [55.5]})
            odds.attrs["odds_audit"] = valid_odds_audit()
            return odds

    monkeypatch.setattr(prop_line_scraper, "NFLPropScraper", FakeScraper)

    result = production_runner.stage_odds(2026, 1)

    assert calls == [(1, 2026, False)]
    assert result["status"] == "ok"
    assert result["stage"] == "odds"
    assert result["odds_count"] == 1
    assert result["odds_validation"]["reason_code"] == "validated"


@pytest.mark.parametrize(
    "audit_update,reason_code",
    [
        ({"source_statuses": ["STALE-ON-ERROR"]}, "stale_cache"),
        ({"covered_event_markets": 2}, "market_coverage"),
    ],
)
def test_odds_stage_rejects_stale_or_partial_snapshots(
    monkeypatch, audit_update, reason_code
) -> None:
    from scripts import prop_line_scraper

    class FakeScraper:
        def run_weekly_update(self, week, season, allow_synthetic=True):
            odds = pd.DataFrame({"line": [55.5]})
            audit = valid_odds_audit()
            audit.update(audit_update)
            odds.attrs["odds_audit"] = audit
            return odds

    monkeypatch.setattr(prop_line_scraper, "NFLPropScraper", FakeScraper)

    result = production_runner.stage_odds(2026, 1)

    assert result["status"] == "error"
    assert result["odds_validation"]["reason_code"] == reason_code


def test_odds_provider_failure_still_records_validation_reason(monkeypatch) -> None:
    from scripts import prop_line_scraper

    class FailingScraper:
        last_weekly_audit = {
            "source_statuses": ["STALE-ON-ERROR"],
            "response_ages_seconds": [900],
            "scheduled_events": 16,
            "covered_events": 0,
            "covered_event_markets": 0,
            "odds_rows": 0,
        }

        def run_weekly_update(self, week, season, allow_synthetic=True):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(prop_line_scraper, "NFLPropScraper", FailingScraper)

    result = production_runner.stage_odds(2026, 1)

    assert result["status"] == "error"
    assert result["odds_validation"]["reason_code"] == "provider_error"
    assert result["odds_validation"]["snapshot_reason_code"] == "stale_cache"
    assert result["odds_validation"]["provider_error"] == "provider unavailable"


def test_odds_scraper_initialization_failure_is_fail_closed(monkeypatch) -> None:
    from scripts import prop_line_scraper

    class FailingScraper:
        def __init__(self):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(prop_line_scraper, "NFLPropScraper", FailingScraper)

    result = production_runner.stage_odds(2026, 1)

    assert result["status"] == "error"
    assert result["odds_validation"]["valid"] is False
    assert result["odds_validation"]["reason_code"] == "provider_error"
    assert result["odds_validation"]["snapshot_reason_code"] == "provenance_missing"


def test_malformed_odds_audit_is_persistable_fail_closed_evidence(monkeypatch) -> None:
    from scripts import prop_line_scraper

    class MalformedAuditScraper:
        def run_weekly_update(self, week, season, allow_synthetic=True):
            odds = pd.DataFrame({"line": [55.5]})
            audit = valid_odds_audit()
            audit["responses_observed"] = "not-an-integer"
            odds.attrs["odds_audit"] = audit
            return odds

    monkeypatch.setattr(prop_line_scraper, "NFLPropScraper", MalformedAuditScraper)

    result = production_runner.stage_odds(2026, 1)

    assert result["status"] == "error"
    assert result["odds_validation"]["valid"] is False
    assert result["odds_validation"]["reason_code"] == "validation_error"
    assert "not-an-integer" in result["odds_validation"]["validation_error"]


def test_weekly_report_refresh_uses_prepare_then_live_odds(monkeypatch) -> None:
    from scripts import run_prop_update

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
    try:
        from scripts import activate_betting
    except ModuleNotFoundError as exc:
        if exc.name not in {"prop_integration", "value_betting_engine"}:
            raise
        pytest.skip(f"private algorithm module is unavailable: {exc}")

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


def test_inline_main_exits_nonzero_when_pipeline_report_has_errors(monkeypatch) -> None:
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

    monkeypatch.setattr(
        sys,
        "argv",
        ["production_runner", "--season", "2026", "--week", "1", "--inline"],
    )
    monkeypatch.setattr(production_runner, "run_production_pipeline", failed_pipeline)

    with pytest.raises(SystemExit) as exc_info:
        production_runner.main()

    assert exc_info.value.code == 1


def test_inline_main_returns_normally_for_successful_pipeline_report(monkeypatch) -> None:
    report: dict[str, Any] = {
        "success": True,
        "stages": [{"status": "ok", "stage": "ingest"}],
        "errors": [],
    }

    def successful_pipeline(*args: object, **kwargs: object) -> dict[str, Any]:
        return report

    monkeypatch.setattr(
        sys,
        "argv",
        ["production_runner", "--season", "2026", "--week", "1", "--inline"],
    )
    monkeypatch.setattr(production_runner, "run_production_pipeline", successful_pipeline)

    production_runner.main()


def test_main_enqueues_by_default(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    class FakeJob:
        job_id = "job-1"
        run_id = "run-1"
        status = "queued"

    class FakeService:
        def create_pipeline_job(self, **kwargs):
            calls.append(kwargs)
            return FakeJob()

    import pipeline_jobs.service

    monkeypatch.setattr(pipeline_jobs.service, "JobService", FakeService)
    monkeypatch.setattr(sys, "argv", ["production_runner", "--season", "2026", "--week", "1"])

    production_runner.main()

    assert calls == [
        {
            "season": 2026,
            "week": 1,
            "source": "cli",
            "skip_ingest": False,
            "skip_odds": False,
        }
    ]
    assert '"status": "queued"' in capsys.readouterr().out
