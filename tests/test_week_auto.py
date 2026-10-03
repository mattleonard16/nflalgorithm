"""The Wednesday job's step order, failure handling, status record, and alert."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import week_auto


class _FakeMake:
    """Records each step and fails the ones named in ``failures``."""

    def __init__(self, failures: dict[str, int] | None = None) -> None:
        self.failures = failures or {}
        self.steps: list[str] = []

    def __call__(self, argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
        step = argv[1]
        self.steps.append(step)
        return subprocess.CompletedProcess(argv, self.failures.get(step, 0))


def _run(tmp_path: Path, week: int, failures: dict[str, int] | None = None):
    make = _FakeMake(failures)
    alerts: list[str] = []
    status_path = tmp_path / "week_auto_status.json"
    exit_code = week_auto.run_week(
        2026,
        week,
        run_step=make,
        notify=alerts.append,
        status_path=status_path,
    )
    return exit_code, make.steps, alerts, json.loads(status_path.read_text())


def test_a_clean_run_records_success_and_sends_no_alert(tmp_path) -> None:
    exit_code, steps, alerts, status = _run(tmp_path, week=4)

    assert exit_code == 0
    assert steps == ["db-analyze", "week-predict", "week-lines", "week-grade", "week-research"]
    assert alerts == []
    assert status["ok"] is True
    assert (status["season"], status["week"], status["failed_step"]) == (2026, 4, None)


def test_a_failed_prediction_stops_the_run_and_alerts(tmp_path) -> None:
    exit_code, steps, alerts, status = _run(tmp_path, week=4, failures={"week-predict": 2})

    assert exit_code == 2
    assert steps == ["db-analyze", "week-predict"]
    assert len(alerts) == 1 and "week-predict" in alerts[0]
    assert status["ok"] is False
    assert (status["failed_step"], status["exit_code"]) == ("week-predict", 2)


def test_a_failed_grade_is_a_warning_and_the_memo_still_runs(tmp_path) -> None:
    exit_code, steps, alerts, status = _run(tmp_path, week=4, failures={"week-grade": 1})

    assert exit_code == 0
    assert steps[-2:] == ["week-grade", "week-research"]
    assert status["ok"] is True
    assert status["warnings"] == ["week-grade 2026 W3 exited 1"]
    assert alerts == []


def test_week_one_has_nothing_to_grade(tmp_path) -> None:
    _, steps, _, _ = _run(tmp_path, week=1)

    assert steps == ["db-analyze", "week-predict", "week-lines"]


def test_a_broken_notifier_does_not_change_the_exit_code(tmp_path) -> None:
    def broken_notify(message: str) -> None:
        raise OSError("osascript missing")

    exit_code = week_auto.run_week(
        2026,
        4,
        run_step=_FakeMake({"week-lines": 2}),
        notify=broken_notify,
        status_path=tmp_path / "status.json",
    )

    assert exit_code == 2


def test_prediction_runs_for_the_target_week_with_context_factors_off(
    tmp_path, monkeypatch
) -> None:
    # The 2025 walk-forward raised yardage MAE with context factors on, so
    # the job must not switch them on.
    monkeypatch.delenv("NFL_FEATURE_CONTEXT_FACTORS", raising=False)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def record(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
        calls.append((argv, env))
        return subprocess.CompletedProcess(argv, 0)

    week_auto.run_week(
        2026, 4, run_step=record, notify=lambda m: None, status_path=tmp_path / "s.json"
    )

    argv, env = next(call for call in calls if call[0][1] == "week-predict")
    assert argv[2:] == ["SEASON=2026", "WEEK=4"]
    assert "NFL_FEATURE_CONTEXT_FACTORS" not in env
    grade_argv = next(call[0] for call in calls if call[0][1] == "week-grade")
    assert grade_argv[2:] == ["SEASON=2026", "WEEK=3"]


def test_an_unresolvable_week_still_records_failure_and_alerts(tmp_path, monkeypatch) -> None:
    from utils import current_week

    alerts: list[str] = []
    monkeypatch.setattr(week_auto, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(week_auto, "_notify_macos", alerts.append)
    monkeypatch.setattr(
        current_week,
        "resolve_current_week",
        lambda: (_ for _ in ()).throw(RuntimeError("no upcoming games")),
    )

    exit_code = week_auto.main([])

    status = json.loads((tmp_path / "status.json").read_text())
    assert exit_code == 1
    assert status["failed_step"] == "resolve-week"
    assert len(alerts) == 1


def test_the_saturday_refresh_only_repredicts_and_republishes(tmp_path) -> None:
    make = _FakeMake()
    status_path = tmp_path / "refresh.json"

    exit_code = week_auto.run_week(
        2026, 4, run_step=make, notify=lambda m: None, status_path=status_path, refresh=True
    )

    assert exit_code == 0
    assert make.steps == ["week-predict", "week-lines"]
    assert json.loads(status_path.read_text())["kind"] == "refresh"
