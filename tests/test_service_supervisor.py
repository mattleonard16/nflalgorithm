"""Production process topology coverage."""

from __future__ import annotations

from scripts.queue_monitor import alert_reasons
from scripts.run_services import service_commands


def test_supervisor_starts_api_and_worker_as_separate_processes(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_PIPELINE_SCHEDULER", raising=False)

    commands = service_commands()

    assert [command[2:] for command in commands] == [
        ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"],
        ["pipeline_jobs.worker"],
    ]


def test_supervisor_can_enable_scheduler(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_PIPELINE_SCHEDULER", "true")

    commands = service_commands()

    assert commands[-1][2:] == ["scripts.pipeline_scheduler"]


def test_supervisor_can_enable_queue_monitor(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_PIPELINE_SCHEDULER", raising=False)
    monkeypatch.setenv("ENABLE_QUEUE_MONITOR", "true")

    assert service_commands()[-1][2:] == ["scripts.queue_monitor"]


def test_queue_monitor_alerts_on_stale_or_old_work(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_MAX_QUEUE_AGE_SECONDS", "60")

    assert alert_reasons({"stale_running": 1, "oldest_queued_seconds": 61}) == [
        "stale worker lease detected",
        "oldest queued job exceeded threshold",
    ]
