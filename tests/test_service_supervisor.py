"""Production process topology coverage."""

from __future__ import annotations

import pytest

from scripts.preflight import Diagnostic
from scripts.queue_monitor import alert_reasons, confirm_delivery, emit_once
from scripts.run_services import service_commands


def test_supervisor_starts_api_and_worker_as_separate_processes(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_PIPELINE_SCHEDULER", raising=False)

    commands = service_commands()

    assert [command[2:] for command in commands] == [
        ["uvicorn", "api.application:app", "--host", "0.0.0.0", "--port", "8000"],
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


def test_queue_monitor_emits_a_synthetic_routing_probe() -> None:
    class Service:
        def operational_metrics(self):
            return {
                "queue": {"queued": 0, "running": 0},
                "stale_running": 0,
                "oldest_queued_seconds": 0,
            }

    payload = emit_once(Service(), synthetic_alert=True)

    assert payload["synthetic"] is True
    assert payload["alerts"] == ["synthetic pipeline alert routing probe"]


def test_queue_monitor_binds_delivery_confirmation_to_probe_sha() -> None:
    confirmation = confirm_delivery(
        {
            "candidate_sha": "a" * 40,
            "synthetic": True,
            "alerts": ["synthetic pipeline alert routing probe"],
        },
        delivery_id="incident-123",
    )

    assert confirmation == {
        "candidate_sha": "a" * 40,
        "passed": True,
        "synthetic_alert_emitted": True,
        "delivery_confirmed": True,
        "delivery_id": "incident-123",
    }


def test_supervisor_stops_before_children_when_preflight_fails(monkeypatch) -> None:
    import scripts.run_services as run_services

    monkeypatch.setattr(run_services, "configure_logging", lambda _service: None)
    monkeypatch.setattr(run_services.MigrationManager, "run", lambda _self: None)
    monkeypatch.setattr(
        run_services,
        "collect_diagnostics",
        lambda **_kwargs: [Diagnostic("database", "fail", "unavailable")],
    )
    monkeypatch.setattr(run_services, "print_diagnostics", lambda _items: None)
    monkeypatch.setattr(
        run_services.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("children must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_services.main()

    assert exc_info.value.code == 2
