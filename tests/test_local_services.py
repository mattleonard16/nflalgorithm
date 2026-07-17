"""Tests for supervised local full-stack startup helpers."""

from __future__ import annotations

import urllib.error

from scripts.run_local_services import local_service_commands, wait_for_readiness


def test_local_commands_use_separate_worker_api_and_frontend(monkeypatch) -> None:
    import scripts.run_local_services as local_services

    monkeypatch.setattr(local_services.config.api, "host", "127.0.0.1")
    monkeypatch.setattr(local_services.config.api, "port", 8123)
    monkeypatch.setenv("FRONTEND_PORT", "3123")

    commands = local_service_commands()

    assert [command.name for command in commands] == ["worker", "api", "frontend"]
    assert commands[0].command[-2:] == ("-m", "pipeline_jobs.worker")
    assert "api.application:app" in commands[1].command
    assert "8123" in commands[1].command
    assert commands[2].command == ("npm", "run", "dev", "--", "--port", "3123")
    assert commands[2].cwd.name == "frontend"


def test_wait_for_readiness_retries_until_success(monkeypatch) -> None:
    import scripts.run_local_services as local_services

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    attempts = iter([urllib.error.URLError("starting"), Response()])

    def open_url(*_args, **_kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(local_services.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(local_services.time, "sleep", lambda _seconds: None)

    wait_for_readiness("http://localhost:8000/readyz", timeout_seconds=1)
