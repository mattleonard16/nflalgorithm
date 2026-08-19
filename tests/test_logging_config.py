"""Tests for the shared structured logging convention."""

from __future__ import annotations

import json
import logging

import pytest

from utils.logging_config import JsonFormatter, configure_logging


def test_json_formatter_emits_stable_context_without_arbitrary_secrets() -> None:
    record = logging.LogRecord(
        name="tests.operations",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="worker claimed job",
        args=(),
        exc_info=None,
    )
    record.event = "job.claimed"
    record.run_id = "run-123"
    record.password = "do-not-log"

    payload = json.loads(JsonFormatter("worker").format(record))

    assert payload["service"] == "worker"
    assert payload["event"] == "job.claimed"
    assert payload["run_id"] == "run-123"
    assert "password" not in payload


def test_configure_logging_rejects_invalid_level(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
        configure_logging("test")


def test_configure_logging_json_format_end_to_end(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    configure_logging("test-service")
    logger = logging.getLogger("tests.end_to_end")
    logger.info("job started", extra={"event": "job.started", "run_id": "run-456", "token": "secret"})

    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["service"] == "test-service"
    assert payload["message"] == "job started"
    assert payload["event"] == "job.started"
    assert payload["run_id"] == "run-456"
    assert "token" not in payload
    assert set(payload) <= {
        "timestamp",
        "level",
        "service",
        "logger",
        "message",
        "exception",
        "event",
        "run_id",
        "job_id",
        "worker_id",
        "season",
        "week",
        "stage",
    }
