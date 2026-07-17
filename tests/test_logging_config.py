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
