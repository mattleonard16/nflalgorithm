"""Tests for the optional, env-gated Sentry error tracking wrapper."""

from __future__ import annotations

import builtins
import logging

import pytest

from utils.error_tracking import init_error_tracking


def test_noop_without_sentry_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)

    assert init_error_tracking("test-service") is False


def test_warns_when_dsn_set_but_sentry_sdk_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://example.invalid/1")

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sentry_sdk":
            raise ImportError("no module named sentry_sdk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with caplog.at_level(logging.WARNING, logger="utils.error_tracking"):
        result = init_error_tracking("test-service")

    assert result is False
    assert any("sentry_sdk not installed" in record.message for record in caplog.records)


def test_initializes_when_dsn_set_and_sentry_sdk_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://example.invalid/1")
    monkeypatch.setenv("ENVIRONMENT", "staging")

    calls = {}

    class _FakeSentrySdk:
        @staticmethod
        def init(**kwargs):
            calls.update(kwargs)

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sentry_sdk":
            return _FakeSentrySdk
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    result = init_error_tracking("test-service")

    assert result is True
    assert calls["dsn"] == "https://example.invalid/1"
    assert calls["environment"] == "staging"
