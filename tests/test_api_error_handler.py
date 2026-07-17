"""Tests for external API failure retry and circuit-breaker guidance."""

from scripts.api_error_handler import APIErrorHandler


def test_transient_failure_returns_bounded_retry_guidance() -> None:
    handler = APIErrorHandler()

    result = handler.handle_api_failure("odds", "receiving_yards", RuntimeError("timeout"))

    assert result == {"should_retry": True, "wait_time": 30, "circuit_open": False}


def test_non_retryable_http_error_does_not_retry() -> None:
    handler = APIErrorHandler()

    result = handler.handle_api_failure("odds", "receiving_yards", RuntimeError("HTTP 401"))

    assert result["should_retry"] is False
    assert result["circuit_open"] is False


def test_repeated_failures_open_circuit() -> None:
    handler = APIErrorHandler()

    for _ in range(5):
        result = handler.handle_api_failure("odds", "receiving_yards", RuntimeError("timeout"))

    assert result["should_retry"] is False
    assert result["circuit_open"] is True
    assert result["wait_time"] == 300
