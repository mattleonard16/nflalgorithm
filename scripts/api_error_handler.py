"""Circuit-breaker state and retry guidance for external API failures."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class APIErrorHandler:
    """Track endpoint failures and open short-lived circuit breakers."""

    def __init__(self) -> None:
        self.failure_count: dict[str, int] = {}
        self.circuit_breakers: dict[str, datetime] = {}

    def handle_api_failure(self, api_type: str, endpoint: str, error: Exception) -> dict[str, Any]:
        """Record an API failure and return retry and fallback guidance."""
        api_key = f"{api_type}:{endpoint}"
        self.failure_count[api_key] = self.failure_count.get(api_key, 0) + 1

        failure_threshold = 5
        timeout_minutes = 30
        if self.failure_count[api_key] >= failure_threshold:
            self.circuit_breakers[api_key] = datetime.now()
            logger.warning(
                "Circuit breaker opened for %s after %d failures",
                api_key,
                failure_threshold,
            )

        if api_key in self.circuit_breakers:
            minutes_open = (datetime.now() - self.circuit_breakers[api_key]).total_seconds() / 60
            if minutes_open > timeout_minutes:
                del self.circuit_breakers[api_key]
                del self.failure_count[api_key]
                logger.info("Circuit breaker reset for %s", api_key)

        return {
            "should_retry": self._should_retry(api_key, error),
            "wait_time": self._calculate_wait_time(api_key),
            "circuit_open": api_key in self.circuit_breakers,
        }

    def _should_retry(self, api_key: str, error: Exception) -> bool:
        if api_key in self.circuit_breakers:
            return False
        if any(code in str(error) for code in ("400", "401", "403", "404")):
            return False
        return self.failure_count.get(api_key, 0) < 3

    def _calculate_wait_time(self, api_key: str) -> int:
        failures = self.failure_count.get(api_key, 0)
        return 0 if failures == 0 else min(300, 30 * (2 ** (failures - 1)))


api_error_handler = APIErrorHandler()
