"""Optional Sentry error tracking, gated entirely by environment.

``sentry-sdk`` is not a declared dependency of this project (install it via
the ``sentry`` extra: ``uv sync --extra sentry``, or add it to the
environment directly). ``init_error_tracking`` no-ops unless both:

- ``SENTRY_DSN`` is set, and
- ``sentry_sdk`` imports successfully.

Setting ``SENTRY_DSN`` without the package installed is a misconfiguration
worth surfacing, so that combination logs one structured warning and
continues rather than crashing the caller.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def init_error_tracking(service: str) -> bool:
    """Initialize Sentry for ``service`` if configured. Returns True if active."""
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return False

    try:
        import sentry_sdk
    except ImportError:
        logger.warning(
            "SENTRY_DSN set but sentry_sdk not installed",
            extra={"event": "error_tracking.misconfigured", "service": service},
        )
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("ENVIRONMENT", os.getenv("ENV", "development")),
        release=os.getenv("SENTRY_RELEASE"),
    )
    logger.info(
        "Error tracking initialized",
        extra={"event": "error_tracking.initialized", "service": service},
    )
    return True
