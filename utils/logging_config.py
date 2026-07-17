"""Shared logging configuration for service and operational entrypoints."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

_CONTEXT_FIELDS = (
    "event",
    "run_id",
    "job_id",
    "worker_id",
    "season",
    "week",
    "stage",
)


class JsonFormatter(logging.Formatter):
    """Format logs as one JSON object per line with a stable field set."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(service: str) -> None:
    """Configure root logging from ``LOG_LEVEL`` and ``LOG_FORMAT``.

    ``LOG_FORMAT=json`` is intended for deployed services. Local commands default
    to a compact console format. Only allow-listed context fields are serialized,
    which avoids accidentally emitting credentials attached to a log record.
    """

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(
            f"Invalid LOG_LEVEL={level_name!r}. Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    handler = logging.StreamHandler()
    if os.getenv("LOG_FORMAT", "console").strip().lower() == "json":
        handler.setFormatter(JsonFormatter(service))
    else:
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s %(levelname)s [{service}] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
