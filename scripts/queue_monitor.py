"""Emit structured durable-pipeline telemetry for platform log alerts."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Mapping

from pipeline_jobs.service import JobService

logger = logging.getLogger(__name__)


def alert_reasons(metrics: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    max_queue_age = float(os.getenv("PIPELINE_MAX_QUEUE_AGE_SECONDS", "300"))
    if int(metrics.get("stale_running", 0)) > 0:
        reasons.append("stale worker lease detected")
    if float(metrics.get("oldest_queued_seconds", 0)) > max_queue_age:
        reasons.append("oldest queued job exceeded threshold")
    return reasons


def emit_once(service: JobService | None = None) -> dict[str, Any]:
    metrics = (service or JobService()).operational_metrics()
    reasons = alert_reasons(metrics)
    payload = {"event": "pipeline_runtime_metrics", "alerts": reasons, **metrics}
    message = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    (logger.warning if reasons else logger.info)(message)
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    interval = max(5.0, float(os.getenv("PIPELINE_MONITOR_INTERVAL_SECONDS", "60")))
    while True:
        emit_once()
        time.sleep(interval)


if __name__ == "__main__":
    main()
