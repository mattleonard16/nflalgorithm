"""Emit structured durable-pipeline telemetry for platform log alerts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
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


def emit_once(
    service: JobService | None = None,
    *,
    synthetic_alert: bool = False,
    candidate_sha: str | None = None,
) -> dict[str, Any]:
    metrics = (service or JobService()).operational_metrics()
    reasons = alert_reasons(metrics)
    if synthetic_alert:
        reasons.append("synthetic pipeline alert routing probe")
    payload = {
        "event": "pipeline_runtime_metrics",
        "synthetic": synthetic_alert,
        "candidate_sha": candidate_sha,
        "alerts": reasons,
        **metrics,
    }
    message = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    (logger.warning if reasons else logger.info)(message)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--synthetic-alert", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-sha")
    args = parser.parse_args()
    if args.synthetic_alert and not args.once:
        parser.error("--synthetic-alert requires --once to prevent repeated test alerts")
    logging.basicConfig(level=logging.INFO)
    if args.once:
        payload = emit_once(
            synthetic_alert=args.synthetic_alert,
            candidate_sha=args.candidate_sha,
        )
        rendered = json.dumps(payload, indent=2, default=str) + "\n"
        if args.output:
            args.output.write_text(rendered)
        print(rendered, end="")
        return

    interval = max(5.0, float(os.getenv("PIPELINE_MONITOR_INTERVAL_SECONDS", "60")))
    while True:
        emit_once()
        time.sleep(interval)


if __name__ == "__main__":
    main()
