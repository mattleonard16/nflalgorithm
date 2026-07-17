"""Enqueue and observe a bounded black-box staging soak."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any

import requests

TERMINAL = {"completed", "failed", "cancelled"}


def run_soak(
    *,
    base_url: str,
    operator_token: str,
    season: int,
    week: int,
    jobs: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {operator_token}"}
    created: dict[str, str] = {}
    soak_id = uuid.uuid4().hex
    for index in range(jobs):
        response = requests.post(
            f"{base}/api/run",
            params={"season": season, "week": week, "skip_ingest": True},
            headers={**headers, "Idempotency-Key": f"soak-{soak_id}-{index}"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        created[str(payload["run_id"])] = str(payload["job_id"])

    if len(created) != jobs or len(set(created.values())) != jobs:
        raise RuntimeError("staging API returned duplicate run or job identifiers")

    deadline = time.monotonic() + timeout_seconds
    states: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        for run_id in created:
            if states.get(run_id, {}).get("status") in TERMINAL:
                continue
            response = requests.get(f"{base}/api/run/{run_id}", headers=headers, timeout=20)
            response.raise_for_status()
            states[run_id] = response.json()
        if len(states) == jobs and all(
            state.get("status") in TERMINAL for state in states.values()
        ):
            break
        time.sleep(poll_seconds)

    stuck = [run_id for run_id in created if states.get(run_id, {}).get("status") not in TERMINAL]
    failed = [run_id for run_id, state in states.items() if state.get("status") != "completed"]
    return {
        "soak_id": soak_id,
        "jobs_requested": jobs,
        "unique_runs": len(created),
        "unique_jobs": len(set(created.values())),
        "stuck_run_ids": stuck,
        "non_completed_run_ids": failed,
        "passed": not stuck and not failed,
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--operator-token", required=True)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    result = run_soak(
        base_url=args.base_url,
        operator_token=args.operator_token,
        season=args.season,
        week=args.week,
        jobs=max(1, args.jobs),
        timeout_seconds=args.timeout_seconds,
        poll_seconds=max(0.1, args.poll_seconds),
    )
    print(json.dumps(result, indent=2, default=str))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
