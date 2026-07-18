"""Enqueue and observe a bounded black-box staging soak."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
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
    candidate_sha: str | None = None,
) -> dict[str, Any]:
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {operator_token}"}
    metrics_url = f"{base}/api/system/pipeline-metrics"
    metrics_before_response = requests.get(metrics_url, headers=headers, timeout=20)
    metrics_before_response.raise_for_status()
    metrics_before = metrics_before_response.json()
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
    identity_mismatches = [
        run_id
        for run_id, expected_job_id in created.items()
        if states.get(run_id, {}).get("run_id") != run_id
        or states.get(run_id, {}).get("job_id") != expected_job_id
    ]
    duplicate_stage_keys: list[str] = []
    for run_id, state in states.items():
        seen: set[tuple[Any, Any]] = set()
        for stage in state.get("stages", []):
            if not isinstance(stage, dict):
                continue
            key = (stage.get("attempt"), stage.get("name"))
            if key in seen:
                duplicate_stage_keys.append(f"{run_id}:{key[0]}:{key[1]}")
            seen.add(key)
    metrics_after_response = requests.get(metrics_url, headers=headers, timeout=20)
    metrics_after_response.raise_for_status()
    metrics_after = metrics_after_response.json()
    return {
        "candidate_sha": candidate_sha,
        "soak_id": soak_id,
        "jobs_requested": jobs,
        "unique_runs": len(created),
        "unique_jobs": len(set(created.values())),
        "stuck_run_ids": stuck,
        "non_completed_run_ids": failed,
        "identity_mismatches": identity_mismatches,
        "duplicate_stage_keys": duplicate_stage_keys,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "passed": not stuck and not failed and not identity_mismatches and not duplicate_stage_keys,
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--operator-token", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
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
        candidate_sha=args.candidate_sha,
    )
    rendered = json.dumps(result, indent=2, default=str) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
