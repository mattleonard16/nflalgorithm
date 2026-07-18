"""Black-box authorization proof for the deployed private pipeline API."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class Check:
    name: str
    expected: int
    actual: int

    @property
    def passed(self) -> bool:
        return self.expected == self.actual


def validate(
    *,
    base_url: str,
    reader_token: str,
    operator_token: str,
    season: int,
    week: int,
    timeout_seconds: float = 15,
) -> list[Check]:
    base = base_url.rstrip("/")
    run_url = f"{base}/api/run"
    metrics_url = f"{base}/api/system/pipeline-metrics"
    checks: list[Check] = []

    anonymous_read = requests.get(metrics_url, timeout=timeout_seconds)
    checks.append(Check("anonymous operational read denied", 401, anonymous_read.status_code))
    anonymous_write = requests.post(
        run_url,
        params={"season": season, "week": week},
        timeout=timeout_seconds,
    )
    checks.append(Check("anonymous mutation denied", 401, anonymous_write.status_code))

    reader_headers = {"Authorization": f"Bearer {reader_token}"}
    reader_read = requests.get(metrics_url, headers=reader_headers, timeout=timeout_seconds)
    checks.append(Check("authenticated reader accepted", 200, reader_read.status_code))
    reader_write = requests.post(
        run_url,
        params={"season": season, "week": week},
        headers={**reader_headers, "Idempotency-Key": f"auth-reader-{uuid.uuid4().hex}"},
        timeout=timeout_seconds,
    )
    checks.append(Check("non-operator mutation denied", 403, reader_write.status_code))

    operator_write = requests.post(
        run_url,
        params={"season": season, "week": week},
        headers={
            "Authorization": f"Bearer {operator_token}",
            "Idempotency-Key": f"auth-operator-{uuid.uuid4().hex}",
        },
        timeout=timeout_seconds,
    )
    checks.append(Check("operator mutation accepted", 200, operator_write.status_code))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reader-token", required=True)
    parser.add_argument("--operator-token", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    args = parser.parse_args()

    checks = validate(
        base_url=args.base_url,
        reader_token=args.reader_token,
        operator_token=args.operator_token,
        season=args.season,
        week=args.week,
    )
    passed = all(check.passed for check in checks)
    result = {
        "candidate_sha": args.candidate_sha,
        "passed": passed,
        "checks": [asdict(check) | {"passed": check.passed} for check in checks],
    }
    rendered = json.dumps(result, indent=2) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
