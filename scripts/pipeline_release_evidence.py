"""Assemble fail-closed, SHA-bound staging promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REQUIRED_EVIDENCE = (
    "database_matrix",
    "staging_failure_safety",
    "authorization",
    "staging_soak",
    "rollback_rehearsal",
    "shadow_weekly_run",
    "monitoring_delivery",
)
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _valid_sha(value: str) -> bool:
    return bool(_SHA_PATTERN.fullmatch(value.lower()))


def build_manifest(
    *,
    candidate_sha: str,
    origin_main_sha: str,
    origin_main_is_ancestor: bool,
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require every release proof to pass and belong to the exact candidate."""
    blockers: list[str] = []
    candidate_sha = candidate_sha.lower()
    origin_main_sha = origin_main_sha.lower()
    known_commit = _valid_sha(candidate_sha)
    if not known_commit:
        blockers.append("candidate SHA is not a full 40-character Git SHA")
    integrated = _valid_sha(origin_main_sha) and origin_main_is_ancestor
    if not _valid_sha(origin_main_sha):
        blockers.append("origin/main SHA is not a full 40-character Git SHA")
    elif not origin_main_is_ancestor:
        blockers.append("origin/main is not an ancestor of the candidate")

    gates: dict[str, Any] = {
        "known_commit_sha": {"passed": known_commit, "candidate_sha": candidate_sha},
        "integrated_with_origin_main": {
            "passed": integrated,
            "origin_main_sha": origin_main_sha,
        },
    }
    for name in REQUIRED_EVIDENCE:
        item = evidence.get(name)
        if not isinstance(item, Mapping):
            blockers.append(f"{name} evidence is missing")
            gates[name] = {"passed": False, "reason": "missing"}
            continue
        item_sha = str(item.get("candidate_sha", "")).lower()
        sha_bound = known_commit and item_sha == candidate_sha
        proof_passed = item.get("passed") is True
        if not sha_bound:
            blockers.append(f"{name} evidence belongs to a different candidate SHA")
        if not proof_passed:
            blockers.append(f"{name} evidence did not pass")
        gates[name] = {
            **dict(item),
            "passed": sha_bound and proof_passed,
            "candidate_sha": item_sha,
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": candidate_sha,
        "origin_main_sha": origin_main_sha,
        "passed": not blockers,
        "blockers": blockers,
        "gates": gates,
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence must be a JSON object: {path}")
    raw = path.read_bytes()
    return {
        **payload,
        "evidence_file": str(path),
        "evidence_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sha")
    parser.add_argument("--origin-main", default="origin/main")
    parser.add_argument("--database-matrix", required=True, type=Path)
    parser.add_argument("--staging-failure-safety", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--staging-soak", required=True, type=Path)
    parser.add_argument("--rollback-rehearsal", required=True, type=Path)
    parser.add_argument("--shadow-weekly-run", required=True, type=Path)
    parser.add_argument("--monitoring-delivery", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate_sha = args.candidate_sha or _git("rev-parse", "HEAD")
    origin_main_sha = _git("rev-parse", args.origin_main)
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", origin_main_sha, candidate_sha],
            check=False,
        ).returncode
        == 0
    )
    paths = {
        "database_matrix": args.database_matrix,
        "staging_failure_safety": args.staging_failure_safety,
        "authorization": args.authorization,
        "staging_soak": args.staging_soak,
        "rollback_rehearsal": args.rollback_rehearsal,
        "shadow_weekly_run": args.shadow_weekly_run,
        "monitoring_delivery": args.monitoring_delivery,
    }
    evidence = {name: _load_evidence(path) for name, path in paths.items()}
    manifest = build_manifest(
        candidate_sha=candidate_sha,
        origin_main_sha=origin_main_sha,
        origin_main_is_ancestor=ancestor,
        evidence=evidence,
    )
    rendered = json.dumps(manifest, indent=2, default=str) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
