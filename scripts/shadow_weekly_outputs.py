"""Capture and compare deterministic weekly outputs from legacy and queued runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from utils.db import fetchall, get_table_columns

OUTPUT_TABLES = (
    "weekly_projections",
    "materialized_value_view",
    "risk_assessments",
    "agent_decisions",
)
VOLATILE_COLUMNS = {
    "generated_at",
    "assessed_at",
    "decided_at",
    "created_at",
    "updated_at",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return round(float(value), 10)
    if isinstance(value, float):
        return round(value, 10)
    return value


def capture_snapshot(season: int, week: int) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for table in OUTPUT_TABLES:
        columns = [name for name in get_table_columns(table) if name not in VOLATILE_COLUMNS]
        selected = ", ".join(columns)
        rows = fetchall(
            f"SELECT {selected} FROM {table} WHERE season = ? AND week = ?",
            (season, week),
        )
        records = [
            {column: _json_value(value) for column, value in zip(columns, row)} for row in rows
        ]
        records.sort(key=lambda record: json.dumps(record, sort_keys=True, default=str))
        canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
        tables[table] = {
            "rows": records,
            "row_count": len(records),
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        }
    return {"season": season, "week": week, "tables": tables}


def compare_snapshots(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    passed = baseline.get("season") == candidate.get("season") and baseline.get(
        "week"
    ) == candidate.get("week")
    for table in OUTPUT_TABLES:
        baseline_table = baseline.get("tables", {}).get(table, {})
        candidate_table = candidate.get("tables", {}).get(table, {})
        matched = baseline_table.get("sha256") == candidate_table.get("sha256")
        passed = passed and matched
        comparisons[table] = {
            "matched": matched,
            "baseline_rows": baseline_table.get("row_count"),
            "candidate_rows": candidate_table.get("row_count"),
            "baseline_sha256": baseline_table.get("sha256"),
            "candidate_sha256": candidate_table.get("sha256"),
        }
    return {"passed": passed, "tables": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--season", required=True, type=int)
    capture.add_argument("--week", required=True, type=int)
    capture.add_argument("--output", required=True, type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()

    if args.command == "capture":
        snapshot = capture_snapshot(args.season, args.week)
        args.output.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
        print(json.dumps({"output": str(args.output), "tables": snapshot["tables"]}, default=str))
        return

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = compare_snapshots(baseline, candidate)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
