"""Capture and compare release-grade weekly shadow evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.db import fetchall, fetchone, get_table_columns

SCHEMA_VERSION = 2
CONTENT_SECTIONS = (
    "input_freshness",
    "projections",
    "odds",
    "candidates",
    "approved_card",
    "rejected_plays",
    "artifacts",
    "api_visible_state",
)
REQUIRED_NONEMPTY_SECTIONS = frozenset(
    {"input_freshness", "projections", "odds", "candidates", "artifacts", "api_visible_state"}
)
VOLATILE_COLUMNS = {
    "generated_at",
    "assessed_at",
    "decided_at",
    "created_at",
    "updated_at",
    "published_run_id",
}
VOLATILE_API_KEYS = frozenset(
    {
        "run_id",
        "job_id",
        "worker_id",
        "claim_token",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
        "available_at",
        "artifact_uri",
    }
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return round(float(value), 10)
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _section(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [dict(record) for record in records]
    normalized.sort(key=_canonical_json)
    canonical = _canonical_json(normalized)
    return {
        "records": normalized,
        "row_count": len(normalized),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _capture_table(
    table: str,
    *,
    season: int,
    week: int,
    extra_where: str = "",
    exclude: set[str] | frozenset[str] = VOLATILE_COLUMNS,
) -> dict[str, Any]:
    columns = [name for name in get_table_columns(table) if name not in exclude]
    selected = ", ".join(columns)
    rows = fetchall(
        f"SELECT {selected} FROM {table} WHERE season = ? AND week = ?{extra_where}",
        (season, week),
    )
    return _section(
        [{column: _json_value(value) for column, value in zip(columns, row)} for row in rows]
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _duration_seconds(started_at: Any, finished_at: Any) -> float | None:
    started = _parse_timestamp(started_at)
    finished = _parse_timestamp(finished_at)
    if not started or not finished or finished < started:
        return None
    return round((finished - started).total_seconds(), 6)


def _normalize_stage_status(value: Any) -> str:
    return {"ok": "completed", "error": "failed"}.get(str(value), str(value))


def _stage_timing_from_run(run_id: str) -> dict[str, Any]:
    run = fetchone(
        "SELECT started_at, finished_at FROM pipeline_runs WHERE run_id = ?",
        (run_id,),
    )
    attempts = fetchone(
        "SELECT MAX(attempt) FROM pipeline_stage_runs WHERE run_id = ?",
        (run_id,),
    )
    attempt = int(attempts[0]) if attempts and attempts[0] is not None else None
    rows = (
        fetchall(
            """
            SELECT stage_name, status, started_at, finished_at
            FROM pipeline_stage_runs
            WHERE run_id = ? AND attempt = ?
            ORDER BY ordinal, stage_name
            """,
            (run_id, attempt),
        )
        if attempt is not None
        else []
    )
    return {
        "total_duration_seconds": _duration_seconds(run[0], run[1]) if run else None,
        "stages": [
            {
                "name": str(name),
                "status": _normalize_stage_status(status),
                "duration_seconds": _duration_seconds(started_at, finished_at),
            }
            for name, status, started_at, finished_at in rows
        ],
    }


def _stage_timing_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    stages = report.get("stages")
    stage_records = stages if isinstance(stages, list) else []
    return {
        "total_duration_seconds": _duration_seconds(
            report.get("started_at"), report.get("finished_at")
        ),
        "stages": [
            {
                "name": str(stage.get("stage", stage.get("name", ""))),
                "status": _normalize_stage_status(stage.get("status")),
                "duration_seconds": (
                    round(float(stage["duration_seconds"]), 6)
                    if stage.get("duration_seconds") is not None
                    else None
                ),
            }
            for stage in stage_records
            if isinstance(stage, Mapping)
        ],
    }


def _capture_artifacts(run_id: str | None, report_path: Path | None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if run_id:
        rows = fetchall(
            """
            SELECT kind, checksum, size_bytes, metadata_json
            FROM pipeline_artifacts WHERE run_id = ?
            """,
            (run_id,),
        )
        for kind, checksum, size_bytes, metadata_json in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else None
            except (json.JSONDecodeError, TypeError):
                metadata = None
            records.append(
                {
                    "kind": str(kind),
                    "checksum": str(checksum) if checksum else None,
                    "size_bytes": int(size_bytes) if size_bytes is not None else None,
                    "metadata": metadata,
                }
            )
    elif report_path and report_path.is_file():
        payload = report_path.read_bytes()
        records.append(
            {
                "kind": "run_report",
                "checksum": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "metadata": None,
            }
        )
    return _section(records)


def _normalize_api_state(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_api_state(item)
            for key, item in value.items()
            if str(key) not in VOLATILE_API_KEYS
        }
    if isinstance(value, list):
        return [_normalize_api_state(item) for item in value]
    return _json_value(value)


def _api_state_from_run(run_id: str) -> dict[str, Any] | None:
    run = fetchone(
        """
        SELECT status, stages_requested, stages_completed, error_message, source,
               report_json, data_health_json
        FROM pipeline_runs WHERE run_id = ?
        """,
        (run_id,),
    )
    if not run:
        return None
    stage_rows = fetchall(
        """
        SELECT stage_name, status, attempt, result_json, error_message
        FROM pipeline_stage_runs WHERE run_id = ? ORDER BY attempt, ordinal, stage_name
        """,
        (run_id,),
    )

    def parsed(value: Any) -> Any:
        try:
            return json.loads(value) if value else None
        except (json.JSONDecodeError, TypeError):
            return None

    return {
        "status": run[0],
        "stages_requested": int(run[1]),
        "stages_completed": int(run[2]),
        "error_message": run[3],
        "source": run[4],
        "report_json": parsed(run[5]),
        "data_health": parsed(run[6]),
        "stages": [
            {
                "name": name,
                "status": _normalize_stage_status(status),
                "attempt": int(attempt),
                "result": parsed(result_json),
                "error_message": error_message,
            }
            for name, status, attempt, result_json, error_message in stage_rows
        ],
    }


def capture_snapshot(
    season: int,
    week: int,
    *,
    run_id: str | None = None,
    run_report: Mapping[str, Any] | None = None,
    report_path: Path | None = None,
    api_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture every behavior-bearing dimension required for weekly shadow proof."""
    sections = {
        "input_freshness": _capture_table(
            "feed_freshness", season=season, week=week, exclude=frozenset()
        ),
        "projections": _capture_table("weekly_projections", season=season, week=week),
        "odds": _capture_table("weekly_odds", season=season, week=week, exclude=frozenset()),
        "candidates": _capture_table("agent_decisions", season=season, week=week),
        "approved_card": _capture_table("materialized_value_view", season=season, week=week),
        "rejected_plays": _capture_table(
            "agent_decisions",
            season=season,
            week=week,
            extra_where=" AND UPPER(decision) = 'REJECTED'",
        ),
        "artifacts": _capture_artifacts(run_id, report_path),
    }
    visible_state = api_state or (_api_state_from_run(run_id) if run_id else None)
    sections["api_visible_state"] = _section(
        [_normalize_api_state(visible_state)] if visible_state is not None else []
    )
    timing = (
        _stage_timing_from_run(run_id) if run_id else _stage_timing_from_report(run_report or {})
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "week": week,
        "sections": sections,
        "stage_timing": timing,
    }


def _snapshot_blockers(label: str, snapshot: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        return [f"{label} schema_version is not {SCHEMA_VERSION}"]
    sections = snapshot.get("sections")
    if not isinstance(sections, Mapping):
        return [f"{label} sections are missing"]
    for name in CONTENT_SECTIONS:
        section = sections.get(name)
        if not isinstance(section, Mapping):
            blockers.append(f"{label} {name} section is missing")
            continue
        if name in REQUIRED_NONEMPTY_SECTIONS and int(section.get("row_count", 0)) < 1:
            blockers.append(f"{label} {name} has no evidence rows")

    for name in ("input_freshness", "odds"):
        section = sections.get(name, {})
        records = section.get("records", []) if isinstance(section, Mapping) else []
        if any(not isinstance(record, Mapping) or not record.get("as_of") for record in records):
            blockers.append(f"{label} {name} is missing source timestamps")

    artifacts = sections.get("artifacts", {})
    artifact_records = artifacts.get("records", []) if isinstance(artifacts, Mapping) else []
    if artifact_records and any(
        not isinstance(record, Mapping) or not record.get("checksum") for record in artifact_records
    ):
        blockers.append(f"{label} artifacts are missing checksums")

    timing = snapshot.get("stage_timing")
    if not isinstance(timing, Mapping) or timing.get("total_duration_seconds") is None:
        blockers.append(f"{label} total stage timing is missing")
    elif not timing.get("stages"):
        blockers.append(f"{label} stage execution evidence is missing")
    return blockers


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _compare_stage_timing(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_total = _number(baseline.get("total_duration_seconds"))
    candidate_total = _number(candidate.get("total_duration_seconds"))
    raw_baseline_stages = baseline.get("stages")
    raw_candidate_stages = candidate.get("stages")
    baseline_stages: list[Mapping[str, Any]] = (
        [stage for stage in raw_baseline_stages if isinstance(stage, Mapping)]
        if isinstance(raw_baseline_stages, list)
        else []
    )
    candidate_stages: list[Mapping[str, Any]] = (
        [stage for stage in raw_candidate_stages if isinstance(stage, Mapping)]
        if isinstance(raw_candidate_stages, list)
        else []
    )
    baseline_shape = [
        (str(stage.get("name")), _normalize_stage_status(stage.get("status")))
        for stage in baseline_stages
    ]
    candidate_shape = [
        (str(stage.get("name")), _normalize_stage_status(stage.get("status")))
        for stage in candidate_stages
    ]
    stage_results: list[dict[str, Any]] = []
    for index in range(max(len(baseline_stages), len(candidate_stages))):
        before = baseline_stages[index] if index < len(baseline_stages) else {}
        after = candidate_stages[index] if index < len(candidate_stages) else {}
        before_duration = _number(before.get("duration_seconds"))
        after_duration = _number(after.get("duration_seconds"))
        stage_results.append(
            {
                "name": str(before.get("name") or after.get("name") or ""),
                "status_matched": (
                    _normalize_stage_status(before.get("status"))
                    == _normalize_stage_status(after.get("status"))
                ),
                "baseline_seconds": before_duration,
                "candidate_seconds": after_duration,
                "delta_seconds": (
                    round(after_duration - before_duration, 6)
                    if before_duration is not None and after_duration is not None
                    else None
                ),
            }
        )
    return {
        "matched_execution_shape": baseline_shape == candidate_shape,
        "baseline_total_seconds": baseline_total,
        "candidate_total_seconds": candidate_total,
        "delta_seconds": (
            round(candidate_total - baseline_total, 6)
            if baseline_total is not None and candidate_total is not None
            else None
        ),
        "ratio": (
            round(candidate_total / baseline_total, 6)
            if baseline_total not in (None, 0) and candidate_total is not None
            else None
        ),
        "stages": stage_results,
    }


def compare_snapshots(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare complete evidence and fail closed when either capture is incomplete."""
    blockers = _snapshot_blockers("baseline", baseline) + _snapshot_blockers("candidate", candidate)
    comparisons: dict[str, Any] = {}
    baseline_sections = baseline.get("sections", {})
    candidate_sections = candidate.get("sections", {})
    for name in CONTENT_SECTIONS:
        before = baseline_sections.get(name, {}) if isinstance(baseline_sections, Mapping) else {}
        after = candidate_sections.get(name, {}) if isinstance(candidate_sections, Mapping) else {}
        matched = bool(before.get("sha256")) and before.get("sha256") == after.get("sha256")
        comparisons[name] = {
            "matched": matched,
            "baseline_rows": before.get("row_count"),
            "candidate_rows": after.get("row_count"),
            "baseline_sha256": before.get("sha256"),
            "candidate_sha256": after.get("sha256"),
        }
    stage_timing = _compare_stage_timing(
        baseline.get("stage_timing", {}), candidate.get("stage_timing", {})
    )
    same_week = baseline.get("season") == candidate.get("season") and baseline.get(
        "week"
    ) == candidate.get("week")
    passed = (
        same_week
        and not blockers
        and all(item["matched"] for item in comparisons.values())
        and stage_timing["matched_execution_shape"]
    )
    return {
        "passed": passed,
        "same_week": same_week,
        "blockers": blockers,
        "sections": comparisons,
        "stage_timing": stage_timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--season", required=True, type=int)
    capture.add_argument("--week", required=True, type=int)
    capture.add_argument("--output", required=True, type=Path)
    capture.add_argument("--run-id")
    capture.add_argument("--run-report", type=Path)
    capture.add_argument("--api-state", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()

    if args.command == "capture":
        report = json.loads(args.run_report.read_text()) if args.run_report else None
        api_state = json.loads(args.api_state.read_text()) if args.api_state else None
        snapshot = capture_snapshot(
            args.season,
            args.week,
            run_id=args.run_id,
            run_report=report,
            report_path=args.run_report,
            api_state=api_state,
        )
        args.output.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
        print(json.dumps({"output": str(args.output), "sections": snapshot["sections"]}))
        return

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = compare_snapshots(baseline, candidate)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
