"""Release-grade weekly shadow evidence tests."""

from __future__ import annotations

import hashlib
import json

import pytest

from schema_migrations import MigrationManager
from scripts.shadow_weekly_outputs import (
    _semantic_artifact_hash,
    capture_snapshot,
    compare_snapshots,
)
from utils.db import execute


@pytest.fixture()
def shadow_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "shadow.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(db_path).run()
    return db_path


def _section(records):
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "records": records,
        "row_count": len(records),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _complete_snapshot(*, odds_timestamp: str, duration: float, commit_sha: str = "a" * 40) -> dict:
    return {
        "schema_version": 2,
        "commit_sha": commit_sha,
        "season": 2025,
        "week": 8,
        "sections": {
            "input_freshness": _section(
                [{"feed": "odds", "season": 2025, "week": 8, "as_of": odds_timestamp}]
            ),
            "projections": _section([{"player_id": "p1", "market": "pass_yds", "mu": 247.5}]),
            "odds": _section(
                [
                    {
                        "event_id": "e1",
                        "player_id": "p1",
                        "market": "pass_yds",
                        "sportsbook": "book-a",
                        "line": 244.5,
                        "price": -110,
                        "as_of": odds_timestamp,
                    }
                ]
            ),
            "candidates": _section(
                [{"player_id": "p1", "market": "pass_yds", "decision": "APPROVED"}]
            ),
            "approved_card": _section([{"player_id": "p1", "market": "pass_yds", "line": 244.5}]),
            "rejected_plays": _section([]),
            "artifacts": _section(
                [
                    {
                        "kind": "run_report",
                        "checksum": "a" * 64,
                        "size_bytes": 100,
                        "semantic_sha256": "c" * 64,
                    }
                ]
            ),
            "api_visible_state": _section([{"status": "completed", "stages_completed": 6}]),
        },
        "stage_timing": {
            "total_duration_seconds": duration,
            "stages": [
                {"name": "prepare_week", "status": "completed", "duration_seconds": duration}
            ],
        },
    }


def test_empty_snapshots_are_not_equivalence_evidence() -> None:
    empty = {
        "season": 2025,
        "week": 8,
        "tables": {
            table: {"records": [], "row_count": 0, "sha256": hashlib.sha256(b"[]").hexdigest()}
            for table in (
                "weekly_projections",
                "materialized_value_view",
                "risk_assessments",
                "agent_decisions",
            )
        },
    }

    result = compare_snapshots(empty, empty)

    assert result["passed"] is False
    assert "baseline schema_version is not 2" in result["blockers"]
    assert "candidate schema_version is not 2" in result["blockers"]


def test_shadow_comparison_reports_stage_timing_without_requiring_equal_duration() -> None:
    baseline = _complete_snapshot(odds_timestamp="2025-10-26T16:00:00+00:00", duration=120.0)
    candidate = _complete_snapshot(odds_timestamp="2025-10-26T16:00:00+00:00", duration=90.0)

    result = compare_snapshots(baseline, candidate)

    assert result["passed"] is True
    assert result["stage_timing"] == {
        "matched_execution_shape": True,
        "baseline_total_seconds": 120.0,
        "candidate_total_seconds": 90.0,
        "delta_seconds": -30.0,
        "ratio": 0.75,
        "stages": [
            {
                "name": "prepare_week",
                "status_matched": True,
                "baseline_seconds": 120.0,
                "candidate_seconds": 90.0,
                "delta_seconds": -30.0,
            }
        ],
    }


def test_shadow_comparison_detects_odds_timestamp_drift() -> None:
    baseline = _complete_snapshot(odds_timestamp="2025-10-26T16:00:00+00:00", duration=100.0)
    candidate = _complete_snapshot(odds_timestamp="2025-10-26T16:01:00+00:00", duration=100.0)

    result = compare_snapshots(baseline, candidate)

    assert result["passed"] is False
    assert result["sections"]["odds"]["matched"] is False
    assert result["sections"]["input_freshness"]["matched"] is False


def test_shadow_comparison_requires_artifact_checksums() -> None:
    baseline = _complete_snapshot(odds_timestamp="2025-10-26T16:00:00+00:00", duration=100.0)
    candidate = _complete_snapshot(odds_timestamp="2025-10-26T16:00:00+00:00", duration=100.0)
    candidate["sections"]["artifacts"] = _section(
        [
            {
                "kind": "run_report",
                "checksum": None,
                "size_bytes": 100,
                "semantic_sha256": "c" * 64,
            }
        ]
    )

    result = compare_snapshots(baseline, candidate)

    assert result["passed"] is False
    assert "candidate artifacts are missing checksums" in result["blockers"]


def test_shadow_comparison_uses_semantic_artifact_content() -> None:
    baseline = _complete_snapshot(odds_timestamp="2025-10-26T16:00:00+00:00", duration=100.0)
    candidate = _complete_snapshot(
        odds_timestamp="2025-10-26T16:00:00+00:00",
        duration=100.0,
        commit_sha="b" * 40,
    )
    candidate["sections"]["artifacts"]["records"][0]["checksum"] = "b" * 64
    candidate["sections"]["artifacts"]["records"][0]["size_bytes"] = 120
    candidate["sections"]["artifacts"] = _section(candidate["sections"]["artifacts"]["records"])

    result = compare_snapshots(baseline, candidate)

    assert result["passed"] is True
    assert result["sections"]["artifacts"]["matched"] is True


def test_run_report_semantic_hash_excludes_execution_timing(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            {
                "success": True,
                "started_at": "2025-10-26T16:00:00Z",
                "finished_at": "2025-10-26T16:02:00Z",
                "stages": [{"stage": "prepare", "status": "ok", "duration_seconds": 120}],
            }
        )
    )
    candidate.write_text(
        json.dumps(
            {
                "success": True,
                "started_at": "2025-10-26T16:05:00Z",
                "finished_at": "2025-10-26T16:06:00Z",
                "stages": [{"stage": "prepare", "status": "ok", "duration_seconds": 60}],
            }
        )
    )

    assert baseline.read_bytes() != candidate.read_bytes()
    assert _semantic_artifact_hash(baseline) == _semantic_artifact_hash(candidate)


def test_shadow_comparison_requires_commit_identity() -> None:
    baseline = _complete_snapshot(odds_timestamp="2025-10-26T16:00:00+00:00", duration=100.0)
    candidate = _complete_snapshot(odds_timestamp="2025-10-26T16:00:00+00:00", duration=100.0)
    baseline["commit_sha"] = None

    result = compare_snapshots(baseline, candidate)

    assert result["passed"] is False
    assert "baseline commit_sha is not a full 40-character Git SHA" in result["blockers"]


def test_capture_collects_every_weekly_evidence_dimension(shadow_db) -> None:
    del shadow_db
    execute(
        "INSERT INTO feed_freshness (feed, season, week, as_of) VALUES (?, ?, ?, ?)",
        ("odds", 2025, 8, "2025-10-26T16:00:00+00:00"),
    )
    execute(
        """
        INSERT INTO weekly_projections (
            season, week, player_id, team, opponent, market, mu, sigma,
            model_version, featureset_hash, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (2025, 8, "p1", "SF", "LAR", "pass_yds", 247.5, 20.0, "v1", "f1", "now"),
    )
    execute(
        """
        INSERT INTO weekly_odds (
            event_id, season, week, player_id, market, sportsbook, line, price, as_of
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "e1",
            2025,
            8,
            "p1",
            "pass_yds",
            "book-a",
            244.5,
            -110,
            "2025-10-26T16:00:00+00:00",
        ),
    )
    execute(
        """
        INSERT INTO agent_decisions (
            season, week, player_id, market, decision, merged_confidence, votes, decided_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (2025, 8, "p1", "pass_yds", "REJECTED", 0.4, "{}", "now"),
    )
    execute(
        """
        INSERT INTO pipeline_runs (
            run_id, season, week, status, stages_requested, stages_completed,
            started_at, finished_at, source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run-1",
            2025,
            8,
            "completed",
            6,
            1,
            "2025-10-26T16:00:00+00:00",
            "2025-10-26T16:02:00+00:00",
            "scheduler",
            "2025-10-26T16:02:00+00:00",
        ),
    )
    execute(
        """
        INSERT INTO pipeline_stage_runs (
            run_id, stage_name, ordinal, status, attempt, started_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run-1",
            "prepare_week",
            0,
            "completed",
            1,
            "2025-10-26T16:00:00+00:00",
            "2025-10-26T16:02:00+00:00",
        ),
    )
    execute(
        """
        INSERT INTO pipeline_artifacts (
            artifact_id, run_id, kind, uri, checksum, size_bytes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("artifact-1", "run-1", "run_report", "report.json", "a" * 64, 100, "now"),
    )

    snapshot = capture_snapshot(2025, 8, run_id="run-1")

    assert snapshot["sections"]["input_freshness"]["row_count"] == 1
    assert snapshot["sections"]["projections"]["records"][0]["mu"] == 247.5
    assert snapshot["sections"]["odds"]["records"][0]["as_of"] == ("2025-10-26T16:00:00+00:00")
    assert snapshot["sections"]["candidates"]["row_count"] == 1
    assert snapshot["sections"]["approved_card"]["row_count"] == 0
    assert snapshot["sections"]["rejected_plays"]["row_count"] == 1
    assert snapshot["sections"]["artifacts"]["records"][0]["checksum"] == "a" * 64
    assert snapshot["sections"]["api_visible_state"]["records"][0]["status"] == "completed"
    assert snapshot["stage_timing"]["total_duration_seconds"] == 120.0
