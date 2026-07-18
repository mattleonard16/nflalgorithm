"""Application and database rollback rehearsal tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.rehearse_pipeline_rollback import rehearse_rollback


def test_rollback_rehearsal_restores_the_exact_predeploy_database() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = rehearse_rollback(
        repo_root=repo_root,
        predeploy_ref="HEAD",
        candidate_sha=candidate_sha,
        application_module="schema_migrations",
        application_attribute="MigrationManager",
    )

    assert result["passed"] is True
    assert result["candidate_sha"] == candidate_sha
    assert result["application_rollback_compatible"] is True
    assert result["database_restore_verified"] is True
    assert result["backup_sha256"] == result["restored_sha256"]
    assert result["steps"] == [
        "predeploy_database_created",
        "candidate_migrations_applied",
        "predeploy_application_started_on_candidate_schema",
        "predeploy_database_restored",
    ]
