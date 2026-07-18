"""Rehearse candidate migration, application rollback, and database restore safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_ref(repo_root: Path, ref: str, destination: Path) -> None:
    archive = subprocess.Popen(
        ["git", "archive", ref],
        cwd=repo_root,
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    try:
        extracted = subprocess.run(
            ["tar", "-x", "-C", str(destination)],
            stdin=archive.stdout,
            check=False,
            capture_output=True,
            text=False,
        )
    finally:
        archive.stdout.close()
    archive_status = archive.wait()
    if archive_status != 0 or extracted.returncode != 0:
        detail = extracted.stderr.decode(errors="replace") if extracted.stderr else ""
        raise RuntimeError(f"could not extract predeploy ref {ref}: {detail}")


def _run_python(source: Path, database: Path, code: str) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "DB_BACKEND": "sqlite",
            "SQLITE_DB_PATH": str(database),
            "PYTHONPATH": str(source),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "rollback rehearsal subprocess failed")


def rehearse_rollback(
    *,
    repo_root: Path,
    predeploy_ref: str,
    candidate_sha: str,
    application_module: str = "api.application",
    application_attribute: str = "app",
) -> dict[str, Any]:
    """Use disposable files only; the configured application database is never touched."""
    steps: list[str] = []
    result: dict[str, Any] = {
        "candidate_sha": candidate_sha,
        "predeploy_ref": predeploy_ref,
        "application_probe": f"{application_module}:{application_attribute}",
        "passed": False,
        "application_rollback_compatible": False,
        "database_restore_verified": False,
        "steps": steps,
    }
    with tempfile.TemporaryDirectory(prefix="pipeline-rollback-") as temporary:
        root = Path(temporary)
        predeploy_source = root / "predeploy"
        predeploy_source.mkdir()
        database = root / "pipeline.db"
        backup = root / "pipeline.predeploy.db"
        try:
            predeploy_sha = subprocess.run(
                ["git", "rev-parse", predeploy_ref],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            result["predeploy_sha"] = predeploy_sha
            _extract_ref(repo_root, predeploy_ref, predeploy_source)

            _run_python(
                predeploy_source,
                database,
                """
from schema_migrations import MigrationManager
from utils.db import execute
import os
path = os.environ["SQLITE_DB_PATH"]
MigrationManager(path).run()
execute(
    "INSERT OR REPLACE INTO feed_freshness (feed, season, week, as_of) VALUES (?, ?, ?, ?)",
    ("rollback_probe", 2025, 8, "2025-10-26T16:00:00+00:00"),
)
""",
            )
            steps.append("predeploy_database_created")
            shutil.copy2(database, backup)
            result["backup_sha256"] = _sha256(backup)

            _run_python(
                repo_root,
                database,
                """
from schema_migrations import MigrationManager
import os, sqlite3
path = os.environ["SQLITE_DB_PATH"]
MigrationManager(path).run()
connection = sqlite3.connect(path)
tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
required = {"pipeline_jobs", "pipeline_stage_runs", "pipeline_artifacts", "pipeline_card_staging"}
assert required <= tables, required - tables
assert connection.execute("SELECT COUNT(*) FROM feed_freshness WHERE feed = 'rollback_probe'").fetchone() == (1,)
connection.close()
""",
            )
            steps.append("candidate_migrations_applied")

            _run_python(
                predeploy_source,
                database,
                f"""
from schema_migrations import MigrationManager
import importlib, os, sqlite3
path = os.environ["SQLITE_DB_PATH"]
MigrationManager(path).run()
module = importlib.import_module({application_module!r})
assert getattr(module, {application_attribute!r}, None) is not None
connection = sqlite3.connect(path)
assert connection.execute("SELECT COUNT(*) FROM feed_freshness WHERE feed = 'rollback_probe'").fetchone() == (1,)
connection.close()
""",
            )
            result["application_rollback_compatible"] = True
            steps.append("predeploy_application_started_on_candidate_schema")

            for suffix in ("-wal", "-shm"):
                Path(f"{database}{suffix}").unlink(missing_ok=True)
            shutil.copy2(backup, database)
            result["restored_sha256"] = _sha256(database)
            result["database_restore_verified"] = (
                result["backup_sha256"] == result["restored_sha256"]
            )
            steps.append("predeploy_database_restored")
            result["passed"] = bool(
                result["application_rollback_compatible"] and result["database_restore_verified"]
            )
        except Exception as exc:
            result["error"] = str(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predeploy-ref", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--application-module", default="api.application")
    parser.add_argument("--application-attribute", default="app")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = rehearse_rollback(
        repo_root=Path(__file__).resolve().parents[1],
        predeploy_ref=args.predeploy_ref,
        candidate_sha=args.candidate_sha,
        application_module=args.application_module,
        application_attribute=args.application_attribute,
    )
    rendered = json.dumps(result, indent=2, default=str) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
