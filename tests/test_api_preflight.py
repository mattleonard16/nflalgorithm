"""Process-level coverage for API migration preflight and startup ordering."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_make(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "--no-print-directory", *arguments],
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def test_api_preflight_backs_up_and_migrates_explicit_sqlite_path(tmp_path: Path) -> None:
    database = tmp_path / "app.db"
    backup_dir = tmp_path / "backups"
    database.touch()

    result = run_make(
        "api-preflight",
        f"PYTHON={sys.executable}",
        "DB_BACKEND=sqlite",
        f"SQLITE_DB_PATH={database}",
        f"MIGRATION_BACKUP_DIR={backup_dir}",
    )

    backups = list(backup_dir.glob("app.db.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b""
    assert result.stdout.index("Backing up") < result.stdout.index("Applying SQLite")
    with sqlite3.connect(database) as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
        ).fetchone()
    assert table == ("pipeline_runs",)


def test_api_preflight_skips_non_sqlite_backend(tmp_path: Path) -> None:
    database = tmp_path / "should-not-exist.db"
    backup_dir = tmp_path / "backups"

    result = run_make(
        "api-preflight",
        f"PYTHON={sys.executable}",
        "DB_BACKEND=mysql",
        f"SQLITE_DB_PATH={database}",
        f"MIGRATION_BACKUP_DIR={backup_dir}",
    )

    assert "Skipping SQLite API preflight" in result.stdout
    assert not database.exists()
    assert not backup_dir.exists()


@pytest.mark.parametrize("target", ["api", "api-prod"])
def test_api_targets_run_preflight_before_uvicorn(target: str, tmp_path: Path) -> None:
    result = run_make(
        "-n",
        target,
        f"PYTHON={sys.executable}",
        "DB_BACKEND=sqlite",
        f"SQLITE_DB_PATH={tmp_path / 'app.db'}",
        f"MIGRATION_BACKUP_DIR={tmp_path / 'backups'}",
    )

    assert result.stdout.index("scripts.run_migrations") < result.stdout.index("uvicorn")


def test_fullstack_stops_when_preflight_fails(tmp_path: Path) -> None:
    result = run_make(
        "fullstack",
        "PYTHON=/usr/bin/false",
        "DB_BACKEND=sqlite",
        f"SQLITE_DB_PATH={tmp_path / 'app.db'}",
        f"MIGRATION_BACKUP_DIR={tmp_path / 'backups'}",
        check=False,
    )

    assert result.returncode != 0
    assert "Starting Next.js frontend" not in result.stdout
