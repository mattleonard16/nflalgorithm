"""Developer-facing Make command contracts."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_make(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "--no-print-directory", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_help_surfaces_primary_startup_and_diagnostic_commands() -> None:
    output = run_make("help").stdout

    for command in ("make doctor", "make migrate", "make fullstack", "make pipeline-worker"):
        assert command in output


def test_obsolete_startup_and_synthetic_activation_targets_are_removed() -> None:
    targets = set(run_make("list-targets").stdout.splitlines())

    assert targets.isdisjoint(
        {
            "start_pipeline",
            "stop_pipeline",
            "activate-betting",
            "activate-all",
            "populate-data",
            "train-models",
            "migrate-to-uv",
            "ingest-ncaab",
            "ingest-ncaab-modifiers",
            "ncaab-bracket",
            "ncaab-predict",
            "ncaab-full",
        }
    )


def test_make_compatible_env_file_configures_api_port(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("API_HOST=127.0.0.1\nAPI_PORT=8123\n", encoding="utf-8")

    output = run_make("-n", "api-serve", f"ENV_FILE={env_file}").stdout

    assert "--host 127.0.0.1 --port 8123" in output
    assert "api.application:app" in output


def test_validate_requires_explicit_season_and_weeks() -> None:
    missing = subprocess.run(
        ["make", "--no-print-directory", "validate", "ENV_FILE=/dev/null"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing.returncode != 0
    assert "SEASON and WEEKS are required" in missing.stderr

    command = run_make(
        "-n",
        "validate",
        "ENV_FILE=/dev/null",
        "SEASON=2025",
        "WEEKS=1 2",
    ).stdout
    assert "scripts.evaluate_nfl_projections evaluate" in command
    assert "--season 2025 --weeks 1 2" in command
