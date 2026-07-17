"""Tests for the tracked runtime configuration fallback."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_tracked_config_loads_without_private_override(tmp_path: Path) -> None:
    database_path = tmp_path / "fresh-checkout.db"
    missing_override = tmp_path / "missing-config.py"
    env = {
        **os.environ,
        "DB_BACKEND": "sqlite",
        "SQLITE_DB_PATH": str(database_path),
        "NFL_CONFIG_PATH": str(missing_override),
    }
    command = (
        "import json; from config import config; "
        "print(json.dumps({'backend': config.database.backend, "
        "'path': config.database.path, 'markets_ready': hasattr(config, 'pipeline')}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).parent.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    loaded = json.loads(completed.stdout)
    assert loaded == {
        "backend": "sqlite",
        "path": str(database_path),
        "markets_ready": True,
    }


def test_private_override_remains_supported(tmp_path: Path) -> None:
    override = tmp_path / "config.py"
    override.write_text(
        "from types import SimpleNamespace\n"
        "config = SimpleNamespace(source='private-override')\n",
        encoding="utf-8",
    )
    env = {**os.environ, "NFL_CONFIG_PATH": str(override)}

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from config import config; print(config.source, hasattr(config, 'pipeline'))",
        ],
        cwd=Path(__file__).parent.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "private-override True"


def test_odds_safety_requirements_are_environment_configurable(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "NFL_CONFIG_PATH": str(tmp_path / "missing.py"),
        "NFL_ODDS_MAX_AGE_SECONDS": "120",
        "NFL_ODDS_MIN_EVENT_COVERAGE": "0.95",
        "NFL_ODDS_MIN_MARKET_COVERAGE": "0.90",
        "NFL_ODDS_REQUIRED_MARKETS": "player_pass_yds,player_rec_yds",
    }
    command = (
        "import json; from config import config; "
        "print(json.dumps({'age': config.pipeline.odds_max_age_seconds, "
        "'events': config.pipeline.odds_min_event_coverage, "
        "'markets': config.pipeline.odds_min_market_coverage, "
        "'required': config.pipeline.odds_required_markets}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).parent.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "age": 120,
        "events": 0.95,
        "markets": 0.9,
        "required": ["player_pass_yds", "player_rec_yds"],
    }
