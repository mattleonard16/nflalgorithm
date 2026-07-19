"""Validate local or deployed runtime configuration before services start."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.logging_config import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_API_TABLES = (
    "feed_freshness",
    "materialized_value_view",
    "pipeline_jobs",
    "pipeline_runs",
    "users",
)
PRIVATE_API_FILE = "api/server.py"
PRIVATE_NFL_FILES = (
    "data_pipeline.py",
    "prop_integration.py",
    "value_betting_engine.py",
    "models/position_specific/weekly.py",
)


@dataclass(frozen=True)
class Diagnostic:
    """One actionable preflight result."""

    name: str
    status: str
    message: str
    action: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == "fail"


def _result(name: str, status: str, message: str, action: str | None = None) -> Diagnostic:
    return Diagnostic(name=name, status=status, message=message, action=action)


def check_python() -> Diagnostic:
    if sys.version_info >= (3, 13):
        return _result("python", "pass", f"Python {sys.version.split()[0]}")
    return _result(
        "python",
        "fail",
        f"Python {sys.version.split()[0]} is unsupported; Python 3.13+ is required.",
        "Run `make install-uv` or install Python 3.13.",
    )


def check_runtime_config() -> tuple[Any | None, list[Diagnostic]]:
    try:
        from config import config
    except Exception as exc:
        return None, [
            _result(
                "runtime_config",
                "fail",
                f"Runtime configuration could not be loaded: {type(exc).__name__}: {exc}",
                "Check .env and NFL_CONFIG_PATH; a private config module must define `config`.",
            )
        ]

    diagnostics: list[Diagnostic] = []
    backend = str(getattr(config.database, "backend", "")).strip().lower()
    if backend not in {"sqlite", "mysql"}:
        diagnostics.append(
            _result(
                "database_config",
                "fail",
                f"Unsupported DB_BACKEND={backend!r}; expected `sqlite` or `mysql`.",
                "Set DB_BACKEND in .env or the deployment environment.",
            )
        )
    elif backend == "mysql":
        db_url = os.getenv("DB_URL", "") or str(getattr(config.database, "db_url", ""))
        if not db_url:
            diagnostics.append(
                _result(
                    "database_config",
                    "fail",
                    "DB_BACKEND=mysql requires DB_URL, but DB_URL is empty.",
                    "Set DB_URL=mysql+pymysql://user:pass@host:3306/database.",
                )
            )
        elif not re.match(r"^mysql(?:\+pymysql)?://", db_url):
            diagnostics.append(
                _result(
                    "database_config",
                    "fail",
                    "DB_URL must use the mysql:// or mysql+pymysql:// scheme.",
                    "Correct DB_URL without printing or committing its credentials.",
                )
            )
        else:
            diagnostics.append(
                _result("database_config", "pass", "MySQL configuration is present.")
            )
    else:
        raw_path = str(getattr(config.database, "path", "")).strip()
        if not raw_path:
            diagnostics.append(
                _result(
                    "database_config",
                    "fail",
                    "DB_BACKEND=sqlite requires a non-empty SQLITE_DB_PATH.",
                    "Set SQLITE_DB_PATH in .env.",
                )
            )
        else:
            diagnostics.append(
                _result("database_config", "pass", f"SQLite path: {Path(raw_path).expanduser()}")
            )
    return config, diagnostics


def check_database(config: Any, *, check_schema: bool) -> list[Diagnostic]:
    backend = str(getattr(config.database, "backend", "")).strip().lower()
    if backend not in {"sqlite", "mysql"}:
        return []

    if backend == "sqlite":
        database_path = Path(str(config.database.path)).expanduser()
        if not database_path.exists():
            return [
                _result(
                    "database",
                    "fail",
                    f"SQLite database does not exist: {database_path}",
                    "Run `make migrate` to create and migrate it.",
                )
            ]
        if not database_path.is_file():
            return [
                _result(
                    "database",
                    "fail",
                    f"SQLITE_DB_PATH is not a file: {database_path}",
                    "Set SQLITE_DB_PATH to a writable database file.",
                )
            ]

    try:
        from utils.db import get_connection, is_sqlite_connection, table_exists

        with get_connection() as connection:
            if is_sqlite_connection(connection):
                connection.execute("SELECT 1")
            else:
                cursor = connection.cursor()
                try:
                    cursor.execute("SELECT 1")
                finally:
                    cursor.close()
            missing = (
                [table for table in REQUIRED_API_TABLES if not table_exists(table, conn=connection)]
                if check_schema
                else []
            )
    except Exception as exc:
        return [
            _result(
                "database",
                "fail",
                f"Database connection failed: {type(exc).__name__}: {exc}",
                "Verify DB_BACKEND/DB_URL/SQLITE_DB_PATH and network permissions.",
            )
        ]

    diagnostics = [_result("database", "pass", f"Connected to {backend} database.")]
    if check_schema:
        if missing:
            diagnostics.append(
                _result(
                    "migrations",
                    "fail",
                    f"Required tables are missing: {', '.join(missing)}.",
                    "Run `make migrate` before starting services.",
                )
            )
        else:
            diagnostics.append(_result("migrations", "pass", "Required API tables are present."))
    return diagnostics


def check_odds_key(config: Any, *, required: bool) -> Diagnostic:
    configured = bool(str(getattr(config.api, "odds_api_key", "")).strip())
    if configured:
        return _result("odds_api_key", "pass", "ODDS_API_KEY is configured.")
    status = "fail" if required else "warn"
    return _result(
        "odds_api_key",
        status,
        "ODDS_API_KEY is not configured; live-odds pipeline runs will fail closed.",
        "Set ODDS_API_KEY in .env or the deployment secret store before running live odds.",
    )


def check_private_api(root: Path = PROJECT_ROOT) -> Diagnostic:
    api_file = root / PRIVATE_API_FILE
    if api_file.is_file():
        return _result("private_api", "pass", "Deployment-supplied API module is available.")
    return _result(
        "private_api",
        "fail",
        f"Private API module is unavailable: {PRIVATE_API_FILE}",
        "Install the deployment-supplied API module before starting API services.",
    )


def check_private_modules(root: Path = PROJECT_ROOT, *, required: bool) -> Diagnostic:
    missing = [relative for relative in PRIVATE_NFL_FILES if not (root / relative).is_file()]
    if not missing:
        return _result("private_modules", "pass", "Private NFL execution modules are available.")
    return _result(
        "private_modules",
        "fail" if required else "warn",
        "Private NFL execution modules are unavailable: " + ", ".join(missing),
        "Install the deployment-supplied private modules; API read-only features can run without them.",
    )


def evaluate_nfl_week_readiness(
    season: int,
    week: int,
    *,
    phase: str,
    counts: dict[str, int],
) -> list[Diagnostic]:
    """Evaluate the database evidence required before or after a weekly run."""
    if phase not in {"pre-run", "post-run"}:
        raise ValueError("phase must be `pre-run` or `post-run`")

    scope = f"season {season}, week {week}"
    diagnostics: list[Diagnostic] = []

    games = counts.get("games", 0)
    games_with_kickoff = counts.get("games_with_kickoff", 0)
    if games <= 0:
        diagnostics.append(
            _result(
                "nfl_schedule",
                "fail",
                f"No scheduled games are loaded for {scope}.",
                f"Run `make week-refresh SEASON={season} WEEK={week}` and verify the schedule source.",
            )
        )
    elif games_with_kickoff != games:
        diagnostics.append(
            _result(
                "nfl_schedule",
                "fail",
                f"Only {games_with_kickoff}/{games} games have kickoff timestamps for {scope}.",
                "Refresh the schedule; point-in-time validation requires every kickoff timestamp.",
            )
        )
    else:
        diagnostics.append(
            _result(
                "nfl_schedule",
                "pass",
                f"{games} scheduled games have kickoff timestamps for {scope}.",
            )
        )

    requirements = (
        (
            "nfl_roster",
            "roster_players",
            "roster players",
            f"Run `make week-refresh SEASON={season} WEEK={week}` to refresh the roster.",
        ),
        (
            "nfl_history",
            "history_rows",
            "historical player-stat rows",
            "Ingest prior completed weeks before generating projections.",
        ),
    )
    for name, key, label, action in requirements:
        count = counts.get(key, 0)
        diagnostics.append(
            _result(
                name,
                "pass" if count > 0 else "fail",
                f"{count} {label} are available for {scope}.",
                None if count > 0 else action,
            )
        )

    if phase == "pre-run":
        return diagnostics

    post_run_requirements = (
        ("nfl_context", "context_rows", "point-in-time player context rows"),
        ("nfl_projections", "projection_rows", "persisted projections"),
        ("nfl_live_odds", "odds_rows", "persisted live-odds rows"),
        ("nfl_worker_run", "completed_runs", "completed worker runs"),
        ("nfl_odds_validation", "valid_odds_runs", "valid odds-validation records"),
        ("nfl_artifacts", "artifact_rows", "registered run artifacts"),
    )
    for name, key, label in post_run_requirements:
        count = counts.get(key, 0)
        diagnostics.append(
            _result(
                name,
                "pass" if count > 0 else "fail",
                f"{count} {label} are available for {scope}.",
                None if count > 0 else "Inspect the failed pipeline run before publishing results.",
            )
        )

    decision_rows = counts.get("decision_rows", 0)
    diagnostics.append(
        _result(
            "nfl_decisions",
            "pass" if decision_rows > 0 else "warn",
            f"{decision_rows} agent decisions are persisted for {scope}.",
            (
                None
                if decision_rows > 0
                else "Confirm the run report records that no candidates reached agent review."
            ),
        )
    )

    card_rows = counts.get("card_rows", 0)
    if card_rows > 0:
        diagnostics.append(
            _result(
                "nfl_card", "pass", f"{card_rows} approved card rows are published for {scope}."
            )
        )
    else:
        diagnostics.append(
            _result(
                "nfl_card",
                "warn",
                f"No plays are published for {scope}; this can be a valid zero-play card.",
                "Confirm that rejection decisions and the run report were persisted.",
            )
        )
    return diagnostics


def collect_nfl_week_counts(season: int, week: int) -> dict[str, int]:
    """Collect portable SQLite/MySQL evidence for one NFL week."""
    from utils.db import fetchone

    def scalar(sql: str, params: tuple[Any, ...]) -> int:
        row = fetchone(sql, params=params)
        return int(row[0] or 0) if row else 0

    counts = {
        "games": scalar("SELECT COUNT(*) FROM games WHERE season = ? AND week = ?", (season, week)),
        "games_with_kickoff": scalar(
            """
            SELECT COUNT(*) FROM games
            WHERE season = ? AND week = ?
              AND kickoff_utc IS NOT NULL AND kickoff_utc <> ''
            """,
            (season, week),
        ),
        "roster_players": scalar(
            "SELECT COUNT(*) FROM nfl_roster_players WHERE season = ? AND roster_week = ?",
            (season, week),
        ),
        "history_rows": scalar(
            """
            SELECT COUNT(*) FROM player_stats_enhanced
            WHERE season < ? OR (season = ? AND week < ?)
            """,
            (season, season, week),
        ),
        "context_rows": scalar(
            "SELECT COUNT(*) FROM nfl_player_context_snapshots WHERE season = ? AND week = ?",
            (season, week),
        ),
        "projection_rows": scalar(
            "SELECT COUNT(*) FROM weekly_projections WHERE season = ? AND week = ?",
            (season, week),
        ),
        "odds_rows": scalar(
            "SELECT COUNT(*) FROM weekly_odds WHERE season = ? AND week = ?",
            (season, week),
        ),
        "decision_rows": scalar(
            "SELECT COUNT(*) FROM agent_decisions WHERE season = ? AND week = ?",
            (season, week),
        ),
        "completed_runs": scalar(
            """
            SELECT COUNT(*) FROM pipeline_runs
            WHERE season = ? AND week = ? AND status = 'completed'
            """,
            (season, week),
        ),
        "valid_odds_runs": scalar(
            """
            SELECT COUNT(*) FROM pipeline_odds_validations validations
            JOIN pipeline_runs runs ON runs.run_id = validations.run_id
            WHERE runs.season = ? AND runs.week = ?
              AND runs.status = 'completed' AND validations.valid = 1
            """,
            (season, week),
        ),
        "artifact_rows": scalar(
            """
            SELECT COUNT(*) FROM pipeline_artifacts artifacts
            JOIN pipeline_runs runs ON runs.run_id = artifacts.run_id
            WHERE runs.season = ? AND runs.week = ? AND runs.status = 'completed'
            """,
            (season, week),
        ),
        "card_rows": scalar(
            "SELECT COUNT(*) FROM materialized_value_view WHERE season = ? AND week = ?",
            (season, week),
        ),
    }
    return counts


def check_nfl_week_readiness(season: int, week: int, *, phase: str) -> list[Diagnostic]:
    """Load and evaluate weekly evidence without hiding schema/query failures."""
    try:
        counts = collect_nfl_week_counts(season, week)
    except Exception as exc:
        return [
            _result(
                "nfl_week_data",
                "fail",
                f"NFL week evidence could not be read: {type(exc).__name__}: {exc}",
                "Run migrations, then refresh the requested NFL week.",
            )
        ]
    return evaluate_nfl_week_readiness(season, week, phase=phase, counts=counts)


def _node_version() -> tuple[int, int, int] | None:
    node = shutil.which("node")
    if not node:
        return None
    completed = subprocess.run(
        [node, "--version"], check=False, capture_output=True, text=True, timeout=5
    )
    match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", completed.stdout.strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_frontend(root: Path = PROJECT_ROOT) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    version = _node_version()
    if version is None:
        diagnostics.append(
            _result(
                "node",
                "fail",
                "Node.js was not found or its version could not be read.",
                "Install Node.js 20.9+.",
            )
        )
    elif version < (20, 9, 0):
        diagnostics.append(
            _result(
                "node",
                "fail",
                f"Node.js {'.'.join(map(str, version))} is too old; Next.js requires 20.9+.",
                "Upgrade Node.js, then run `make frontend-install`.",
            )
        )
    else:
        diagnostics.append(_result("node", "pass", f"Node.js {'.'.join(map(str, version))}"))

    if shutil.which("npm") is None:
        diagnostics.append(
            _result("npm", "fail", "npm was not found.", "Install npm with Node.js 20.9+.")
        )
    else:
        diagnostics.append(_result("npm", "pass", "npm is available."))

    if not (root / "frontend" / "node_modules" / "next" / "package.json").is_file():
        diagnostics.append(
            _result(
                "frontend_dependencies",
                "fail",
                "Frontend dependencies are not installed.",
                "Run `make frontend-install`.",
            )
        )
    else:
        diagnostics.append(
            _result("frontend_dependencies", "pass", "Frontend dependencies are installed.")
        )
    return diagnostics


def collect_diagnostics(
    *,
    check_schema: bool = False,
    check_frontend_dependencies: bool = False,
    require_live_odds: bool = False,
    require_private_modules: bool = False,
    season: int | None = None,
    week: int | None = None,
    season_phase: str = "pre-run",
) -> list[Diagnostic]:
    diagnostics = [check_python()]
    config, config_diagnostics = check_runtime_config()
    diagnostics.extend(config_diagnostics)
    if config is not None and not any(item.failed for item in config_diagnostics):
        database_diagnostics = check_database(config, check_schema=check_schema)
        diagnostics.extend(database_diagnostics)
        diagnostics.append(check_odds_key(config, required=require_live_odds))
        if (
            season is not None
            and week is not None
            and not any(item.failed for item in database_diagnostics)
        ):
            diagnostics.extend(check_nfl_week_readiness(season, week, phase=season_phase))
    diagnostics.append(check_private_api())
    diagnostics.append(check_private_modules(required=require_private_modules))
    if check_frontend_dependencies:
        diagnostics.extend(check_frontend())
    return diagnostics


def print_diagnostics(diagnostics: Iterable[Diagnostic], *, as_json: bool = False) -> None:
    items = list(diagnostics)
    if as_json:
        print(
            json.dumps(
                {
                    "ok": not any(item.failed for item in items),
                    "checks": [asdict(item) for item in items],
                }
            )
        )
        return

    labels = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    for item in items:
        print(f"[{labels[item.status]}] {item.name}: {item.message}")
        if item.action:
            print(f"       Action: {item.action}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-schema", action="store_true", help="Require migrated API tables")
    parser.add_argument(
        "--check-frontend", action="store_true", help="Check Node/npm/frontend install"
    )
    parser.add_argument("--season", type=int, help="NFL season for weekly readiness checks")
    parser.add_argument("--week", type=int, help="NFL week for weekly readiness checks")
    parser.add_argument(
        "--season-phase",
        choices=("pre-run", "post-run"),
        default="pre-run",
        help="Require inputs before a run or persisted evidence after a run",
    )
    parser.add_argument(
        "--require-live-odds", action="store_true", help="Fail if ODDS_API_KEY is empty"
    )
    parser.add_argument(
        "--require-private-modules",
        action="store_true",
        help="Fail if deployment-supplied NFL modules are missing",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable diagnostics")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.season is None) != (args.week is None):
        raise SystemExit("--season and --week must be supplied together")
    configure_logging("preflight")
    diagnostics = collect_diagnostics(
        check_schema=args.check_schema,
        check_frontend_dependencies=args.check_frontend,
        require_live_odds=args.require_live_odds,
        require_private_modules=args.require_private_modules,
        season=args.season,
        week=args.week,
        season_phase=args.season_phase,
    )
    print_diagnostics(diagnostics, as_json=args.json)
    return 1 if any(item.failed for item in diagnostics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
