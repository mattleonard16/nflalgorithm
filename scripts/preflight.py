"""Validate local or deployed runtime configuration before services start."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sqlite3
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
API_SERVER_FILE = "api/server.py"
API_VISIBILITY_CONTRACT = "publication-safe-v1"
PRIVATE_NFL_FILES = (
    "data_pipeline.py",
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


def check_sqlite_wal(config: Any) -> list[Diagnostic]:
    """Confirm the SQLite database is in WAL mode.

    Journal mode is persistent, set once by MigrationManager. A database left in
    the `delete` default serializes readers against the writer, so a dashboard
    read blocks for the length of a materialization.
    """
    backend = str(getattr(config.database, "backend", "")).strip().lower()
    if backend != "sqlite":
        return []
    database_path = Path(str(config.database.path)).expanduser()
    if not database_path.is_file():
        return []

    try:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
            journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    except Exception as exc:
        return [
            _result(
                "sqlite_wal",
                "fail",
                f"Could not read journal_mode from {database_path}: {type(exc).__name__}: {exc}",
                "Verify SQLITE_DB_PATH points at a readable database file.",
            )
        ]

    if journal_mode == "wal":
        return [_result("sqlite_wal", "pass", "SQLite journal_mode is WAL.")]
    return [
        _result(
            "sqlite_wal",
            "fail",
            f"SQLite journal_mode is {journal_mode!r}; concurrent reads will block on writes.",
            "Run `make migrate` to set WAL mode.",
        )
    ]


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


def check_demo_mode(config: Any, *, production: bool) -> Diagnostic:
    enabled = bool(getattr(config.api, "demo_mode", False))
    if not enabled:
        return _result("demo_mode", "pass", "DEMO_MODE is disabled.")
    return _result(
        "demo_mode",
        "fail" if production else "warn",
        "DEMO_MODE exposes fixture and unpublished value rows.",
        "Set DEMO_MODE=false before running production checks.",
    )


def check_allowed_origins(*, production: bool) -> Diagnostic:
    """Confirm CORS is configured for a real frontend origin.

    api/server.py falls back to localhost origins when ALLOWED_ORIGINS is
    unset, which silently breaks every browser call from the deployed frontend.
    """
    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    action = (
        "Set ALLOWED_ORIGINS to a comma-separated list of deployed frontend origins, "
        'e.g. ALLOWED_ORIGINS="https://app.example.com".'
    )
    if not origins:
        return _result(
            "allowed_origins",
            "fail" if production else "warn",
            "ALLOWED_ORIGINS is unset; CORS will only allow localhost dev origins.",
            action,
        )
    localhost_only = all(
        "localhost" in origin or "127.0.0.1" in origin for origin in origins
    )
    if localhost_only:
        return _result(
            "allowed_origins",
            "fail" if production else "warn",
            "ALLOWED_ORIGINS contains only localhost origins; deployed browsers will be blocked.",
            action,
        )
    return _result(
        "allowed_origins", "pass", f"ALLOWED_ORIGINS configures {len(origins)} origin(s)."
    )


def check_api_server(root: Path = PROJECT_ROOT) -> Diagnostic:
    """Validate that api/server.py is present and applies the current visibility contract.

    The file is tracked, so a missing copy means an incomplete checkout rather than a
    public clone. A copy that parses but declares a stale contract is worse than a
    missing one: it serves legacy unjoinable and SimBook rows as if they were real.
    """
    api_file = root / API_SERVER_FILE
    if not api_file.is_file():
        return _result(
            "api_server",
            "fail",
            f"{API_SERVER_FILE} is missing.",
            "The file is tracked in git. Check `git status` for a deletion, "
            "or re-clone the repository.",
        )

    try:
        module = ast.parse(api_file.read_text(encoding="utf-8"), filename=str(api_file))
    except (OSError, SyntaxError) as exc:
        return _result(
            "api_server",
            "fail",
            f"{API_SERVER_FILE} cannot be validated: {type(exc).__name__}: {exc}",
            "Restore the file from git.",
        )

    contract = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PUBLIC_VALUE_VISIBILITY_CONTRACT"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                contract = node.value.value
            break
    if contract != API_VISIBILITY_CONTRACT:
        return _result(
            "api_server",
            "fail",
            f"{API_SERVER_FILE} does not implement the current public visibility contract.",
            f"Set PUBLIC_VALUE_VISIBILITY_CONTRACT={API_VISIBILITY_CONTRACT!r} in {API_SERVER_FILE}.",
        )
    return _result(
        "api_server",
        "pass",
        f"{API_SERVER_FILE} implements {API_VISIBILITY_CONTRACT}.",
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

    empty_team_rows = counts.get("history_empty_team_rows", 0)
    if empty_team_rows > 0:
        diagnostics.append(
            _result(
                "nfl_history_team_scope",
                "fail",
                f"{empty_team_rows} historical player-stat rows have an empty team for {scope}.",
                "Re-ingest prior seasons through week 22 so LA/Rams rows canonicalize to LAR.",
            )
        )
    else:
        diagnostics.append(
            _result(
                "nfl_history_team_scope",
                "pass",
                f"Historical player-stat rows are team-scoped for {scope}.",
            )
        )

    incomplete_franchise_seasons = counts.get("history_incomplete_franchise_seasons", 0)
    if incomplete_franchise_seasons > 0:
        diagnostics.append(
            _result(
                "nfl_history_franchises",
                "fail",
                (
                    f"{incomplete_franchise_seasons} prior season(s) are missing a full "
                    f"32-team skill-stat slate for {scope}."
                ),
                "Re-ingest 2024 and 2025; empty-team Rams rows previously hid an entire club.",
            )
        )
    else:
        diagnostics.append(
            _result(
                "nfl_history_franchises",
                "pass",
                f"Prior seasons include all 32 franchises for {scope}.",
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
    from utils.db import get_connection

    with get_connection() as connection:
        return _collect_nfl_week_counts(connection, season, week)


def _collect_nfl_week_counts(connection: Any, season: int, week: int) -> dict[str, int]:
    from utils.db import fetchone

    def scalar(sql: str, params: tuple[Any, ...]) -> int:
        row = fetchone(sql, params=params, conn=connection)
        return int(row[0] or 0) if row else 0

    latest_run = fetchone(
        """
        SELECT run_id FROM pipeline_runs
        WHERE season = ? AND week = ? AND status = 'completed'
        ORDER BY finished_at DESC, run_id DESC LIMIT 1
        """,
        params=(season, week),
        conn=connection,
    )
    latest_run_id = str(latest_run[0]) if latest_run else None

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
        "history_empty_team_rows": scalar(
            """
            SELECT COUNT(*) FROM player_stats_enhanced
            WHERE (season < ? OR (season = ? AND week < ?))
              AND (team IS NULL OR TRIM(team) = '')
            """,
            (season, season, week),
        ),
        "history_incomplete_franchise_seasons": scalar(
            """
            SELECT COUNT(*) FROM (
                SELECT season
                FROM player_stats_enhanced
                WHERE season < ?
                GROUP BY season
                HAVING COUNT(DISTINCT CASE
                    WHEN team IS NOT NULL AND TRIM(team) <> '' THEN team
                END) < 32
            ) incomplete
            """,
            (season,),
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
        "completed_runs": 1 if latest_run_id else 0,
        "valid_odds_runs": scalar(
            """
            SELECT COUNT(*) FROM pipeline_odds_validations
            WHERE run_id = ? AND valid = 1
            """,
            (latest_run_id,),
        ),
        "artifact_rows": scalar(
            "SELECT COUNT(*) FROM pipeline_artifacts WHERE run_id = ?",
            (latest_run_id,),
        ),
        "card_rows": scalar(
            """
            SELECT COUNT(*) FROM materialized_value_view
            WHERE season = ? AND week = ? AND published_run_id = ?
            """,
            (season, week, latest_run_id),
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
    require_demo_mode_off: bool = False,
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
        if not any(item.failed for item in database_diagnostics):
            diagnostics.extend(check_sqlite_wal(config))
        diagnostics.append(check_odds_key(config, required=require_live_odds))
        diagnostics.append(check_demo_mode(config, production=require_demo_mode_off))
        diagnostics.append(check_allowed_origins(production=require_demo_mode_off))
        if (
            season is not None
            and week is not None
            and not any(item.failed for item in database_diagnostics)
        ):
            diagnostics.extend(check_nfl_week_readiness(season, week, phase=season_phase))
    diagnostics.append(check_api_server())
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
    parser.add_argument(
        "--require-demo-mode-off",
        action="store_true",
        help="Fail if fixture visibility is enabled",
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
        require_demo_mode_off=args.require_demo_mode_off,
        season=args.season,
        week=args.week,
        season_phase=args.season_phase,
    )
    print_diagnostics(diagnostics, as_json=args.json)
    return 1 if any(item.failed for item in diagnostics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
