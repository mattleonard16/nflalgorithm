#!/usr/bin/env python3
"""Run the Wednesday NFL job step by step, record how it went, and alert on failure.

launchd runs ``make week-auto`` with nobody watching, and before this runner a
failed week was found only by reading the log days later. Every run now writes
``logs/week_auto_status.json``; a failed run also posts a macOS notification.

Grading and the research memo describe the previous week, so their failures
are warnings: new lines still publish. Any other failed step stops the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

logger = logging.getLogger(__name__)

STATUS_PATH = Path("logs/week_auto_status.json")

RunStep = Callable[[list[str], dict[str, str]], subprocess.CompletedProcess]
Notify = Callable[[str], None]


class Step(NamedTuple):
    target: str
    season: int
    week: int
    fatal: bool
    extra_env: dict[str, str]


def plan_steps(season: int, week: int) -> list[Step]:
    """Return the Wednesday steps in order for the upcoming ``season``/``week``."""
    steps = [
        Step("db-analyze", season, week, True, {}),
        Step("week-predict", season, week, True, {"NFL_FEATURE_CONTEXT_FACTORS": "1"}),
        Step("week-lines", season, week, True, {}),
    ]
    if week > 1:
        steps += [
            Step("week-grade", season, week - 1, False, {}),
            Step("week-research", season, week - 1, False, {}),
        ]
    return steps


def _run_make(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env, check=False)


def _notify_macos(message: str) -> None:
    script = f'display notification {json.dumps(message)} with title "NFL week-auto"'
    subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=30)


def _write_status(path: Path, status: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_week(
    season: int,
    week: int,
    *,
    run_step: RunStep = _run_make,
    notify: Notify = _notify_macos,
    status_path: Path = STATUS_PATH,
) -> int:
    """Run every step for ``season``/``week`` and return the job's exit code."""
    make = os.environ.get("MAKE", "make")
    status: dict[str, object] = {
        "season": season,
        "week": week,
        "started_at": _now(),
        "finished_at": None,
        "ok": False,
        "failed_step": None,
        "exit_code": 0,
        "warnings": [],
    }
    warnings: list[str] = []
    print(f"=== week-auto {season} W{week} started {status['started_at']} ===", flush=True)

    for step in plan_steps(season, week):
        argv = [make, step.target]
        if step.target != "db-analyze":
            argv += [f"SEASON={step.season}", f"WEEK={step.week}"]
        returncode = run_step(argv, {**os.environ, **step.extra_env}).returncode
        if returncode == 0:
            continue
        if not step.fatal:
            warning = f"{step.target} {step.season} W{step.week} exited {returncode}"
            logger.warning("%s; lines were still published", warning)
            warnings.append(warning)
            continue
        status.update(failed_step=step.target, exit_code=returncode)
        break

    status.update(ok=status["failed_step"] is None, warnings=warnings)
    _finish(status, status_path, notify)
    return int(status["exit_code"])


def _finish(status: dict[str, object], status_path: Path, notify: Notify) -> None:
    status["finished_at"] = _now()
    _write_status(status_path, status)
    print(f"=== week-auto finished {status['finished_at']} ===", flush=True)
    if status["ok"]:
        return
    message = (
        f"{status['season']} W{status['week']} failed at {status['failed_step']} "
        f"(exit {status['exit_code']})"
    )
    try:
        notify(message)
    except Exception as exc:  # The alert is best effort; the status file is the record.
        logger.warning("Could not post the failure notification: %s", exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from utils.current_week import resolve_current_week

    try:
        season, week = resolve_current_week()
    except Exception:
        logger.exception("Could not resolve the upcoming NFL week")
        status: dict[str, object] = {
            "season": None,
            "week": None,
            "started_at": _now(),
            "ok": False,
            "failed_step": "resolve-week",
            "exit_code": 1,
            "warnings": [],
        }
        _finish(status, STATUS_PATH, _notify_macos)
        return 1
    print(f"Resolved upcoming week: {season} W{week}", flush=True)
    return run_week(season, week, notify=_notify_macos, status_path=STATUS_PATH)


if __name__ == "__main__":
    sys.exit(main())
