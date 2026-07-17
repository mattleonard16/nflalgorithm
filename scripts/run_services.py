"""Run the API and durable worker as separate supervised processes."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence

from config import config
from config.runtime import env_flag
from schema_migrations import MigrationManager
from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def service_commands() -> list[list[str]]:
    """Return production child-process commands from environment configuration."""
    python = sys.executable
    commands = [
        [
            python,
            "-m",
            "uvicorn",
            "api.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            os.getenv("PORT", os.getenv("API_PORT", "8000")),
        ],
        [python, "-m", "pipeline_jobs.worker"],
    ]
    if env_flag("ENABLE_PIPELINE_SCHEDULER"):
        commands.append([python, "-m", "scripts.pipeline_scheduler"])
    if env_flag("ENABLE_QUEUE_MONITOR"):
        commands.append([python, "-m", "scripts.queue_monitor"])
    return commands


def _terminate(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> None:
    configure_logging("service-supervisor")
    processes: list[subprocess.Popen[bytes]] = []

    logger.info(
        "Applying database migrations before service startup",
        extra={"event": "startup.migrations"},
    )
    MigrationManager(config.database.path).run()

    def stop(_signum: int, _frame: object) -> None:
        _terminate(processes)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        for command in service_commands():
            logger.info(
                "Starting service process: %s",
                " ".join(command[2:]),
                extra={"event": "startup.child_start"},
            )
            processes.append(subprocess.Popen(command))

        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    raise SystemExit(return_code or 1)
            time.sleep(1)
    finally:
        _terminate(processes)


if __name__ == "__main__":
    main()
