"""Start the API, durable worker, and frontend with readiness supervision."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import config
from scripts.preflight import collect_diagnostics, print_diagnostics
from scripts.run_services import _terminate
from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ServiceCommand:
    name: str
    command: tuple[str, ...]
    cwd: Path = PROJECT_ROOT


def local_service_commands() -> list[ServiceCommand]:
    """Return local child commands without starting them."""
    python = sys.executable
    host = str(getattr(config.api, "host", "0.0.0.0"))
    port = str(getattr(config.api, "port", 8000))
    frontend_port = os.getenv("FRONTEND_PORT", "3000")
    return [
        ServiceCommand("worker", (python, "-m", "pipeline_jobs.worker")),
        ServiceCommand(
            "api",
            (
                python,
                "-m",
                "uvicorn",
                "api.application:app",
                "--host",
                host,
                "--port",
                port,
                "--reload",
            ),
        ),
        ServiceCommand(
            "frontend",
            ("npm", "run", "dev", "--", "--port", frontend_port),
            PROJECT_ROOT / "frontend",
        ),
    ]


def port_is_available(port: int, host: str = "127.0.0.1") -> bool:
    """Return whether a TCP port can be bound locally."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def wait_for_readiness(url: str, timeout_seconds: float = 30.0) -> None:
    """Wait for an HTTP 2xx readiness response or raise a clear timeout."""
    deadline = time.monotonic() + timeout_seconds
    last_error = "service did not respond"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"API readiness timed out after {timeout_seconds:.0f}s: {last_error}")


def main() -> int:
    configure_logging("local-services")
    diagnostics = collect_diagnostics(check_schema=True, check_frontend_dependencies=True)
    print_diagnostics(diagnostics)
    if any(item.failed for item in diagnostics):
        logger.error(
            "Local startup preflight failed",
            extra={"event": "startup.preflight_failed"},
        )
        return 2

    api_port = int(getattr(config.api, "port", 8000))
    frontend_port = int(os.getenv("FRONTEND_PORT", "3000"))
    occupied = [port for port in (api_port, frontend_port) if not port_is_available(port)]
    if occupied:
        logger.error(
            "Required local ports are already in use: %s",
            ", ".join(map(str, occupied)),
            extra={"event": "startup.port_conflict"},
        )
        return 2

    commands = local_service_commands()
    processes: list[subprocess.Popen[bytes]] = []

    def stop(_signum: int, _frame: object) -> None:
        logger.info("Stopping local services", extra={"event": "startup.stop"})
        _terminate(processes)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        for spec in commands[:2]:
            logger.info(
                "Starting %s: %s",
                spec.name,
                " ".join(spec.command),
                extra={"event": "startup.child_start"},
            )
            processes.append(subprocess.Popen(spec.command, cwd=spec.cwd))

        readiness_url = f"http://127.0.0.1:{api_port}/readyz"
        wait_for_readiness(readiness_url)
        logger.info(
            "API ready at %s",
            readiness_url,
            extra={"event": "startup.ready"},
        )

        frontend = commands[2]
        processes.append(subprocess.Popen(frontend.command, cwd=frontend.cwd))
        logger.info(
            "Full stack ready: frontend=http://localhost:%s api=http://localhost:%s",
            frontend_port,
            api_port,
            extra={"event": "startup.complete"},
        )

        while True:
            for spec, process in zip(commands, processes):
                return_code = process.poll()
                if return_code is not None:
                    logger.error(
                        "%s exited with status %s; stopping remaining services",
                        spec.name,
                        return_code,
                        extra={"event": "startup.child_exit"},
                    )
                    return return_code or 1
            time.sleep(0.5)
    except RuntimeError as exc:
        logger.error("%s", exc, extra={"event": "startup.readiness_failed"})
        return 2
    finally:
        _terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
