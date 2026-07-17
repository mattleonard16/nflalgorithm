"""Dedicated NFL pipeline worker for durable jobs."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from pipeline_jobs.service import JobService, LeaseLostError, PipelineJob
from scripts.production_runner import run_production_pipeline

logger = logging.getLogger(__name__)

PipelineRunner = Callable[..., Mapping[str, Any]]
LeaseLossHandler = Callable[[str], None]
LEASE_LOST_EXIT_CODE = 75


def _terminate_process_for_lease_loss(reason: str) -> None:
    """Hard-stop production execution when cooperative cancellation is impossible."""
    logger.critical("Terminating worker after lease loss: %s", reason)
    os._exit(LEASE_LOST_EXIT_CODE)


def _automatic_retry_is_safe(report: Mapping[str, Any]) -> bool:
    """Require every executed stage to declare retry-idempotent side effects."""
    stages = report.get("stages")
    if not isinstance(stages, list):
        return False
    executed = [
        stage
        for stage in stages
        if isinstance(stage, Mapping) and stage.get("status") in {"ok", "error"}
    ]
    return bool(executed) and all(stage.get("retry_safe") is True for stage in executed)


class PipelineWorker:
    """Claim one durable job at a time and execute it outside the API process."""

    def __init__(
        self,
        *,
        worker_id: str | None = None,
        service: JobService | None = None,
        runner: PipelineRunner = run_production_pipeline,
        heartbeat_seconds: float = 15.0,
        lease_loss_handler: LeaseLossHandler | None = None,
    ) -> None:
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.service = service or JobService()
        self.runner = runner
        self.heartbeat_seconds = max(0.01, heartbeat_seconds)
        self.lease_loss_handler = lease_loss_handler

    def process_once(self) -> bool:
        job = self.service.claim_next(self.worker_id)
        if not job:
            return False
        self._execute(job)
        return True

    def _execute(self, job: PipelineJob) -> None:
        payload = job.payload
        heartbeat_stop = threading.Event()
        lease_lost = threading.Event()
        terminal_written = threading.Event()
        lease_operation_lock = threading.Lock()

        def mark_lease_lost(reason: str) -> None:
            if terminal_written.is_set():
                return
            first_loss = not lease_lost.is_set()
            if first_loss:
                logger.error("Pipeline job %s lost its lease: %s", job.job_id, reason)
            lease_lost.set()
            if first_loss and self.lease_loss_handler:
                self.lease_loss_handler(reason)

        def ensure_lease() -> None:
            if lease_lost.is_set():
                raise LeaseLostError("worker attempt lease was lost")

        def on_stage_start(stage_name: str, ordinal: int) -> None:
            ensure_lease()
            self.service.record_stage_started(
                job,
                stage_name,
                ordinal,
            )

        def on_stage_result(stage_name: str, ordinal: int, result: Mapping[str, Any]) -> None:
            ensure_lease()
            self.service.record_stage_result(
                job,
                stage_name,
                ordinal,
                result,
            )

        def execution_should_stop() -> bool:
            if lease_lost.is_set():
                return True
            try:
                return self.service.cancellation_requested(job)
            except LeaseLostError as exc:
                mark_lease_lost(str(exc))
                return True

        def write_terminal(action: Callable[[], Any], *, renew_first: bool = False) -> Any:
            """Serialize the last renewal and terminal write against the heartbeat loop."""
            with lease_operation_lock:
                ensure_lease()
                if renew_first:
                    try:
                        renewed = self.service.heartbeat(job)
                    except Exception as exc:
                        mark_lease_lost(f"final heartbeat raised {type(exc).__name__}: {exc}")
                        raise LeaseLostError("final heartbeat failed") from exc
                    if not renewed:
                        mark_lease_lost("final lease renewal updated zero rows")
                        raise LeaseLostError("final heartbeat lost the worker attempt lease")
                result = action()
                terminal_written.set()
                return result

        def renew_lease() -> None:
            while not heartbeat_stop.wait(self.heartbeat_seconds):
                with lease_operation_lock:
                    if terminal_written.is_set():
                        return
                    try:
                        renewed = self.service.heartbeat(job)
                    except Exception as exc:
                        mark_lease_lost(f"heartbeat raised {type(exc).__name__}: {exc}")
                        return
                    if not renewed:
                        mark_lease_lost("heartbeat renewal updated zero rows")
                        return

        heartbeat_thread = threading.Thread(target=renew_lease, daemon=True)
        heartbeat_thread.start()
        try:
            try:
                report = dict(
                    self.runner(
                        int(payload["season"]),
                        int(payload["week"]),
                        skip_ingest=bool(payload.get("skip_ingest", False)),
                        skip_odds=bool(payload.get("skip_odds", False)),
                        on_stage_start=on_stage_start,
                        on_stage_result=on_stage_result,
                        cancellation_requested=execution_should_stop,
                        run_id=job.run_id,
                        attempt=job.attempts,
                    )
                )
            except LeaseLostError as exc:
                mark_lease_lost(str(exc))
                return
            except Exception as exc:
                logger.exception("Pipeline job %s raised unexpectedly", job.job_id)
                if lease_lost.is_set():
                    return
                try:
                    # An unhandled runner exception may follow an external side
                    # effect whose acknowledgement was lost. Without an explicit
                    # stage result proving retry safety, fail closed.
                    write_terminal(lambda: self.service.fail(job, str(exc), retryable=False))
                except LeaseLostError as lease_exc:
                    mark_lease_lost(str(lease_exc))
                return

            if lease_lost.is_set():
                return
            try:
                cancellation_requested = self.service.cancellation_requested(job)
            except LeaseLostError as exc:
                mark_lease_lost(str(exc))
                return
            if cancellation_requested or report.get("cancelled"):
                try:
                    write_terminal(lambda: self.service.cancel_running(job, report))
                except LeaseLostError as exc:
                    mark_lease_lost(str(exc))
                return

            if report.get("success"):
                health = None
                try:
                    from api.data_health import run_all_checks

                    health = run_all_checks(int(payload["season"]), int(payload["week"]))
                except Exception as exc:
                    logger.warning("Post-run health checks failed for %s: %s", job.run_id, exc)
                if lease_lost.is_set():
                    return
                try:
                    artifact_uri = report.get("artifact_uri")
                    artifact = (
                        {
                            "kind": "run_report",
                            "uri": str(artifact_uri),
                            "metadata": {"season": payload["season"], "week": payload["week"]},
                        }
                        if artifact_uri
                        else None
                    )
                    completed = write_terminal(
                        lambda: self.service.complete(
                            job,
                            report,
                            data_health=health,
                            artifact=artifact,
                        ),
                        renew_first=True,
                    )
                except LeaseLostError as exc:
                    mark_lease_lost(str(exc))
                    return
                if not completed:
                    logger.warning("Discarding completion from stale worker for %s", job.run_id)
                    return
                try:
                    from api.cache import value_bets_cache

                    value_bets_cache.invalidate_all()
                except Exception as exc:
                    logger.warning("Value cache invalidation failed for %s: %s", job.run_id, exc)
                return

            errors = report.get("errors") or []
            detail = "; ".join(str(item.get("error", "pipeline failed")) for item in errors)
            automatic_retry_safe = _automatic_retry_is_safe(report)
            report["automatic_retry_safe"] = automatic_retry_safe
            try:
                write_terminal(
                    lambda: self.service.fail(
                        job,
                        detail
                        or (
                            "pipeline stopped before final card"
                            if report.get("incomplete")
                            else "pipeline failed"
                        ),
                        report,
                        retryable=not bool(report.get("incomplete"))
                        and automatic_retry_safe,
                    )
                )
            except LeaseLostError as exc:
                mark_lease_lost(str(exc))
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=self.heartbeat_seconds + 1)

    def run_forever(self, *, poll_seconds: float = 2.0) -> None:
        logger.info("Pipeline worker %s started", self.worker_id)
        while True:
            processed = self.process_once()
            if not processed:
                time.sleep(max(0.1, poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL durable pipeline worker")
    parser.add_argument("--once", action="store_true", help="Process at most one job")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--worker-id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    worker = PipelineWorker(
        worker_id=args.worker_id,
        lease_loss_handler=_terminate_process_for_lease_loss,
    )
    if args.once:
        worker.process_once()
    else:
        worker.run_forever(poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
