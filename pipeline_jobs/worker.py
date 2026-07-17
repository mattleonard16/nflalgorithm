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

from pipeline_jobs.service import JobService, PipelineJob
from scripts.production_runner import run_production_pipeline

logger = logging.getLogger(__name__)

PipelineRunner = Callable[..., Mapping[str, Any]]


class PipelineWorker:
    """Claim one durable job at a time and execute it outside the API process."""

    def __init__(
        self,
        *,
        worker_id: str | None = None,
        service: JobService | None = None,
        runner: PipelineRunner = run_production_pipeline,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.service = service or JobService()
        self.runner = runner
        self.heartbeat_seconds = max(1.0, heartbeat_seconds)

    def process_once(self) -> bool:
        job = self.service.claim_next(self.worker_id)
        if not job:
            return False
        self._execute(job)
        return True

    def _execute(self, job: PipelineJob) -> None:
        payload = job.payload

        def on_stage_start(stage_name: str, ordinal: int) -> None:
            self.service.record_stage_started(
                job.run_id,
                stage_name,
                ordinal,
                job_id=job.job_id,
                worker_id=self.worker_id,
            )

        def on_stage_result(stage_name: str, ordinal: int, result: Mapping[str, Any]) -> None:
            self.service.record_stage_result(
                job.run_id,
                stage_name,
                ordinal,
                result,
                job_id=job.job_id,
                worker_id=self.worker_id,
            )

        heartbeat_stop = threading.Event()

        def renew_lease() -> None:
            while not heartbeat_stop.wait(self.heartbeat_seconds):
                self.service.heartbeat(job.job_id, self.worker_id)

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
                        cancellation_requested=lambda: self.service.cancellation_requested(
                            job.job_id
                        ),
                    )
                )
            except Exception as exc:
                logger.exception("Pipeline job %s raised unexpectedly", job.job_id)
                self.service.fail(job, str(exc))
                return
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=self.heartbeat_seconds + 1)

        if self.service.cancellation_requested(job.job_id) or report.get("cancelled"):
            self.service.cancel_running(job, report)
            return

        if report.get("success"):
            health = None
            try:
                from api.data_health import run_all_checks

                health = run_all_checks(int(payload["season"]), int(payload["week"]))
            except Exception as exc:
                logger.warning("Post-run health checks failed for %s: %s", job.run_id, exc)
            if not self.service.complete(job, report, data_health=health):
                logger.warning("Discarding completion from stale worker for %s", job.run_id)
                return
            try:
                from api.cache import value_bets_cache

                value_bets_cache.invalidate_all()
            except Exception as exc:
                logger.warning("Value cache invalidation failed for %s: %s", job.run_id, exc)
            artifact_uri = report.get("artifact_uri")
            if artifact_uri:
                self.service.register_artifact(
                    run_id=job.run_id,
                    kind="run_report",
                    uri=str(artifact_uri),
                    metadata={"season": payload["season"], "week": payload["week"]},
                )
            return

        errors = report.get("errors") or []
        detail = "; ".join(str(item.get("error", "pipeline failed")) for item in errors)
        self.service.fail(
            job,
            detail
            or (
                "pipeline stopped before final card"
                if report.get("incomplete")
                else "pipeline failed"
            ),
            report,
            retryable=not bool(report.get("incomplete")),
        )

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

    worker = PipelineWorker(worker_id=args.worker_id)
    if args.once:
        worker.process_once()
    else:
        worker.run_forever(poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
