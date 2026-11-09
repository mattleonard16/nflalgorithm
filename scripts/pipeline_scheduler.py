"""Scheduler for the canonical NFL pregame production workflow."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import config
from pipeline_jobs.service import QUEUED_STATUSES, JobService
from utils.current_week import SeasonOverError, resolve_current_week
from utils.db import fetchone

config.logs_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.logs_dir / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = (*QUEUED_STATUSES, "running")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _has_active_job(season: int, week: int) -> bool:
    """True when a queued, retry-scheduled or running job exists for the season/week."""
    marks = ", ".join("?" for _ in _ACTIVE_STATUSES)
    row = fetchone(
        f"""
        SELECT 1 FROM pipeline_jobs j
        JOIN pipeline_runs r ON r.run_id = j.run_id
        WHERE r.season = ? AND r.week = ? AND j.status IN ({marks})
        LIMIT 1
        """,
        (season, week, *_ACTIVE_STATUSES),
    )
    return row is not None


class PipelineScheduler:
    """Periodically run the same validated pregame path used by the CLI/API."""

    def __init__(self) -> None:
        self.scheduler = BlockingScheduler()
        self.setup_jobs()

    def setup_jobs(self) -> None:
        minutes = config.pipeline.update_interval_minutes
        if minutes <= 0:
            # The run key divides time by the interval; zero or less has no window.
            raise ValueError(f"NFL_PIPELINE_INTERVAL_MINUTES must be positive, got {minutes}")
        self.scheduler.add_job(
            func=self.run_pregame_pipeline,
            trigger=IntervalTrigger(minutes=minutes),
            id="canonical_pregame_refresh",
            name="Canonical NFL Pregame Refresh",
            replace_existing=True,
        )
        logger.info("Canonical pregame refresh scheduled")

    def run_pregame_pipeline(self) -> dict[str, Any]:
        """Resolve the active NFL week and enqueue the fail-closed production path."""
        try:
            season, week = resolve_current_week()
        except SeasonOverError as exc:
            # Only a fully played schedule is the offseason. A missing or
            # unreadable schedule raises, so it cannot pass as a quiet skip.
            logger.info("Skipping pregame run, no upcoming NFL week: %s", exc)
            return {"success": True, "status": "skipped", "reason": "offseason"}
        if _has_active_job(season, week):
            logger.info("Skipping pregame run, a job is already active for %s week %s", season, week)
            return {"success": True, "status": "skipped", "reason": "active_job"}
        # One key per interval window, so repeated ticks in a window dedupe to one job.
        window_seconds = config.pipeline.update_interval_minutes * 60
        bucket = int(_utcnow().timestamp()) // window_seconds
        job = JobService().create_pipeline_job(
            season=season,
            week=week,
            source="scheduler",
            idempotency_key=f"scheduler:nfl:{season}:{week}:{bucket}",
        )
        logger.info("Queued canonical pregame run %s for %s week %s", job.run_id, season, week)
        return {"success": True, "status": job.status, "run_id": job.run_id, "job_id": job.job_id}

    def start(self) -> None:
        logger.info("Starting NFL pregame pipeline scheduler")
        for job in self.scheduler.get_jobs():
            logger.info("Scheduled %s: %s", job.name, job.trigger)
        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown()


if __name__ == "__main__":
    PipelineScheduler().start()
