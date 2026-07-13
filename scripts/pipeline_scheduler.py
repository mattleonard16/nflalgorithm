"""Scheduler for the canonical NFL pregame production workflow."""

from __future__ import annotations

import logging
from typing import Any

import nflreadpy as nfl
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import config
from scripts.production_runner import run_production_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.logs_dir / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Periodically run the same validated pregame path used by the CLI/API."""

    def __init__(self) -> None:
        self.scheduler = BlockingScheduler()
        self.setup_jobs()

    def setup_jobs(self) -> None:
        self.scheduler.add_job(
            func=self.run_pregame_pipeline,
            trigger=IntervalTrigger(minutes=config.pipeline.update_interval_minutes),
            id="canonical_pregame_refresh",
            name="Canonical NFL Pregame Refresh",
            replace_existing=True,
        )
        logger.info("Canonical pregame refresh scheduled")

    def run_pregame_pipeline(self) -> dict[str, Any]:
        """Resolve the active NFL week and run the fail-closed production path."""
        season = int(nfl.get_current_season(roster=True))
        week = int(nfl.get_current_week())
        logger.info("Starting canonical pregame pipeline for %s week %s", season, week)
        report = run_production_pipeline(season, week)
        if not report.get("success"):
            logger.error("Canonical pregame pipeline failed: %s", report.get("errors", []))
        return report

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
