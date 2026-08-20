"""Publish our own prop lines for a week's top-N player universe.

Usage:
    python -m scripts.publish_internal_lines                    # upcoming week
    python -m scripts.publish_internal_lines --season 2026 --week 1
    python -m scripts.publish_internal_lines --dry-run

Season and week are optional on purpose. Omitted, they come from
``utils.current_week``, which reads the schedule already in the database. That
is what lets a cron entry run the weekly chain with nobody around to look up
which week it is.

Republishing a week replaces that week's rows; see
``utils.internal_lines.persist_internal_lines``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from utils.current_week import resolve_current_week
from utils.db import get_connection
from utils.internal_lines import load_internal_lines, persist_internal_lines
from utils.player_universe import DEFAULT_TRAILING_WEEKS, DEFAULT_UNIVERSE_SIZE

logger = logging.getLogger(__name__)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--season", type=int, help="NFL season; resolved from the schedule if omitted"
    )
    parser.add_argument("--week", type=int, help="NFL week; resolved from the schedule if omitted")
    parser.add_argument("--size", type=int, default=DEFAULT_UNIVERSE_SIZE, help="universe size")
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_TRAILING_WEEKS,
        help="trailing played weeks used to rank usage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and report the lines without writing them",
    )
    args = parser.parse_args(argv)

    # One without the other is a typo, not a request. Resolving the missing half
    # would publish a week the caller did not name.
    if (args.season is None) != (args.week is None):
        parser.error("--season and --week must be given together, or both omitted")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    with get_connection() as conn:
        if args.season is None:
            season, week = resolve_current_week(conn=conn)
            logger.info("no season/week given; resolved %s week %s from the schedule", season, week)
        else:
            season, week = args.season, args.week

        lines = load_internal_lines(season, week, conn=conn, size=args.size, window=args.window)

        if args.dry_run:
            logger.info(
                "dry run: %d lines for %s week %s covering %d players; nothing written",
                len(lines),
                season,
                week,
                lines["player_id"].nunique(),
            )
            return 0

        written = persist_internal_lines(lines, conn)

    logger.info("published %d internal lines for %s week %s", written, season, week)
    return 0


if __name__ == "__main__":
    sys.exit(main())
