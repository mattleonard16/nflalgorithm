"""Deterministic weekly trend-research memo.

Runs on Wednesday between grading and the next week's prediction: it reads what
the week that just finished actually produced and writes a dated memo, plus a
machine-readable twin beside it.

    uv run python -m scripts.weekly_research --season 2025 --week 10

``--season``/``--week`` name the week that JUST COMPLETED. The three
forward-looking sections (usage trends, game scripts, odds/schedule freshness)
are about ``week + 1``, which is the week about to be projected.

The five sections live in :mod:`scripts.research_review` and :mod:`scripts.research_outlook`; this module decides
whether the database can support a memo at all, orders the sections, and writes
the two files. Only two tables are load-bearing enough to abort the run when
missing -- ``player_stats_enhanced`` and ``games``. Every other table (the
grading trio, ``weekly_projections``, ``weekly_odds``) is optional and reads as
empty, because an empty August database must still produce a valid memo.

Nothing here writes to the database. The memo is a read-only artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from utils.context_factors import DEFAULT_PARAMS, ContextParams
from utils.db import DBConnection, get_connection, table_exists

from scripts.research_outlook import (
    DEFAULT_TREND_LIMIT,
    data_freshness,
    next_week_game_scripts,
    usage_trends,
)
from scripts.research_review import grading_recap, projection_accuracy

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Tables without which the memo would be a page of "no data" notes. Their
# absence means the database was never migrated, which is an operator error,
# not a quiet season.
REQUIRED_TABLES: Tuple[str, ...] = ("player_stats_enhanced", "games")

# NFL weeks including the postseason. Anything outside this is a typo.
MIN_WEEK = 1
MAX_WEEK = 22
MIN_SEASON = 1999


class MemoInputError(RuntimeError):
    """The database cannot support a memo at all (missing core tables)."""


def build_memo(
    season: int,
    week: int,
    *,
    conn: Optional[DBConnection] = None,
    params: ContextParams = DEFAULT_PARAMS,
    limit: int = DEFAULT_TREND_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    """Assemble all five sections into ``(markdown, payload)``."""
    missing = [name for name in REQUIRED_TABLES if not table_exists(name, conn=conn)]
    if missing:
        raise MemoInputError(
            f"Cannot build a memo: required tables are missing: {missing}. "
            "Run `make migrate` against this database first."
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    sections = [
        grading_recap(season, week, conn=conn),
        projection_accuracy(season, week, conn=conn),
        usage_trends(season, week, conn=conn, params=params, limit=limit),
        next_week_game_scripts(season, week, conn=conn),
        data_freshness(season, week, conn=conn),
    ]

    header = [
        f"# Weekly Research Memo - {season} Week {week:02d}",
        "",
        f"- Completed week: **{season} W{week:02d}**",
        f"- Next week (the one being projected): **{season} W{week + 1:02d}**",
        f"- Generated: {generated_at}",
        "",
        "---",
        "",
    ]
    body = "\n\n---\n\n".join(text for text, _ in sections)
    markdown = "\n".join(header) + "\n" + body + "\n"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "season": season,
        "week": week,
        "next_week": week + 1,
        "sections": {data["section"]: data for _, data in sections},
    }
    return markdown, payload


def memo_paths(output_dir: Path, season: int, week: int) -> Tuple[Path, Path]:
    stem = f"weekly_research_{season}_W{week:02d}"
    return output_dir / f"{stem}.md", output_dir / f"{stem}.json"


def write_memo(
    markdown: str, payload: Mapping[str, Any], output_dir: Path, season: int, week: int
) -> Tuple[Path, Path]:
    markdown_path, json_path = memo_paths(output_dir, season, week)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return markdown_path, json_path


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="weekly_research",
        description="Write the weekly trend-research memo for a completed NFL week.",
    )
    parser.add_argument(
        "--season", type=int, required=True, help="season of the week that just completed"
    )
    parser.add_argument(
        "--week", type=int, required=True, help="week that just completed (1-22)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="directory for the memo files (default: reports)",
    )
    parser.add_argument(
        "--trend-limit",
        type=int,
        default=DEFAULT_TREND_LIMIT,
        help=f"risers/fallers to list (default: {DEFAULT_TREND_LIMIT})",
    )
    args = parser.parse_args(argv)

    if args.season < MIN_SEASON:
        parser.error(f"--season {args.season} is before {MIN_SEASON}; that is not a real season")
    if not MIN_WEEK <= args.week <= MAX_WEEK:
        parser.error(
            f"--week {args.week} is outside {MIN_WEEK}-{MAX_WEEK}; --week names the week "
            "that just completed, including postseason weeks"
        )
    if args.trend_limit < 1:
        parser.error("--trend-limit must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)

    with get_connection() as conn:
        markdown, payload = build_memo(
            args.season, args.week, conn=conn, limit=args.trend_limit
        )

    markdown_path, json_path = write_memo(
        markdown, payload, args.output_dir, args.season, args.week
    )
    logger.info("weekly_research: wrote machine-readable memo to %s", json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
