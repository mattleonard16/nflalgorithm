"""Data health invariant checks run after pipeline completion."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from utils.db import fetchall, fetchone

logger = logging.getLogger(__name__)


def check_missing_player_info(season: int, week: int) -> Dict[str, Any]:
    """Check rate of missing player_name/position in materialized_value_view."""
    try:
        row = fetchone(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN pd.player_name IS NULL THEN 1 ELSE 0 END) AS missing_name,
                SUM(CASE WHEN pd.position IS NULL THEN 1 ELSE 0 END) AS missing_position
            FROM materialized_value_view v
            LEFT JOIN player_dim pd ON v.player_id = pd.player_id
            WHERE v.season = ? AND v.week = ?
            """,
            params=(season, week),
        )
        total = row[0] if row else 0
        missing_name = row[1] if row else 0
        missing_pos = row[2] if row else 0

        return {
            "check": "missing_player_info",
            "total_bets": total,
            "missing_name": missing_name,
            "missing_position": missing_pos,
            "missing_name_rate": round(missing_name / total, 4) if total > 0 else 0.0,
            "missing_position_rate": round(missing_pos / total, 4) if total > 0 else 0.0,
            "status": "pass" if (missing_name / total if total else 0) < 0.1 else "warn",
        }
    except Exception as exc:
        logger.error("check_missing_player_info failed: %s", exc)
        return {
            "check": "missing_player_info",
            "status": "fail",
            "error": str(exc),
            "total_bets": 0,
            "missing_name": 0,
            "missing_position": 0,
            "missing_name_rate": 0.0,
            "missing_position_rate": 0.0,
        }


def check_duplicate_player_dim() -> Dict[str, Any]:
    """Check for duplicate player_id entries in player_dim."""
    try:
        row = fetchone(
            """
            SELECT COUNT(*) AS total, COUNT(DISTINCT player_id) AS distinct_ids
            FROM player_dim
            """
        )
        total = row[0] if row else 0
        distinct = row[1] if row else 0
        duplicates = total - distinct

        return {
            "check": "duplicate_player_dim",
            "total_rows": total,
            "distinct_ids": distinct,
            "duplicates": duplicates,
            "status": "pass" if duplicates == 0 else "fail",
        }
    except Exception as exc:
        logger.error("check_duplicate_player_dim failed: %s", exc)
        return {
            "check": "duplicate_player_dim",
            "status": "fail",
            "error": str(exc),
            "total_rows": 0,
            "distinct_ids": 0,
            "duplicates": 0,
        }


def check_null_lines(season: int, week: int) -> Dict[str, Any]:
    """Check rate of null line values in materialized_value_view."""
    try:
        row = fetchone(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN line IS NULL THEN 1 ELSE 0 END) AS null_line,
                SUM(CASE WHEN mu IS NULL THEN 1 ELSE 0 END) AS null_mu
            FROM materialized_value_view
            WHERE season = ? AND week = ?
            """,
            params=(season, week),
        )
        total = row[0] if row else 0
        null_line = row[1] or 0 if row else 0
        null_mu = row[2] or 0 if row else 0

        return {
            "check": "null_lines_projections",
            "total_bets": total,
            "null_line": null_line,
            "null_mu": null_mu,
            "null_line_rate": round(null_line / total, 4) if total > 0 else 0.0,
            "null_mu_rate": round(null_mu / total, 4) if total > 0 else 0.0,
            "status": "pass" if total == 0 or (null_line == 0 and null_mu == 0) else "warn",
        }
    except Exception as exc:
        logger.error("check_null_lines failed: %s", exc)
        return {
            "check": "null_lines_projections",
            "status": "fail",
            "error": str(exc),
            "total_bets": 0,
            "null_line": 0,
            "null_mu": 0,
            "null_line_rate": 0.0,
            "null_mu_rate": 0.0,
        }


def check_projection_coverage(season: int, week: int) -> Dict[str, Any]:
    """Check how many bets have matching projections."""
    try:
        row = fetchone(
            """
            SELECT
                COUNT(*) AS total_bets,
                SUM(CASE WHEN wp.player_id IS NOT NULL THEN 1 ELSE 0 END) AS with_projection
            FROM materialized_value_view v
            LEFT JOIN weekly_projections wp
                ON v.season = wp.season AND v.week = wp.week
                AND v.player_id = wp.player_id AND v.market = wp.market
            WHERE v.season = ? AND v.week = ?
            """,
            params=(season, week),
        )
        total = row[0] if row else 0
        with_proj = row[1] if row else 0

        return {
            "check": "projection_coverage",
            "total_bets": total,
            "with_projection": with_proj,
            "coverage_rate": round(with_proj / total, 4) if total > 0 else 0.0,
            "status": "pass" if total == 0 or (with_proj / total) > 0.8 else "warn",
        }
    except Exception as exc:
        logger.error("check_projection_coverage failed: %s", exc)
        return {
            "check": "projection_coverage",
            "status": "fail",
            "error": str(exc),
            "total_bets": 0,
            "with_projection": 0,
            "coverage_rate": 0.0,
        }


def run_all_checks(season: int, week: int) -> Dict[str, Any]:
    """Run all data health invariant checks and return a summary."""
    checks: List[Dict[str, Any]] = [
        check_missing_player_info(season, week),
        check_duplicate_player_dim(),
        check_null_lines(season, week),
        check_projection_coverage(season, week),
    ]

    statuses = [c["status"] for c in checks]
    if "fail" in statuses:
        overall = "fail"
    elif "warn" in statuses:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "overall": overall,
        "checks": checks,
        "season": season,
        "week": week,
    }
