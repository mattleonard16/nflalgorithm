#!/usr/bin/env python3
"""Tight End Market Bias Analysis.

Analyzes whether TE receiving yards props are systematically overpriced,
particularly in playoff contexts. Compares EWMA projections vs market
lines, runs statistical significance tests, and outputs a JSON report.

Usage:
    python -m scripts.te_market_bias [--season SEASON] [--output DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

from config import config
from utils.db import read_dataframe

logger = logging.getLogger(__name__)

# Playoff weeks are 19-22 (wild card, divisional, conference, super bowl)
PLAYOFF_WEEKS = frozenset(range(19, 23))
REGULAR_SEASON_WEEKS = frozenset(range(1, 19))


def load_te_projections_vs_odds(
    seasons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Load TE projection/odds pairs for receiving_yards."""
    if seasons is None:
        seasons = config.pipeline.default_seasons

    placeholders = ",".join("?" for _ in seasons)
    query = f"""
    SELECT
        p.season, p.week, p.player_id, p.team, p.mu, p.sigma,
        o.line, o.sportsbook
    FROM weekly_projections p
    INNER JOIN weekly_odds o
        ON p.season = o.season
        AND p.week = o.week
        AND p.player_id = o.player_id
        AND p.market = o.market
    INNER JOIN player_stats_enhanced s
        ON p.player_id = s.player_id
        AND p.season = s.season
        AND p.week = s.week
    WHERE p.market = 'receiving_yards'
        AND s.position = 'TE'
        AND p.season IN ({placeholders})
    ORDER BY p.season, p.week
    """
    try:
        return read_dataframe(query, params=tuple(seasons))
    except Exception as exc:
        logger.warning("Failed to load TE projection/odds data: %s", exc)
        return pd.DataFrame()


def load_te_actuals(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """Load actual TE receiving yards from player_stats_enhanced."""
    if seasons is None:
        seasons = config.pipeline.default_seasons

    placeholders = ",".join("?" for _ in seasons)
    query = f"""
    SELECT player_id, season, week, receiving_yards AS actual_yards
    FROM player_stats_enhanced
    WHERE position = 'TE'
        AND season IN ({placeholders})
    ORDER BY season, week
    """
    try:
        return read_dataframe(query, params=tuple(seasons))
    except Exception as exc:
        logger.warning("Failed to load TE actuals: %s", exc)
        return pd.DataFrame()


def classify_game_type(week: int) -> str:
    """Classify a week as 'playoff' or 'regular'."""
    if week in PLAYOFF_WEEKS:
        return "playoff"
    return "regular"


def merge_projections_with_actuals(
    projections: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Merge projection/odds data with actual results."""
    if projections.empty or actuals.empty:
        return pd.DataFrame()

    merged = projections.merge(
        actuals,
        on=["player_id", "season", "week"],
        how="inner",
    )
    merged = merged.copy()
    merged["game_type"] = merged["week"].apply(classify_game_type)
    merged["line_diff"] = merged["line"] - merged["actual_yards"]
    merged["mu_diff"] = merged["mu"] - merged["actual_yards"]
    merged["hit_over"] = (merged["actual_yards"] > merged["line"]).astype(int)
    return merged


def compute_bias_metrics(df: pd.DataFrame, label: str) -> Dict:
    """Compute bias metrics for a subset of TE data."""
    if df.empty:
        return {
            "label": label,
            "sample_size": 0,
            "mean_line_diff": None,
            "median_line_diff": None,
            "mean_mu_diff": None,
            "over_hit_rate": None,
            "std_line_diff": None,
        }

    line_diffs = df["line_diff"].values
    mu_diffs = df["mu_diff"].values
    hit_over = df["hit_over"].values

    return {
        "label": label,
        "sample_size": len(df),
        "mean_line_diff": float(np.mean(line_diffs)),
        "median_line_diff": float(np.median(line_diffs)),
        "mean_mu_diff": float(np.mean(mu_diffs)),
        "over_hit_rate": float(np.mean(hit_over)),
        "std_line_diff": float(np.std(line_diffs)),
    }


def run_significance_tests(
    regular: pd.DataFrame,
    playoff: pd.DataFrame,
) -> Dict:
    """Run statistical tests for systematic bias.

    Returns dict with t-test and chi-squared results comparing
    regular season vs playoff TE pricing.
    """
    results: Dict = {"t_test": None, "chi_squared": None}

    # T-test on line_diff (line - actual) between regular and playoff
    if len(regular) >= 2 and len(playoff) >= 2:
        t_stat, p_value = ttest_ind(
            regular["line_diff"].values,
            playoff["line_diff"].values,
            equal_var=False,
        )
        results["t_test"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_05": bool(p_value < 0.05),
            "significant_at_10": bool(p_value < 0.10),
        }

    # Chi-squared test on over/under hit rates
    if len(regular) >= 5 and len(playoff) >= 5:
        reg_over = int(regular["hit_over"].sum())
        reg_under = len(regular) - reg_over
        plo_over = int(playoff["hit_over"].sum())
        plo_under = len(playoff) - plo_over

        # Only run if all cells have expected count > 0
        if min(reg_over, reg_under, plo_over, plo_under) > 0:
            contingency = np.array([
                [reg_over, reg_under],
                [plo_over, plo_under],
            ])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            results["chi_squared"] = {
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "significant_at_05": bool(p_value < 0.05),
                "contingency_table": {
                    "regular": {"over": reg_over, "under": reg_under},
                    "playoff": {"over": plo_over, "under": plo_under},
                },
            }

    return results


def compute_suggested_adjustment(
    playoff_metrics: Dict,
    overall_metrics: Dict,
) -> Optional[float]:
    """Compute a suggested TE market bias adjustment for playoff context.

    Returns negative value (yards to subtract from line) if TEs are
    systematically overpriced in playoffs, None if insufficient data.
    """
    if playoff_metrics["sample_size"] < 5:
        return None

    mean_diff = playoff_metrics["mean_line_diff"]
    if mean_diff is None:
        return None

    # Only suggest adjustment if bias > 1.5 yards overpriced
    if mean_diff > 1.5:
        return round(-mean_diff, 1)

    return None


def build_report(
    overall: Dict,
    regular: Dict,
    playoff: Dict,
    significance: Dict,
    adjustment: Optional[float],
    seasons: List[int],
) -> Dict:
    """Build the full JSON analysis report."""
    return {
        "report_type": "te_market_bias_analysis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons_analyzed": seasons,
        "overall_metrics": overall,
        "regular_season_metrics": regular,
        "playoff_metrics": playoff,
        "statistical_tests": significance,
        "suggested_adjustment": {
            "value": adjustment,
            "applies_to": "playoff_te_receiving_yards",
            "rationale": (
                "Negative adjustment reduces projected value for TE "
                "receiving yards props in playoff games where lines "
                "are systematically inflated."
            )
            if adjustment is not None
            else "Insufficient data or no significant bias detected.",
        },
    }


def print_terminal_summary(report: Dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("TE Market Bias Analysis Report")
    print("=" * 60)
    print(f"Seasons: {report['seasons_analyzed']}")
    print(f"Generated: {report['generated_at']}")

    for section_key in ("overall_metrics", "regular_season_metrics", "playoff_metrics"):
        section = report[section_key]
        print(f"\n--- {section['label']} (n={section['sample_size']}) ---")
        if section["sample_size"] == 0:
            print("  No data available.")
            continue
        print(f"  Mean line - actual: {section['mean_line_diff']:+.2f} yards")
        print(f"  Median line - actual: {section['median_line_diff']:+.2f} yards")
        print(f"  Mean mu - actual: {section['mean_mu_diff']:+.2f} yards")
        print(f"  Over hit rate: {section['over_hit_rate']:.1%}")

    tests = report["statistical_tests"]
    print("\n--- Statistical Tests ---")
    if tests["t_test"]:
        t = tests["t_test"]
        sig_label = "YES" if t["significant_at_05"] else "NO"
        print(f"  T-test (regular vs playoff line_diff):")
        print(f"    t={t['t_statistic']:.3f}, p={t['p_value']:.4f}, significant(0.05): {sig_label}")
    else:
        print("  T-test: insufficient data")

    if tests["chi_squared"]:
        c = tests["chi_squared"]
        sig_label = "YES" if c["significant_at_05"] else "NO"
        print(f"  Chi-squared (over/under rates):")
        print(f"    chi2={c['chi2_statistic']:.3f}, p={c['p_value']:.4f}, significant(0.05): {sig_label}")
    else:
        print("  Chi-squared: insufficient data")

    adj = report["suggested_adjustment"]
    print("\n--- Suggested Adjustment ---")
    if adj["value"] is not None:
        print(f"  Adjustment: {adj['value']:+.1f} yards for playoff TE props")
    else:
        print(f"  {adj['rationale']}")

    print("=" * 60)


def run_analysis(
    seasons: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> Dict:
    """Run the full TE market bias analysis pipeline.

    Returns the report dict and optionally writes JSON to output_dir.
    """
    if seasons is None:
        seasons = list(config.pipeline.default_seasons)

    projections = load_te_projections_vs_odds(seasons)
    actuals = load_te_actuals(seasons)
    merged = merge_projections_with_actuals(projections, actuals)

    if merged.empty:
        logger.warning("No TE data found for analysis.")
        report = build_report(
            overall=compute_bias_metrics(pd.DataFrame(), "Overall"),
            regular=compute_bias_metrics(pd.DataFrame(), "Regular Season"),
            playoff=compute_bias_metrics(pd.DataFrame(), "Playoff"),
            significance={"t_test": None, "chi_squared": None},
            adjustment=None,
            seasons=seasons,
        )
    else:
        regular_df = merged[merged["game_type"] == "regular"]
        playoff_df = merged[merged["game_type"] == "playoff"]

        overall = compute_bias_metrics(merged, "Overall")
        regular = compute_bias_metrics(regular_df, "Regular Season")
        playoff = compute_bias_metrics(playoff_df, "Playoff")
        significance = run_significance_tests(regular_df, playoff_df)
        adjustment = compute_suggested_adjustment(playoff, overall)

        report = build_report(
            overall=overall,
            regular=regular,
            playoff=playoff,
            significance=significance,
            adjustment=adjustment,
            seasons=seasons,
        )

    print_terminal_summary(report)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_file = out_path / "te_market_bias_report.json"
        report_file.write_text(json.dumps(report, indent=2))
        logger.info("Report written to %s", report_file)

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TE Market Bias Analysis")
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated seasons (default: config default_seasons)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for JSON report (default: reports)",
    )
    args = parser.parse_args()

    seasons = None
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]

    logging.basicConfig(level=logging.INFO)
    run_analysis(seasons=seasons, output_dir=args.output)


if __name__ == "__main__":
    main()
