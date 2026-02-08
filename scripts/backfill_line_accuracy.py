"""Historical line accuracy backfill for NFL value engine.

Cross-references closing lines from weekly_odds with actual player stats
from player_stats_enhanced and projected deltas from weekly_projections.
Quantifies where the model historically beats closing lines.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import config
from utils.db import get_connection, read_dataframe, write_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKET_TO_STAT = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical line accuracy analysis"
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2024,2025",
        help="Comma-separated seasons to analyze",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist results to line_accuracy_history table",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for JSON report output",
    )
    return parser.parse_args()


def load_closing_lines(seasons: List[int]) -> pd.DataFrame:
    """Load the latest (closing) odds snapshot per player/market/sportsbook/week."""
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
    SELECT
        o.season, o.week, o.player_id, o.market, o.sportsbook,
        o.line, o.price, o.as_of
    FROM weekly_odds o
    INNER JOIN (
        SELECT season, week, player_id, market, sportsbook,
               MAX(as_of) AS max_as_of
        FROM weekly_odds
        WHERE season IN ({placeholders})
        GROUP BY season, week, player_id, market, sportsbook
    ) latest
        ON o.season = latest.season
        AND o.week = latest.week
        AND o.player_id = latest.player_id
        AND o.market = latest.market
        AND o.sportsbook = latest.sportsbook
        AND o.as_of = latest.max_as_of
    WHERE o.season IN ({placeholders})
    ORDER BY o.season, o.week, o.player_id
    """
    params = tuple(seasons) + tuple(seasons)
    try:
        return read_dataframe(query, params=params)
    except Exception as e:
        logger.error("Failed to load closing lines: %s", e)
        return pd.DataFrame()


def load_opening_lines(seasons: List[int]) -> pd.DataFrame:
    """Load the earliest (opening) odds snapshot per player/market/sportsbook/week."""
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
    SELECT
        o.season, o.week, o.player_id, o.market, o.sportsbook,
        o.line AS open_line, o.as_of AS open_as_of
    FROM weekly_odds o
    INNER JOIN (
        SELECT season, week, player_id, market, sportsbook,
               MIN(as_of) AS min_as_of
        FROM weekly_odds
        WHERE season IN ({placeholders})
        GROUP BY season, week, player_id, market, sportsbook
    ) earliest
        ON o.season = earliest.season
        AND o.week = earliest.week
        AND o.player_id = earliest.player_id
        AND o.market = earliest.market
        AND o.sportsbook = earliest.sportsbook
        AND o.as_of = earliest.min_as_of
    WHERE o.season IN ({placeholders})
    ORDER BY o.season, o.week, o.player_id
    """
    params = tuple(seasons) + tuple(seasons)
    try:
        return read_dataframe(query, params=params)
    except Exception as e:
        logger.error("Failed to load opening lines: %s", e)
        return pd.DataFrame()


def load_actual_stats(seasons: List[int]) -> pd.DataFrame:
    """Load actual player stats for comparison with lines."""
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
    SELECT player_id, season, week, rushing_yards, receiving_yards
    FROM player_stats_enhanced
    WHERE season IN ({placeholders})
    ORDER BY season, week, player_id
    """
    try:
        df = read_dataframe(query, params=tuple(seasons))
        return df
    except Exception as e:
        logger.error("Failed to load actual stats: %s", e)
        return pd.DataFrame()


def load_projections(seasons: List[int]) -> pd.DataFrame:
    """Load model projections for comparison."""
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
    SELECT season, week, player_id, market, mu, sigma
    FROM weekly_projections
    WHERE season IN ({placeholders})
    ORDER BY season, week, player_id
    """
    try:
        return read_dataframe(query, params=tuple(seasons))
    except Exception as e:
        logger.error("Failed to load projections: %s", e)
        return pd.DataFrame()


def _melt_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    """Reshape actual stats into long format with market column."""
    rows: List[pd.DataFrame] = []
    for market, stat_col in MARKET_TO_STAT.items():
        if stat_col not in actuals.columns:
            continue
        subset = actuals[["player_id", "season", "week", stat_col]].copy()
        subset = subset.rename(columns={stat_col: "actual"})
        subset["market"] = market
        rows.append(subset)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_accuracy_dataset(
    closing: pd.DataFrame,
    opening: pd.DataFrame,
    actuals: pd.DataFrame,
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """Join closing lines, opening lines, actuals, and projections."""
    if closing.empty or actuals.empty:
        return pd.DataFrame()

    melted = _melt_actuals(actuals)
    if melted.empty:
        return pd.DataFrame()

    join_keys = ["season", "week", "player_id", "market"]

    merged = closing.merge(melted, on=join_keys, how="inner")

    if not projections.empty:
        merged = merged.merge(
            projections[["season", "week", "player_id", "market", "mu", "sigma"]],
            on=join_keys,
            how="left",
        )
    else:
        merged["mu"] = np.nan
        merged["sigma"] = np.nan

    if not opening.empty:
        open_keys = ["season", "week", "player_id", "market", "sportsbook"]
        merged = merged.merge(
            opening[["season", "week", "player_id", "market", "sportsbook", "open_line"]],
            on=open_keys,
            how="left",
        )
    else:
        merged["open_line"] = np.nan

    return merged


def compute_hit_rate_by_market(df: pd.DataFrame) -> Dict[str, Dict]:
    """Calculate hit rate for over bets by market type."""
    results = {}
    for market in df["market"].unique():
        subset = df[df["market"] == market]
        if subset.empty:
            continue
        over_hits = (subset["actual"] > subset["line"]).sum()
        total = len(subset)
        results[market] = {
            "total_lines": int(total),
            "over_hits": int(over_hits),
            "hit_rate": round(float(over_hits / total), 4) if total > 0 else 0.0,
        }
    return results


def compute_delta_vs_closing(df: pd.DataFrame) -> Dict[str, Dict]:
    """Average delta between projected mu and closing line by market."""
    results = {}
    proj_rows = df.dropna(subset=["mu"])
    for market in proj_rows["market"].unique():
        subset = proj_rows[proj_rows["market"] == market]
        if subset.empty:
            continue
        deltas = subset["mu"] - subset["line"]
        results[market] = {
            "count": int(len(subset)),
            "avg_delta": round(float(deltas.mean()), 3),
            "median_delta": round(float(deltas.median()), 3),
            "std_delta": round(float(deltas.std()), 3),
            "pct_model_higher": round(float((deltas > 0).mean()), 4),
        }
    return results


def compute_edge_decay(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze edge decay from opening to closing lines."""
    rows_with_open = df.dropna(subset=["open_line", "mu"])
    results = {}
    for market in rows_with_open["market"].unique():
        subset = rows_with_open[rows_with_open["market"] == market]
        if subset.empty:
            continue
        open_delta = subset["mu"] - subset["open_line"]
        close_delta = subset["mu"] - subset["line"]
        decay = open_delta - close_delta
        results[market] = {
            "count": int(len(subset)),
            "avg_open_delta": round(float(open_delta.mean()), 3),
            "avg_close_delta": round(float(close_delta.mean()), 3),
            "avg_decay": round(float(decay.mean()), 3),
            "decay_pct": round(
                float(decay.mean() / open_delta.mean()) if open_delta.mean() != 0 else 0.0, 4
            ),
        }
    return results


def compute_season_type_accuracy(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compare accuracy between regular season (weeks 1-18) and playoffs (weeks 19+)."""
    results = {}
    regular = df[df["week"] <= 18]
    playoff = df[df["week"] > 18]

    for label, subset in [("regular_season", regular), ("playoffs", playoff)]:
        if subset.empty:
            results[label] = {
                "count": 0,
                "over_hit_rate": 0.0,
                "avg_abs_error_vs_line": 0.0,
            }
            continue
        over_hits = (subset["actual"] > subset["line"]).mean()
        abs_error = (subset["actual"] - subset["line"]).abs().mean()
        results[label] = {
            "count": int(len(subset)),
            "over_hit_rate": round(float(over_hits), 4),
            "avg_abs_error_vs_line": round(float(abs_error), 3),
        }

    proj_rows = df.dropna(subset=["mu"])
    for label, subset in [
        ("regular_season", proj_rows[proj_rows["week"] <= 18]),
        ("playoffs", proj_rows[proj_rows["week"] > 18]),
    ]:
        if subset.empty:
            results[label]["model_mae_vs_actual"] = 0.0
            results[label]["model_beat_line_rate"] = 0.0
            continue
        model_error = (subset["mu"] - subset["actual"]).abs().mean()
        line_error = (subset["line"] - subset["actual"]).abs().mean()
        model_beats = float(model_error < line_error) if len(subset) > 0 else 0.0
        results[label]["model_mae_vs_actual"] = round(float(model_error), 3)
        results[label]["line_mae_vs_actual"] = round(float(line_error), 3)
        results[label]["model_beat_line_rate"] = round(
            float(
                (
                    (subset["mu"] - subset["actual"]).abs()
                    < (subset["line"] - subset["actual"]).abs()
                ).mean()
            ),
            4,
        )

    return results


def build_report(
    hit_rates: Dict,
    deltas: Dict,
    edge_decay: Dict,
    season_type: Dict,
    total_records: int,
) -> Dict:
    """Assemble structured report from computed analyses."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records_analyzed": total_records,
        "hit_rate_by_market": hit_rates,
        "avg_delta_vs_closing_line": deltas,
        "edge_decay_analysis": edge_decay,
        "season_type_accuracy": season_type,
    }


def persist_results(df: pd.DataFrame) -> int:
    """Write per-row accuracy records to line_accuracy_history table."""
    if df.empty:
        return 0

    persist_df = df[
        ["season", "week", "player_id", "market", "sportsbook", "line", "actual"]
    ].copy()
    persist_df["mu"] = df.get("mu", np.nan)
    persist_df["sigma"] = df.get("sigma", np.nan)
    persist_df["open_line"] = df.get("open_line", np.nan)
    persist_df["delta_vs_close"] = persist_df["mu"] - persist_df["line"]
    persist_df["actual_vs_line"] = persist_df["actual"] - persist_df["line"]
    persist_df["model_abs_error"] = (persist_df["mu"] - persist_df["actual"]).abs()
    persist_df["line_abs_error"] = (persist_df["line"] - persist_df["actual"]).abs()
    persist_df["model_beats_line"] = (
        persist_df["model_abs_error"] < persist_df["line_abs_error"]
    ).astype(int)
    persist_df["is_over_hit"] = (persist_df["actual"] > persist_df["line"]).astype(int)
    persist_df["computed_at"] = datetime.now(timezone.utc).isoformat()

    try:
        write_dataframe(persist_df, "line_accuracy_history", if_exists="replace")
        logger.info("Persisted %d records to line_accuracy_history", len(persist_df))
        return len(persist_df)
    except Exception as e:
        logger.error("Failed to persist results: %s", e)
        return 0


def save_report(report: Dict, output_dir: Path) -> Path:
    """Save JSON report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"line_accuracy_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", path)
    return path


def print_report(report: Dict) -> None:
    """Print a human-readable summary of the report."""
    print("\n" + "=" * 60)
    print("LINE ACCURACY BACKFILL REPORT")
    print("=" * 60)
    print(f"Generated: {report['generated_at']}")
    print(f"Total records analyzed: {report['total_records_analyzed']}")

    print("\n--- Hit Rate by Market ---")
    for market, stats in report.get("hit_rate_by_market", {}).items():
        print(f"  {market}: {stats['hit_rate']:.1%} ({stats['over_hits']}/{stats['total_lines']})")

    print("\n--- Average Delta vs Closing Line ---")
    for market, stats in report.get("avg_delta_vs_closing_line", {}).items():
        print(
            f"  {market}: avg={stats['avg_delta']:+.1f}  "
            f"model_higher={stats['pct_model_higher']:.1%}  "
            f"(n={stats['count']})"
        )

    print("\n--- Edge Decay (Open -> Close) ---")
    for market, stats in report.get("edge_decay_analysis", {}).items():
        print(
            f"  {market}: open_delta={stats['avg_open_delta']:+.1f}  "
            f"close_delta={stats['avg_close_delta']:+.1f}  "
            f"decay={stats['avg_decay']:+.1f}"
        )

    print("\n--- Season Type Accuracy ---")
    for label, stats in report.get("season_type_accuracy", {}).items():
        if stats["count"] == 0:
            print(f"  {label}: no data")
            continue
        model_beat = stats.get("model_beat_line_rate", 0)
        print(
            f"  {label}: n={stats['count']}  "
            f"over_hit={stats['over_hit_rate']:.1%}  "
            f"model_beats_line={model_beat:.1%}"
        )

    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    logger.info("Backfilling line accuracy for seasons: %s", seasons)

    closing = load_closing_lines(seasons)
    opening = load_opening_lines(seasons)
    actuals = load_actual_stats(seasons)
    projections = load_projections(seasons)

    logger.info(
        "Loaded: %d closing lines, %d opening lines, %d actual stat rows, %d projections",
        len(closing),
        len(opening),
        len(actuals),
        len(projections),
    )

    dataset = build_accuracy_dataset(closing, opening, actuals, projections)
    if dataset.empty:
        logger.warning("No matched records found. Ensure odds and stats tables are populated.")
        return

    logger.info("Matched %d records for analysis", len(dataset))

    hit_rates = compute_hit_rate_by_market(dataset)
    deltas = compute_delta_vs_closing(dataset)
    edge_decay = compute_edge_decay(dataset)
    season_type = compute_season_type_accuracy(dataset)

    report = build_report(hit_rates, deltas, edge_decay, season_type, len(dataset))
    print_report(report)

    output_dir = Path(args.output_dir) if args.output_dir else config.reports_dir
    save_report(report, output_dir)

    if args.persist:
        count = persist_results(dataset)
        logger.info("Persisted %d accuracy records", count)


if __name__ == "__main__":
    main()
