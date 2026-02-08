"""Dry-run validation pipeline for prior playoff slates.

Replays the full pipeline on past playoff weeks, measures edge decay
under simulated line movement, and compares agent-filtered vs raw
algorithm picks to validate collaboration quality.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.coordinator import run_all_agents
from confidence_engine import compute_confidence_score, assign_tier, score_plays
from config import config
from risk_manager import assess_risk
from utils.db import read_dataframe
from value_betting_engine import rank_weekly_value

logger = logging.getLogger(__name__)

PLAYOFF_WEEKS = [19, 20, 21]
LINE_MOVEMENT_OFFSETS = [-3, -2, -1, 1, 2, 3]


# ── Core replay ──────────────────────────────────────────────────────


def replay_week(
    season: int,
    week: int,
    min_edge: Optional[float] = None,
) -> Dict[str, Any]:
    """Replay the full pipeline for a single week.

    Returns a dict with raw picks, agent-filtered picks, and metadata.
    """
    threshold = min_edge if min_edge is not None else config.betting.min_edge_threshold

    raw_picks = rank_weekly_value(season, week, min_edge=threshold)

    if raw_picks.empty:
        return {
            "season": season,
            "week": week,
            "raw_picks": pd.DataFrame(),
            "scored_picks": pd.DataFrame(),
            "risk_assessed": pd.DataFrame(),
            "agent_decisions": [],
            "agent_filtered": pd.DataFrame(),
            "metrics": _empty_metrics(),
        }

    scored = score_plays(raw_picks)

    risk_cols = {"kelly_fraction", "p_win", "price", "team", "player_id", "market"}
    has_risk_cols = risk_cols.issubset(set(scored.columns))

    if has_risk_cols:
        try:
            risk_assessed = assess_risk(scored)
        except Exception as exc:
            logger.warning("Risk assessment failed for s=%d w=%d: %s", season, week, exc)
            risk_assessed = scored.copy()
    else:
        risk_assessed = scored.copy()

    try:
        agent_decisions = run_all_agents(season, week)
    except Exception as exc:
        logger.warning("Agent coordinator failed for s=%d w=%d: %s", season, week, exc)
        agent_decisions = []

    agent_filtered = _apply_agent_filter(risk_assessed, agent_decisions)

    actuals = _load_actuals(season, week)
    metrics = _compute_week_metrics(raw_picks, agent_filtered, actuals)

    return {
        "season": season,
        "week": week,
        "raw_picks": raw_picks,
        "scored_picks": scored,
        "risk_assessed": risk_assessed,
        "agent_decisions": agent_decisions,
        "agent_filtered": agent_filtered,
        "metrics": metrics,
    }


def replay_playoff_slate(
    season: int,
    weeks: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Replay full pipeline across multiple playoff weeks."""
    target_weeks = weeks if weeks is not None else PLAYOFF_WEEKS
    results = []
    for week in target_weeks:
        logger.info("Replaying season=%d week=%d", season, week)
        result = replay_week(season, week)
        results.append(result)
    return results


# ── Edge decay ───────────────────────────────────────────────────────


def measure_edge_decay(
    season: int,
    week: int,
    offsets: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Simulate line movement and measure how edges decay.

    Returns a DataFrame with columns:
        player_id, market, original_edge, offset, shifted_edge, edge_change
    """
    moves = offsets if offsets is not None else LINE_MOVEMENT_OFFSETS
    raw = rank_weekly_value(season, week, min_edge=0.0)

    if raw.empty:
        return pd.DataFrame(columns=[
            "player_id", "market", "original_edge",
            "offset", "shifted_edge", "edge_change",
        ])

    rows: List[Dict[str, Any]] = []
    for _, pick in raw.iterrows():
        original_edge = float(pick["edge_percentage"])
        mu = float(pick["mu"])
        sigma = float(pick["sigma"])
        line = float(pick["line"])
        price = int(pick["price"])

        from value_betting_engine import prob_over, _implied_probability

        implied = _implied_probability(price)

        for offset in moves:
            shifted_line = line + offset
            shifted_p = prob_over(mu, sigma, shifted_line)
            shifted_edge = shifted_p - implied
            rows.append({
                "player_id": pick["player_id"],
                "market": pick["market"],
                "original_edge": original_edge,
                "offset": offset,
                "shifted_edge": shifted_edge,
                "edge_change": shifted_edge - original_edge,
            })

    return pd.DataFrame(rows)


def edge_decay_summary(decay_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize edge decay by market type and offset magnitude."""
    if decay_df.empty:
        return pd.DataFrame()

    return (
        decay_df
        .groupby(["market", "offset"])
        .agg(
            mean_edge_change=("edge_change", "mean"),
            median_edge_change=("edge_change", "median"),
            count=("edge_change", "count"),
        )
        .reset_index()
        .sort_values(["market", "offset"])
    )


# ── Agent A/B comparison ─────────────────────────────────────────────


def compare_agent_vs_raw(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """A/B comparison: agent-filtered picks vs raw algorithm picks.

    Returns structured metrics for both approaches.
    """
    raw_all: List[pd.DataFrame] = []
    filtered_all: List[pd.DataFrame] = []

    for r in results:
        if not r["raw_picks"].empty:
            raw_all.append(r["raw_picks"])
        if not r["agent_filtered"].empty:
            filtered_all.append(r["agent_filtered"])

    raw_combined = pd.concat(raw_all, ignore_index=True) if raw_all else pd.DataFrame()
    filtered_combined = pd.concat(filtered_all, ignore_index=True) if filtered_all else pd.DataFrame()

    raw_metrics = _summarize_picks(raw_combined, "raw")
    filtered_metrics = _summarize_picks(filtered_combined, "agent_filtered")

    return {
        "raw": raw_metrics,
        "agent_filtered": filtered_metrics,
        "improvement": {
            "edge_delta": filtered_metrics.get("avg_edge", 0) - raw_metrics.get("avg_edge", 0),
            "pick_reduction": raw_metrics.get("total_picks", 0) - filtered_metrics.get("total_picks", 0),
        },
        "weeks_analyzed": len(results),
    }


# ── Validation report ────────────────────────────────────────────────


def generate_validation_report(
    season: int,
    results: List[Dict[str, Any]],
    decay_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Produce a structured validation report for a playoff slate."""
    ab_comparison = compare_agent_vs_raw(results)

    week_summaries = []
    for r in results:
        week_summaries.append({
            "week": r["week"],
            "raw_pick_count": len(r["raw_picks"]),
            "filtered_pick_count": len(r["agent_filtered"]),
            "agent_decision_count": len(r["agent_decisions"]),
            "approved_count": sum(
                1 for d in r["agent_decisions"] if d.get("decision") == "APPROVED"
            ),
            "rejected_count": sum(
                1 for d in r["agent_decisions"] if d.get("decision") == "REJECTED"
            ),
            "metrics": r["metrics"],
        })

    decay_summary = {}
    if decay_df is not None and not decay_df.empty:
        summary = edge_decay_summary(decay_df)
        decay_summary = summary.to_dict(orient="records")

    return {
        "season": season,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weeks": week_summaries,
        "ab_comparison": ab_comparison,
        "edge_decay": decay_summary,
    }


# ── Helpers ──────────────────────────────────────────────────────────


def _apply_agent_filter(
    picks: pd.DataFrame,
    decisions: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Keep only picks that agents approved."""
    if picks.empty or not decisions:
        return picks.copy()

    approved_keys = {
        (d.get("player_id"), d.get("market"))
        for d in decisions
        if d.get("decision") == "APPROVED"
    }

    if not approved_keys:
        return pd.DataFrame(columns=picks.columns)

    mask = picks.apply(
        lambda row: (row.get("player_id"), row.get("market")) in approved_keys,
        axis=1,
    )
    return picks[mask].copy()


def _load_actuals(season: int, week: int) -> pd.DataFrame:
    """Load actual player stats for grading."""
    query = """
    SELECT player_id, season, week, rushing_yards, receiving_yards,
           passing_yards, receptions, targets
    FROM player_stats_enhanced
    WHERE season = ? AND week = ?
    """
    try:
        return read_dataframe(query, params=(season, week))
    except Exception:
        return pd.DataFrame()


MARKET_TO_STAT = {
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "passing_yards": "passing_yards",
    "receptions": "receptions",
    "targets": "targets",
}


def _grade_picks(picks: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Grade picks against actual results. Returns picks with result column."""
    if picks.empty or actuals.empty:
        return picks.assign(actual=np.nan, result="unknown")

    results = []
    for _, pick in picks.iterrows():
        pid = pick["player_id"]
        market = pick["market"]
        line = float(pick["line"])
        stat_col = MARKET_TO_STAT.get(market)

        player_row = actuals[actuals["player_id"] == pid]
        if player_row.empty or stat_col is None:
            results.append({"actual": np.nan, "result": "unknown"})
            continue

        actual = float(player_row.iloc[0].get(stat_col, np.nan))
        if np.isnan(actual):
            results.append({"actual": np.nan, "result": "unknown"})
        elif actual > line:
            results.append({"actual": actual, "result": "win"})
        elif actual < line:
            results.append({"actual": actual, "result": "loss"})
        else:
            results.append({"actual": actual, "result": "push"})

    graded_df = pd.DataFrame(results, index=picks.index)
    return picks.assign(actual=graded_df["actual"], result=graded_df["result"])


def _compute_week_metrics(
    raw: pd.DataFrame,
    filtered: pd.DataFrame,
    actuals: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute hit rate, avg edge, simulated ROI for a single week."""
    metrics: Dict[str, Any] = _empty_metrics()

    for label, df in [("raw", raw), ("filtered", filtered)]:
        if df.empty:
            continue

        graded = _grade_picks(df, actuals)
        decided = graded[graded["result"].isin(["win", "loss"])]

        total = len(decided)
        wins = int((decided["result"] == "win").sum())
        hit_rate = wins / total if total > 0 else 0.0
        avg_edge = float(df["edge_percentage"].mean()) if "edge_percentage" in df.columns else 0.0

        metrics[label] = {
            "total_picks": len(df),
            "graded": total,
            "wins": wins,
            "hit_rate": round(hit_rate, 4),
            "avg_edge": round(avg_edge, 4),
        }

    return metrics


def _empty_metrics() -> Dict[str, Any]:
    return {
        "raw": {"total_picks": 0, "graded": 0, "wins": 0, "hit_rate": 0.0, "avg_edge": 0.0},
        "filtered": {"total_picks": 0, "graded": 0, "wins": 0, "hit_rate": 0.0, "avg_edge": 0.0},
    }


def _summarize_picks(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Summarize a combined picks DataFrame."""
    if df.empty:
        return {"label": label, "total_picks": 0, "avg_edge": 0.0, "avg_kelly": 0.0}

    return {
        "label": label,
        "total_picks": len(df),
        "avg_edge": round(float(df["edge_percentage"].mean()), 4) if "edge_percentage" in df.columns else 0.0,
        "avg_kelly": round(float(df["kelly_fraction"].mean()), 4) if "kelly_fraction" in df.columns else 0.0,
        "total_stake": round(float(df["stake"].sum()), 2) if "stake" in df.columns else 0.0,
    }


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run validation pipeline")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--weeks", nargs="*", type=int, default=None,
                        help="Weeks to replay (default: 19 20 21)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for report JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    weeks = args.weeks if args.weeks else PLAYOFF_WEEKS

    print(f"Dry-run validation: season={args.season} weeks={weeks}")
    results = replay_playoff_slate(args.season, weeks)

    all_decay: List[pd.DataFrame] = []
    for week in weeks:
        decay = measure_edge_decay(args.season, week)
        if not decay.empty:
            all_decay.append(decay)

    combined_decay = pd.concat(all_decay, ignore_index=True) if all_decay else pd.DataFrame()
    report = generate_validation_report(args.season, results, combined_decay)

    output_dir = Path(args.output) if args.output else config.reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"dry_run_validation_{args.season}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nValidation report saved to {report_path}")
    _print_summary(report)


def _print_summary(report: Dict[str, Any]) -> None:
    """Print human-readable summary of validation report."""
    print(f"\nDry-Run Validation Summary (Season {report['season']})")
    print("=" * 60)

    for ws in report.get("weeks", []):
        w = ws["week"]
        raw_n = ws["raw_pick_count"]
        filt_n = ws["filtered_pick_count"]
        approved = ws["approved_count"]
        rejected = ws["rejected_count"]
        print(f"  Week {w}: {raw_n} raw -> {filt_n} filtered "
              f"({approved} approved, {rejected} rejected)")

    ab = report.get("ab_comparison", {})
    raw_m = ab.get("raw", {})
    filt_m = ab.get("agent_filtered", {})
    print(f"\nA/B Comparison:")
    print(f"  Raw:    {raw_m.get('total_picks', 0)} picks, avg edge={raw_m.get('avg_edge', 0):.4f}")
    print(f"  Agents: {filt_m.get('total_picks', 0)} picks, avg edge={filt_m.get('avg_edge', 0):.4f}")

    improvement = ab.get("improvement", {})
    print(f"  Edge delta: {improvement.get('edge_delta', 0):+.4f}")
    print(f"  Pick reduction: {improvement.get('pick_reduction', 0)}")


if __name__ == "__main__":
    main()
