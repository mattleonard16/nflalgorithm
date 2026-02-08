"""Post-game learning loop and outcome attribution.

Tracks model accuracy, agent performance, and tunes confidence
thresholds based on historical results via Bayesian updating.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import config
from utils.db import execute, executemany, get_connection, read_dataframe

logger = logging.getLogger(__name__)


# ── Market-to-stat mapping ───────────────────────────────────────────

MARKET_TO_STAT = {
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "passing_yards": "passing_yards",
    "receptions": "receptions",
    "targets": "targets",
}


# ── Outcome attribution ─────────────────────────────────────────────


def _empty_attribution(
    player_id: str,
    market: str,
    reason: str,
) -> Dict[str, Any]:
    """Return a stub attribution when data is missing."""
    return {
        "player_id": player_id,
        "market": market,
        "season": 0,
        "week": 0,
        "actual": None,
        "mu": 0.0,
        "sigma": 0.0,
        "line": None,
        "model_error": 0.0,
        "line_value": None,
        "variance_component": False,
        "attribution": reason,
    }


def attribute_outcome(
    season: int,
    week: int,
    player_id: str,
    market: str,
) -> Dict[str, Any]:
    """Attribute the outcome of a single prediction to its components.

    Returns
    -------
    dict with keys:
        model_error   : |mu - actual| / sigma  (z-score of model miss)
        line_value    : float, positive if we beat closing line
        variance_component : bool, True if result within 1-sigma
        attribution   : str, one of "model_accurate", "model_miss", "high_variance"
        actual        : float or None
        mu            : float
        sigma         : float
        line          : float
    """
    projection = _load_projection(season, week, player_id, market)
    if projection is None:
        return _empty_attribution(player_id, market, "no_projection")

    actual = _load_actual_stat(season, week, player_id, market)
    if actual is None:
        return _empty_attribution(player_id, market, "no_actual")

    mu = float(projection["mu"])
    sigma = max(float(projection["sigma"]), 0.1)
    line = _load_best_line(season, week, player_id, market)

    model_error = abs(mu - actual) / sigma
    line_value = (actual - line) if line is not None else 0.0
    within_one_sigma = abs(actual - mu) <= sigma

    if model_error <= 1.0:
        attribution = "model_accurate"
    elif model_error <= 2.0:
        attribution = "model_miss"
    else:
        attribution = "high_variance"

    return {
        "player_id": player_id,
        "market": market,
        "season": season,
        "week": week,
        "actual": actual,
        "mu": mu,
        "sigma": sigma,
        "line": line,
        "model_error": round(model_error, 4),
        "line_value": round(line_value, 4) if line is not None else None,
        "variance_component": within_one_sigma,
        "attribution": attribution,
    }


def batch_attribute_outcomes(
    season: int,
    week: int,
) -> List[Dict[str, Any]]:
    """Attribute outcomes for all projections in a given week."""
    projections = _load_all_projections(season, week)
    if projections.empty:
        logger.warning("No projections found for s=%d w=%d", season, week)
        return []

    results = []
    for _, proj in projections.iterrows():
        result = attribute_outcome(
            season, week, proj["player_id"], proj["market"]
        )
        results.append(result)

    return results


# ── Agent performance tracking ───────────────────────────────────────


def update_agent_performance(
    season: int,
    week: int,
    outcomes: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Track which agents contributed to winning vs losing decisions.

    Loads agent decisions for the week, joins with bet outcomes,
    and updates the agent_performance table.

    Returns the number of performance records inserted.
    """
    decisions = _load_agent_decisions(season, week)
    if decisions.empty:
        logger.warning("No agent decisions for s=%d w=%d", season, week)
        return 0

    bet_outcomes = _load_bet_outcomes(season, week)
    if bet_outcomes.empty:
        logger.warning("No bet outcomes for s=%d w=%d", season, week)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    records: List[Tuple] = []

    for _, dec in decisions.iterrows():
        pid = dec["player_id"]
        market = dec["market"]
        decision = dec["decision"]

        matching = bet_outcomes[
            (bet_outcomes["player_id"] == pid) &
            (bet_outcomes["market"] == market)
        ]

        if matching.empty:
            continue

        outcome_row = matching.iloc[0]
        result = outcome_row.get("result", "unknown")
        profit = float(outcome_row.get("profit_units", 0.0))

        agent_reports_raw = dec.get("agent_reports", "[]")
        try:
            agent_reports = json.loads(agent_reports_raw) if isinstance(agent_reports_raw, str) else agent_reports_raw
        except (json.JSONDecodeError, TypeError):
            agent_reports = []

        if not isinstance(agent_reports, list):
            agent_reports = []

        for ar in agent_reports:
            agent_name = ar.get("agent", "unknown")
            recommendation = ar.get("recommendation", "NEUTRAL")
            confidence = float(ar.get("confidence", 0.5))

            correct = _was_recommendation_correct(recommendation, decision, result)

            records.append((
                season, week, agent_name, pid, market,
                recommendation, confidence, decision, result,
                profit, int(correct), now,
            ))

    if not records:
        return 0

    insert_sql = """
    INSERT OR REPLACE INTO agent_performance (
        season, week, agent_name, player_id, market,
        recommendation, confidence, final_decision, outcome,
        profit_units, correct, recorded_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    executemany(insert_sql, records)
    logger.info("Inserted %d agent performance records for s=%d w=%d",
                len(records), season, week)
    return len(records)


def _was_recommendation_correct(
    recommendation: str,
    decision: str,
    result: str,
) -> bool:
    """Determine if an agent's recommendation was correct.

    Correct means:
      - APPROVE + APPROVED + win  -> correct
      - REJECT  + REJECTED + loss -> correct (avoided a loser)
      - APPROVE + APPROVED + loss -> incorrect
      - REJECT  + APPROVED + win  -> incorrect (disagreed with winner)
    """
    if result not in ("win", "loss"):
        return False

    if recommendation == "APPROVE" and decision == "APPROVED" and result == "win":
        return True
    if recommendation == "REJECT" and result == "loss":
        return True
    if recommendation == "APPROVE" and result == "loss":
        return False
    if recommendation == "REJECT" and result == "win":
        return False

    return False


def get_agent_accuracy_summary(
    n_weeks: int = 8,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Summarize agent accuracy over recent weeks.

    Returns DataFrame with columns: agent_name, total, correct, accuracy, avg_confidence.
    """
    where_clauses = []
    params: List[Any] = []

    if season is not None:
        where_clauses.append("season = ?")
        params.append(season)

    where_str = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    query = f"""
    SELECT agent_name,
           COUNT(*) as total,
           SUM(correct) as correct,
           ROUND(CAST(SUM(correct) AS REAL) / COUNT(*), 4) as accuracy,
           ROUND(AVG(confidence), 4) as avg_confidence
    FROM agent_performance
    {where_str}
    GROUP BY agent_name
    ORDER BY accuracy DESC
    """
    try:
        return read_dataframe(query, params=tuple(params) if params else None)
    except Exception:
        return pd.DataFrame(columns=["agent_name", "total", "correct", "accuracy", "avg_confidence"])


# ── Confidence threshold tightening ──────────────────────────────────


def recommend_threshold_updates(
    n_weeks: int = 4,
    season: Optional[int] = None,
) -> Dict[str, Any]:
    """Bayesian update of confidence thresholds based on actual outcomes.

    Analyzes bet outcomes by confidence tier and recommends raising
    thresholds for tiers with poor performance.

    Returns
    -------
    dict with:
        tier_performance : list of per-tier stats
        recommendations  : list of suggested config changes
        current_thresholds : current config values
    """
    where_clauses = ["result IN ('win', 'loss')"]
    params: List[Any] = []

    if season is not None:
        where_clauses.append("season = ?")
        params.append(season)

    where_str = " WHERE " + " AND ".join(where_clauses)

    query = f"""
    SELECT confidence_tier, result, profit_units, edge_at_placement
    FROM bet_outcomes
    {where_str}
    """
    try:
        df = read_dataframe(query, params=tuple(params) if params else None)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return {
            "tier_performance": [],
            "recommendations": [],
            "current_thresholds": _current_thresholds(),
        }

    tier_stats = []
    recommendations = []

    for tier in df["confidence_tier"].unique():
        tier_df = df[df["confidence_tier"] == tier]
        total = len(tier_df)
        wins = int((tier_df["result"] == "win").sum())
        losses = total - wins
        hit_rate = wins / total if total > 0 else 0.0
        avg_edge = float(tier_df["edge_at_placement"].mean())
        avg_profit = float(tier_df["profit_units"].mean())

        stat = {
            "tier": tier,
            "total": total,
            "wins": wins,
            "losses": losses,
            "hit_rate": round(hit_rate, 4),
            "avg_edge": round(avg_edge, 4),
            "avg_profit": round(avg_profit, 4),
        }
        tier_stats.append(stat)

        if hit_rate < 0.45 and total >= 5:
            recommendations.append({
                "tier": tier,
                "action": "raise_threshold",
                "reason": f"Hit rate {hit_rate:.1%} below 45% over {total} bets",
                "suggested_min_edge": round(avg_edge * 1.25, 4),
            })
        elif hit_rate > 0.60 and avg_profit > 0 and total >= 5:
            recommendations.append({
                "tier": tier,
                "action": "maintain_or_lower",
                "reason": f"Hit rate {hit_rate:.1%} with positive profit over {total} bets",
            })

    return {
        "tier_performance": tier_stats,
        "recommendations": recommendations,
        "current_thresholds": _current_thresholds(),
    }


def _current_thresholds() -> Dict[str, Any]:
    """Return current betting/confidence thresholds from config."""
    return {
        "min_edge_threshold": config.betting.min_edge_threshold,
        "min_confidence": config.betting.min_confidence,
        "confidence_min_tier": getattr(config.confidence, "min_tier", 2),
    }


# ── Learning report ──────────────────────────────────────────────────


def generate_learning_report(
    season: int,
    week_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Comprehensive learning report for a season/week range.

    Includes:
        - Performance by confidence tier
        - Performance by agent recommendation
        - Overall model accuracy trends
        - Threshold recommendations
    """
    if week_range is not None:
        week_start, week_end = week_range
    else:
        week_start, week_end = 1, 22

    # Tier performance
    tier_perf = _tier_performance(season, week_start, week_end)

    # Agent performance
    agent_summary = get_agent_accuracy_summary(season=season)

    # Model accuracy trends
    accuracy_trends = _model_accuracy_trends(season, week_start, week_end)

    # Threshold recommendations
    threshold_recs = recommend_threshold_updates(season=season)

    # Attribution summary
    attributions = _attribution_summary(season, week_start, week_end)

    return {
        "season": season,
        "week_range": [week_start, week_end],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier_performance": tier_perf,
        "agent_performance": agent_summary.to_dict(orient="records") if not agent_summary.empty else [],
        "model_accuracy_trends": accuracy_trends,
        "threshold_recommendations": threshold_recs,
        "attribution_summary": attributions,
    }


def _tier_performance(
    season: int,
    week_start: int,
    week_end: int,
) -> List[Dict[str, Any]]:
    """Performance breakdown by confidence tier."""
    query = """
    SELECT confidence_tier, result, profit_units, edge_at_placement
    FROM bet_outcomes
    WHERE season = ? AND week BETWEEN ? AND ?
      AND result IN ('win', 'loss')
    """
    try:
        df = read_dataframe(query, params=(season, week_start, week_end))
    except Exception:
        return []

    if df.empty:
        return []

    results = []
    for tier, group in df.groupby("confidence_tier"):
        total = len(group)
        wins = int((group["result"] == "win").sum())
        profit = float(group["profit_units"].sum())
        results.append({
            "tier": tier,
            "total": total,
            "wins": wins,
            "losses": total - wins,
            "hit_rate": round(wins / total, 4) if total > 0 else 0.0,
            "total_profit": round(profit, 2),
            "avg_edge": round(float(group["edge_at_placement"].mean()), 4),
        })

    return sorted(results, key=lambda x: x.get("hit_rate", 0), reverse=True)


def _model_accuracy_trends(
    season: int,
    week_start: int,
    week_end: int,
) -> List[Dict[str, Any]]:
    """Weekly model accuracy (MAE) trends."""
    query = """
    SELECT p.week, p.market,
           AVG(ABS(p.mu - s.{stat_col})) as mae,
           COUNT(*) as n
    FROM weekly_projections p
    INNER JOIN player_stats_enhanced s
        ON p.player_id = s.player_id
        AND p.season = s.season
        AND p.week = s.week
    WHERE p.season = ? AND p.week BETWEEN ? AND ?
    GROUP BY p.week, p.market
    ORDER BY p.week, p.market
    """
    results = []
    for market, stat_col in MARKET_TO_STAT.items():
        formatted_query = query.format(stat_col=stat_col)
        try:
            df = read_dataframe(formatted_query, params=(season, week_start, week_end))
            for _, row in df.iterrows():
                results.append({
                    "week": int(row["week"]),
                    "market": market,
                    "mae": round(float(row["mae"]), 2),
                    "sample_size": int(row["n"]),
                })
        except Exception:
            continue

    return results


def _attribution_summary(
    season: int,
    week_start: int,
    week_end: int,
) -> Dict[str, Any]:
    """Summarize outcome attributions across weeks."""
    all_attributions: List[Dict[str, Any]] = []
    for week in range(week_start, week_end + 1):
        attributions = batch_attribute_outcomes(season, week)
        all_attributions.extend(attributions)

    if not all_attributions:
        return {"total": 0, "breakdown": {}}

    df = pd.DataFrame(all_attributions)
    valid = df[df["attribution"] != "no_projection"]
    valid = valid[valid["attribution"] != "no_actual"]

    if valid.empty:
        return {"total": 0, "breakdown": {}}

    breakdown = {}
    for attr_type, group in valid.groupby("attribution"):
        breakdown[attr_type] = {
            "count": len(group),
            "avg_model_error": round(float(group["model_error"].mean()), 4),
            "pct_within_sigma": round(float(group["variance_component"].mean()), 4),
        }

    return {
        "total": len(valid),
        "breakdown": breakdown,
    }


# ── Data loading helpers ─────────────────────────────────────────────


def _load_projection(
    season: int, week: int, player_id: str, market: str,
) -> Optional[pd.Series]:
    query = """
    SELECT mu, sigma FROM weekly_projections
    WHERE season = ? AND week = ? AND player_id = ? AND market = ?
    """
    try:
        df = read_dataframe(query, params=(season, week, player_id, market))
        return df.iloc[0] if not df.empty else None
    except Exception:
        return None


def _load_actual_stat(
    season: int, week: int, player_id: str, market: str,
) -> Optional[float]:
    stat_col = MARKET_TO_STAT.get(market)
    if stat_col is None:
        return None

    query = f"""
    SELECT {stat_col} FROM player_stats_enhanced
    WHERE season = ? AND week = ? AND player_id = ?
    """
    try:
        df = read_dataframe(query, params=(season, week, player_id))
        if df.empty:
            return None
        val = df.iloc[0][stat_col]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _load_best_line(
    season: int, week: int, player_id: str, market: str,
) -> Optional[float]:
    query = """
    SELECT line FROM weekly_odds
    WHERE season = ? AND week = ? AND player_id = ? AND market = ?
    ORDER BY as_of DESC
    LIMIT 1
    """
    try:
        df = read_dataframe(query, params=(season, week, player_id, market))
        return float(df.iloc[0]["line"]) if not df.empty else None
    except Exception:
        return None


def _load_all_projections(season: int, week: int) -> pd.DataFrame:
    query = """
    SELECT player_id, market, mu, sigma
    FROM weekly_projections
    WHERE season = ? AND week = ?
    """
    try:
        return read_dataframe(query, params=(season, week))
    except Exception:
        return pd.DataFrame()


def _load_agent_decisions(season: int, week: int) -> pd.DataFrame:
    query = """
    SELECT player_id, market, decision, merged_confidence,
           votes, agent_reports
    FROM agent_decisions
    WHERE season = ? AND week = ?
    """
    try:
        return read_dataframe(query, params=(season, week))
    except Exception:
        return pd.DataFrame()


def _load_bet_outcomes(season: int, week: int) -> pd.DataFrame:
    query = """
    SELECT player_id, market, result, profit_units,
           confidence_tier, edge_at_placement
    FROM bet_outcomes
    WHERE season = ? AND week = ?
    """
    try:
        return read_dataframe(query, params=(season, week))
    except Exception:
        return pd.DataFrame()


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Post-game learning loop")
    sub = parser.add_subparsers(dest="command")

    learn_p = sub.add_parser("learn", help="Run learning loop for a week")
    learn_p.add_argument("--season", type=int, required=True)
    learn_p.add_argument("--week", type=int, required=True)

    report_p = sub.add_parser("report", help="Generate learning report")
    report_p.add_argument("--season", type=int, required=True)
    report_p.add_argument("--week-start", type=int, default=1)
    report_p.add_argument("--week-end", type=int, default=22)

    threshold_p = sub.add_parser("thresholds", help="Recommend threshold updates")
    threshold_p.add_argument("--season", type=int, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "learn":
        print(f"Running learning loop for {args.season} Week {args.week}")
        attributions = batch_attribute_outcomes(args.season, args.week)
        print(f"  Attributed {len(attributions)} outcomes")

        n_perf = update_agent_performance(args.season, args.week)
        print(f"  Updated {n_perf} agent performance records")

    elif args.command == "report":
        report = generate_learning_report(
            args.season,
            week_range=(args.week_start, args.week_end),
        )
        import json
        output_dir = config.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"learning_report_{args.season}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Learning report saved to {path}")
        _print_report_summary(report)

    elif args.command == "thresholds":
        recs = recommend_threshold_updates(season=args.season)
        print(f"\nCurrent thresholds: {recs['current_thresholds']}")
        print(f"\nTier performance ({len(recs['tier_performance'])} tiers):")
        for tp in recs["tier_performance"]:
            print(f"  {tp['tier']}: {tp['wins']}/{tp['total']} "
                  f"({tp['hit_rate']:.1%}), avg profit={tp['avg_profit']:.2f}")
        if recs["recommendations"]:
            print(f"\nRecommendations:")
            for rec in recs["recommendations"]:
                print(f"  [{rec['tier']}] {rec['action']}: {rec['reason']}")

    else:
        parser.print_help()


def _print_report_summary(report: Dict[str, Any]) -> None:
    """Print human-readable learning report summary."""
    print(f"\nLearning Report: Season {report['season']} "
          f"(Weeks {report['week_range'][0]}-{report['week_range'][1]})")
    print("=" * 60)

    print("\nTier Performance:")
    for tp in report.get("tier_performance", []):
        print(f"  {tp['tier']}: {tp['wins']}/{tp['total']} "
              f"({tp['hit_rate']:.1%}), profit={tp['total_profit']:.2f}")

    print("\nAgent Performance:")
    for ap in report.get("agent_performance", []):
        print(f"  {ap['agent_name']}: {ap.get('correct', 0)}/{ap.get('total', 0)} "
              f"({ap.get('accuracy', 0):.1%})")

    attr = report.get("attribution_summary", {})
    if attr.get("total", 0) > 0:
        print(f"\nAttribution Summary ({attr['total']} outcomes):")
        for atype, data in attr.get("breakdown", {}).items():
            print(f"  {atype}: {data['count']} "
                  f"(avg z-score={data['avg_model_error']:.2f})")

    recs = report.get("threshold_recommendations", {})
    if recs.get("recommendations"):
        print(f"\nThreshold Recommendations:")
        for rec in recs["recommendations"]:
            print(f"  [{rec['tier']}] {rec['action']}: {rec['reason']}")


if __name__ == "__main__":
    main()
