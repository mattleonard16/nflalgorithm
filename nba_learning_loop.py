"""NBA post-game learning loop and outcome attribution.

Tracks model accuracy, agent performance, and tunes confidence
thresholds based on historical results via Bayesian updating.

Adapted from the NFL learning_loop.py for NBA's date-based time
dimensions and NBA-specific markets (pts, reb, ast, fg3m).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.db import execute, executemany, read_dataframe

logger = logging.getLogger(__name__)

# ── NBA market-to-stat mapping ──────────────────────────────────────

NBA_MARKET_TO_STAT = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "fg3m": "fg3m",
}

# ── Hardcoded NBA defaults (no config import) ───────────────────────

NBA_MIN_EDGE_THRESHOLD = 0.08
NBA_MIN_CONFIDENCE = 0.75
NBA_CONFIDENCE_MIN_TIER = 2
NBA_REPORTS_DIR = "reports"


# ── Outcome attribution ─────────────────────────────────────────────


def _empty_nba_attribution(
    player_id: int,
    market: str,
    reason: str,
) -> Dict[str, Any]:
    """Return a stub attribution when data is missing."""
    return {
        "player_id": player_id,
        "market": market,
        "game_date": None,
        "actual": None,
        "mu": 0.0,
        "sigma": 0.0,
        "line": None,
        "model_error": 0.0,
        "line_value": None,
        "variance_component": False,
        "attribution": reason,
    }


def attribute_nba_outcome(
    game_date: str,
    player_id: int,
    market: str,
) -> Dict[str, Any]:
    """Attribute the outcome of a single NBA prediction to its components.

    Returns
    -------
    dict with keys:
        model_error      : |mu - actual| / sigma  (z-score of model miss)
        line_value        : float, positive if we beat closing line
        variance_component: bool, True if result within 1-sigma
        attribution       : str, one of "model_accurate", "model_miss", "high_variance"
        actual            : float or None
        mu                : float
        sigma             : float
        line              : float or None
    """
    projection = _load_nba_projection(game_date, player_id, market)
    if projection is None:
        return _empty_nba_attribution(player_id, market, "no_projection")

    actual = _load_nba_actual_stat(game_date, player_id, market)
    if actual is None:
        return _empty_nba_attribution(player_id, market, "no_actual")

    mu = float(projection["projected_value"])
    sigma = max(float(projection["sigma"]) if pd.notna(projection.get("sigma")) else 1.0, 0.1)
    line = _load_nba_best_line(game_date, player_id, market)

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
        "game_date": game_date,
        "actual": actual,
        "mu": mu,
        "sigma": sigma,
        "line": line,
        "model_error": round(model_error, 4),
        "line_value": round(line_value, 4) if line is not None else None,
        "variance_component": within_one_sigma,
        "attribution": attribution,
    }


def batch_attribute_nba_outcomes(
    game_date: str,
) -> List[Dict[str, Any]]:
    """Attribute outcomes for all NBA projections on a given date."""
    projections = _load_all_nba_projections(game_date)
    if projections.empty:
        logger.warning("No projections found for %s", game_date)
        return []

    results = []
    for _, proj in projections.iterrows():
        result = attribute_nba_outcome(
            game_date, int(proj["player_id"]), proj["market"]
        )
        results.append(result)

    return results


# ── Agent performance tracking ───────────────────────────────────────


def update_nba_agent_performance(
    game_date: str,
) -> int:
    """Track which agents contributed to winning vs losing NBA decisions.

    Loads agent decisions for the date, joins with bet outcomes,
    and updates the nba_agent_performance table.

    Returns the number of performance records inserted.
    """
    decisions = _load_nba_agent_decisions(game_date)
    if decisions.empty:
        logger.warning("No agent decisions for %s", game_date)
        return 0

    bet_outcomes = _load_nba_bet_outcomes(game_date)
    if bet_outcomes.empty:
        logger.warning("No bet outcomes for %s", game_date)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    records: List[Tuple] = []

    for _, dec in decisions.iterrows():
        pid = int(dec["player_id"])
        market = dec["market"]
        decision = dec["decision"]

        matching = bet_outcomes[
            (bet_outcomes["player_id"] == pid)
            & (bet_outcomes["market"] == market)
        ]

        if matching.empty:
            continue

        outcome_row = matching.iloc[0]
        result = outcome_row.get("result", "unknown")
        profit = float(outcome_row.get("profit_units", 0.0))

        agent_reports_raw = dec.get("agent_reports", "[]")
        try:
            agent_reports = (
                json.loads(agent_reports_raw)
                if isinstance(agent_reports_raw, str)
                else agent_reports_raw
            )
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
                game_date,
                agent_name,
                pid,
                market,
                recommendation,
                confidence,
                decision,
                result,
                profit,
                int(correct),
                now,
            ))

    if not records:
        return 0

    insert_sql = """
    INSERT OR REPLACE INTO nba_agent_performance (
        game_date, agent_name, player_id, market,
        recommendation, confidence, final_decision, outcome,
        profit_units, correct, recorded_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    executemany(insert_sql, records)
    logger.info(
        "Inserted %d agent performance records for %s",
        len(records),
        game_date,
    )
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


def get_nba_agent_accuracy_summary(
    n_days: int = 30,
) -> pd.DataFrame:
    """Summarize NBA agent accuracy over recent days.

    Returns DataFrame with columns: agent_name, total, correct, accuracy, avg_confidence.
    """
    query = """
    SELECT agent_name,
           COUNT(*) as total,
           SUM(correct) as correct,
           ROUND(CAST(SUM(correct) AS REAL) / COUNT(*), 4) as accuracy,
           ROUND(AVG(confidence), 4) as avg_confidence
    FROM nba_agent_performance
    GROUP BY agent_name
    ORDER BY accuracy DESC
    """
    try:
        return read_dataframe(query)
    except Exception:
        return pd.DataFrame(
            columns=["agent_name", "total", "correct", "accuracy", "avg_confidence"]
        )


# ── Confidence threshold tightening ──────────────────────────────────


def recommend_nba_threshold_updates(
    n_days: int = 30,
) -> Dict[str, Any]:
    """Bayesian update of confidence thresholds based on actual NBA outcomes.

    Analyzes bet outcomes by confidence tier and recommends raising
    thresholds for tiers with poor performance.

    Returns
    -------
    dict with:
        tier_performance   : list of per-tier stats
        recommendations    : list of suggested config changes
        current_thresholds : current hardcoded NBA defaults
    """
    query = """
    SELECT confidence_tier, result, profit_units, edge_at_placement
    FROM nba_bet_outcomes
    WHERE result IN ('win', 'loss')
    """
    try:
        df = read_dataframe(query)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return {
            "tier_performance": [],
            "recommendations": [],
            "current_thresholds": _nba_current_thresholds(),
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
        "current_thresholds": _nba_current_thresholds(),
    }


def _nba_current_thresholds() -> Dict[str, Any]:
    """Return current NBA betting/confidence thresholds (hardcoded defaults)."""
    return {
        "min_edge_threshold": NBA_MIN_EDGE_THRESHOLD,
        "min_confidence": NBA_MIN_CONFIDENCE,
        "confidence_min_tier": NBA_CONFIDENCE_MIN_TIER,
    }


# ── Injury edge performance ─────────────────────────────────────────


def _injury_edge_performance(
    n_days: int = 30,
) -> Dict[str, Any]:
    """Measure whether injury-boosted bets generate alpha.

    Queries bets where injury_boost_multiplier > 1.0 from
    nba_materialized_value_view and compares hit rate and profit
    vs non-injury bets.

    Returns
    -------
    dict with injury_bets and non_injury_bets performance stats.
    """
    # Get bets that had injury boost
    injury_query = """
    SELECT bo.result, bo.profit_units, bo.edge_at_placement
    FROM nba_bet_outcomes bo
    INNER JOIN nba_materialized_value_view mv
        ON bo.player_id = mv.player_id
        AND bo.market = mv.market
        AND bo.game_date = mv.game_date
        AND bo.sportsbook = mv.sportsbook
    WHERE bo.result IN ('win', 'loss')
      AND mv.injury_boost_multiplier > 1.0
    """

    non_injury_query = """
    SELECT bo.result, bo.profit_units, bo.edge_at_placement
    FROM nba_bet_outcomes bo
    INNER JOIN nba_materialized_value_view mv
        ON bo.player_id = mv.player_id
        AND bo.market = mv.market
        AND bo.game_date = mv.game_date
        AND bo.sportsbook = mv.sportsbook
    WHERE bo.result IN ('win', 'loss')
      AND (mv.injury_boost_multiplier IS NULL OR mv.injury_boost_multiplier <= 1.0)
    """

    def _summarize(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"total": 0, "wins": 0, "hit_rate": 0.0, "avg_profit": 0.0, "total_profit": 0.0}
        total = len(df)
        wins = int((df["result"] == "win").sum())
        return {
            "total": total,
            "wins": wins,
            "hit_rate": round(wins / total, 4) if total > 0 else 0.0,
            "avg_profit": round(float(df["profit_units"].mean()), 4),
            "total_profit": round(float(df["profit_units"].sum()), 4),
        }

    try:
        injury_df = read_dataframe(injury_query)
    except Exception:
        injury_df = pd.DataFrame()

    try:
        non_injury_df = read_dataframe(non_injury_query)
    except Exception:
        non_injury_df = pd.DataFrame()

    injury_stats = _summarize(injury_df)
    non_injury_stats = _summarize(non_injury_df)

    alpha = injury_stats["hit_rate"] - non_injury_stats["hit_rate"]

    return {
        "injury_bets": injury_stats,
        "non_injury_bets": non_injury_stats,
        "injury_alpha": round(alpha, 4),
        "injury_generates_alpha": alpha > 0 and injury_stats["total"] >= 5,
    }


# ── Learning report ──────────────────────────────────────────────────


def generate_nba_learning_report(
    game_date_start: str,
    game_date_end: str,
) -> Dict[str, Any]:
    """Comprehensive NBA learning report for a date range.

    Includes:
        - Performance by confidence tier
        - Performance by agent recommendation
        - Overall model accuracy trends
        - Threshold recommendations
        - Injury edge analysis
    """
    # Tier performance
    tier_perf = _nba_tier_performance(game_date_start, game_date_end)

    # Agent performance
    agent_summary = get_nba_agent_accuracy_summary()

    # Model accuracy trends
    accuracy_trends = _nba_model_accuracy_trends(game_date_start, game_date_end)

    # Threshold recommendations
    threshold_recs = recommend_nba_threshold_updates()

    # Attribution summary
    attributions = _nba_attribution_summary(game_date_start, game_date_end)

    # Injury edge analysis
    injury_perf = _injury_edge_performance()

    return {
        "date_range": [game_date_start, game_date_end],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier_performance": tier_perf,
        "agent_performance": (
            agent_summary.to_dict(orient="records") if not agent_summary.empty else []
        ),
        "model_accuracy_trends": accuracy_trends,
        "threshold_recommendations": threshold_recs,
        "attribution_summary": attributions,
        "injury_edge_analysis": injury_perf,
    }


def _nba_tier_performance(
    game_date_start: str,
    game_date_end: str,
) -> List[Dict[str, Any]]:
    """Performance breakdown by confidence tier for NBA bets."""
    query = """
    SELECT confidence_tier, result, profit_units, edge_at_placement
    FROM nba_bet_outcomes
    WHERE game_date BETWEEN ? AND ?
      AND result IN ('win', 'loss')
    """
    try:
        df = read_dataframe(query, params=(game_date_start, game_date_end))
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


def _nba_model_accuracy_trends(
    game_date_start: str,
    game_date_end: str,
) -> List[Dict[str, Any]]:
    """Daily model accuracy (MAE) trends for NBA markets."""
    results = []
    for market, stat_col in NBA_MARKET_TO_STAT.items():
        query = f"""
        SELECT p.game_date, p.market,
               AVG(ABS(p.projected_value - g.{stat_col})) as mae,
               COUNT(*) as n
        FROM nba_projections p
        INNER JOIN nba_player_game_logs g
            ON p.player_id = g.player_id
            AND p.game_date = g.game_date
        WHERE p.game_date BETWEEN ? AND ?
          AND p.market = ?
        GROUP BY p.game_date, p.market
        ORDER BY p.game_date
        """
        try:
            df = read_dataframe(query, params=(game_date_start, game_date_end, market))
            for _, row in df.iterrows():
                results.append({
                    "game_date": row["game_date"],
                    "market": market,
                    "mae": round(float(row["mae"]), 2),
                    "sample_size": int(row["n"]),
                })
        except Exception:
            continue

    return results


def _nba_attribution_summary(
    game_date_start: str,
    game_date_end: str,
) -> Dict[str, Any]:
    """Summarize NBA outcome attributions across a date range."""
    # Get all distinct game dates with projections
    dates_query = """
    SELECT DISTINCT game_date
    FROM nba_projections
    WHERE game_date BETWEEN ? AND ?
    ORDER BY game_date
    """
    try:
        dates_df = read_dataframe(dates_query, params=(game_date_start, game_date_end))
    except Exception:
        return {"total": 0, "breakdown": {}}

    all_attributions: List[Dict[str, Any]] = []
    for _, date_row in dates_df.iterrows():
        gd = date_row["game_date"]
        attributions = batch_attribute_nba_outcomes(gd)
        all_attributions.extend(attributions)

    if not all_attributions:
        return {"total": 0, "breakdown": {}}

    df = pd.DataFrame(all_attributions)
    valid = df[~df["attribution"].isin(["no_projection", "no_actual"])]

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


def _load_nba_projection(
    game_date: str,
    player_id: int,
    market: str,
) -> Optional[pd.Series]:
    query = """
    SELECT projected_value, sigma
    FROM nba_projections
    WHERE game_date = ? AND player_id = ? AND market = ?
    """
    try:
        df = read_dataframe(query, params=(game_date, player_id, market))
        return df.iloc[0] if not df.empty else None
    except Exception:
        return None


def _load_nba_actual_stat(
    game_date: str,
    player_id: int,
    market: str,
) -> Optional[float]:
    stat_col = NBA_MARKET_TO_STAT.get(market)
    if stat_col is None:
        return None

    query = f"""
    SELECT {stat_col}
    FROM nba_player_game_logs
    WHERE game_date = ? AND player_id = ?
    """
    try:
        df = read_dataframe(query, params=(game_date, player_id))
        if df.empty:
            return None
        val = df.iloc[0][stat_col]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _load_nba_best_line(
    game_date: str,
    player_id: int,
    market: str,
) -> Optional[float]:
    query = """
    SELECT line FROM nba_odds
    WHERE game_date = ? AND player_id = ? AND market = ?
    ORDER BY as_of DESC
    LIMIT 1
    """
    try:
        df = read_dataframe(query, params=(game_date, player_id, market))
        return float(df.iloc[0]["line"]) if not df.empty else None
    except Exception:
        return None


def _load_all_nba_projections(game_date: str) -> pd.DataFrame:
    query = """
    SELECT player_id, market, projected_value, sigma
    FROM nba_projections
    WHERE game_date = ?
    """
    try:
        return read_dataframe(query, params=(game_date,))
    except Exception:
        return pd.DataFrame()


def _load_nba_agent_decisions(game_date: str) -> pd.DataFrame:
    query = """
    SELECT player_id, market, decision, merged_confidence,
           votes, agent_reports
    FROM nba_agent_decisions
    WHERE game_date = ?
    """
    try:
        return read_dataframe(query, params=(game_date,))
    except Exception:
        return pd.DataFrame()


def _load_nba_bet_outcomes(game_date: str) -> pd.DataFrame:
    query = """
    SELECT player_id, market, result, profit_units,
           confidence_tier, edge_at_placement
    FROM nba_bet_outcomes
    WHERE game_date = ?
    """
    try:
        return read_dataframe(query, params=(game_date,))
    except Exception:
        return pd.DataFrame()


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="NBA post-game learning loop")
    sub = parser.add_subparsers(dest="command")

    learn_p = sub.add_parser("learn", help="Run learning loop for a game date")
    learn_p.add_argument("--date", type=str, required=True, help="Game date YYYY-MM-DD")

    report_p = sub.add_parser("report", help="Generate learning report")
    report_p.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    report_p.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")

    threshold_p = sub.add_parser("thresholds", help="Recommend threshold updates")
    threshold_p.add_argument("--days", type=int, default=30, help="Number of days to analyze")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "learn":
        print(f"Running NBA learning loop for {args.date}")
        attributions = batch_attribute_nba_outcomes(args.date)
        print(f"  Attributed {len(attributions)} outcomes")

        n_perf = update_nba_agent_performance(args.date)
        print(f"  Updated {n_perf} agent performance records")

    elif args.command == "report":
        report = generate_nba_learning_report(args.start_date, args.end_date)
        output_dir = Path(NBA_REPORTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"nba_learning_report_{args.start_date}_{args.end_date}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"NBA learning report saved to {path}")
        _print_nba_report_summary(report)

    elif args.command == "thresholds":
        recs = recommend_nba_threshold_updates(n_days=args.days)
        print(f"\nCurrent thresholds: {recs['current_thresholds']}")
        print(f"\nTier performance ({len(recs['tier_performance'])} tiers):")
        for tp in recs["tier_performance"]:
            print(
                f"  {tp['tier']}: {tp['wins']}/{tp['total']} "
                f"({tp['hit_rate']:.1%}), avg profit={tp['avg_profit']:.2f}"
            )
        if recs["recommendations"]:
            print("\nRecommendations:")
            for rec in recs["recommendations"]:
                print(f"  [{rec['tier']}] {rec['action']}: {rec['reason']}")

        # Also show injury edge performance
        injury_perf = _injury_edge_performance(n_days=args.days)
        print(f"\nInjury Edge Analysis:")
        print(f"  Injury bets: {injury_perf['injury_bets']}")
        print(f"  Non-injury bets: {injury_perf['non_injury_bets']}")
        print(f"  Injury alpha: {injury_perf['injury_alpha']:.1%}")
        print(f"  Generates alpha: {injury_perf['injury_generates_alpha']}")

    else:
        parser.print_help()


def _print_nba_report_summary(report: Dict[str, Any]) -> None:
    """Print human-readable NBA learning report summary."""
    print(
        f"\nNBA Learning Report: {report['date_range'][0]} "
        f"to {report['date_range'][1]}"
    )
    print("=" * 60)

    print("\nTier Performance:")
    for tp in report.get("tier_performance", []):
        print(
            f"  {tp['tier']}: {tp['wins']}/{tp['total']} "
            f"({tp['hit_rate']:.1%}), profit={tp['total_profit']:.2f}"
        )

    print("\nAgent Performance:")
    for ap in report.get("agent_performance", []):
        print(
            f"  {ap['agent_name']}: {ap.get('correct', 0)}/{ap.get('total', 0)} "
            f"({ap.get('accuracy', 0):.1%})"
        )

    attr = report.get("attribution_summary", {})
    if attr.get("total", 0) > 0:
        print(f"\nAttribution Summary ({attr['total']} outcomes):")
        for atype, data in attr.get("breakdown", {}).items():
            print(
                f"  {atype}: {data['count']} "
                f"(avg z-score={data['avg_model_error']:.2f})"
            )

    recs = report.get("threshold_recommendations", {})
    if recs.get("recommendations"):
        print("\nThreshold Recommendations:")
        for rec in recs["recommendations"]:
            print(f"  [{rec['tier']}] {rec['action']}: {rec['reason']}")

    injury = report.get("injury_edge_analysis", {})
    if injury:
        print(f"\nInjury Edge Analysis:")
        inj = injury.get("injury_bets", {})
        non_inj = injury.get("non_injury_bets", {})
        print(
            f"  Injury bets: {inj.get('total', 0)} bets, "
            f"hit rate={inj.get('hit_rate', 0):.1%}"
        )
        print(
            f"  Non-injury bets: {non_inj.get('total', 0)} bets, "
            f"hit rate={non_inj.get('hit_rate', 0):.1%}"
        )
        print(f"  Injury alpha: {injury.get('injury_alpha', 0):.1%}")


if __name__ == "__main__":
    main()
