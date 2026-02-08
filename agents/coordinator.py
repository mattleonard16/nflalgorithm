"""Agent coordinator: runs all agents and produces consensus decisions.

The coordinator executes each specialized agent, merges their reports
per player/market combination, resolves conflicts using consensus logic,
and persists final decisions to the ``agent_decisions`` table.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agents import AgentReport, validate_report
from agents.base_agent import BaseAgent
from agents.market_bias_agent import MarketBiasAgent
from agents.model_diagnostics_agent import ModelDiagnosticsAgent
from agents.odds_agent import OddsAgent
from agents.risk_agent import RiskAgent
from utils.db import execute, get_connection

logger = logging.getLogger(__name__)

# Minimum number of agents that must agree for consensus approval
CONSENSUS_THRESHOLD = 3
# Total number of voting agents
TOTAL_AGENTS = 4


def _group_reports(
    all_reports: List[AgentReport],
) -> Dict[Tuple[Optional[str], Optional[str]], List[AgentReport]]:
    """Group reports by (player_id, market) key."""
    groups: Dict[Tuple[Optional[str], Optional[str]], List[AgentReport]] = (
        defaultdict(list)
    )
    for report in all_reports:
        key = (report.player_id, report.market)
        groups[key].append(report)
    return dict(groups)


def _resolve_consensus(
    reports: List[AgentReport],
) -> Dict[str, Any]:
    """Resolve a set of reports for one player/market into a decision.

    Returns a dict with:
        decision: "APPROVED" | "REJECTED"
        merged_confidence: weighted average confidence
        votes: {APPROVE: n, REJECT: n, NEUTRAL: n}
        rationale: merged explanation
        override: bool (coordinator override applied)
        agent_reports: list of per-agent summaries
    """
    votes: Dict[str, int] = {"APPROVE": 0, "REJECT": 0, "NEUTRAL": 0}
    weighted_conf_sum = 0.0
    total_conf = 0.0
    rationale_parts: List[str] = []
    agent_summaries: List[Dict[str, Any]] = []

    for r in reports:
        votes[r.recommendation] = votes.get(r.recommendation, 0) + 1
        weighted_conf_sum += r.confidence
        total_conf += 1.0
        rationale_parts.append(f"[{r.agent_name}] {r.rationale}")
        agent_summaries.append({
            "agent": r.agent_name,
            "recommendation": r.recommendation,
            "confidence": r.confidence,
        })

    approve_count = votes["APPROVE"]
    reject_count = votes["REJECT"]

    merged_confidence = (
        weighted_conf_sum / total_conf if total_conf > 0 else 0.0
    )

    # Consensus: approved if >= CONSENSUS_THRESHOLD agents approve
    override = False
    if approve_count >= CONSENSUS_THRESHOLD:
        decision = "APPROVED"
    elif reject_count >= CONSENSUS_THRESHOLD:
        decision = "REJECTED"
    else:
        # No clear consensus -- coordinator tiebreak
        # Approve if more approvals than rejections and merged confidence > 0.6
        if approve_count > reject_count and merged_confidence > 0.6:
            decision = "APPROVED"
            override = True
        else:
            decision = "REJECTED"
            override = True

    return {
        "decision": decision,
        "merged_confidence": round(merged_confidence, 4),
        "votes": votes,
        "rationale": " | ".join(rationale_parts),
        "override": override,
        "agent_reports": agent_summaries,
    }


def _persist_decisions(
    decisions: List[Dict[str, Any]],
    season: int,
    week: int,
) -> int:
    """Write decisions to the agent_decisions table. Returns row count."""
    if not decisions:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for d in decisions:
        rows.append((
            season,
            week,
            d.get("player_id") or "",
            d.get("market") or "",
            d["decision"],
            d["merged_confidence"],
            json.dumps(d["votes"]),
            d["rationale"][:2000],
            int(d["override"]),
            json.dumps(d["agent_reports"]),
            now,
        ))

    try:
        with get_connection() as conn:
            for row in rows:
                execute(
                    """
                    INSERT OR REPLACE INTO agent_decisions
                    (season, week, player_id, market, decision,
                     merged_confidence, votes, rationale, coordinator_override,
                     agent_reports, decided_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                    conn=conn,
                )
            conn.commit()
        return len(rows)
    except Exception as exc:
        logger.error("Failed to persist agent decisions: %s", exc)
        return 0


def run_all_agents(
    season: int,
    week: int,
    player_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run all agents, merge reports, resolve conflicts, persist results.

    Parameters
    ----------
    season : int
    week : int
    player_id : str, optional
        Limit analysis to a single player.

    Returns
    -------
    list of dict
        One decision dict per player/market with keys: player_id, market,
        decision, merged_confidence, votes, rationale, override,
        agent_reports.
    """
    agents: List[BaseAgent] = [
        OddsAgent(),
        ModelDiagnosticsAgent(),
        MarketBiasAgent(),
        RiskAgent(),
    ]

    all_reports: List[AgentReport] = []
    for agent in agents:
        try:
            reports = agent.analyze(season, week, player_id)
            # Validate each report
            for r in reports:
                errors = validate_report(r)
                if errors:
                    logger.warning(
                        "Invalid report from %s: %s", agent.name, errors
                    )
                    continue
                all_reports.append(r)
        except Exception as exc:
            logger.error("Agent %s failed: %s", agent.name, exc)

    if not all_reports:
        logger.warning("No agent reports produced for s=%d w=%d", season, week)
        return []

    grouped = _group_reports(all_reports)

    decisions: List[Dict[str, Any]] = []
    for (pid, market), reports in grouped.items():
        consensus = _resolve_consensus(reports)
        consensus["player_id"] = pid
        consensus["market"] = market
        decisions.append(consensus)

    persisted = _persist_decisions(decisions, season, week)
    logger.info(
        "Coordinator: %d decisions (%d approved, %d rejected), %d persisted",
        len(decisions),
        sum(1 for d in decisions if d["decision"] == "APPROVED"),
        sum(1 for d in decisions if d["decision"] == "REJECTED"),
        persisted,
    )

    return decisions


def _print_decision_summary(decisions: List[Dict[str, Any]]) -> None:
    """Print human-readable summary of coordinator decisions."""
    approved = [d for d in decisions if d["decision"] == "APPROVED"]
    rejected = [d for d in decisions if d["decision"] == "REJECTED"]

    print(f"\nAgent Coordinator Summary")
    print(f"========================")
    print(f"Total decisions: {len(decisions)}")
    print(f"Approved: {len(approved)}")
    print(f"Rejected: {len(rejected)}")

    if approved:
        print(f"\nApproved plays:")
        for d in sorted(approved, key=lambda x: x["merged_confidence"], reverse=True):
            pid = d.get("player_id", "?")
            mkt = d.get("market", "?")
            conf = d["merged_confidence"]
            override = " [OVERRIDE]" if d["override"] else ""
            print(f"  {pid} {mkt}: conf={conf:.2f}{override}")

    if rejected:
        print(f"\nRejected plays:")
        for d in rejected[:10]:
            pid = d.get("player_id", "?")
            mkt = d.get("market", "?")
            votes = d["votes"]
            print(f"  {pid} {mkt}: votes={votes}")


def main() -> None:
    """CLI entry point for the agent coordinator."""
    import argparse

    parser = argparse.ArgumentParser(description="Run agent coordinator")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--player-id", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    decisions = run_all_agents(args.season, args.week, args.player_id)
    _print_decision_summary(decisions)


if __name__ == "__main__":
    main()
