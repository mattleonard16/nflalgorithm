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

from agents import AgentReport, validate_report
from agents.base_agent import BaseAgent
from agents.market_bias_agent import MarketBiasAgent
from agents.model_diagnostics_agent import ModelDiagnosticsAgent
from agents.odds_agent import OddsAgent
from agents.risk_agent import RiskAgent
from utils.db import execute, get_backend, get_connection

logger = logging.getLogger(__name__)

# Minimum number of agents that must agree for consensus approval
CONSENSUS_THRESHOLD = 3

# Lower is more cautious. An agent that rejects one report on a prop and
# approves another has not approved the prop.
_CAUTION = {"REJECT": 0, "NEUTRAL": 1, "APPROVE": 2}


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


def _one_vote_per_agent(reports: List[AgentReport]) -> List[AgentReport]:
    """Keep each agent's most cautious report, so the threshold counts agents.

    The risk agent reads the value card, which can price one bet at several
    books and list both sides of a prop. Each row used to be its own vote.
    """
    kept: Dict[str, AgentReport] = {}
    for r in reports:
        current = kept.get(r.agent_name)
        if current is None or _CAUTION[r.recommendation] < _CAUTION[current.recommendation]:
            kept[r.agent_name] = r
    return list(kept.values())


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

    for r in _one_vote_per_agent(reports):
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
    *,
    replace_week: bool = False,
) -> int:
    """Write decisions to the agent_decisions table. Returns row count.

    With ``replace_week`` the week's earlier verdicts go first, in the same
    transaction, so a bet that left the card does not keep a stale verdict.
    """
    if not decisions and not replace_week:
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

    # `INSERT OR REPLACE` is SQLite-only; MySQL rejected every write here.
    # Same upsert on the same primary key, spelled per backend.
    columns = """
        season, week, player_id, market, decision, merged_confidence, votes,
        rationale, coordinator_override, agent_reports, decided_at
    """
    if get_backend() == "mysql":
        sql = f"""
            INSERT INTO agent_decisions ({columns})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                decision=VALUES(decision),
                merged_confidence=VALUES(merged_confidence),
                votes=VALUES(votes),
                rationale=VALUES(rationale),
                coordinator_override=VALUES(coordinator_override),
                agent_reports=VALUES(agent_reports),
                decided_at=VALUES(decided_at)
        """
    else:
        sql = f"""
            INSERT INTO agent_decisions ({columns})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(season, week, player_id, market) DO UPDATE SET
                decision=excluded.decision,
                merged_confidence=excluded.merged_confidence,
                votes=excluded.votes,
                rationale=excluded.rationale,
                coordinator_override=excluded.coordinator_override,
                agent_reports=excluded.agent_reports,
                decided_at=excluded.decided_at
        """

    try:
        with get_connection() as conn:
            if replace_week:
                execute(
                    "DELETE FROM agent_decisions WHERE season = ? AND week = ?",
                    (season, week),
                    conn=conn,
                )
            for row in rows:
                execute(sql, row, conn=conn)
            conn.commit()
        return len(rows)
    except Exception as exc:
        logger.error("Failed to persist agent decisions: %s", exc)
        return 0


def run_all_agents(
    season: int,
    week: int,
    player_id: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
    attempt: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run all agents, merge reports, resolve conflicts, persist results.

    Parameters
    ----------
    season : int
    week : int
    player_id : str, optional
        Limit analysis to a single player.
    run_id, attempt : optional
        A durable run's attempt. Agents then judge the card it staged
        instead of the published one.

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
        RiskAgent(run_id=run_id, attempt=attempt),
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

    decisions: List[Dict[str, Any]] = []
    for (pid, market), reports in _group_reports(all_reports).items():
        consensus = _resolve_consensus(reports)
        consensus["player_id"] = pid
        consensus["market"] = market
        decisions.append(consensus)

    # A single-player run must not wipe the rest of the week's verdicts.
    persisted = _persist_decisions(
        decisions, season, week, replace_week=player_id is None
    )
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
