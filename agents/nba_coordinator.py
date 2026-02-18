"""NBA agent coordinator: runs all NBA agents and produces consensus decisions.

Reuses sport-agnostic functions ``_group_reports`` and ``_resolve_consensus``
from the NFL coordinator, with NBA-specific agent instantiation and
persistence to ``nba_agent_decisions``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents import AgentReport, validate_report
from agents.coordinator import _group_reports, _resolve_consensus
from agents.nba_base_agent import NbaBaseAgent
from agents.nba_market_bias_agent import NbaMarketBiasAgent
from agents.nba_model_diagnostics_agent import NbaModelDiagnosticsAgent
from agents.nba_odds_agent import NbaOddsAgent
from agents.nba_risk_agent import NbaRiskAgent
from utils.db import execute, get_connection

logger = logging.getLogger(__name__)


def _persist_nba_decisions(
    decisions: List[Dict[str, Any]],
    game_date: str,
) -> int:
    """Write decisions to nba_agent_decisions. Returns row count."""
    if not decisions:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for d in decisions:
        pid = d.get("player_id") or ""
        # NBA player IDs are integers; try to convert
        try:
            pid_int: Optional[int] = int(pid)
        except (ValueError, TypeError):
            pid_int = None
        rows.append((
            game_date,
            pid_int,
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
                    INSERT OR REPLACE INTO nba_agent_decisions
                    (game_date, player_id, market, decision,
                     merged_confidence, votes, rationale, coordinator_override,
                     agent_reports, decided_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                    conn=conn,
                )
            conn.commit()
        return len(rows)
    except Exception as exc:
        logger.error("Failed to persist NBA agent decisions: %s", exc)
        return 0


def run_nba_agents(
    game_date: str,
    player_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run all NBA agents, merge reports, resolve conflicts, persist results.

    Returns list of decision dicts (one per player/market).
    """
    agents: List[NbaBaseAgent] = [
        NbaOddsAgent(),
        NbaModelDiagnosticsAgent(),
        NbaMarketBiasAgent(),
        NbaRiskAgent(),
    ]

    all_reports: List[AgentReport] = []
    for agent in agents:
        try:
            reports = agent.analyze(game_date, player_id)
            for r in reports:
                errors = validate_report(r)
                if errors:
                    logger.warning(
                        "Invalid report from %s: %s", agent.name, errors
                    )
                    continue
                all_reports.append(r)
        except Exception as exc:
            logger.error("NBA Agent %s failed: %s", agent.name, exc)

    if not all_reports:
        logger.warning("No NBA agent reports for game_date=%s", game_date)
        return []

    grouped = _group_reports(all_reports)

    decisions: List[Dict[str, Any]] = []
    for (pid, market), reports in grouped.items():
        consensus = _resolve_consensus(reports)
        decisions.append({**consensus, "player_id": pid, "market": market})

    persisted = _persist_nba_decisions(decisions, game_date)
    logger.info(
        "NBA Coordinator: %d decisions (%d approved, %d rejected), %d persisted",
        len(decisions),
        sum(1 for d in decisions if d["decision"] == "APPROVED"),
        sum(1 for d in decisions if d["decision"] == "REJECTED"),
        persisted,
    )

    return decisions


def _print_decision_summary(decisions: List[Dict[str, Any]]) -> None:
    """Log human-readable summary of NBA coordinator decisions."""
    approved = [d for d in decisions if d["decision"] == "APPROVED"]
    rejected = [d for d in decisions if d["decision"] == "REJECTED"]

    logger.info("NBA Agent Coordinator Summary")
    logger.info("=============================")
    logger.info("Total decisions: %d", len(decisions))
    logger.info("Approved: %d", len(approved))
    logger.info("Rejected: %d", len(rejected))

    if approved:
        logger.info("Approved plays:")
        for d in sorted(approved, key=lambda x: x["merged_confidence"], reverse=True):
            pid = d.get("player_id", "?")
            mkt = d.get("market", "?")
            conf = d["merged_confidence"]
            override = " [OVERRIDE]" if d["override"] else ""
            logger.info("  %s %s: conf=%.2f%s", pid, mkt, conf, override)

    if rejected:
        logger.info("Rejected plays:")
        for d in rejected[:10]:
            pid = d.get("player_id", "?")
            mkt = d.get("market", "?")
            votes = d["votes"]
            logger.info("  %s %s: votes=%s", pid, mkt, votes)


def main() -> None:
    """CLI entry point for the NBA agent coordinator."""
    import argparse

    parser = argparse.ArgumentParser(description="Run NBA agent coordinator")
    parser.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Game date to analyze",
    )
    parser.add_argument("--player-id", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    decisions = run_nba_agents(args.date, args.player_id)
    _print_decision_summary(decisions)


if __name__ == "__main__":
    main()
