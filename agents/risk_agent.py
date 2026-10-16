"""Risk management agent.

Wraps the risk_manager module to enforce exposure rules, detect
correlations, and report warnings about bankroll risk.

A bet breaching a team/game/player exposure cap is REJECTed; a bet that is
only part of a correlation group is NEUTRAL with the group in its rationale.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from agents import AgentReport
from agents.base_agent import BaseAgent
from risk_manager import (
    assess_risk,
    detect_team_stacks,
)


class RiskAgent(BaseAgent):
    """Enforces exposure rules and reports correlation warnings."""

    def __init__(self) -> None:
        super().__init__("risk_agent")

    def analyze(
        self,
        season: int,
        week: int,
        player_id: Optional[str] = None,
    ) -> List[AgentReport]:
        value_df = self._load_value_view(season, week, player_id)
        if value_df.empty:
            self.logger.info(
                "No value data for season=%d week=%d", season, week
            )
            return []

        # Check required columns exist
        required = {"kelly_fraction", "p_win", "price", "player_id", "market"}
        missing = required - set(value_df.columns)
        if missing:
            self.logger.warning(
                "Value view missing columns for risk assessment: %s", missing
            )
            return _fallback_reports(self.name, value_df)

        assessed = assess_risk(value_df)
        stacks = detect_team_stacks(assessed)

        reports: List[AgentReport] = []
        for _, row in assessed.iterrows():
            pid = row.get("player_id", "")
            market = row.get("market", "")
            corr_group = row.get("correlation_group")
            exposure_warn = row.get("exposure_warning")
            risk_kelly = row.get("risk_adjusted_kelly", 0)
            orig_kelly = row.get("kelly_fraction", 0)

            kelly_reduced = (
                risk_kelly < orig_kelly
                if orig_kelly > 0
                else False
            )

            # Exposure rules are enforced; correlation is reported. Every
            # correlation group sits inside one team (risk_manager requires
            # it), so correlated stake past a limit always also breaches the
            # team cap and arrives here as an exposure warning. Rejecting on
            # the label alone vetoed nearly every multi-bet card.
            if pd.notna(exposure_warn):
                recommendation = "REJECT"
                confidence = 0.80
                warnings = []
                if pd.notna(corr_group):
                    warnings.append(f"correlated({corr_group})")
                warnings.append(str(exposure_warn))
                rationale = (
                    f"Risk warnings for {pid} {market}: "
                    + "; ".join(warnings)
                )
            elif pd.notna(corr_group):
                recommendation = "NEUTRAL"
                confidence = 0.65
                rationale = (
                    f"Correlated exposure for {pid} {market}: correlated({corr_group}), "
                    "within exposure limits"
                )
                if kelly_reduced:
                    rationale += (
                        f"; Kelly reduced {orig_kelly:.4f} -> {risk_kelly:.4f} (drawdown risk)"
                    )
            elif kelly_reduced:
                recommendation = "NEUTRAL"
                confidence = 0.65
                rationale = (
                    f"Kelly reduced for {pid} {market}: "
                    f"{orig_kelly:.4f} -> {risk_kelly:.4f} (drawdown risk)"
                )
            else:
                recommendation = "APPROVE"
                confidence = 0.75
                rationale = (
                    f"No risk warnings for {pid} {market}. "
                    f"Kelly={orig_kelly:.4f}"
                )

            reports.append(AgentReport(
                agent_name=self.name,
                recommendation=recommendation,
                confidence=confidence,
                rationale=rationale,
                player_id=pid,
                market=market,
                data={
                    "correlation_group": corr_group if pd.notna(corr_group) else None,
                    "exposure_warning": exposure_warn if pd.notna(exposure_warn) else None,
                    "kelly_fraction": float(orig_kelly),
                    "risk_adjusted_kelly": float(risk_kelly),
                    "team_stacks": {
                        t: len(idxs) for t, idxs in stacks.items()
                    },
                },
            ))

        self.logger.info(
            "RiskAgent: %d reports, %d with warnings, %d team stacks",
            len(reports),
            sum(1 for r in reports if r.recommendation == "REJECT"),
            len(stacks),
        )
        return reports


def _fallback_reports(agent_name: str, df: pd.DataFrame) -> List[AgentReport]:
    """Generate neutral reports when risk assessment cannot run."""
    reports = []
    seen = set()
    for _, row in df.iterrows():
        pid = row.get("player_id", "")
        market = row.get("market", "")
        key = (pid, market)
        if key in seen:
            continue
        seen.add(key)
        reports.append(AgentReport(
            agent_name=agent_name,
            recommendation="NEUTRAL",
            confidence=0.40,
            rationale=(
                f"Insufficient data for risk assessment of {pid} {market}"
            ),
            player_id=pid,
            market=market,
            data={},
        ))
    return reports
