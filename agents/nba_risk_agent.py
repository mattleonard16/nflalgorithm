"""NBA risk management agent.

Wraps ``nba_risk_manager`` to enforce exposure rules, detect
correlations, and report warnings about bankroll risk for NBA bets.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from agents import AgentReport
from agents.nba_base_agent import NbaBaseAgent
from config import config
from nba_risk_manager import assess_risk, detect_team_stacks


class NbaRiskAgent(NbaBaseAgent):
    """Enforces NBA exposure rules and reports correlation warnings."""

    def __init__(self) -> None:
        super().__init__("nba_risk_agent")

    def analyze(
        self,
        game_date: str,
        player_id: Optional[int] = None,
    ) -> List[AgentReport]:
        value_df = self._load_value_view(game_date, player_id)
        if value_df.empty:
            self.logger.info("No NBA value data for game_date=%s", game_date)
            return []

        # Check required columns
        required = {"kelly_fraction", "p_win", "over_price", "player_id", "market"}
        missing = required - set(value_df.columns)
        if missing:
            self.logger.warning(
                "NBA value view missing columns for risk assessment: %s", missing
            )
            return _fallback_reports(self.name, value_df)

        # Derive stake for exposure calculation
        bankroll = config.betting.bankroll
        value_df = value_df.assign(stake=value_df["kelly_fraction"] * bankroll)

        assessed = assess_risk(value_df, bankroll)
        stacks = detect_team_stacks(assessed)

        reports: List[AgentReport] = []
        for _, row in assessed.iterrows():
            pid = row.get("player_id")
            pid_str = str(pid) if pd.notna(pid) else ""
            market = str(row.get("market", ""))
            corr_group = row.get("correlation_group")
            exposure_warn = row.get("exposure_warning")
            risk_kelly = row.get("risk_adjusted_kelly", 0)
            orig_kelly = row.get("kelly_fraction", 0)

            has_warning = pd.notna(corr_group) or pd.notna(exposure_warn)
            kelly_reduced = risk_kelly < orig_kelly if orig_kelly > 0 else False

            if has_warning:
                recommendation = "REJECT"
                confidence = 0.80
                warnings = []
                if pd.notna(corr_group):
                    warnings.append(f"correlated({corr_group})")
                if pd.notna(exposure_warn):
                    warnings.append(str(exposure_warn))
                rationale = (
                    f"Risk warnings for {pid_str} {market}: "
                    + "; ".join(warnings)
                )
            elif kelly_reduced:
                recommendation = "NEUTRAL"
                confidence = 0.65
                rationale = (
                    f"Kelly reduced for {pid_str} {market}: "
                    f"{orig_kelly:.4f} -> {risk_kelly:.4f} (drawdown risk)"
                )
            else:
                recommendation = "APPROVE"
                confidence = 0.75
                rationale = (
                    f"No risk warnings for {pid_str} {market}. "
                    f"Kelly={orig_kelly:.4f}"
                )

            reports.append(AgentReport(
                agent_name=self.name,
                recommendation=recommendation,
                confidence=confidence,
                rationale=rationale,
                player_id=pid_str,
                market=market,
                data={
                    "correlation_group": corr_group if pd.notna(corr_group) else None,
                    "exposure_warning": exposure_warn if pd.notna(exposure_warn) else None,
                    "kelly_fraction": float(orig_kelly),
                    "risk_adjusted_kelly": float(risk_kelly),
                    "team_stacks": {t: len(idxs) for t, idxs in stacks.items()},
                },
            ))

        self.logger.info(
            "NbaRiskAgent: %d reports, %d with warnings, %d team stacks",
            len(reports),
            sum(1 for r in reports if r.recommendation == "REJECT"),
            len(stacks),
        )
        return reports


def _fallback_reports(agent_name: str, df: pd.DataFrame) -> List[AgentReport]:
    """Generate neutral reports when risk assessment cannot run."""
    reports = []
    seen: set = set()
    for _, row in df.iterrows():
        pid = row.get("player_id", "")
        market = row.get("market", "")
        key = (str(pid), market)
        if key in seen:
            continue
        seen.add(key)
        reports.append(AgentReport(
            agent_name=agent_name,
            recommendation="NEUTRAL",
            confidence=0.40,
            rationale=f"Insufficient data for risk assessment of {pid} {market}",
            player_id=str(pid),
            market=market,
            data={},
        ))
    return reports
