"""NBA model diagnostics agent.

Flags players where the model's projection deviates suspiciously
from the market line, using an inline sigma estimate.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from agents import AgentReport
from agents.nba_base_agent import NbaBaseAgent

# Flag projections where |projected_value - line| > SUSPICIOUS_MULTIPLIER * sigma
SUSPICIOUS_MULTIPLIER = 1.5


class NbaModelDiagnosticsAgent(NbaBaseAgent):
    """Flags projections that deviate suspiciously from NBA market lines."""

    def __init__(self) -> None:
        super().__init__("nba_model_diagnostics_agent")

    def analyze(
        self,
        game_date: str,
        player_id: Optional[int] = None,
    ) -> List[AgentReport]:
        projections = self._load_projections(game_date, player_id)
        odds = self._load_odds(game_date, player_id)

        if projections.empty:
            self.logger.info("No projections for game_date=%s", game_date)
            return []

        # Build best lines per player/market from odds
        best_lines: dict[tuple, float] = {}
        if not odds.empty and "player_id" in odds.columns:
            for (pid, mkt), grp in odds.groupby(["player_id", "market"]):
                if pd.notna(pid):
                    best_lines[(pid, mkt)] = float(grp["line"].min())

        reports: List[AgentReport] = []
        suspicious_count = 0

        for _, row in projections.iterrows():
            pid = row["player_id"]
            market = str(row["market"])
            projected = float(row["projected_value"])

            # Inline sigma estimate (same as nba_value_engine)
            sigma = max(projected * 0.20, 3.0)

            line = best_lines.get((pid, market))
            if line is None:
                # No odds to compare — approve with lower confidence
                reports.append(AgentReport(
                    agent_name=self.name,
                    recommendation="APPROVE",
                    confidence=0.50,
                    rationale=f"No odds for {pid} {market}. proj={projected:.1f}",
                    player_id=str(pid),
                    market=market,
                    data={"projected_value": projected, "sigma": sigma, "is_suspicious": False},
                ))
                continue

            gap = abs(projected - line)
            threshold = SUSPICIOUS_MULTIPLIER * sigma
            is_suspicious = gap > threshold

            if is_suspicious:
                suspicious_count += 1
                confidence = min(0.90, 0.60 + (gap / (sigma + 1e-6)) * 0.10)
                reports.append(AgentReport(
                    agent_name=self.name,
                    recommendation="REJECT",
                    confidence=confidence,
                    rationale=(
                        f"Suspicious gap for {pid} {market}: "
                        f"|proj({projected:.1f}) - line({line:.1f})| = {gap:.1f} > "
                        f"{SUSPICIOUS_MULTIPLIER}*sigma({sigma:.1f})"
                    ),
                    player_id=str(pid),
                    market=market,
                    data={
                        "projected_value": projected, "line": line,
                        "gap": gap, "sigma": sigma, "is_suspicious": True,
                    },
                ))
            else:
                reports.append(AgentReport(
                    agent_name=self.name,
                    recommendation="APPROVE",
                    confidence=max(0.50, 0.80 - gap / (sigma + 1e-6) * 0.10),
                    rationale=(
                        f"Projection for {pid} {market} within range. "
                        f"proj={projected:.1f}, line={line:.1f}, sigma={sigma:.1f}"
                    ),
                    player_id=str(pid),
                    market=market,
                    data={
                        "projected_value": projected, "line": line,
                        "gap": gap, "sigma": sigma, "is_suspicious": False,
                    },
                ))

        self.logger.info(
            "NbaModelDiagnosticsAgent: %d reports, %d suspicious",
            len(reports), suspicious_count,
        )
        return reports
