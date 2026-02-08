"""Model diagnostics agent.

Investigates projection-vs-market gaps and flags players where the
model's projection deviates suspiciously from the market line. Uses
volatility scores and confidence scoring to determine which projections
are least reliable.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from agents import AgentReport
from agents.base_agent import BaseAgent

# Flag projections where |mu - line| > SUSPICIOUS_MULTIPLIER * sigma
SUSPICIOUS_MULTIPLIER = 1.5


def _flag_suspicious(
    projections: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    """Return rows where model and market disagree suspiciously.

    Joins projections and latest odds, then filters to rows where
    ``|mu - line| > 1.5 * sigma``.
    """
    if projections.empty or odds.empty:
        return pd.DataFrame()

    # Take the latest odds snapshot per player/market/sportsbook
    latest_odds = (
        odds.sort_values("as_of")
        .groupby(["player_id", "market", "sportsbook"])
        .tail(1)
    )

    # Use the best (lowest) line per player/market for comparison
    best_lines = (
        latest_odds.groupby(["player_id", "market"])["line"]
        .min()
        .reset_index()
        .rename(columns={"line": "best_line"})
    )

    merged = projections.merge(
        best_lines,
        on=["player_id", "market"],
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    merged = merged.copy()
    merged["gap"] = (merged["mu"] - merged["best_line"]).abs()
    merged["threshold"] = SUSPICIOUS_MULTIPLIER * merged["sigma"]
    suspicious = merged[merged["gap"] > merged["threshold"]]
    return suspicious.reset_index(drop=True)


class ModelDiagnosticsAgent(BaseAgent):
    """Flags projections that deviate suspiciously from market lines."""

    def __init__(self) -> None:
        super().__init__("model_diagnostics_agent")

    def analyze(
        self,
        season: int,
        week: int,
        player_id: Optional[str] = None,
    ) -> List[AgentReport]:
        projections = self._load_projections(season, week, player_id)
        odds = self._load_odds(season, week, player_id)

        if projections.empty:
            self.logger.info("No projections for season=%d week=%d", season, week)
            return []

        suspicious = _flag_suspicious(projections, odds)
        suspicious_keys = set()
        if not suspicious.empty:
            suspicious_keys = set(
                zip(suspicious["player_id"], suspicious["market"])
            )

        reports: List[AgentReport] = []
        for _, row in projections.iterrows():
            pid = row["player_id"]
            market = row["market"]
            mu = float(row["mu"])
            sigma = float(row["sigma"])
            volatility = float(row.get("volatility_score", 50.0) or 50.0)

            is_suspicious = (pid, market) in suspicious_keys

            if is_suspicious:
                sus_row = suspicious[
                    (suspicious["player_id"] == pid)
                    & (suspicious["market"] == market)
                ]
                gap = float(sus_row.iloc[0]["gap"]) if not sus_row.empty else 0
                recommendation = "REJECT"
                confidence = min(0.90, 0.60 + (gap / (sigma + 1e-6)) * 0.10)
                rationale = (
                    f"Suspicious gap for {pid} {market}: "
                    f"|mu({mu:.1f}) - line| = {gap:.1f} > "
                    f"{SUSPICIOUS_MULTIPLIER}*sigma({sigma:.1f})"
                )
            else:
                recommendation = "APPROVE"
                confidence = max(0.50, 1.0 - volatility / 100.0)
                rationale = (
                    f"Projection for {pid} {market} within expected range. "
                    f"mu={mu:.1f}, sigma={sigma:.1f}, vol={volatility:.0f}"
                )

            reports.append(AgentReport(
                agent_name=self.name,
                recommendation=recommendation,
                confidence=confidence,
                rationale=rationale,
                player_id=pid,
                market=market,
                data={
                    "mu": mu,
                    "sigma": sigma,
                    "volatility_score": volatility,
                    "is_suspicious": is_suspicious,
                },
            ))

        self.logger.info(
            "ModelDiagnosticsAgent: %d reports, %d suspicious",
            len(reports),
            len(suspicious_keys),
        )
        return reports
