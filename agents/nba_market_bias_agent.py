"""NBA market bias detection agent.

Detects fg3m recency staleness: when a player's last-5 game average for
3-pointers differs significantly from their last-10 average, the market
line may lag behind the trend. Returns NEUTRAL for non-fg3m markets.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from agents import AgentReport
from agents.nba_base_agent import NbaBaseAgent
from utils.db import read_dataframe

# Threshold: last-5 vs last-10 divergence > 1.5 three-pointers
FG3M_DIVERGENCE_THRESHOLD = 1.5


class NbaMarketBiasAgent(NbaBaseAgent):
    """Detects fg3m recency staleness in NBA markets."""

    def __init__(self) -> None:
        super().__init__("nba_market_bias_agent")

    def analyze(
        self,
        game_date: str,
        player_id: Optional[int] = None,
    ) -> List[AgentReport]:
        value_df = self._load_value_view(game_date, player_id)
        if value_df.empty:
            self.logger.info("No value data for game_date=%s", game_date)
            return []

        # Get unique player/market combos
        player_markets = (
            value_df[["player_id", "market"]]
            .drop_duplicates()
            .values.tolist()
        )

        # Preload recency stats for fg3m analysis
        fg3m_stats: dict[int, dict] = {}
        fg3m_players = [
            pid for pid, mkt in player_markets if mkt == "fg3m" and pd.notna(pid)
        ]
        if fg3m_players:
            fg3m_stats = self._load_fg3m_recency(game_date, fg3m_players)

        reports: List[AgentReport] = []

        for pid, market in player_markets:
            pid_str = str(pid) if pd.notna(pid) else "unknown"

            if market != "fg3m":
                # Non-fg3m markets: always NEUTRAL
                reports.append(AgentReport(
                    agent_name=self.name,
                    recommendation="NEUTRAL",
                    confidence=0.40,
                    rationale=f"Market bias check not applicable for {pid_str} {market}.",
                    player_id=pid_str,
                    market=market,
                    data={},
                ))
                continue

            stats = fg3m_stats.get(int(pid), {}) if pd.notna(pid) else {}
            last5 = stats.get("last5_avg")
            last10 = stats.get("last10_avg")

            if last5 is None or last10 is None:
                reports.append(AgentReport(
                    agent_name=self.name,
                    recommendation="NEUTRAL",
                    confidence=0.35,
                    rationale=f"Insufficient fg3m history for {pid_str}.",
                    player_id=pid_str,
                    market=market,
                    data={"last5_avg": last5, "last10_avg": last10},
                ))
                continue

            divergence = last5 - last10

            if abs(divergence) > FG3M_DIVERGENCE_THRESHOLD:
                if divergence > 0:
                    # Player running hot — market may be stale/low
                    recommendation = "APPROVE"
                    confidence = min(0.85, 0.60 + abs(divergence) * 0.05)
                    rationale = (
                        f"fg3m recency bias for {pid_str}: "
                        f"L5={last5:.1f} > L10={last10:.1f} by {divergence:+.1f}. "
                        f"Market may be stale."
                    )
                else:
                    # Player running cold — market may be stale/high
                    recommendation = "REJECT"
                    confidence = min(0.80, 0.55 + abs(divergence) * 0.05)
                    rationale = (
                        f"fg3m cold streak for {pid_str}: "
                        f"L5={last5:.1f} < L10={last10:.1f} by {divergence:+.1f}. "
                        f"Market may not have adjusted down."
                    )
            else:
                recommendation = "NEUTRAL"
                confidence = 0.50
                rationale = (
                    f"fg3m stable for {pid_str}: "
                    f"L5={last5:.1f}, L10={last10:.1f}, divergence={divergence:+.1f}"
                )

            reports.append(AgentReport(
                agent_name=self.name,
                recommendation=recommendation,
                confidence=confidence,
                rationale=rationale,
                player_id=pid_str,
                market=market,
                data={
                    "last5_avg": last5,
                    "last10_avg": last10,
                    "divergence": divergence,
                },
            ))

        self.logger.info(
            "NbaMarketBiasAgent produced %d reports for game_date=%s",
            len(reports), game_date,
        )
        return reports

    def _load_fg3m_recency(
        self,
        game_date: str,
        player_ids: list[int],
    ) -> dict[int, dict]:
        """Load last-5 and last-10 fg3m averages for players."""
        if not player_ids:
            return {}

        placeholders = ",".join("?" * len(player_ids))
        query = (
            "SELECT player_id, "
            "AVG(CASE WHEN rn <= 5 THEN fg3m END) as last5_avg, "
            "AVG(CASE WHEN rn <= 10 THEN fg3m END) as last10_avg "
            "FROM ("
            "  SELECT player_id, fg3m, "
            "  ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) as rn "
            "  FROM nba_player_game_logs "
            f"  WHERE player_id IN ({placeholders}) AND game_date < ?"
            ") sub WHERE rn <= 10 "
            "GROUP BY player_id"
        )
        params = player_ids + [game_date]

        try:
            df = read_dataframe(query, params=params)
        except Exception as exc:
            self.logger.warning("Failed to load fg3m recency: %s", exc)
            return {}

        result: dict[int, dict] = {}
        for _, row in df.iterrows():
            pid = int(row["player_id"])
            result[pid] = {
                "last5_avg": float(row["last5_avg"]) if pd.notna(row["last5_avg"]) else None,
                "last10_avg": float(row["last10_avg"]) if pd.notna(row["last10_avg"]) else None,
            }
        return result
