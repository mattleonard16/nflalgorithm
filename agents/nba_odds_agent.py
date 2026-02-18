"""NBA odds monitoring agent.

Tracks line movement from nba_odds, detects steam moves
(large line shifts in short windows), and reports best available prices.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from agents import AgentReport
from agents.nba_base_agent import NbaBaseAgent

# NBA lines are tighter than NFL — lower threshold for steam detection
STEAM_MOVE_THRESHOLD = 1.5


def _detect_steam_moves(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Identify player/market combos with large line swings."""
    if odds_df.empty or "as_of" not in odds_df.columns:
        return pd.DataFrame()

    sorted_odds = odds_df.sort_values("as_of")
    groups = sorted_odds.groupby(["player_name", "market", "sportsbook"])
    records: List[Dict] = []
    for (pname, market, book), group in groups:
        if len(group) < 2:
            continue
        first_line = float(group.iloc[0]["line"])
        last_line = float(group.iloc[-1]["line"])
        movement = abs(last_line - first_line)
        if movement >= STEAM_MOVE_THRESHOLD:
            records.append({
                "player_name": pname,
                "market": market,
                "sportsbook": book,
                "line_open": first_line,
                "line_close": last_line,
                "movement": movement,
                "player_id": group.iloc[0].get("player_id"),
            })
    return pd.DataFrame(records)


class NbaOddsAgent(NbaBaseAgent):
    """Monitors NBA odds movement and reports best available prices."""

    def __init__(self) -> None:
        super().__init__("nba_odds_agent")

    def analyze(
        self,
        game_date: str,
        player_id: Optional[int] = None,
    ) -> List[AgentReport]:
        odds_df = self._load_odds(game_date, player_id)
        if odds_df.empty:
            self.logger.info("No NBA odds for game_date=%s", game_date)
            return []

        steam = _detect_steam_moves(odds_df)
        steam_keys = set()
        if not steam.empty:
            steam_keys = set(zip(steam["player_name"], steam["market"]))

        reports: List[AgentReport] = []

        player_markets = (
            odds_df[["player_id", "player_name", "market"]]
            .drop_duplicates()
            .values.tolist()
        )

        for pid, pname, market in player_markets:
            is_steam = (pname, market) in steam_keys

            if is_steam:
                recommendation = "REJECT"
                confidence = 0.85
                rationale = (
                    f"Steam move detected for {pname} {market}: "
                    f"line shifted >= {STEAM_MOVE_THRESHOLD} pts"
                )
            else:
                recommendation = "APPROVE"
                confidence = 0.70
                rationale = f"Line stable for {pname} {market}."

            reports.append(AgentReport(
                agent_name=self.name,
                recommendation=recommendation,
                confidence=confidence,
                rationale=rationale,
                player_id=str(pid) if pd.notna(pid) else str(pname),
                market=market,
                data={"steam_move": is_steam},
            ))

        self.logger.info(
            "NbaOddsAgent produced %d reports (%d steam moves)",
            len(reports), len(steam),
        )
        return reports
