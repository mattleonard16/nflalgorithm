"""Odds monitoring agent.

Tracks line movement from the weekly_odds table, detects steam moves
(large line shifts in short windows), and reports best available prices
per player/market.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from agents import AgentReport
from agents.base_agent import BaseAgent

# A line movement exceeding this threshold (in stat points) is a steam move.
STEAM_MOVE_THRESHOLD = 2.0


def _detect_steam_moves(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Identify player/market combos with large line swings.

    Returns a DataFrame of steam move records with columns:
        player_id, market, sportsbook, line_open, line_close, movement
    """
    if odds_df.empty or "as_of" not in odds_df.columns:
        return pd.DataFrame()

    sorted_odds = odds_df.sort_values("as_of")

    groups = sorted_odds.groupby(["player_id", "market", "sportsbook"])
    records: List[Dict] = []
    for (pid, market, book), group in groups:
        if len(group) < 2:
            continue
        first_line = float(group.iloc[0]["line"])
        last_line = float(group.iloc[-1]["line"])
        movement = abs(last_line - first_line)
        if movement >= STEAM_MOVE_THRESHOLD:
            records.append({
                "player_id": pid,
                "market": market,
                "sportsbook": book,
                "line_open": first_line,
                "line_close": last_line,
                "movement": movement,
            })
    return pd.DataFrame(records)


def _best_prices(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Return the best (most favorable) line per player/market.

    For an OVER bettor, the best line is the lowest available line.
    Returns one row per player/market with the best sportsbook and line.
    """
    if odds_df.empty:
        return pd.DataFrame()

    latest = odds_df.sort_values("as_of").groupby(
        ["player_id", "market", "sportsbook"]
    ).tail(1)

    idx = latest.groupby(["player_id", "market"])["line"].idxmin()
    return latest.loc[idx].reset_index(drop=True)


class OddsAgent(BaseAgent):
    """Monitors odds movement and reports best available prices."""

    def __init__(self) -> None:
        super().__init__("odds_agent")

    def analyze(
        self,
        season: int,
        week: int,
        player_id: Optional[str] = None,
    ) -> List[AgentReport]:
        odds_df = self._load_odds(season, week, player_id)
        if odds_df.empty:
            self.logger.info("No odds data for season=%d week=%d", season, week)
            return []

        steam = _detect_steam_moves(odds_df)
        best = _best_prices(odds_df)

        steam_ids = set(steam["player_id"]) if not steam.empty else set()

        reports: List[AgentReport] = []

        player_markets = (
            odds_df[["player_id", "market"]]
            .drop_duplicates()
            .values.tolist()
        )

        for pid, market in player_markets:
            is_steam = pid in steam_ids
            best_row = (
                best[(best["player_id"] == pid) & (best["market"] == market)]
                if not best.empty
                else pd.DataFrame()
            )
            best_line = (
                float(best_row.iloc[0]["line"]) if not best_row.empty else None
            )
            best_book = (
                str(best_row.iloc[0]["sportsbook"])
                if not best_row.empty
                else None
            )

            if is_steam:
                recommendation = "REJECT"
                confidence = 0.85
                rationale = (
                    f"Steam move detected for {pid} {market}: "
                    f"line shifted >= {STEAM_MOVE_THRESHOLD} pts"
                )
            else:
                recommendation = "APPROVE"
                confidence = 0.70
                rationale = (
                    f"Line stable for {pid} {market}. "
                    f"Best price: {best_line} at {best_book}"
                )

            reports.append(AgentReport(
                agent_name=self.name,
                recommendation=recommendation,
                confidence=confidence,
                rationale=rationale,
                player_id=pid,
                market=market,
                data={
                    "steam_move": is_steam,
                    "best_line": best_line,
                    "best_sportsbook": best_book,
                },
            ))

        self.logger.info(
            "OddsAgent produced %d reports (%d steam moves)",
            len(reports),
            len(steam),
        )
        return reports
