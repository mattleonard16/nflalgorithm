"""Base agent class for the NBA betting agent system.

All NBA agents inherit from ``NbaBaseAgent`` and implement ``analyze``
using ``game_date`` instead of NFL's ``(season, week)`` pattern.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from agents import AgentReport
from config import config
from utils.db import read_dataframe


class NbaBaseAgent(ABC):
    """Abstract base for every NBA-specific agent.

    Subclasses must implement ``analyze`` which returns a list of
    ``AgentReport`` objects.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"agents.nba.{name}")
        self.config = config

    @abstractmethod
    def analyze(
        self,
        game_date: str,
        player_id: Optional[int] = None,
    ) -> List[AgentReport]:
        """Run analysis and return a list of reports."""

    def _load_projections(
        self,
        game_date: str,
        player_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load NBA projections from the database."""
        query = (
            "SELECT id, player_id, player_name, team, season, game_date, game_id,"
            " market, projected_value, confidence, created_at"
            " FROM nba_projections WHERE game_date = ?"
        )
        params: list = [game_date]
        if player_id is not None:
            query += " AND player_id = ?"
            params.append(player_id)
        try:
            return read_dataframe(query, params=params)
        except Exception as exc:
            self.logger.warning("Failed to load NBA projections: %s", exc)
            return pd.DataFrame()

    def _load_odds(
        self,
        game_date: str,
        player_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load NBA odds from the database."""
        query = (
            "SELECT event_id, season, game_date, player_id, player_name, team,"
            " market, sportsbook, line, over_price, under_price, as_of"
            " FROM nba_odds WHERE game_date = ?"
        )
        params: list = [game_date]
        if player_id is not None:
            query += " AND player_id = ?"
            params.append(player_id)
        try:
            return read_dataframe(query, params=params)
        except Exception as exc:
            self.logger.warning("Failed to load NBA odds: %s", exc)
            return pd.DataFrame()

    def _load_value_view(
        self,
        game_date: str,
        player_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load NBA materialized value view from the database."""
        query = (
            "SELECT season, game_date, player_id, player_name, team, event_id,"
            " market, sportsbook, line, over_price, under_price, mu, sigma,"
            " p_win, edge_percentage, expected_roi, kelly_fraction, confidence,"
            " generated_at"
            " FROM nba_materialized_value_view WHERE game_date = ?"
        )
        params: list = [game_date]
        if player_id is not None:
            query += " AND player_id = ?"
            params.append(player_id)
        try:
            return read_dataframe(query, params=params)
        except Exception as exc:
            self.logger.warning("Failed to load NBA value view: %s", exc)
            return pd.DataFrame()
