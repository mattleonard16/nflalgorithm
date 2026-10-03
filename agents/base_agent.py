"""Base agent class for the NFL betting agent system.

All specialized agents inherit from ``BaseAgent`` and implement the
``analyze`` method.  The base class provides shared DB access, config
loading, and logging helpers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from agents import AgentReport
from config import config
from utils.db import read_dataframe


class BaseAgent(ABC):
    """Abstract base for every specialized agent.

    Subclasses must implement ``analyze`` which returns a list of
    ``AgentReport`` objects -- one per player/market combination that
    the agent has an opinion about.
    """

    def __init__(
        self,
        name: str,
        *,
        run_id: str | None = None,
        attempt: int | None = None,
    ) -> None:
        self.name = name
        self.logger = logging.getLogger(f"agents.{name}")
        self.config = config
        # Set for a durable run, so the agent judges the card that run staged
        # rather than the one already published.
        self.run_id = run_id
        self.attempt = attempt

    @abstractmethod
    def analyze(
        self,
        season: int,
        week: int,
        player_id: Optional[str] = None,
    ) -> List[AgentReport]:
        """Run analysis and return a list of reports.

        Parameters
        ----------
        season : int
        week : int
        player_id : str, optional
            When provided, limit analysis to this player.
        """

    def _load_projections(
        self,
        season: int,
        week: int,
        player_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load weekly projections from the database."""
        query = (
            "SELECT * FROM weekly_projections WHERE season = ? AND week = ?"
        )
        params: tuple = (season, week)
        if player_id is not None:
            query += " AND player_id = ?"
            params = (season, week, player_id)
        try:
            return read_dataframe(query, params=params)
        except Exception as exc:
            self.logger.warning("Failed to load projections: %s", exc)
            return pd.DataFrame()

    def _load_odds(
        self,
        season: int,
        week: int,
        player_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load weekly odds from the database."""
        query = "SELECT * FROM weekly_odds WHERE season = ? AND week = ?"
        params: tuple = (season, week)
        if player_id is not None:
            query += " AND player_id = ?"
            params = (season, week, player_id)
        try:
            return read_dataframe(query, params=params)
        except Exception as exc:
            self.logger.warning("Failed to load odds: %s", exc)
            return pd.DataFrame()

    def _load_value_view(
        self,
        season: int,
        week: int,
        player_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load this run's card: the staged one for a durable run, else the published one."""
        from pipeline_jobs.cards import load_card

        try:
            return load_card(
                season,
                week,
                run_id=self.run_id,
                attempt=self.attempt,
                player_id=player_id,
            )
        except Exception as exc:
            self.logger.warning("Failed to load value view: %s", exc)
            return pd.DataFrame()
