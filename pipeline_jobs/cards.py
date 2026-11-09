"""Database-only staging and promotion helpers for durable final cards."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from utils.db import execute, fetchone, read_dataframe


def load_card(
    season: int,
    week: int,
    *,
    run_id: str | None = None,
    attempt: int | None = None,
    player_id: Optional[str] = None,
) -> pd.DataFrame:
    """Read one attempt's staged card, or the published card when no attempt is given.

    Risk and agents run after staging and before publication, so a durable run
    must judge its own staged card. The published card is last run's.
    """
    if (run_id is None) != (attempt is None):
        raise ValueError("run_id and attempt must be provided together")
    if run_id is not None:
        query = (
            "SELECT * FROM pipeline_card_staging "
            "WHERE run_id = ? AND attempt = ? AND season = ? AND week = ?"
        )
        params: tuple[Any, ...] = (run_id, attempt, season, week)
    else:
        query = "SELECT * FROM materialized_value_view WHERE season = ? AND week = ?"
        params = (season, week)
    if player_id is not None:
        query += " AND player_id = ?"
        params = (*params, player_id)
    return read_dataframe(query, params=params)


def promote_staged_card(
    conn: Any,
    *,
    run_id: str,
    attempt: int,
    season: int,
    week: int,
) -> int:
    """Replace the active weekly card from one fenced attempt in its transaction."""
    row = fetchone(
        """
        SELECT COUNT(*) FROM pipeline_card_staging
        WHERE run_id = ? AND attempt = ? AND season = ? AND week = ?
        """,
        (run_id, attempt, season, week),
        conn=conn,
    )
    count = int(row[0]) if row else 0
    execute(
        "DELETE FROM materialized_value_view WHERE season = ? AND week = ?",
        (season, week),
        conn=conn,
    )
    execute(
        """
        INSERT INTO materialized_value_view (
            season, week, player_id, event_id, team, team_odds, market, sportsbook,
            line, price, side, mu, sigma, p_win, implied_prob, implied_prob_under,
            edge_percentage, expected_roi, kelly_fraction, stake, confidence_score,
            confidence_tier, generated_at, published_run_id
        )
        SELECT season, week, player_id, event_id, team, team_odds, market, sportsbook,
               line, price, side, mu, sigma, p_win, implied_prob, implied_prob_under,
               edge_percentage, expected_roi, kelly_fraction, stake, confidence_score,
               confidence_tier, generated_at, ?
        FROM pipeline_card_staging
        WHERE run_id = ? AND attempt = ? AND season = ? AND week = ?
        """,
        (run_id, run_id, attempt, season, week),
        conn=conn,
    )
    return count
