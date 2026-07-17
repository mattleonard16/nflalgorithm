"""Database-only staging and promotion helpers for durable final cards."""

from __future__ import annotations

from typing import Any

from utils.db import execute, fetchone


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
        "SELECT COUNT(*) FROM pipeline_card_staging WHERE run_id = ? AND attempt = ?",
        (run_id, attempt),
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

