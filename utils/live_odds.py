"""Select the live odds snapshot used for ranking and kickoff-aware CLV.

``weekly_odds`` is append-only scrape history. Taking ``MAX(as_of)`` before
dropping stale and post-kickoff quotes would discard a good pre-kickoff line
when a later in-game scrape exists. Filter first, then keep the newest
remaining row per snapshot key.
"""

from __future__ import annotations

import pandas as pd

from utils.matching import filter_stale_snapshots, latest_snapshot_per_key


def kickoffs_from_games(games_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return ``event_id``, ``kickoff_utc`` from a ``games`` frame.

    Canonical odds keys match ``games.game_id``. Callers that already renamed
    the column to ``event_id`` are accepted as-is.
    """
    if games_df is None or games_df.empty:
        return pd.DataFrame(columns=["event_id", "kickoff_utc"])

    if "event_id" in games_df.columns:
        frame = games_df
    elif "game_id" in games_df.columns:
        frame = games_df.rename(columns={"game_id": "event_id"})
    else:
        raise ValueError("games_df missing required column: game_id or event_id")
    if "kickoff_utc" not in frame.columns:
        raise ValueError("games_df missing required column: kickoff_utc")
    return frame[["event_id", "kickoff_utc"]].drop_duplicates().reset_index(drop=True)


def select_live_odds(
    odds_df: pd.DataFrame,
    kickoffs_df: pd.DataFrame,
    *,
    max_age_hours: float = 48.0,
) -> pd.DataFrame:
    """Drop stale and post-kickoff quotes, then keep the newest remaining snapshot."""
    filtered = filter_stale_snapshots(odds_df, kickoffs_df, max_age_hours=max_age_hours)
    return latest_snapshot_per_key(filtered.frame)
