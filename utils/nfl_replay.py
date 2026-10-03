"""Pieces of the 2025 replay that need no database.

The walk-forward backtest predicts from history alone, so it cannot see an
injury report or a depth chart. The replay rebuilds each week's roster and
context snapshot as they stood at that week's first kickoff, then predicts
through the roster path production uses. It writes rosters and snapshots, so
it runs only against a scratch copy of the database.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Weekly rosters mark game-day inactives, which clubs announce 90 minutes
# before kickoff. Production predicts days earlier and never sees them.
GAME_DAY_STATUSES = frozenset({"INA"})


def first_kickoff_cutoffs(games: pd.DataFrame, *, season: int, week: int) -> dict[int, str]:
    """Return ``{season: first kickoff}`` for the week, as ISO-8601 UTC."""
    week_games = games[(games["season"] == season) & (games["week"] == week)]
    kickoffs = pd.to_datetime(week_games["kickoff_utc"], errors="coerce", utc=True).dropna()
    if kickoffs.empty:
        raise ValueError(f"No kickoff times for season {season} week {week}")
    return {season: kickoffs.min().isoformat()}


def roster_for_week(rosters: pd.DataFrame, *, season: int, week: int) -> pd.DataFrame:
    """Return the weekly roster for ``week``, with game-day inactives marked active."""
    roster = rosters[(rosters["season"] == season) & (rosters["week"] == week)].copy()
    status = roster["status"].fillna("").astype(str).str.upper().str.strip()
    roster.loc[status.isin(GAME_DAY_STATUSES), "status"] = "ACT"
    return roster.reset_index(drop=True)


def require_scratch_database(path: Path, production_path: Path) -> Path:
    """Return ``path`` resolved, or raise if it is the production database."""
    resolved = Path(path).resolve()
    if resolved == Path(production_path).resolve():
        raise ValueError(
            f"{path} is the production database; the replay rewrites rosters and "
            "snapshots, so point it at a scratch copy"
        )
    if not resolved.is_file():
        raise ValueError(f"Scratch database {path} does not exist; copy nfl_data.db there first")
    return resolved


def players_without_stats(
    predictions: pd.DataFrame, actuals: pd.DataFrame, *, season: int, week: int
) -> int:
    """Count projected players who have no stat row for the week."""
    week_actuals = actuals[(actuals["season"] == season) & (actuals["week"] == week)]
    played = set(week_actuals["player_id"].astype(str))
    projected = set(predictions["player_id"].astype(str))
    return len(projected - played)


def rekey_to_stat_ids(predictions: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Swap roster player ids for the ids the week's stat rows use.

    Rosters key players by full name (``ARI_greg_dortch``) and nflverse stats
    by short name (``ARI_g_dortch``); ``id_map`` pairs them through gsis_id.
    A player with no stat row keeps his roster id and goes unmatched.
    """
    mapping = dict(zip(id_map["roster_id"].astype(str), id_map["stat_id"].astype(str)))
    roster_ids = predictions["player_id"].astype(str)
    return predictions.assign(player_id=roster_ids.map(mapping).fillna(roster_ids))
