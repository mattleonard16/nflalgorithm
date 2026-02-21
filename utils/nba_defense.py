"""NBA defense adjustment multipliers.

Calculates how players perform RELATIVE to their own average against
specific opponents. Used to boost/reduce projections based on matchup
strength.

Modeled on ``utils/defense_adjustments.py`` (NFL) but adapted for NBA
markets (pts, reb, ast, fg3m) and date-based scheduling.

Example:
  - If players typically get 15% MORE pts than their average vs SAC -> 1.15x (weak D)
  - If players typically get 10% LESS pts than their average vs BOS -> 0.90x (strong D)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple

import pandas as pd

from utils.db import read_dataframe

logger = logging.getLogger(__name__)

# NBA markets for defense multipliers
_NBA_MARKETS = ("pts", "reb", "ast", "fg3m")

# Tighter clamp range than NFL (0.70-1.30) since NBA has more games
_CLAMP_MIN = 0.75
_CLAMP_MAX = 1.25


@lru_cache(maxsize=32)
def compute_nba_defense_multipliers(
    season: int,
    through_date: str,
    min_games: int = 3,
) -> Dict[Tuple[str, str], float]:
    """Compute NBA defense multipliers from relative player performance.

    For each (opponent, market) pair, calculates how much players over/under
    perform their season average when facing that opponent.

    Parameters
    ----------
    season : int
        NBA season year.
    through_date : str
        Only use games before this date (YYYY-MM-DD).
    min_games : int
        Minimum number of player-games vs an opponent to compute a multiplier.

    Returns
    -------
    dict
        Mapping of (opponent, market) -> multiplier float.
        Multiplier > 1.0 = weak defense, < 1.0 = strong defense.
    """
    stats = read_dataframe(
        "SELECT player_id, player_name, team_abbreviation, game_date, "
        "matchup, pts, reb, ast, fg3m "
        "FROM nba_player_game_logs "
        "WHERE season = ? AND game_date < ? "
        "AND pts IS NOT NULL",
        [season, through_date],
    )

    if stats.empty:
        logger.warning("No NBA game logs for defense multipliers (season=%d)", season)
        return {}

    # Extract opponent from matchup string
    stats = stats.copy()
    stats["opponent"] = stats["matchup"].str.strip().str[-3:]

    # Player season averages (their baseline)
    player_avgs = (
        stats.groupby("player_id")[list(_NBA_MARKETS)]
        .agg(["mean", "count"])
    )
    # Flatten multi-index columns
    player_avgs.columns = [
        f"{col}_{agg}" for col, agg in player_avgs.columns
    ]
    player_avgs = player_avgs.reset_index()

    # Only use players with enough games for reliable baseline
    games_col = "pts_count"  # all markets have same count
    player_avgs = player_avgs[player_avgs[games_col] >= 3]

    # Merge averages back
    avg_cols = {m: f"{m}_mean" for m in _NBA_MARKETS}
    stats = stats.merge(
        player_avgs[["player_id"] + list(avg_cols.values())],
        on="player_id",
        how="inner",
    )

    multipliers: Dict[Tuple[str, str], float] = {}

    for market in _NBA_MARKETS:
        avg_col = avg_cols[market]
        # Relative performance: actual / player's season average
        rel_col = f"{market}_rel"
        stats[rel_col] = stats[market] / stats[avg_col].clip(lower=0.1)

        # Filter outliers (10x normal is noise)
        valid = stats[stats[rel_col] < 10.0]

        for opponent in valid["opponent"].unique():
            opp_games = valid[valid["opponent"] == opponent]

            if len(opp_games) < min_games:
                continue

            avg_rel_perf = float(opp_games[rel_col].mean())
            multiplier = max(_CLAMP_MIN, min(_CLAMP_MAX, avg_rel_perf))
            multipliers[(opponent, market)] = multiplier

    logger.info(
        "Computed %d NBA defense multipliers (season=%d, through=%s)",
        len(multipliers), season, through_date,
    )
    return multipliers


def get_nba_defense_multiplier(
    opponent: str,
    market: str,
    season: int,
    through_date: str,
) -> float:
    """Get the defense adjustment multiplier for a specific NBA matchup.

    Parameters
    ----------
    opponent : str
        Opponent team abbreviation (e.g. 'BOS').
    market : str
        Stat market (pts, reb, ast, fg3m).
    season : int
        NBA season year.
    through_date : str
        Use data through this date.

    Returns
    -------
    float
        Multiplier (1.0 = league average).
    """
    multipliers = compute_nba_defense_multipliers(season, through_date)
    return multipliers.get((opponent, market), 1.0)
