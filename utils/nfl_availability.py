"""Find games a player left early, and keep them out of his averages.

A starter who leaves in the first quarter posts a stat line far below his role.
Averaged in, that game drags the next weeks' expected volume down even though
his role never changed. The mask uses the game's own snap share and the games
before it, so flagging week W reads nothing from week W+1 or later.
"""

from __future__ import annotations

import pandas as pd

PRIOR_GAMES = 6
MIN_PRIOR_GAMES = 3
MIN_PRIOR_MEDIAN_SHARE = 60.0
SHORTENED_SHARE_RATIO = 0.5


def shortened_game_mask(
    frame: pd.DataFrame,
    *,
    player_col: str = "player_id",
    share_col: str = "snap_percentage",
) -> pd.Series:
    """Return True for games played at under half the player's usual snap share.

    Only regulars count: the median over the prior six games must be at least
    60%. A share of 0 means the snap merge missed the row, not that the player
    sat (see ``scripts/ingest_real_nfl_data.py``), so it is never flagged and
    never enters a median. ``frame`` must be in game order within each player.
    """
    share = pd.to_numeric(frame[share_col], errors="coerce")
    share = share.where(share > 0)
    prior_median = share.groupby(frame[player_col]).transform(
        lambda s: s.shift(1).rolling(PRIOR_GAMES, min_periods=MIN_PRIOR_GAMES).median()
    )
    regular = prior_median >= MIN_PRIOR_MEDIAN_SHARE
    return (regular & (share < prior_median * SHORTENED_SHARE_RATIO)).fillna(False).astype(bool)


def masked_lagged_ewm(
    values: pd.Series, groups: pd.Series, mask: pd.Series, *, span: int
) -> pd.Series:
    """Lagged EWM per group that skips flagged rows entirely.

    ``ignore_na=True`` makes the result equal to the EWM over the unflagged
    rows alone. With nothing flagged it equals ``shift(1).ewm(span)``.
    """
    kept = values.where(~mask.to_numpy())
    return kept.groupby(groups).transform(
        lambda s: s.shift(1).ewm(span=span, min_periods=1, ignore_na=True).mean()
    )
