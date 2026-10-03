"""Availability adjustments to expected volume.

Shortened games: a starter who leaves in the first quarter posts a stat line
far below his role. Averaged in, that game drags the next weeks' expected
volume down even though his role never changed. The mask uses the game's own
snap share and the games before it, so flagging week W reads nothing from
week W+1 or later.

Out players: when a player is ruled out, his targets and carries go to
same-team, same-position teammates, damped and capped as in
``utils/nba_injury_adjustments.py``. A QB's attempts go whole to the depth-2
QB, since exactly one QB plays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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


# Share of an Out player's volume that reaches teammates; the rest is lost to
# players off the projected slate and to scheme changes.
OUT_VOLUME_DAMPING = 0.6
MAX_VOLUME_MULTIPLIER = 1.5
SHARED_VOLUME_COLUMNS = ("expected_targets", "expected_rushing_attempts")
BACKUP_QB_START_PROBABILITY = 0.02
_STARTER_START_PROBABILITY = {"OUT": 0.0, "IR": 0.0, "DOUBTFUL": 0.25, "QUESTIONABLE": 0.75}


def redistribute_out_volume(
    frame: pd.DataFrame,
    out_mask: pd.Series,
    *,
    columns: tuple[str, ...] = SHARED_VOLUME_COLUMNS,
    team_col: str = "team",
    position_col: str = "position",
) -> pd.DataFrame:
    """Return ``frame`` with active teammates absorbing the Out players' volume.

    Each active player gets the damped freed volume times his share of the
    active group's volume, capped at ``MAX_VOLUME_MULTIPLIER`` times his own.
    Out rows keep their values; the caller drops them.
    """
    result = frame.copy()
    out = pd.Series(out_mask.to_numpy(dtype=bool), index=frame.index)
    if not out.any():
        return result
    keys = [frame[team_col], frame[position_col]]
    for column in columns:
        if column not in frame.columns:
            continue
        raw = pd.to_numeric(frame[column], errors="coerce")
        volume = raw.fillna(0.0)
        freed = volume.where(out, 0.0).groupby(keys).transform("sum")
        active = volume.where(~out, 0.0)
        active_total = active.groupby(keys).transform("sum")
        share = (active / active_total.where(active_total > 0)).fillna(0.0)
        boosted = np.minimum(
            volume + freed * OUT_VOLUME_DAMPING * share, volume * MAX_VOLUME_MULTIPLIER
        )
        result[column] = raw.where(out | raw.isna(), boosted)
    return result


def _depth(value: Any) -> int:
    depth = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 1 if pd.isna(depth) else int(depth)


def promote_backup_qbs(
    frame: pd.DataFrame,
    out_mask: pd.Series,
    *,
    team_col: str = "team",
    position_col: str = "position",
) -> pd.DataFrame:
    """Add ``p_start`` and hand an Out starter's attempts to the depth-2 QB.

    Starters keep the injury-status probabilities the model already used;
    backups start with ``BACKUP_QB_START_PROBABILITY`` unless the depth-1 QB
    on their team is ruled out, in which case the depth-2 QB starts.
    Non-QB rows get ``p_start`` 1.0.
    """
    result = frame.copy()
    out = pd.Series(out_mask.to_numpy(dtype=bool), index=frame.index)
    is_qb = frame[position_col].astype(str).str.upper().str.strip() == "QB"
    depth = frame.get("depth_rank", pd.Series(np.nan, index=frame.index)).map(_depth)
    status = (
        frame.get("injury_status", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    attempts = pd.to_numeric(
        frame.get("expected_passing_attempts", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    out_starters = is_qb & (depth == 1) & out
    freed_attempts = attempts.where(out_starters).groupby(frame[team_col]).max()
    promoted = is_qb & (depth == 2) & frame[team_col].isin(freed_attempts.dropna().index)

    p_start = pd.Series(1.0, index=frame.index)
    p_start[is_qb & (depth == 1)] = status.map(_STARTER_START_PROBABILITY).fillna(1.0)
    p_start[is_qb & (depth > 1)] = BACKUP_QB_START_PROBABILITY
    p_start[promoted] = 1.0
    result["p_start"] = p_start
    if "expected_passing_attempts" in frame.columns:
        inherited = frame.loc[promoted, team_col].map(freed_attempts)
        result.loc[promoted, "expected_passing_attempts"] = np.fmax(attempts[promoted], inherited)
    return result
