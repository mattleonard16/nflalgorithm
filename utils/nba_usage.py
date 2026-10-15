"""NBA usage rate features for projection models.

Computes player usage share metrics (FGA share, minutes share) and
trend signals (usage delta) that serve as volume certainty inputs
for the confidence engine.
"""

from __future__ import annotations

import pandas as pd

# Trailing window (games) for the pre-game fga_share / min_share features.
USAGE_SHARE_WINDOW = 10


def compute_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add usage rate columns to a player game log DataFrame.

    Every output column is a **pre-game** value: it uses only the player's
    earlier games (``shift(1)`` before rolling), like the other rolling
    features in ``models/nba/stat_model.py``. These columns are model
    features, so a value computed from the game being predicted would leak
    that game's box score into training.

    Adds:
      - fga_share: mean of per-game (player FGA / team FGA) over the previous
        ``USAGE_SHARE_WINDOW`` games
      - min_share: mean of per-game (player MIN / team MIN, ~240 for a
        regulation game) over the previous ``USAGE_SHARE_WINDOW`` games
      - usage_delta: L5 fga_share - L10 fga_share (positive = trending up)

    A player's first game has no history, so its values are NaN.

    Parameters
    ----------
    df : DataFrame
        Must contain: player_id, team_abbreviation, game_id, fga, min.
        Sorted by (player_id, game_date).

    Returns
    -------
    DataFrame
        New DataFrame with usage columns added. Original is not mutated.
    """
    result = df.copy()

    # Compute team totals per game
    team_totals = (
        result.groupby(["team_abbreviation", "game_id"])
        .agg(team_fga=("fga", "sum"), team_min=("min", "sum"))
        .reset_index()
    )

    result = result.merge(
        team_totals,
        on=["team_abbreviation", "game_id"],
        how="left",
    )

    # Same-game shares (avoid division by zero). Intermediate only: these use
    # the current game's box score and must not be exposed as features.
    result["_fga_share_game"] = result["fga"] / result["team_fga"].clip(lower=1)
    # Minutes share (team_min ~ 240 for a regulation game)
    result["_min_share_game"] = result["min"] / result["team_min"].clip(lower=1)

    def _trailing_mean(col: str, window: int) -> pd.Series:
        # shift(1) so a row only sees the player's earlier games
        return result.groupby("player_id")[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    result["fga_share"] = _trailing_mean("_fga_share_game", USAGE_SHARE_WINDOW)
    result["min_share"] = _trailing_mean("_min_share_game", USAGE_SHARE_WINDOW)

    # Usage delta: L5 fga_share - L10 fga_share
    result["usage_delta"] = _trailing_mean("_fga_share_game", 5) - _trailing_mean(
        "_fga_share_game", 10
    )

    # Drop intermediate columns
    result = result.drop(
        columns=["team_fga", "team_min", "_fga_share_game", "_min_share_game"],
        errors="ignore",
    )

    return result


def is_usage_spike(
    fga_share_l5: float,
    fga_share_l10: float,
    threshold: float = 0.15,
) -> bool:
    """Flag an unusual usage increase.

    Returns True when the recent (L5) FGA share exceeds the longer-term
    (L10) share by more than *threshold* (absolute difference).

    Parameters
    ----------
    fga_share_l5 : float
        Last-5-game average FGA share.
    fga_share_l10 : float
        Last-10-game average FGA share.
    threshold : float
        Minimum absolute increase to flag as a spike.
    """
    if fga_share_l5 is None or fga_share_l10 is None:
        return False

    return (fga_share_l5 - fga_share_l10) > threshold
