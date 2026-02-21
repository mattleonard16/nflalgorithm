"""NBA usage rate features for projection models.

Computes player usage share metrics (FGA share, minutes share) and
trend signals (usage delta) that serve as volume certainty inputs
for the confidence engine.
"""

from __future__ import annotations

import pandas as pd


def compute_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add usage rate columns to a player game log DataFrame.

    Adds:
      - fga_share: player FGA / team FGA per game
      - min_share: player MIN / team MIN per game (approx 240 for 5-man)
      - usage_delta: L5 fga_share - L10 fga_share (positive = trending up)

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

    # FGA share (avoid division by zero)
    result["fga_share"] = result["fga"] / result["team_fga"].clip(lower=1)

    # Minutes share (team_min ~ 240 for a regulation game)
    result["min_share"] = result["min"] / result["team_min"].clip(lower=1)

    # Usage delta: L5 fga_share - L10 fga_share (shift by 1 to avoid leakage)
    result["fga_share_l5"] = (
        result.groupby("player_id")["fga_share"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    result["fga_share_l10"] = (
        result.groupby("player_id")["fga_share"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    )
    result["usage_delta"] = result["fga_share_l5"] - result["fga_share_l10"]

    # Drop intermediate columns
    result = result.drop(
        columns=["team_fga", "team_min", "fga_share_l5", "fga_share_l10"],
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
