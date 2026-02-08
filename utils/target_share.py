"""Target share calculation and low-volume gating.

Computes target_share as player_targets / team_total_targets for a given
week, and flags low-volume receivers whose projection confidence should
be capped.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config import config


def calculate_target_share(
    player_targets: float,
    team_total_targets: float,
) -> float:
    """Return target_share (0.0-1.0) for a player.

    Returns 0.0 when team_total_targets is zero or negative to avoid
    division-by-zero.
    """
    if team_total_targets <= 0 or player_targets < 0:
        return 0.0
    return min(player_targets / team_total_targets, 1.0)


def compute_team_target_shares(
    stats: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Compute target_share for every receiver row in a week's stats.

    Expects ``stats`` to contain at least ``team``, ``position``,
    ``targets``, ``player_id``, ``season``, and ``week`` columns.

    Returns a new DataFrame with an updated ``target_share`` column.
    The original DataFrame is never mutated.
    """
    if stats.empty or "targets" not in stats.columns:
        return stats.copy()

    week_stats = stats[
        (stats["season"] == season) & (stats["week"] == week)
    ].copy()

    if week_stats.empty:
        return stats.copy()

    receiver_positions = {"WR", "TE", "RB"}
    receivers = week_stats[week_stats["position"].isin(receiver_positions)]

    team_totals = (
        receivers.groupby("team")["targets"]
        .sum()
        .rename("team_total_targets")
    )

    merged = week_stats.merge(team_totals, on="team", how="left")
    merged["team_total_targets"] = merged["team_total_targets"].fillna(0)

    merged["target_share"] = merged.apply(
        lambda r: calculate_target_share(
            float(r.get("targets", 0) or 0),
            float(r["team_total_targets"]),
        ),
        axis=1,
    )

    result = stats.copy()
    update_idx = result[
        (result["season"] == season) & (result["week"] == week)
    ].index
    if len(update_idx) > 0 and "target_share" in merged.columns:
        result.loc[update_idx, "target_share"] = merged["target_share"].values

    return result


def is_low_volume(
    target_share: float,
    threshold: Optional[float] = None,
) -> bool:
    """Return True if target_share is below the minimum threshold.

    Uses ``config.betting.target_share_min_threshold`` when no explicit
    threshold is provided.
    """
    if threshold is None:
        threshold = getattr(
            config.betting, "target_share_min_threshold", 0.08
        )
    return target_share < threshold


def cap_confidence_for_low_volume(
    confidence: float,
    target_share: float,
    threshold: Optional[float] = None,
) -> float:
    """Cap projection confidence for low-volume receivers.

    When target_share is below the threshold the confidence is scaled
    down proportionally so projections remain but are flagged as risky.
    """
    if threshold is None:
        threshold = getattr(
            config.betting, "target_share_min_threshold", 0.08
        )
    if target_share >= threshold:
        return confidence

    if threshold <= 0:
        return confidence

    scale = target_share / threshold
    return confidence * scale
