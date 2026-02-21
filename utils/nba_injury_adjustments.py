"""NBA Injury-Aware Projection Adjustments.

When key teammates are OUT, their expected production is redistributed
proportionally among active players.  A damping factor prevents over-
crediting (not all production transfers).

Functions
---------
compute_teammate_absence_boost
    Per-player boost when teammates are out.
apply_injury_adjustments
    Batch-modify value rows with injury info before p_win computation.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

from utils.db import read_dataframe

logger = logging.getLogger(__name__)

# Damping factor — fraction of OUT player's production that actually
# redistributes.  0.6 means 60% of their share is split among actives.
DAMPING_FACTOR = 0.6

# Sigma inflation when an injury boost is applied — less certainty.
SIGMA_INFLATION = 0.20  # +20%

# ── Market → game log column ─────────────────────────────────────────────────

_MARKET_TO_STAT: Dict[str, str] = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "fg3m": "fg3m",
}


def _season_from_date(game_date: str) -> int:
    from datetime import datetime

    dt = datetime.strptime(game_date, "%Y-%m-%d")
    return dt.year if dt.month >= 10 else dt.year - 1


def _get_team_market_shares(
    team: str,
    market: str,
    season: int,
) -> Dict[int, float]:
    """Return {player_id: fraction_of_team_total} for *market* on *team*.

    E.g. if player A averaged 25 pts and the team averaged 110 pts per game,
    player A's share = 25 / 110 ≈ 0.227.
    """
    stat_col = _MARKET_TO_STAT.get(market)
    if stat_col is None:
        return {}

    try:
        df = read_dataframe(
            f"SELECT player_id, AVG({stat_col}) as avg_stat "
            "FROM nba_player_game_logs "
            "WHERE team_abbreviation = ? AND season = ? AND min > 0 "
            "GROUP BY player_id",
            params=(team, season),
        )
    except Exception as exc:
        logger.warning("Could not load market shares for %s/%s: %s", team, market, exc)
        return {}

    if df.empty:
        return {}

    total = df["avg_stat"].sum()
    if total <= 0:
        return {}

    return {
        int(row["player_id"]): float(row["avg_stat"]) / total
        for row in df.to_dict("records")
    }


def _get_out_players(team: str, game_date: str) -> List[Dict]:
    """Return list of OUT players for *team* on *game_date*."""
    try:
        df = read_dataframe(
            "SELECT player_id, player_name FROM nba_injuries "
            "WHERE team = ? AND game_date = ? AND status = 'OUT'",
            params=(team, game_date),
        )
        return df.to_dict("records")
    except Exception as exc:
        logger.debug("No injury data for %s on %s: %s", team, game_date, exc)
        return []


def compute_teammate_absence_boost(
    player_id: int,
    team: str,
    game_date: str,
    market: str,
    base_mu: float,
) -> Tuple[float, float, List[str]]:
    """Compute adjusted mu when teammates are OUT.

    Returns (adjusted_mu, boost_multiplier, list_of_out_player_names).

    Algorithm:
        1. Identify OUT teammates for the same team & game_date.
        2. Look up each OUT player's historical market share.
        3. Sum the freed share, apply damping factor.
        4. Redistribute proportionally to the active player based on
           their own share relative to remaining actives.
        5. adjusted_mu = base_mu * (1 + redistribution_fraction).
    """
    season = _season_from_date(game_date)
    out_players = _get_out_players(team, game_date)

    if not out_players:
        return base_mu, 1.0, []

    # Filter to actual teammates (not self)
    out_teammates = [p for p in out_players if int(p["player_id"]) != player_id]
    if not out_teammates:
        return base_mu, 1.0, []

    market_shares = _get_team_market_shares(team, market, season)
    if not market_shares:
        return base_mu, 1.0, []

    # Sum freed share from OUT teammates
    freed_share = sum(
        market_shares.get(int(p["player_id"]), 0.0) for p in out_teammates
    )

    if freed_share <= 0:
        return base_mu, 1.0, []

    # Player's own share (if missing, use uniform estimate)
    player_share = market_shares.get(player_id, 0.0)
    active_total = 1.0 - freed_share
    if active_total <= 0:
        active_total = 1.0

    # Player's proportion of remaining actives
    player_proportion = player_share / active_total if active_total > 0 else 0.0

    # Redistribution: damped freed share * player's proportion of actives
    redistribution = freed_share * DAMPING_FACTOR * player_proportion

    boost_multiplier = 1.0 + redistribution
    adjusted_mu = base_mu * boost_multiplier

    out_names = [p["player_name"] for p in out_teammates]

    logger.debug(
        "Injury boost for player %d (%s %s): freed=%.3f, proportion=%.3f, "
        "boost=%.3f, mu %.1f -> %.1f, out=%s",
        player_id,
        team,
        market,
        freed_share,
        player_proportion,
        boost_multiplier,
        base_mu,
        adjusted_mu,
        out_names,
    )

    return adjusted_mu, boost_multiplier, out_names


def apply_injury_adjustments(
    value_rows: List[Dict],
    game_date: str,
) -> List[Dict]:
    """Add injury-adjusted fields to each value row.

    For every row that has a valid player_id and team, computes the
    teammate-absence boost and adds:
        - base_mu                (original projected_value)
        - injury_adjusted_mu     (after boost)
        - injury_boost_multiplier
        - injury_boost_players   (JSON list of OUT teammate names)

    Returns a **new** list of dicts (immutability).
    """
    adjusted: List[Dict] = []

    for row in value_rows:
        new_row = dict(row)
        player_id = new_row.get("player_id")
        team = new_row.get("team")
        market = new_row.get("market")
        mu = new_row.get("projected_value") or new_row.get("mu")

        if player_id is None or team is None or mu is None:
            new_row["base_mu"] = mu
            new_row["injury_adjusted_mu"] = mu
            new_row["injury_boost_multiplier"] = 1.0
            new_row["injury_boost_players"] = None
            adjusted.append(new_row)
            continue

        adj_mu, boost, out_names = compute_teammate_absence_boost(
            player_id=int(player_id),
            team=str(team),
            game_date=game_date,
            market=str(market),
            base_mu=float(mu),
        )

        new_row["base_mu"] = float(mu)
        new_row["injury_adjusted_mu"] = adj_mu
        new_row["injury_boost_multiplier"] = boost
        new_row["injury_boost_players"] = json.dumps(out_names) if out_names else None

        # Update the projection mu so downstream sigma/p_win use the adjusted value
        if "projected_value" in new_row:
            new_row["projected_value"] = adj_mu

        adjusted.append(new_row)

    boosted_count = sum(1 for r in adjusted if r.get("injury_boost_multiplier", 1.0) > 1.0)
    if boosted_count:
        logger.info(
            "Applied injury boost to %d / %d rows for %s",
            boosted_count,
            len(adjusted),
            game_date,
        )

    return adjusted
