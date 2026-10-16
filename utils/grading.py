"""Shared grading utilities for bet outcome evaluation.

Used by both NFL (scripts/record_outcomes.py) and NBA (scripts/record_nba_outcomes.py)
grading pipelines.
"""

from __future__ import annotations

import pandas as pd


def grade_bet(actual: float, line: float, side: str) -> str:
    """Grade a single bet based on actual result vs line.

    Args:
        actual: Actual stat value
        line: Bet line
        side: Bet side ('over' or 'under')

    Returns:
        Result: 'win', 'loss', or 'push'
    """
    if pd.isna(actual):
        return "push"

    if actual == line:
        return "push"

    if side.lower() == "over":
        return "win" if actual > line else "loss"
    else:
        return "win" if actual < line else "loss"


def calculate_profit_units(result: str, price: int) -> float:
    """Calculate profit in units for a bet.

    Args:
        result: Bet result ('win', 'loss', or 'push')
        price: American odds (e.g., -110, +150)

    Returns:
        Profit in units (1 unit = stake)
    """
    if result == "push":
        return 0.0
    elif result == "loss":
        return -1.0
    elif result == "win":
        if price < 0:
            return 100.0 / abs(price)
        else:
            return price / 100.0
    else:
        return 0.0


def get_confidence_tier(edge_percentage: float) -> str:
    """Determine confidence tier based on edge percentage.

    Args:
        edge_percentage: Edge percentage at time of placement

    Returns:
        Confidence tier: HIGH, MEDIUM, LOW, or MINIMAL
    """
    if edge_percentage >= 15.0:
        return "HIGH"
    elif edge_percentage >= 8.0:
        return "MEDIUM"
    elif edge_percentage >= 3.0:
        return "LOW"
    else:
        return "MINIMAL"


def _clean_ids(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def align_actuals_to_bets(
    actuals: pd.DataFrame,
    roster: pd.DataFrame,
    snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-key actual stat rows to the player ids the bets are stored under.

    Two tables mint ``player_id`` from different spellings of the same name and
    the two never match. ``scripts/ingest_real_nfl_data.transform_to_enhanced_stats``
    builds it from nflverse's abbreviated ``player_name`` ("M.Stafford"), giving
    ``LAR_m_stafford``, while ``upsert_roster_players`` builds it from the full
    ``player_name`` on the roster feed, giving ``LAR_matthew_stafford``. Bets
    carry the roster form, so grading matched nothing and recorded every bet as
    a push with zero profit. That is indistinguishable from a settled push once
    it lands in ``bet_outcomes`` and the CLV average.

    Both tables also carry nflverse's ``gsis_id``, which is stable across name
    spellings, suffixes and mid-season trades, so that is what this joins on.
    It is unique per season on the roster and per player-week in the stats, so
    the mapping is unambiguous.

    The id also embeds the club, and ``nfl_roster_players`` is keyed
    ``(season, gsis_id)``: its upsert overwrites ``player_id`` when a player
    changes teams. A bet minted as ``NYJ_garrett_wilson`` before a trade would
    find only ``DAL_garrett_wilson`` there. ``snapshots`` (rows of
    ``nfl_player_context_snapshots``: ``gsis_id``, ``player_id`` and, when the
    actuals carry them, ``season``/``week``) record the id the projection was
    minted under *for that week*, so they win; the current roster fills in
    players the snapshots do not cover. When both name an id for a stat row and
    the two differ, the row is emitted under each, snapshot first, so a bet
    stored under either still matches. Within one ``(season, week)`` an id is
    kept once, by the highest-priority source.

    A stat row whose ``gsis_id`` is missing from both keeps the id it already
    has rather than being dropped: some feeds carry players the roster
    snapshot does not, and an id that was already aligned still matches.
    """
    for column in ("player_id", "gsis_id"):
        if column not in actuals.columns:
            raise ValueError(f"actuals missing required column: {column}")
        if column not in roster.columns:
            raise ValueError(f"roster missing required column: {column}")
        if snapshots is not None and column not in snapshots.columns:
            raise ValueError(f"snapshots missing required column: {column}")

    sources = [frame for frame in (snapshots, roster) if frame is not None and not frame.empty]
    if actuals.empty or not sources:
        return actuals

    base = actuals.reset_index(drop=True)
    rows = pd.DataFrame({"_row": range(len(base)), "_gsis": _clean_ids(base["gsis_id"])})
    for column in ("season", "week"):
        if column in base.columns:
            rows[column] = base[column].to_numpy()

    candidates = []
    for priority, source in enumerate(sources):
        keys = ["_gsis"] + [
            column for column in ("season", "week") if column in source and column in rows
        ]
        usable = source.assign(_gsis=_clean_ids(source["gsis_id"]))
        usable = usable[usable["_gsis"] != ""]
        usable = usable[[*keys, "player_id"]].drop_duplicates(keys, keep="last")
        matched = rows[rows["_gsis"] != ""].merge(usable, on=keys, how="inner")
        candidates.append(matched[["_row", "player_id"]].assign(_priority=priority))
    own = pd.DataFrame(
        {"_row": rows["_row"], "player_id": base["player_id"], "_priority": len(sources)}
    )

    aliases = pd.concat([*candidates, own], ignore_index=True)
    aliases = aliases.sort_values(["_row", "_priority"], kind="stable")
    # The stat row's own id is only a fallback: an abbreviated id such as
    # BUF_j_smith can belong to two players, so it never rides along as an alias.
    has_bridge = aliases.groupby("_row")["_priority"].transform("min") < len(sources)
    aliases = aliases[~(has_bridge & (aliases["_priority"] == len(sources)))]
    aliases = aliases.drop_duplicates(["_row", "player_id"])

    out = base.drop(columns=["player_id"]).iloc[aliases["_row"].to_numpy()].copy()
    out["player_id"] = aliases["player_id"].to_numpy()
    out["_priority"] = aliases["_priority"].to_numpy()
    # An alias never displaces a higher-priority claim on the same id that
    # week. Rows tied at one priority all stay, exactly as before bridging.
    week_keys = [column for column in ("season", "week") if column in out.columns]
    best = out.groupby([*week_keys, "player_id"], dropna=False)["_priority"].transform("min")
    out = out[out["_priority"] == best]
    out = out[[*actuals.columns]]
    return out.reset_index(drop=True)


def known_bet_ids(
    roster: pd.DataFrame,
    snapshots: pd.DataFrame | None = None,
) -> set[str]:
    """Every player id an identity source can vouch for.

    A bet whose id is in this set but has no stat row is a player who did not
    play (a genuine push). A bet whose id is in neither this set nor the
    aligned stats was never tied to a player at all, and must not be graded as
    a settled push.
    """
    ids: set[str] = set()
    for frame in (roster, snapshots):
        if frame is None or frame.empty or "player_id" not in frame.columns:
            continue
        ids.update(value for value in _clean_ids(frame["player_id"]) if value)
    return ids
