"""Who belongs on the public algorithm slate.

The predict path writes every rotation-eligible row. The Board should not.
This helper keeps players the depth chart says will see the field: starting
QBs, RB1/RB2, WR1–WR3, TE1/TE2. Receptions follow the receiving rule and
anytime touchdown keeps anyone likely to rush or catch. Inactive roster
statuses and OUT/IR/Doubtful injuries are excluded.
"""

from __future__ import annotations

from typing import Any, Mapping

from sports.nfl import INACTIVE_ROSTER_STATUSES, MARKET_MIN_EXPECTED_VOLUME

OUT_INJURY_STATUSES = frozenset({"OUT", "IR", "DOUBTFUL"})
# Statuses that mean the player will not play. Doubtful is left out: props void
# when a player sits, so an if-he-plays projection stays valid for him.
RULED_OUT_STATUSES = frozenset({"OUT", "IR", "INJURED RESERVE"})


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip().upper()


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _depth(value: Any) -> int | None:
    number = _number(value)
    if number is None:
        return None
    return int(number)


def ruled_out(row: Mapping[str, Any], target_week: int) -> bool:
    """Return True when the target week's own injury report rules the player out.

    A status carried forward from an earlier week's report does not count: the
    Wednesday run comes before clubs file, and a player back at practice would
    lose his projection until the next run.
    """
    if _text(row.get("injury_status")) not in RULED_OUT_STATUSES:
        return False
    report_week = _number(row.get("injury_report_week"))
    return report_week is not None and int(report_week) == target_week


def backup_qb(row: Mapping[str, Any]) -> bool:
    """Return True for a QB the depth chart lists behind the starter.

    Models trained on his past starts project him like a starter: on the 2025
    replay, backups ran +61 passing yards high and carried the whole passing
    bias. A QB with no depth entry is not a backup, since missing data is not
    evidence that he sits.
    """
    if _text(row.get("position")) != "QB":
        return False
    if int(_number(row.get("is_starter")) or 0) == 1:
        return False
    depth = _depth(row.get("depth_rank"))
    return depth is not None and depth > 1


def likely_to_play(row: Mapping[str, Any], market: str) -> bool:
    """Return True when the row is a plausible week-of participant for ``market``."""
    roster_status = _text(row.get("roster_status")) or "ACT"
    if roster_status in INACTIVE_ROSTER_STATUSES:
        return False
    if _text(row.get("injury_status")) in OUT_INJURY_STATUSES:
        return False

    depth = _depth(row.get("depth_rank"))
    starter = int(_number(row.get("is_starter")) or 0) == 1
    position = _text(row.get("position"))

    if market == "passing_yards":
        return starter or depth == 1

    if market == "rushing_yards":
        return _likely_rusher(row, position, starter, depth)

    if market in {"receiving_yards", "receptions"}:
        return _likely_receiver(row, position, starter, depth)

    if market == "anytime_touchdown":
        # A quarterback scores on the ground, so the backup-QB rule applies.
        if position == "QB":
            return _likely_rusher(row, position, starter, depth)
        return _likely_rusher(row, position, starter, depth) or _likely_receiver(
            row, position, starter, depth
        )

    return False


def _likely_rusher(row: Mapping[str, Any], position: str, starter: bool, depth: int | None) -> bool:
    if position == "QB":
        return starter or depth == 1
    if depth is not None and depth <= 2:
        return True
    rush = _number(row.get("expected_rushing_attempts")) or 0.0
    return rush >= MARKET_MIN_EXPECTED_VOLUME["rushing_yards"]


def _likely_receiver(
    row: Mapping[str, Any], position: str, starter: bool, depth: int | None
) -> bool:
    max_depth = 2 if position in {"TE", "RB", "FB"} else 3
    if starter or (depth is not None and depth <= max_depth):
        return True
    targets = _number(row.get("expected_targets")) or 0.0
    return targets >= MARKET_MIN_EXPECTED_VOLUME["receiving_yards"]
