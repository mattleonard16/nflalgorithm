"""Who belongs on the public algorithm slate.

The predict path writes every rotation-eligible row. The Board should not.
This helper keeps players the depth chart says will see the field: starting
QBs, RB1/RB2, WR1–WR3, TE1/TE2. Inactive roster statuses and OUT/IR/Doubtful
injuries are excluded.
"""

from __future__ import annotations

from typing import Any, Mapping

from sports.nfl import INACTIVE_ROSTER_STATUSES, MARKET_MIN_EXPECTED_VOLUME

OUT_INJURY_STATUSES = frozenset({"OUT", "IR", "DOUBTFUL"})


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
        if position == "QB":
            return starter or depth == 1
        if depth is not None and depth <= 2:
            return True
        rush = _number(row.get("expected_rushing_attempts")) or 0.0
        return rush >= MARKET_MIN_EXPECTED_VOLUME["rushing_yards"]

    if market == "receiving_yards":
        max_depth = 2 if position in {"TE", "RB", "FB"} else 3
        if starter or (depth is not None and depth <= max_depth):
            return True
        targets = _number(row.get("expected_targets")) or 0.0
        return targets >= MARKET_MIN_EXPECTED_VOLUME["receiving_yards"]

    return False
