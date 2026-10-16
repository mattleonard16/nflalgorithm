"""Pair an Over quote with the Under quote for the *same* line.

Pure functions only — no HTTP, no database — so the pairing rule stays
testable in CI even though the scraper's transport code is not exercised
there.

No-vig probability is only meaningful when both prices describe the same
wager. The Odds API returns every alternate line for a player as a separate
outcome in one flat list, all sharing the same ``description``. Matching the
opposite side on player name alone therefore pairs an Over at 55.5 with an
Under at 70.5 whenever a book posts alternates, and
``implied_probability_no_vig`` will happily consume that pair and return a
confident, wrong number. The line has to be part of the match key.

Yes/No markets (``player_anytime_td``) quote "Yes"/"No" with no line. "Yes"
is the Over of one-or-more, so a Yes/No pair is stored as Over/Under at
``YES_NO_LINE`` (0.5); a Yes/No outcome that does carry a line keeps it.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

OVER = "Over"
UNDER = "Under"
YES = "Yes"
NO = "No"

# The line a Yes/No (at-least-one) market grades against: 1+ is over 0.5.
YES_NO_LINE = 0.5

# Side held -> the outcome name of the opposite side of the same wager.
_OPPOSITE = {OVER: UNDER, UNDER: OVER, YES: NO, NO: YES}
# Outcome name -> the stored side it prices.
_STORED_SIDE = {OVER: OVER, UNDER: UNDER, YES: OVER, NO: UNDER}

# Books quote points as floats; compare with a tolerance rather than ==, since
# a JSON round-trip can leave 55.5 and 55.500000000000004.
_POINT_TOLERANCE = 1e-6


def _point(outcome: Mapping[str, Any]) -> Optional[float]:
    value = outcome.get("point")
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _same_point(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        # A quote with no line cannot be shown to be the same wager as one
        # with a line. Refuse rather than assume.
        return left is None and right is None
    return abs(left - right) <= _POINT_TOLERANCE


def find_opposite_side(
    outcomes: Iterable[Mapping[str, Any]],
    *,
    description: Optional[str],
    side: str,
    point: Optional[float],
) -> Optional[Mapping[str, Any]]:
    """Return the quote for the opposite side of the same player-and-line.

    Args:
        outcomes: All outcomes in the market, alternates included.
        description: The player description to match, compared exactly as the
            book spells it. ``None`` matches only another quote that also
            carries no description.
        side: The side already held — ``"Over"``/``"Under"``, or
            ``"Yes"``/``"No"`` for an at-least-one market.
        point: The line already held. ``None`` matches only another quote that
            also carries no line.

    Returns:
        The matching outcome, or ``None`` when the book quoted only one side
        at this line.
    """
    if side not in _OPPOSITE:
        raise ValueError(f"side must be one of {sorted(_OPPOSITE)!r}, got {side!r}")

    wanted = _OPPOSITE[side]
    for candidate in outcomes:
        if candidate.get("description") != description:
            continue
        if candidate.get("name") != wanted:
            continue
        if _same_point(point, _point(candidate)):
            return candidate
    return None


def pair_two_sided_prices(
    outcome: Mapping[str, Any], outcomes: Iterable[Mapping[str, Any]]
) -> Optional[tuple[float, int, int]]:
    """Resolve ``outcome`` into ``(line, over_price, under_price)``.

    Returns ``None`` when the quote is unusable: an unrecognized side, a
    missing price or line, or no opposite side at the same line. A quote with
    no line is not a line of 0.0, and storing it as one would grade every
    stat against zero; the one exception is a Yes/No pair, whose missing line
    is ``YES_NO_LINE`` by definition of the market. Callers skip ``None``
    rows — a one-sided quote (a book posting only "Yes") cannot have its vig
    removed, and guessing the other side would invent a market that was never
    offered.
    """
    side = outcome.get("name")
    if side not in _OPPOSITE:
        return None

    price = outcome.get("price")
    point = _point(outcome)
    if price is None:
        return None
    if point is None and side not in (YES, NO):
        return None

    opposite = find_opposite_side(
        outcomes,
        description=outcome.get("description"),
        side=side,
        point=point,
    )
    if opposite is None or opposite.get("price") is None:
        return None

    is_over = _STORED_SIDE[side] == OVER
    over_price = price if is_over else opposite["price"]
    under_price = opposite["price"] if is_over else price
    line = YES_NO_LINE if point is None else point

    try:
        return (line, int(over_price), int(under_price))
    except (TypeError, ValueError):
        return None
