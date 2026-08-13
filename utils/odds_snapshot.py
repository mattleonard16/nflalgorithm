"""Build a ``weekly_odds`` snapshot row from a scraper quote.

Tracked on purpose. The writer that calls this lives in a gitignored module,
so without a tracked home the rule that decides whether a price survives to
the database would be invisible to CI.

Two-sided capture is the point. ``implied_probability_no_vig`` needs both
sides of the same wager; with only the over price it cannot run, and the
consumer falls back to the vig-inclusive number — which is biased high by
the bookmaker margin on every single bet. The scraper already resolves both
sides (see ``utils.two_sided_odds``); this module is what carries the under
price the rest of the way into storage.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# The NFL table names the over side ``price``; NBA's equivalent table calls it
# ``over_price``. Anything reading across both has to know that.
OVER_PRICE_COLUMN = "price"
UNDER_PRICE_COLUMN = "under_price"


class UnusableQuoteError(ValueError):
    """A scraper quote that cannot become a snapshot row."""


def _optional_price(value: Any) -> Optional[int]:
    """Coerce a price to int, or ``None`` when the book did not quote one.

    A missing under price is normal — plenty of books post one side only —
    so this returns ``None`` rather than raising. A *malformed* price is not
    normal and is rejected: storing a silently dropped price would present a
    one-sided market as if the book had never quoted the other side.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise UnusableQuoteError(f"price must be numeric, got bool: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise UnusableQuoteError(f"unparseable price: {value!r}") from exc


def build_snapshot_row(
    quote: Mapping[str, Any],
    *,
    season: int,
    week: int,
    player_id: str,
    event_id: str,
    as_of: str,
    sportsbook_default: str = "RealBook",
) -> dict:
    """Normalize one scraper quote into a ``weekly_odds`` row.

    Args:
        quote: A scraper result carrying ``stat``, ``line``, ``over_odds`` and
            optionally ``under_odds`` and ``book``.
        season: NFL season the snapshot belongs to.
        week: NFL week the snapshot belongs to.
        player_id: Canonical player id, already resolved by the caller.
        event_id: Canonical game key, already resolved by the caller. This is
            what joins the snapshot to ``games`` for a kickoff.
        as_of: ISO-8601 capture timestamp. Snapshot ordering — and therefore
            which line counts as closing — depends on it.
        sportsbook_default: Book name to use when the quote carries none.

    Returns:
        A row dict ready for ``weekly_odds``. ``under_price`` is ``None`` when
        the book quoted only one side.

    Raises:
        UnusableQuoteError: The market or line is missing or unparseable, or
            the over price is absent. Such a row cannot be priced or graded,
            and storing it would inflate coverage counts with rows no
            downstream stage can use.
    """
    market = str(quote.get("stat") or "").strip()
    if not market:
        raise UnusableQuoteError("quote has no market")

    raw_line = quote.get("line")
    if raw_line is None:
        raise UnusableQuoteError(f"quote for {market} has no line")
    try:
        line = float(raw_line)
    except (TypeError, ValueError) as exc:
        raise UnusableQuoteError(f"unparseable line for {market}: {raw_line!r}") from exc

    over_price = _optional_price(quote.get("over_odds"))
    if over_price is None:
        raise UnusableQuoteError(f"quote for {market} has no over price")

    book = str(quote.get("book") or "").strip() or sportsbook_default

    return {
        "event_id": event_id,
        "season": season,
        "week": week,
        "player_id": player_id,
        "market": market,
        "sportsbook": book,
        "line": line,
        OVER_PRICE_COLUMN: over_price,
        UNDER_PRICE_COLUMN: _optional_price(quote.get("under_odds")),
        "as_of": as_of,
    }


def is_two_sided(row: Mapping[str, Any]) -> bool:
    """Whether a snapshot row carries both sides, so no-vig can run on it."""
    return row.get(OVER_PRICE_COLUMN) is not None and row.get(UNDER_PRICE_COLUMN) is not None
