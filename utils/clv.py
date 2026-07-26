"""Closing Line Value (CLV) math for NFL weekly bets.

Pure functions only — no database access — so the logic stays testable in CI
even though the caller (``scripts/record_outcomes.py``) is gitignored.

CLV answers the only question that separates a lucky week from a real edge:
did the market move toward the number we took? Two representations are
returned because they answer different questions:

- ``clv_points``  — line movement in stat units, sign-corrected for side.
  Intuitive, but not comparable across markets (2 receiving yards is noise,
  2 receptions is enormous).
- ``clv_bp``      — movement in no-vig probability space, in basis points.
  Comparable across markets and books; this is the honest metric.

Missing or single-snapshot inputs return an explicit ``insufficient_snapshots``
status. CLV of "unknown" must never be silently recorded as zero — a zero would
average into weekly CLV and understate or overstate the edge.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from value_betting_engine import implied_probability_no_vig, prob_over

# Key identifying one book's quote on one player prop.
SNAPSHOT_KEY = ["event_id", "player_id", "market", "sportsbook"]

# Columns carried from the closing snapshot.
_CLOSE_COLUMNS = ["line", "price", "under_price", "as_of"]

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_snapshots"


def resolve_closing_lines(odds_df: pd.DataFrame) -> pd.DataFrame:
    """Return the closing snapshot per ``SNAPSHOT_KEY``.

    Closing is defined as the row at ``MAX(as_of)`` for the key. The ``games``
    table carries no populated ``kickoff_utc``, so "last snapshot before
    kickoff" is not expressible yet; this matches the NBA reference
    (``scripts/record_nba_outcomes.py``) and ``scripts/backfill_line_accuracy.py``.

    Adds ``snapshot_count`` so callers can distinguish a real close from a
    single-scrape key where CLV is undefined.

    Args:
        odds_df: Rows from ``weekly_odds``. Must contain ``SNAPSHOT_KEY``,
            ``line``, ``price`` and ``as_of``; ``under_price`` is optional.

    Returns:
        One row per key with ``close_line``, ``close_price``,
        ``close_under_price``, ``closed_at`` and ``snapshot_count``.
    """
    out_columns = [
        *SNAPSHOT_KEY,
        "close_line",
        "close_price",
        "close_under_price",
        "closed_at",
        "snapshot_count",
    ]

    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=out_columns)

    missing = [c for c in [*SNAPSHOT_KEY, "line", "price", "as_of"] if c not in odds_df.columns]
    if missing:
        raise ValueError(f"odds_df missing required columns: {missing}")

    frame = odds_df.copy()
    if "under_price" not in frame.columns:
        frame["under_price"] = pd.NA

    # as_of is an ISO-8601 string; lexical max equals chronological max only for
    # a uniform offset, so sort on parsed timestamps instead.
    frame["_as_of_ts"] = pd.to_datetime(frame["as_of"], errors="coerce", utc=True, format="mixed")
    if frame["_as_of_ts"].isna().any():
        bad = frame.loc[frame["_as_of_ts"].isna(), "as_of"].unique()[:5]
        raise ValueError(f"weekly_odds.as_of values are not parseable timestamps: {list(bad)}")

    counts = frame.groupby(SNAPSHOT_KEY, dropna=False).size().rename("snapshot_count")
    latest_idx = frame.groupby(SNAPSHOT_KEY, dropna=False)["_as_of_ts"].idxmax()

    closing = frame.loc[latest_idx, [*SNAPSHOT_KEY, *_CLOSE_COLUMNS]].rename(
        columns={
            "line": "close_line",
            "price": "close_price",
            "under_price": "close_under_price",
            "as_of": "closed_at",
        }
    )
    closing = closing.merge(counts.reset_index(), on=SNAPSHOT_KEY, how="left")
    return closing.reset_index(drop=True)[out_columns]


def _fair_prob(
    line: float,
    price: Any,
    under_price: Any,
    side: str,
    *,
    mu: float | None,
    sigma: float | None,
) -> float:
    """Fair (no-vig) probability that ``side`` wins at ``line``.

    Uses the two-sided book quote when both prices exist, which removes the
    bookmaker margin. When only one price is quoted, vig cannot be isolated
    from a single number, so fall back to the model's own distribution — that
    keeps both sides of the CLV comparison on the same scale.

    Callers must guarantee one of the two paths is available; the raise is a
    programming-error guard, not an expected branch.
    """
    over_odds = _as_odds(price)
    under_odds = _as_odds(under_price)

    if over_odds is not None and under_odds is not None:
        p_over, p_under = implied_probability_no_vig(over_odds, under_odds)
    elif mu is not None and sigma is not None and sigma > 0:
        p_over = prob_over(mu, sigma, float(line))
        p_under = 1.0 - p_over
    else:
        raise ValueError("fair probability needs a two-sided quote or a positive sigma")

    return p_over if side == "over" else p_under


def _as_odds(value: Any) -> int | None:
    """Coerce an American-odds cell to int, or None when absent/unusable."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    try:
        odds = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return odds or None


def compute_clv(entry: Mapping[str, Any], close: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compute CLV for one bet against its closing snapshot.

    Args:
        entry: The bet as placed. Requires ``line``, ``side``; optionally
            ``price``, ``under_price``, ``mu``, ``sigma``.
        close: The closing snapshot from :func:`resolve_closing_lines`, or
            ``None`` when no snapshot matched.

    Returns:
        On success: ``status='ok'`` with ``clv_points``, ``clv_bp``,
        ``close_line``, ``close_price``, ``closed_at``.
        Otherwise ``status='insufficient_snapshots'`` with a ``reason`` and
        ``clv_points``/``clv_bp`` set to ``None`` — never 0.
    """
    side = str(entry.get("side") or "over").lower()
    if side not in ("over", "under"):
        raise ValueError(f"unsupported bet side: {entry.get('side')!r}")

    if close is None:
        return _insufficient("no matching odds snapshot for bet key")

    snapshot_count = close.get("snapshot_count")
    if snapshot_count is not None and not pd.isna(snapshot_count) and int(snapshot_count) < 2:
        return _insufficient("only one odds snapshot recorded for this key")

    entry_line = _as_float(entry.get("line"))
    close_line = _as_float(close.get("close_line"))
    if entry_line is None or close_line is None:
        return _insufficient("entry or closing line missing")

    # Points CLV: an over bettor gains when the line moves down, an under
    # bettor when it moves up.
    raw_points = entry_line - close_line
    clv_points = raw_points if side == "over" else -raw_points

    mu = _as_float(entry.get("mu"))
    sigma = _as_float(entry.get("sigma"))
    has_model = mu is not None and sigma is not None and sigma > 0

    two_sided_entry = (
        _as_odds(entry.get("price")) is not None and _as_odds(entry.get("under_price")) is not None
    )
    two_sided_close = (
        _as_odds(close.get("close_price")) is not None
        and _as_odds(close.get("close_under_price")) is not None
    )

    if (two_sided_entry or has_model) and (two_sided_close or has_model):
        entry_prob = _fair_prob(
            entry_line, entry.get("price"), entry.get("under_price"), side, mu=mu, sigma=sigma
        )
        close_prob = _fair_prob(
            close_line,
            close.get("close_price"),
            close.get("close_under_price"),
            side,
            mu=mu,
            sigma=sigma,
        )
        # We beat the close when the fair probability of our side rose after we
        # took it: the number we hold is now better than the market's.
        clv_bp = round((close_prob - entry_prob) * 10_000.0, 4)
    else:
        # One-sided quote and no model distribution: probability-space CLV is
        # not computable. Report unknown rather than inventing a 0.
        clv_bp = None

    return {
        "status": STATUS_OK,
        "clv_points": round(clv_points, 4),
        "clv_bp": clv_bp,
        "close_line": close_line,
        "close_price": _as_odds(close.get("close_price")),
        "closed_at": close.get("closed_at"),
    }


def _insufficient(reason: str) -> dict[str, Any]:
    return {
        "status": STATUS_INSUFFICIENT,
        "reason": reason,
        "clv_points": None,
        "clv_bp": None,
        "close_line": None,
        "close_price": None,
        "closed_at": None,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
