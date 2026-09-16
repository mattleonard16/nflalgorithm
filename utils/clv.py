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

from utils.nfl_markets import prob_over

# Key identifying one book's quote on one player prop.
SNAPSHOT_KEY = ["event_id", "player_id", "market", "sportsbook"]

# Columns carried from the closing snapshot.
_CLOSE_COLUMNS = ["line", "price", "under_price", "as_of"]

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_snapshots"


def resolve_closing_lines(
    odds_df: pd.DataFrame, kickoffs: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the closing snapshot per ``SNAPSHOT_KEY``.

    Closing is the last snapshot taken *before kickoff* when ``kickoffs`` is
    supplied. That is the real definition: a quote captured after the game
    started is not a closing line, and grading against one would score the
    model on information the market already had.

    Without ``kickoffs`` the definition degrades to ``MAX(as_of)``, matching
    the NBA reference (``scripts/record_nba_outcomes.py``) and
    ``scripts/backfill_line_accuracy.py``. That fallback existed because
    ``event_id`` used to hold per-player strings that joined to no game; now
    that snapshots carry canonical game keys, callers should pass kickoffs.

    A key whose every snapshot is post-kickoff yields no closing row at all,
    rather than a row silently graded off a stale quote.

    Adds ``snapshot_count`` so callers can distinguish a real close from a
    single-scrape key where CLV is undefined. The count reflects only the
    pre-kickoff snapshots actually eligible to close.

    Args:
        odds_df: Rows from ``weekly_odds``. Must contain ``SNAPSHOT_KEY``,
            ``line``, ``price`` and ``as_of``; ``under_price`` is optional.
        kickoffs: Optional rows with ``event_id`` and ``kickoff_utc``. Keys
            with no matching kickoff keep the ``MAX(as_of)`` definition, so a
            partially-populated schedule degrades per-key rather than wholesale.

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

    frame = _drop_post_kickoff_snapshots(frame, kickoffs)
    if frame.empty:
        return pd.DataFrame(columns=out_columns)

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


def _drop_post_kickoff_snapshots(
    frame: pd.DataFrame, kickoffs: pd.DataFrame | None
) -> pd.DataFrame:
    """Keep only snapshots taken at or before their game's kickoff.

    Keys with no known kickoff are kept unchanged: a missing schedule row must
    not silently discard a book's whole quote history. ``frame`` is expected to
    carry the parsed ``_as_of_ts`` column.
    """
    if kickoffs is None or kickoffs.empty:
        return frame

    missing = [c for c in ("event_id", "kickoff_utc") if c not in kickoffs.columns]
    if missing:
        raise ValueError(f"kickoffs missing required columns: {missing}")

    schedule = kickoffs[["event_id", "kickoff_utc"]].drop_duplicates(subset="event_id").copy()
    schedule["_kickoff_ts"] = pd.to_datetime(
        schedule["kickoff_utc"], errors="coerce", utc=True, format="mixed"
    )
    schedule = schedule.loc[schedule["_kickoff_ts"].notna(), ["event_id", "_kickoff_ts"]]
    if schedule.empty:
        return frame

    merged = frame.merge(schedule, on="event_id", how="left")
    # NaT compares False, which is what an unknown kickoff should mean here:
    # the row is kept rather than judged against a timestamp we do not have.
    post_kickoff = merged["_kickoff_ts"].notna() & (merged["_as_of_ts"] > merged["_kickoff_ts"])
    return merged.loc[~post_kickoff].drop(columns="_kickoff_ts")


def _stored_fair_prob(row: Any, side: str) -> float | None:
    """Fair probability of ``side`` from de-vigged probabilities on the row.

    ``materialized_value_view`` stores ``implied_prob``/``implied_prob_under``,
    already de-vigged at the time the bet was priced, but it does not store the
    under price the pair came from. Without this, an entry row has no market
    probability at all and the comparison silently falls back to the model.

    Returns ``None`` unless both cells are present and sum to 1 within a
    thousandth: a pair that does not sum to 1 was never de-vigged, and treating
    a vigged number as fair biases every CLV in the same direction.
    """
    over = _as_float(_cell(row, "implied_prob"))
    under = _as_float(_cell(row, "implied_prob_under"))
    if over is None or under is None:
        return None
    if not 0.0 < over < 1.0 or not 0.0 < under < 1.0:
        return None
    if abs(over + under - 1.0) > 1e-3:
        return None
    return over if side == "over" else under


def _market_fair_prob(row: Any, price_key: str, under_key: str, side: str) -> float | None:
    """Fair probability of ``side`` from the market alone, or ``None``.

    Never falls back to the model: a CLV that subtracts a market probability
    from a model one measures the model's edge, not line movement.
    """
    over_odds = _as_odds(_cell(row, price_key))
    under_odds = _as_odds(_cell(row, under_key))
    if over_odds is not None and under_odds is not None:
        from value_betting_engine import implied_probability_no_vig

        p_over, p_under = implied_probability_no_vig(over_odds, under_odds)
        # value_betting_engine is gitignored and untyped, so both are Any here.
        return float(p_over if side == "over" else p_under)
    return _stored_fair_prob(row, side)


def _cell(row: Any, key: str) -> Any:
    """Read ``key`` off a mapping or Series, returning None when absent."""
    try:
        return row.get(key)
    except AttributeError:
        return None


def _fair_prob(
    line: float,
    price: Any,
    under_price: Any,
    side: str,
    *,
    mu: float | None,
    sigma: float | None,
    market: str | None = None,
) -> float:
    """Fair (no-vig) probability that ``side`` wins at ``line``.

    Uses the two-sided book quote when both prices exist, which removes the
    bookmaker margin. When only one price is quoted, vig cannot be isolated
    from a single number, so fall back to the model's own distribution — that
    keeps both sides of the CLV comparison on the same scale. ``market`` is
    forwarded so anytime-touchdown rows price via Poisson survival instead of
    the Gaussian CDF.

    Callers must guarantee one of the two paths is available; the raise is a
    programming-error guard, not an expected branch.

    ``implied_probability_no_vig`` is imported here rather than at module scope:
    it still lives in gitignored ``value_betting_engine``, so a top-level import
    makes this module — and every test that touches it — fail to import in CI,
    which is the opposite of why the math lives in a tracked file. Tests that
    exercise the no-vig path must inject prices and are skipped when the private
    module is absent. The single-price fallback has no such constraint: it takes
    ``prob_over`` from tracked ``utils.nfl_markets``, so CI covers it. That is
    only true if the import sits inside the two-sided branch: at the top of the
    function it raised ImportError before the fallback was ever reached, which
    took the model path down in CI too.
    """
    over_odds = _as_odds(price)
    under_odds = _as_odds(under_price)

    if over_odds is not None and under_odds is not None:
        from value_betting_engine import implied_probability_no_vig

        p_over, p_under = implied_probability_no_vig(over_odds, under_odds)
    elif mu is not None and sigma is not None and sigma > 0:
        p_over = prob_over(mu, sigma, float(line), market=market)
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

    # The market rides along so model-fallback probabilities price count
    # props (anytime TD) with Poisson survival rather than a Gaussian CDF.
    # Normalized to str-or-None: a NaN cell must not reach prob_over, whose
    # `"touchdown" in market` check assumes an iterable.
    candidate = entry.get("market") or close.get("market")
    market = (
        None
        if candidate is None or (not isinstance(candidate, str) and pd.isna(candidate))
        else str(candidate)
    )

    # Both probabilities must come from the same source. Pricing the entry off
    # the model and the close off the market subtracts a model probability from
    # a market one, which is the model's edge with the sign flipped, not closing
    # line value. That produced a -1929 bp weekly average on 2026 week 1 where
    # entry and close were the identical line at the identical price, so the
    # true answer was 0.
    entry_market = _market_fair_prob(entry, "price", "under_price", side)
    close_market = _market_fair_prob(close, "close_price", "close_under_price", side)

    if entry_market is not None and close_market is not None:
        entry_prob: float | None = entry_market
        close_prob: float | None = close_market
    elif has_model:
        # No market probability on one side. The model's own distribution is a
        # consistent basis for both, and it still measures the move: the same
        # curve read at the entry line against the closing line.
        entry_prob = _fair_prob(
            entry_line, None, None, side, mu=mu, sigma=sigma, market=market
        )
        close_prob = _fair_prob(
            close_line, None, None, side, mu=mu, sigma=sigma, market=market
        )
    else:
        # One-sided quote and no model distribution: probability-space CLV is
        # not computable. Report unknown rather than inventing a 0.
        entry_prob = close_prob = None

    if entry_prob is None or close_prob is None:
        clv_bp = None
    else:
        # We beat the close when the fair probability of our side rose after we
        # took it: the number we hold is now better than the market's.
        clv_bp = round((close_prob - entry_prob) * 10_000.0, 4)

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
