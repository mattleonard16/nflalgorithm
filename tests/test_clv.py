"""Tests for closing-line-value math (utils/clv.py).

Hand-built frames only — no database. The consumer
(`scripts/record_outcomes.py`) is gitignored, so this is the CI-reachable
contract for the CLV computation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from utils.clv import STATUS_INSUFFICIENT, STATUS_OK, compute_clv, resolve_closing_lines


def _odds_row(as_of: str, line: float, price: int = -110, under_price: int | None = -110) -> dict:
    return {
        "event_id": "2025_W13_evt",
        "player_id": "P1",
        "market": "receiving_yards",
        "sportsbook": "SimBook",
        "line": line,
        "price": price,
        "under_price": under_price,
        "as_of": as_of,
    }


# ---------------------------------------------------------------------------
# resolve_closing_lines
# ---------------------------------------------------------------------------


def test_resolve_closing_lines_takes_latest_snapshot():
    """Closing line is the row at MAX(as_of) for the key."""
    odds = pd.DataFrame(
        [
            _odds_row("2025-11-25T12:00:00+00:00", 50.5),
            _odds_row("2025-11-27T18:00:00+00:00", 54.5),
            _odds_row("2025-11-26T09:00:00+00:00", 52.5),
        ]
    )

    closing = resolve_closing_lines(odds)

    assert len(closing) == 1
    row = closing.iloc[0]
    assert row["close_line"] == 54.5
    assert row["closed_at"] == "2025-11-27T18:00:00+00:00"
    assert row["snapshot_count"] == 3


def test_resolve_closing_lines_separates_sportsbooks():
    """Each (event, player, market, book) key closes independently."""
    odds = pd.DataFrame(
        [
            _odds_row("2025-11-25T12:00:00+00:00", 50.5),
            _odds_row("2025-11-27T18:00:00+00:00", 54.5),
            {**_odds_row("2025-11-27T18:00:00+00:00", 48.5), "sportsbook": "OtherBook"},
        ]
    )

    closing = resolve_closing_lines(odds).set_index("sportsbook")

    assert closing.loc["SimBook", "close_line"] == 54.5
    assert closing.loc["OtherBook", "close_line"] == 48.5
    assert closing.loc["OtherBook", "snapshot_count"] == 1


def test_resolve_closing_lines_handles_empty_frame():
    closing = resolve_closing_lines(pd.DataFrame())
    assert closing.empty
    assert "close_line" in closing.columns


def test_resolve_closing_lines_rejects_missing_columns():
    """Fail loud at the boundary rather than silently dropping keys."""
    with pytest.raises(ValueError, match="missing required columns"):
        resolve_closing_lines(pd.DataFrame([{"event_id": "e", "line": 1.0}]))


def test_resolve_closing_lines_rejects_unparseable_as_of():
    odds = pd.DataFrame([_odds_row("not-a-timestamp", 50.5)])
    with pytest.raises(ValueError, match="not parseable timestamps"):
        resolve_closing_lines(odds)


def test_resolve_closing_lines_orders_by_time_not_string():
    """Mixed UTC offsets must compare chronologically, not lexically."""
    odds = pd.DataFrame(
        [
            # Later in real time (17:00Z) but lexically smaller than "2025-11-27T18:00".
            _odds_row("2025-11-27T12:00:00-05:00", 60.5),
            _odds_row("2025-11-27T16:00:00+00:00", 54.5),
        ]
    )

    closing = resolve_closing_lines(odds)
    assert closing.iloc[0]["close_line"] == 60.5


# ---------------------------------------------------------------------------
# compute_clv
# ---------------------------------------------------------------------------


def test_compute_clv_over_beats_close_when_line_rises():
    """Took over 50.5, market closed at 54.5 → negative points CLV."""
    entry = {"line": 50.5, "side": "over", "price": -110, "under_price": -110}
    close = {
        "close_line": 54.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 3,
    }

    result = compute_clv(entry, close)

    assert result["status"] == STATUS_OK
    assert result["clv_points"] == pytest.approx(-4.0)
    assert result["close_line"] == 54.5
    assert result["closed_at"] == "2025-11-27T18:00:00+00:00"


def test_compute_clv_over_gains_when_line_drops():
    entry = {"line": 54.5, "side": "over", "price": -110, "under_price": -110}
    close = {
        "close_line": 50.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 4,
    }

    assert compute_clv(entry, close)["clv_points"] == pytest.approx(4.0)


def test_compute_clv_under_sign_is_corrected():
    """An under bettor gains when the line moves up — opposite sign to over."""
    close = {
        "close_line": 54.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 3,
    }

    over = compute_clv({"line": 50.5, "side": "over", "price": -110, "under_price": -110}, close)
    under = compute_clv({"line": 50.5, "side": "under", "price": -110, "under_price": -110}, close)

    assert over["clv_points"] == pytest.approx(-4.0)
    assert under["clv_points"] == pytest.approx(4.0)


def test_compute_clv_bp_uses_no_vig_probabilities():
    """Price movement alone moves clv_bp, with the book margin removed.

    Raw implied probs at -140/+120 sum to >1.0; the bp figure must be computed
    from the normalized pair, so a symmetric -110/-110 entry against a -140
    close yields exactly the no-vig probability delta.
    """
    entry = {"line": 50.5, "side": "over", "price": -110, "under_price": -110}
    close = {
        "close_line": 50.5,
        "close_price": -140,
        "close_under_price": 120,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 2,
    }

    result = compute_clv(entry, close)

    # entry fair p_over = 0.5; close fair p_over from -140/+120.
    raw_over = 140 / 240
    raw_under = 100 / 220
    expected_close = raw_over / (raw_over + raw_under)
    assert result["clv_bp"] == pytest.approx((expected_close - 0.5) * 10_000, abs=1e-2)
    # Sanity: the vig-included figure would be materially larger.
    assert result["clv_bp"] < (raw_over - 0.5) * 10_000


def test_compute_clv_bp_zero_line_and_price_unchanged():
    """No movement at all → exactly zero bp, not noise."""
    quote = {"price": -110, "under_price": -110}
    entry = {"line": 50.5, "side": "over", **quote}
    close = {
        "close_line": 50.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 5,
    }

    result = compute_clv(entry, close)
    assert result["clv_points"] == pytest.approx(0.0)
    assert result["clv_bp"] == pytest.approx(0.0)


def test_compute_clv_single_snapshot_is_insufficient():
    """One scrape means no close to compare against — never report 0."""
    entry = {"line": 50.5, "side": "over", "price": -110, "under_price": -110}
    close = {
        "close_line": 50.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 1,
    }

    result = compute_clv(entry, close)

    assert result["status"] == STATUS_INSUFFICIENT
    assert result["clv_points"] is None
    assert result["clv_bp"] is None
    assert "one odds snapshot" in result["reason"]


def test_compute_clv_missing_close_is_insufficient():
    result = compute_clv({"line": 50.5, "side": "over"}, None)
    assert result["status"] == STATUS_INSUFFICIENT
    assert result["clv_bp"] is None


def test_compute_clv_missing_line_is_insufficient():
    close = {"close_line": None, "closed_at": "x", "snapshot_count": 3}
    result = compute_clv({"line": 50.5, "side": "over"}, close)
    assert result["status"] == STATUS_INSUFFICIENT


def test_compute_clv_one_sided_quote_without_model_reports_unknown_bp():
    """Points CLV still resolves; probability CLV is unknown, not zero."""
    entry = {"line": 50.5, "side": "over", "price": -110, "under_price": None}
    close = {
        "close_line": 48.5,
        "close_price": -115,
        "close_under_price": None,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 3,
    }

    result = compute_clv(entry, close)

    assert result["status"] == STATUS_OK
    assert result["clv_points"] == pytest.approx(2.0)
    assert result["clv_bp"] is None


def test_compute_clv_one_sided_quote_falls_back_to_model_distribution():
    """With mu/sigma available, a one-sided book still yields a bp figure."""
    entry = {
        "line": 50.5,
        "side": "over",
        "price": -110,
        "under_price": None,
        "mu": 55.0,
        "sigma": 20.0,
    }
    close = {
        "close_line": 54.5,
        "close_price": -110,
        "close_under_price": None,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 3,
    }

    result = compute_clv(entry, close)

    assert result["status"] == STATUS_OK
    # Line moved against the over bettor, so fair p_over at the close is lower.
    assert result["clv_bp"] < 0


def test_compute_clv_rejects_unknown_side():
    with pytest.raises(ValueError, match="unsupported bet side"):
        compute_clv({"line": 50.5, "side": "middle"}, None)


def test_resolve_then_compute_end_to_end():
    """The two functions compose on a realistic multi-snapshot frame."""
    odds = pd.DataFrame(
        [
            _odds_row("2025-11-25T12:00:00+00:00", 50.5),
            _odds_row("2025-11-26T12:00:00+00:00", 52.5),
            _odds_row("2025-11-27T18:00:00+00:00", 54.5),
        ]
    )

    closing = resolve_closing_lines(odds).iloc[0].to_dict()
    result = compute_clv(
        {"line": 50.5, "side": "over", "price": -110, "under_price": -110}, closing
    )

    assert result["status"] == STATUS_OK
    assert result["clv_points"] == pytest.approx(-4.0)
