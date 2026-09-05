"""Tests for closing-line-value math (utils/clv.py).

Hand-built frames only — no database. The consumer
(`scripts/record_outcomes.py`) is gitignored, so this is the CI-reachable
contract for the CLV computation.

`resolve_closing_lines` is pure pandas and always runs. The probability
conversion inside `compute_clv` delegates to the gitignored
`value_betting_engine`, so tests that reach it are marked `needs_engine` and
skip when that module is absent — the snapshot-resolution and
insufficient-data contracts stay covered either way.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from utils.clv import STATUS_INSUFFICIENT, STATUS_OK, compute_clv, resolve_closing_lines

needs_engine = pytest.mark.skipif(
    importlib.util.find_spec("value_betting_engine") is None,
    reason="value_betting_engine is gitignored and absent in this checkout",
)


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


@needs_engine
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


@needs_engine
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


@needs_engine
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


@needs_engine
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


@needs_engine
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


@needs_engine
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


@needs_engine
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


@needs_engine
def test_fair_prob_prices_anytime_touchdown_with_poisson_survival():
    """Model-fallback CLV on a TD row must use Poisson, not the Gaussian CDF."""
    import math

    from utils.clv import _fair_prob

    mu, sigma, line = 0.65, 0.45, 0.5
    assert _fair_prob(
        line, -110, None, "over", mu=mu, sigma=sigma, market="anytime_touchdown"
    ) == pytest.approx(1.0 - math.exp(-mu))


@needs_engine
def test_fair_prob_keeps_gaussian_pricing_for_yardage_markets():
    """The market parameter must not disturb continuous-prop pricing."""
    from scipy.stats import norm

    from utils.clv import _fair_prob

    mu, sigma, line = 55.0, 20.0, 50.5
    assert _fair_prob(
        line, -110, None, "over", mu=mu, sigma=sigma, market="receiving_yards"
    ) == pytest.approx(float(1.0 - norm.cdf(line, loc=mu, scale=sigma)))


@needs_engine
def test_compute_clv_carries_market_into_model_fallback():
    """One-sided TD quotes fall back to the model distribution — via Poisson."""
    import math

    entry = {
        "line": 0.5,
        "side": "over",
        "price": -110,
        "under_price": None,
        "mu": 0.65,
        "sigma": 0.45,
        "market": "anytime_touchdown",
    }
    close = {
        "close_line": 0.5,
        "close_price": -110,
        "close_under_price": None,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 3,
        "market": "anytime_touchdown",
    }

    result = compute_clv(entry, close)

    assert result["status"] == STATUS_OK
    # Same line at entry and close with the same model distribution: no movement.
    assert result["clv_bp"] == pytest.approx(0.0)
    # And the probability actually used is the Poisson figure, not Gaussian.
    from utils.clv import _fair_prob

    assert _fair_prob(
        0.5, -110, None, "over", mu=0.65, sigma=0.45, market="anytime_touchdown"
    ) == pytest.approx(1.0 - math.exp(-0.65))


@needs_engine
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


# ---------------------------------------------------------------------------
# Kickoff-aware closing line
# ---------------------------------------------------------------------------


def _kickoff_odds(as_of: str, line: float) -> dict:
    row = _odds_row(as_of, line)
    row["event_id"] = "2025_13_KC_BUF"
    return row


def _kickoffs(kickoff_utc: str = "2025-11-27T18:00:00+00:00") -> pd.DataFrame:
    return pd.DataFrame([{"event_id": "2025_13_KC_BUF", "kickoff_utc": kickoff_utc}])


def test_closing_line_ignores_snapshots_taken_after_kickoff():
    """A quote captured mid-game is not a closing line; grading against it
    would score the model on information the market already had."""
    odds = pd.DataFrame(
        [
            _kickoff_odds("2025-11-27T17:00:00+00:00", 54.5),
            _kickoff_odds("2025-11-27T21:00:00+00:00", 80.5),  # in-game
        ]
    )

    closing = resolve_closing_lines(odds, _kickoffs())

    assert closing.iloc[0]["close_line"] == 54.5
    # The discarded in-game row must not inflate the count either.
    assert closing.iloc[0]["snapshot_count"] == 1


def test_snapshot_exactly_at_kickoff_still_closes():
    odds = pd.DataFrame([_kickoff_odds("2025-11-27T18:00:00+00:00", 54.5)])

    closing = resolve_closing_lines(odds, _kickoffs())

    assert closing.iloc[0]["close_line"] == 54.5


def test_key_with_only_post_kickoff_snapshots_yields_no_closing_row():
    """Better to report nothing than to grade off a stale in-game quote."""
    odds = pd.DataFrame([_kickoff_odds("2025-11-27T23:00:00+00:00", 80.5)])

    assert resolve_closing_lines(odds, _kickoffs()).empty


def test_key_with_no_known_kickoff_keeps_the_latest_snapshot():
    """A partially populated schedule degrades per key, not wholesale."""
    odds = pd.DataFrame(
        [
            _odds_row("2025-11-25T12:00:00+00:00", 50.5),
            _odds_row("2025-11-27T21:00:00+00:00", 56.5),
        ]
    )

    closing = resolve_closing_lines(odds, _kickoffs())

    assert closing.iloc[0]["close_line"] == 56.5


def test_unparseable_kickoff_does_not_discard_the_quote_history():
    odds = pd.DataFrame([_kickoff_odds("2025-11-27T21:00:00+00:00", 56.5)])

    closing = resolve_closing_lines(odds, _kickoffs("not a timestamp"))

    assert closing.iloc[0]["close_line"] == 56.5


def test_omitting_kickoffs_preserves_the_max_as_of_definition():
    odds = pd.DataFrame(
        [
            _kickoff_odds("2025-11-27T17:00:00+00:00", 54.5),
            _kickoff_odds("2025-11-27T21:00:00+00:00", 80.5),
        ]
    )

    assert resolve_closing_lines(odds).iloc[0]["close_line"] == 80.5


def test_kickoffs_missing_required_columns_fails_loud():
    odds = pd.DataFrame([_kickoff_odds("2025-11-27T17:00:00+00:00", 54.5)])

    with pytest.raises(ValueError, match="kickoff_utc"):
        resolve_closing_lines(odds, pd.DataFrame({"event_id": ["2025_13_KC_BUF"]}))


def test_kickoff_filter_applies_per_event_not_globally():
    """One game's kickoff must not truncate another game's quote history."""
    early = _kickoff_odds("2025-11-27T21:00:00+00:00", 80.5)
    late = dict(early)
    late["event_id"] = "2025_13_SF_SEA"

    closing = resolve_closing_lines(
        pd.DataFrame([early, late]),
        pd.DataFrame(
            [
                {"event_id": "2025_13_KC_BUF", "kickoff_utc": "2025-11-27T18:00:00+00:00"},
                {"event_id": "2025_13_SF_SEA", "kickoff_utc": "2025-11-28T18:00:00+00:00"},
            ]
        ),
    )

    assert list(closing["event_id"]) == ["2025_13_SF_SEA"]
