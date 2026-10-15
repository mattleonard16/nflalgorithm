"""Tests for closing-line-value math (utils/clv.py).

Hand-built frames only, no database. The consumer is
`scripts/record_outcomes.py`; its database path is tested in
`tests/test_record_outcomes.py`.
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
    """Took over 50.5, market closed at 54.5 → the over bettor holds the
    easier number, +4 points."""
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
    assert result["clv_points"] == pytest.approx(4.0)
    assert result["close_line"] == 54.5
    assert result["closed_at"] == "2025-11-27T18:00:00+00:00"


def test_compute_clv_over_loses_when_line_drops():
    entry = {"line": 54.5, "side": "over", "price": -110, "under_price": -110}
    close = {
        "close_line": 50.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 4,
    }

    assert compute_clv(entry, close)["clv_points"] == pytest.approx(-4.0)


def test_compute_clv_under_sign_is_corrected():
    """An under bettor gains when the line moves down — opposite sign to over."""
    close = {
        "close_line": 54.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 3,
    }

    over = compute_clv({"line": 50.5, "side": "over", "price": -110, "under_price": -110}, close)
    under = compute_clv({"line": 50.5, "side": "under", "price": -110, "under_price": -110}, close)

    assert over["clv_points"] == pytest.approx(4.0)
    assert under["clv_points"] == pytest.approx(-4.0)


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
    # Over 50.5 against a 48.5 close: the over bettor took the worse number.
    assert result["clv_points"] == pytest.approx(-2.0)
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
    # The line rose to 54.5: the over at 50.5 is likelier to win than the over
    # at the close, so the bettor beat it.
    assert result["clv_bp"] > 0


def test_an_unmoved_line_with_only_stored_entry_probabilities_is_zero():
    """The 2026 week 1 bug. materialized_value_view stores de-vigged
    probabilities but no under price, so the entry priced off the model while
    the close priced off the market. Subtracting those gave -3608 bp on a bet
    whose line and price never moved."""
    entry = {
        "line": 20.5,
        "side": "over",
        "price": -116,
        "implied_prob": 0.537037,
        "implied_prob_under": 0.462963,
        "mu": 40.7,
        "sigma": 19.6,
    }
    close = {
        "close_line": 20.5,
        "close_price": -116,
        "close_under_price": -108,
        "closed_at": "2026-09-10T00:00:00+00:00",
        "snapshot_count": 5,
    }

    result = compute_clv(entry, close)

    assert result["status"] == STATUS_OK
    assert result["clv_points"] == pytest.approx(0.0)
    assert abs(result["clv_bp"]) < 500


def test_stored_entry_probabilities_are_ignored_when_they_still_carry_vig():
    """A pair that does not sum to 1 was never de-vigged. Using it anyway
    biases every bet the same way."""
    entry = {
        "line": 20.5,
        "side": "over",
        "price": -116,
        "implied_prob": 0.537,
        "implied_prob_under": 0.537,
        "mu": 20.5,
        "sigma": 10.0,
    }
    close = {
        "close_line": 20.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2026-09-10T00:00:00+00:00",
        "snapshot_count": 5,
    }

    result = compute_clv(entry, close)

    # Falls back to the model on both sides: same line, same curve, no movement.
    assert result["clv_bp"] == pytest.approx(0.0, abs=1e-6)


def test_a_model_priced_entry_is_never_compared_against_a_market_priced_close():
    """Mixing the two measures the model's edge, not line movement. With no
    market probability on the entry and no model to fall back on, the answer is
    unknown."""
    entry = {"line": 20.5, "side": "over", "price": -116}
    close = {
        "close_line": 20.5,
        "close_price": -110,
        "close_under_price": -110,
        "closed_at": "2026-09-10T00:00:00+00:00",
        "snapshot_count": 5,
    }

    result = compute_clv(entry, close)

    assert result["status"] == STATUS_OK
    assert result["clv_bp"] is None


def test_compute_clv_rejects_unknown_side():
    with pytest.raises(ValueError, match="unsupported bet side"):
        compute_clv({"line": 50.5, "side": "middle"}, None)


def test_fair_prob_prices_anytime_touchdown_with_poisson_survival():
    """Model-fallback CLV on a TD row must use Poisson, not the Gaussian CDF."""
    import math

    from utils.clv import _fair_prob

    mu, sigma, line = 0.65, 0.45, 0.5
    assert _fair_prob(
        line, -110, None, "over", mu=mu, sigma=sigma, market="anytime_touchdown"
    ) == pytest.approx(1.0 - math.exp(-mu))


def test_fair_prob_prices_yardage_markets_the_same_way_the_model_does():
    """CLV must read the same curve the bet was priced on. Yardage moved from
    the normal CDF to gamma survival in utils.nfl_markets; pricing CLV off the
    old curve would book a probability change that never happened."""
    from utils.nfl_markets import prob_over

    from utils.clv import _fair_prob

    mu, sigma, line = 55.0, 20.0, 50.5
    assert _fair_prob(
        line, -110, None, "over", mu=mu, sigma=sigma, market="receiving_yards"
    ) == pytest.approx(prob_over(mu, sigma, line, market="receiving_yards"))


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
    assert result["clv_points"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Line movement in the market branch
# ---------------------------------------------------------------------------


def _two_sided_close(line: float, price: int = -110, under_price: int = -110) -> dict:
    return {
        "close_line": line,
        "close_price": price,
        "close_under_price": under_price,
        "closed_at": "2025-11-27T18:00:00+00:00",
        "snapshot_count": 3,
    }


def test_market_clv_prices_a_line_move_at_the_entry_line():
    """Over 50.5 at -110/-110 against a 54.5 close at -110/-110 used to read
    0 bp: two fair probabilities at two different lines. The close is now
    translated to our line through the mean it implies."""
    from utils.market_blend import implied_mean
    from utils.nfl_markets import prob_over

    entry = {
        "line": 50.5,
        "side": "over",
        "price": -110,
        "under_price": -110,
        "mu": 60.0,
        "sigma": 20.0,
        "market": "receiving_yards",
    }

    result = compute_clv(entry, _two_sided_close(54.5))

    close_mean = implied_mean(54.5, 20.0, 0.5, market="receiving_yards")
    expected = prob_over(close_mean, 20.0, 50.5, market="receiving_yards") - 0.5
    assert result["clv_bp"] == pytest.approx(expected * 10_000, abs=1e-2)
    assert result["clv_bp"] > 0


def test_market_clv_on_a_moved_line_without_sigma_is_unknown():
    """Without a curve the close cannot be moved to our line. Unknown, not 0."""
    entry = {"line": 50.5, "side": "over", "price": -110, "under_price": -110}

    result = compute_clv(entry, _two_sided_close(54.5))

    assert result["status"] == STATUS_OK
    assert result["clv_points"] == pytest.approx(4.0)
    assert result["clv_bp"] is None


_MODEL = {"mu": 52.0, "sigma": 18.0, "market": "receiving_yards"}


@pytest.mark.parametrize("side", ["over", "under"])
@pytest.mark.parametrize("branch", ["market", "model"])
def test_a_line_move_toward_our_side_is_positive_clv(side, branch):
    """The close moved to a number harder for our side: we beat it."""
    entry_line = 50.5
    close_line = entry_line + 4.0 if side == "over" else entry_line - 4.0
    entry = {"line": entry_line, "side": side, "price": -110, **_MODEL}
    close = _two_sided_close(close_line)
    if branch == "market":
        entry["under_price"] = -110
    else:
        close["close_under_price"] = None

    result = compute_clv(entry, close)

    assert result["clv_points"] == pytest.approx(4.0)
    assert result["clv_bp"] > 0

    # And the mirror move is negative.
    away = dict(close, close_line=2 * entry_line - close_line)
    mirrored = compute_clv(entry, away)
    assert mirrored["clv_points"] == pytest.approx(-4.0)
    assert mirrored["clv_bp"] < 0


@pytest.mark.parametrize("side", ["over", "under"])
def test_a_price_move_toward_our_side_is_positive_clv(side):
    """Same line, the market shortened our side: we beat the close."""
    entry = {"line": 50.5, "side": side, "price": -110, "under_price": -110, **_MODEL}
    shorter_over = _two_sided_close(50.5, price=-140, under_price=120)
    shorter_under = _two_sided_close(50.5, price=120, under_price=-140)

    toward = compute_clv(entry, shorter_over if side == "over" else shorter_under)
    away = compute_clv(entry, shorter_under if side == "over" else shorter_over)

    assert toward["clv_points"] == pytest.approx(0.0)
    assert toward["clv_bp"] > 0
    assert away["clv_bp"] < 0


def test_out_of_range_odds_cells_are_treated_as_missing():
    """A 0 or 50 in a price cell is not American odds; it must not crash the
    grading run or price a probability."""
    entry = {"line": 50.5, "side": "over", "price": 50, "under_price": -110}

    result = compute_clv(entry, _two_sided_close(50.5, price=0))

    assert result["status"] == STATUS_OK
    assert result["clv_bp"] is None
    assert result["close_price"] is None


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
