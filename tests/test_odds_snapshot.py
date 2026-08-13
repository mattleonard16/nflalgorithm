"""Behavior of scraper-quote to snapshot-row normalization."""

from __future__ import annotations

import pytest

from utils.odds_snapshot import UnusableQuoteError, build_snapshot_row, is_two_sided

KEYS = {
    "season": 2026,
    "week": 1,
    "player_id": "BUF_alpha_receiver",
    "event_id": "2026_01_KC_BUF",
    "as_of": "2026-09-10T18:00:00+00:00",
}


def _quote(**overrides):
    quote = {
        "stat": "receiving_yards",
        "line": 55.5,
        "over_odds": -110,
        "under_odds": -105,
        "book": "DraftKings",
    }
    quote.update(overrides)
    return quote


class TestTwoSidedCapture:
    def test_carries_both_prices_through(self):
        """The whole point: without the under price, no-vig cannot run and the
        consumer falls back to a number biased by the bookmaker margin."""
        row = build_snapshot_row(_quote(), **KEYS)
        assert row["price"] == -110
        assert row["under_price"] == -105
        assert is_two_sided(row)

    def test_one_sided_quote_stores_a_null_under_price(self):
        row = build_snapshot_row(_quote(under_odds=None), **KEYS)
        assert row["price"] == -110
        assert row["under_price"] is None
        assert not is_two_sided(row)

    def test_absent_under_key_is_the_same_as_a_null_one(self):
        quote = _quote()
        del quote["under_odds"]
        assert build_snapshot_row(quote, **KEYS)["under_price"] is None

    def test_positive_american_odds_survive(self):
        row = build_snapshot_row(_quote(over_odds=150, under_odds=-180), **KEYS)
        assert (row["price"], row["under_price"]) == (150, -180)

    def test_string_prices_are_coerced(self):
        row = build_snapshot_row(_quote(over_odds="-110", under_odds="-105"), **KEYS)
        assert (row["price"], row["under_price"]) == (-110, -105)


class TestKeysAndIdentity:
    def test_row_carries_the_resolved_game_key(self):
        assert build_snapshot_row(_quote(), **KEYS)["event_id"] == "2026_01_KC_BUF"

    def test_row_carries_season_week_and_player(self):
        row = build_snapshot_row(_quote(), **KEYS)
        assert (row["season"], row["week"]) == (2026, 1)
        assert row["player_id"] == "BUF_alpha_receiver"

    def test_as_of_is_preserved_exactly(self):
        # Snapshot ordering decides which line is closing; a rewritten
        # timestamp would silently reorder that.
        assert build_snapshot_row(_quote(), **KEYS)["as_of"] == KEYS["as_of"]

    def test_book_name_is_kept(self):
        assert build_snapshot_row(_quote(), **KEYS)["sportsbook"] == "DraftKings"

    def test_missing_book_falls_back_to_the_default(self):
        row = build_snapshot_row(_quote(book=None), **KEYS, sportsbook_default="RealBook")
        assert row["sportsbook"] == "RealBook"

    def test_blank_book_falls_back_too(self):
        assert build_snapshot_row(_quote(book="   "), **KEYS)["sportsbook"] == "RealBook"

    def test_line_is_stored_as_a_float(self):
        row = build_snapshot_row(_quote(line="55.5"), **KEYS)
        assert row["line"] == 55.5
        assert isinstance(row["line"], float)


class TestRejection:
    def test_missing_market_is_rejected(self):
        with pytest.raises(UnusableQuoteError, match="no market"):
            build_snapshot_row(_quote(stat=""), **KEYS)

    def test_missing_line_is_rejected(self):
        with pytest.raises(UnusableQuoteError, match="no line"):
            build_snapshot_row(_quote(line=None), **KEYS)

    def test_unparseable_line_is_rejected(self):
        with pytest.raises(UnusableQuoteError, match="unparseable line"):
            build_snapshot_row(_quote(line="fifty"), **KEYS)

    def test_missing_over_price_is_rejected(self):
        """A row with no over price cannot be priced or graded; storing it
        would inflate coverage with rows nothing downstream can use."""
        with pytest.raises(UnusableQuoteError, match="no over price"):
            build_snapshot_row(_quote(over_odds=None), **KEYS)

    def test_unparseable_under_price_fails_loud(self):
        # Not silently dropped to NULL: that would present a two-sided market
        # as one-sided and quietly disable no-vig for the row.
        with pytest.raises(UnusableQuoteError, match="unparseable price"):
            build_snapshot_row(_quote(under_odds="even"), **KEYS)

    def test_boolean_price_is_rejected(self):
        with pytest.raises(UnusableQuoteError, match="bool"):
            build_snapshot_row(_quote(under_odds=True), **KEYS)
