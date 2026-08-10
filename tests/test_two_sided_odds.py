"""Behavior of over/under pairing, which no-vig probability depends on."""

from __future__ import annotations

import pytest

from utils.two_sided_odds import find_opposite_side, pair_two_sided_prices


def _outcome(name, point, price, description="Alpha Receiver"):
    return {"name": name, "point": point, "price": price, "description": description}


ALT_MARKET = [
    _outcome("Over", 55.5, -110),
    _outcome("Under", 55.5, -105),
    _outcome("Over", 70.5, +150),
    _outcome("Under", 70.5, -180),
]


class TestFindOppositeSide:
    def test_pairs_the_under_at_the_same_line(self):
        found = find_opposite_side(
            ALT_MARKET, description="Alpha Receiver", side="Over", point=55.5
        )
        assert found["price"] == -105

    def test_does_not_pair_across_alternate_lines(self):
        """An Over at 55.5 paired with an Under at 70.5 is a different wager;
        no-vig on that pair returns a confident wrong number."""
        found = find_opposite_side(
            ALT_MARKET, description="Alpha Receiver", side="Over", point=70.5
        )
        assert found["price"] == -180

    def test_pairs_the_over_when_holding_the_under(self):
        found = find_opposite_side(
            ALT_MARKET, description="Alpha Receiver", side="Under", point=70.5
        )
        assert found["price"] == 150

    def test_does_not_cross_players(self):
        market = [_outcome("Over", 55.5, -110), _outcome("Under", 55.5, -105, "Other Guy")]
        assert (
            find_opposite_side(market, description="Alpha Receiver", side="Over", point=55.5)
            is None
        )

    def test_one_sided_line_has_no_match(self):
        market = [_outcome("Over", 55.5, -110), _outcome("Under", 70.5, -180)]
        assert (
            find_opposite_side(market, description="Alpha Receiver", side="Over", point=55.5)
            is None
        )

    def test_float_noise_still_matches(self):
        market = [_outcome("Under", 55.500000000000004, -105)]
        found = find_opposite_side(market, description="Alpha Receiver", side="Over", point=55.5)
        assert found is not None

    def test_a_quote_without_a_line_matches_only_another_without_one(self):
        market = [_outcome("Under", None, -105)]
        assert (
            find_opposite_side(market, description="Alpha Receiver", side="Over", point=None)
            is not None
        )
        assert (
            find_opposite_side(market, description="Alpha Receiver", side="Over", point=55.5)
            is None
        )

    def test_a_quote_without_a_description_matches_only_another_without_one(self):
        # A book that omits the player name gives no evidence the two quotes
        # are the same wager; pairing them would invent that evidence.
        market = [_outcome("Under", 55.5, -105, description=None)]
        assert find_opposite_side(market, description=None, side="Over", point=55.5) is not None
        assert (
            find_opposite_side(market, description="Alpha Receiver", side="Over", point=55.5)
            is None
        )

    def test_unknown_side_fails_loud(self):
        with pytest.raises(ValueError, match="side must be"):
            find_opposite_side(ALT_MARKET, description="Alpha Receiver", side="Yes", point=55.5)


class TestPairTwoSidedPrices:
    def test_returns_line_and_both_prices(self):
        assert pair_two_sided_prices(ALT_MARKET[0], ALT_MARKET) == (55.5, -110, -105)

    def test_orders_prices_by_side_not_by_argument(self):
        # Starting from the Under must yield the same tuple as starting from
        # the Over, or the two sides get swapped in the no-vig call.
        assert pair_two_sided_prices(ALT_MARKET[1], ALT_MARKET) == (55.5, -110, -105)

    def test_alternate_line_pairs_with_its_own_counterpart(self):
        assert pair_two_sided_prices(ALT_MARKET[2], ALT_MARKET) == (70.5, 150, -180)

    def test_one_sided_quote_is_rejected(self):
        market = [_outcome("Over", 55.5, -110)]
        assert pair_two_sided_prices(market[0], market) is None

    def test_missing_price_is_rejected(self):
        market = [_outcome("Over", 55.5, None), _outcome("Under", 55.5, -105)]
        assert pair_two_sided_prices(market[0], market) is None

    def test_opposite_side_missing_its_price_is_rejected(self):
        market = [_outcome("Over", 55.5, -110), _outcome("Under", 55.5, None)]
        assert pair_two_sided_prices(market[0], market) is None

    def test_non_over_under_outcome_is_rejected(self):
        market = [_outcome("Yes", 55.5, -110)]
        assert pair_two_sided_prices(market[0], market) is None
