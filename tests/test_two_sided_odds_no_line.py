"""A quote with no line is unusable, not a line of 0.0 (utils/two_sided_odds.py).

Kept apart from tests/test_two_sided_odds.py, which covers the same-line
pairing rule. Storing a lineless pair at 0.0 would grade every stat against
zero and call every over a win.
"""

from __future__ import annotations

from utils.two_sided_odds import pair_two_sided_prices


def _outcome(name, point, price, description="Alpha Receiver"):
    return {"name": name, "point": point, "price": price, "description": description}


def test_a_two_sided_quote_with_no_line_is_skipped():
    market = [_outcome("Over", None, -110), _outcome("Under", None, -110)]

    assert pair_two_sided_prices(market[0], market) is None
    assert pair_two_sided_prices(market[1], market) is None


def test_a_line_of_zero_is_still_a_line():
    market = [_outcome("Over", 0.0, -110), _outcome("Under", 0.0, -110)]

    assert pair_two_sided_prices(market[0], market) == (0.0, -110, -110)
