"""One card row per bet: the book with the best edge, kept whole."""

from __future__ import annotations

import pandas as pd

from utils.best_line import best_line_per_bet


def _row(sportsbook: str, edge: float, **overrides) -> dict:
    row = {
        "player_id": "p1",
        "market": "receiving_yards",
        "side": "over",
        "sportsbook": sportsbook,
        "line": 64.5,
        "price": -110,
        "edge_percentage": edge,
        "confidence_score": 0.8,
    }
    row.update(overrides)
    return row


def test_keeps_only_the_highest_edge_book_for_each_bet() -> None:
    card = pd.DataFrame(
        [_row("DraftKings", 0.08), _row("FanDuel", 0.12, line=63.5), _row("Bovada", 0.10)]
    )

    best = best_line_per_bet(card)

    assert best[["sportsbook", "line"]].to_dict("records") == [
        {"sportsbook": "FanDuel", "line": 63.5}
    ]


def test_never_fills_a_missing_value_from_another_book() -> None:
    # groupby().first() takes the first non-null value per column, so a best row
    # with a null field used to borrow that field from a worse book.
    card = pd.DataFrame([_row("FanDuel", 0.12, confidence_score=None), _row("DraftKings", 0.08)])

    best = best_line_per_bet(card)

    assert best.iloc[0]["sportsbook"] == "FanDuel"
    assert pd.isna(best.iloc[0]["confidence_score"])


def test_over_and_under_on_the_same_prop_are_separate_bets() -> None:
    card = pd.DataFrame([_row("FanDuel", 0.12), _row("DraftKings", 0.09, side="under", line=66.5)])

    best = best_line_per_bet(card)

    assert sorted(best["side"]) == ["over", "under"]


def test_equal_edges_pick_the_same_book_on_every_run() -> None:
    card = pd.DataFrame([_row("FanDuel", 0.10), _row("DraftKings", 0.10), _row("Bovada", 0.10)])

    best = best_line_per_bet(card)

    assert best.iloc[0]["sportsbook"] == "Bovada"


def test_result_is_ordered_by_edge_descending() -> None:
    card = pd.DataFrame(
        [
            _row("FanDuel", 0.05, player_id="a"),
            _row("FanDuel", 0.20, player_id="b"),
            _row("FanDuel", 0.10, player_id="c"),
        ]
    )

    best = best_line_per_bet(card)

    assert list(best["player_id"]) == ["b", "c", "a"]


def test_bet_key_can_leave_out_side_for_one_sided_cards() -> None:
    # The NBA card prices overs only and has no side column.
    card = pd.DataFrame([_row("FanDuel", 0.12), _row("DraftKings", 0.08)]).drop(columns="side")

    best = best_line_per_bet(card, keys=("player_id", "market"))

    assert list(best["sportsbook"]) == ["FanDuel"]
