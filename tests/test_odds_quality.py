"""Behavior of the odds screen that guards the value and CLV path."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.odds_quality import (
    describe_excluded,
    filter_gradeable_odds,
    is_synthetic_book,
)


def _odds(rows):
    return pd.DataFrame(rows, columns=["event_id", "player_id", "sportsbook", "line"])


REAL = ("2025_10_KC_BUF", "BUF_alpha_receiver", "DraftKings", 55.5)
SYNTH = ("2025_10_KC_BUF", "BUF_alpha_receiver", "SimBook", 55.5)
LEGACY = ("2025_W22_NE_a_hooper", "NE_a_hooper", "FanDuel", 21.5)
LEGACY_SYNTH = ("2025_W10_alpha_receiver", "alpha_receiver", "SimBook", 50.0)


class TestIsSyntheticBook:
    def test_identifies_the_generator(self):
        assert is_synthetic_book("SimBook")

    def test_is_case_and_whitespace_insensitive(self):
        # Book names arrive from several writers with inconsistent casing.
        assert is_synthetic_book(" simbook ")

    def test_real_books_are_not_synthetic(self):
        assert not is_synthetic_book("DraftKings")

    def test_missing_book_is_not_synthetic(self):
        assert not is_synthetic_book(None)


class TestFilterGradeableOdds:
    def test_keeps_real_rows_with_a_joinable_key(self):
        assert len(filter_gradeable_odds(_odds([REAL]))) == 1

    def test_drops_synthetic_rows_by_default(self):
        # A SimBook line is derived from the same week's realized yardage, so
        # any edge measured against it is circular.
        assert filter_gradeable_odds(_odds([SYNTH])).empty

    def test_drops_rows_whose_key_joins_to_no_game(self):
        assert filter_gradeable_odds(_odds([LEGACY])).empty

    def test_synthetic_rows_survive_when_the_caller_opts_in(self):
        kept = filter_gradeable_odds(_odds([SYNTH]), drop_synthetic=False)
        assert len(kept) == 1

    def test_unjoinable_rows_are_dropped_even_when_synthetic_is_allowed(self):
        # Joinability is not a preference: without a kickoff there is nothing
        # to grade against.
        assert filter_gradeable_odds(_odds([LEGACY_SYNTH]), drop_synthetic=False).empty

    def test_mixed_frame_keeps_only_the_gradeable_rows(self):
        kept = filter_gradeable_odds(_odds([REAL, SYNTH, LEGACY, LEGACY_SYNTH]))
        assert list(kept["sportsbook"]) == ["DraftKings"]

    def test_does_not_mutate_the_caller_s_frame(self):
        original = _odds([REAL, SYNTH])
        filter_gradeable_odds(original)
        assert len(original) == 2

    def test_empty_frame_passes_through(self):
        assert filter_gradeable_odds(_odds([])).empty

    def test_missing_event_id_column_fails_loud(self):
        with pytest.raises(KeyError, match="event_id"):
            filter_gradeable_odds(pd.DataFrame({"sportsbook": ["DraftKings"]}))

    def test_missing_sportsbook_column_fails_loud_when_screening_synthetic(self):
        with pytest.raises(KeyError, match="sportsbook"):
            filter_gradeable_odds(pd.DataFrame({"event_id": ["2025_10_KC_BUF"]}))


class TestDescribeExcluded:
    def test_counts_each_reason_separately(self):
        counts = describe_excluded(_odds([REAL, SYNTH, LEGACY, LEGACY_SYNTH]))
        assert counts == {"total": 4, "unjoinable": 2, "synthetic": 2, "gradeable": 1}

    def test_empty_frame_reports_zeroes(self):
        assert describe_excluded(_odds([])) == {
            "total": 0,
            "unjoinable": 0,
            "synthetic": 0,
            "gradeable": 0,
        }
