"""Tests for utils/matching.py — position guards, suffix-preserving names,
latest-snapshot dedup, and the stale-line filter.

Imports only tracked modules (utils/), so the whole file runs in CI.

Fixtures mirror real collisions found in nfl_roster_players:
- Lamar Jackson: BAL QB vs ATL DB
- Justin Jefferson: MIN WR vs CLE LB
- David Long Jr. (TEN LB) vs David Long (IND DB)
- Michael Carter II (NYJ DB) vs Michael Carter (ARI RB)
"""

from __future__ import annotations

import pandas as pd
import pytest

from utils.clv import SNAPSHOT_KEY
from utils.matching import (
    StaleFilterResult,
    filter_stale_snapshots,
    latest_snapshot_per_key,
    name_variants,
    normalize_name,
    positions_compatible,
    strip_name_suffix,
    suffix_conflict,
)


# ======================================================================
# positions_compatible
# ======================================================================


class TestPositionsCompatible:
    def test_lamar_jackson_qb_vs_db_blocked_for_passing(self) -> None:
        # BAL QB projection vs ATL DB odds row: same name, different players.
        assert positions_compatible("QB", "DB", "passing_yards") is False

    def test_lamar_jackson_qb_vs_qb_allowed(self) -> None:
        assert positions_compatible("QB", "QB", "passing_yards") is True

    def test_db_with_unknown_counterpart_blocked_for_passing(self) -> None:
        # Only the ATL DB side is known: a DB cannot hold a passing_yards prop.
        assert positions_compatible(None, "DB", "passing_yards") is False
        assert positions_compatible("DB", "", "passing_yards") is False

    def test_qb_with_unknown_counterpart_plausible_for_passing(self) -> None:
        assert positions_compatible("QB", None, "passing_yards") is True
        assert positions_compatible("", "QB", "passing_yards") is True

    def test_justin_jefferson_wr_vs_lb_blocked_for_receiving(self) -> None:
        # MIN WR projection vs CLE LB odds row.
        assert positions_compatible("WR", "LB", "receiving_yards") is False

    def test_lb_with_unknown_counterpart_blocked_for_receiving(self) -> None:
        assert positions_compatible("LB", None, "receiving_yards") is False

    def test_receiving_plausible_positions(self) -> None:
        for pos in ("WR", "TE", "RB"):
            assert positions_compatible(pos, None, "receiving_yards") is True

    def test_rushing_plausible_positions(self) -> None:
        for pos in ("RB", "QB", "WR", "TE"):
            assert positions_compatible(pos, None, "rushing_yards") is True
        assert positions_compatible("DB", None, "rushing_yards") is False

    def test_unknown_vs_unknown_passes_but_is_callers_job_to_tag(self) -> None:
        assert positions_compatible(None, None, "receiving_yards") is True
        assert positions_compatible("", float("nan"), "passing_yards") is True

    def test_unmodeled_market_cannot_be_disproven(self) -> None:
        assert positions_compatible("TE", None, "receptions") is True

    def test_both_known_equality_is_case_insensitive(self) -> None:
        assert positions_compatible("wr", "WR ", "receiving_yards") is True

    @pytest.mark.parametrize(
        "proj,odds,market,expected",
        [
            # Both known: exact equality required, market irrelevant.
            ("QB", "QB", "passing_yards", True),
            ("QB", "DB", "passing_yards", False),
            ("RB", "WR", "rushing_yards", False),  # both plausible, still blocked
            ("WR", "WR", "receiving_yards", True),
            # One known: market plausibility decides.
            ("QB", None, "passing_yards", True),
            (None, "QB", "passing_yards", True),
            ("DB", None, "passing_yards", False),
            (None, "DB", "passing_yards", False),
            ("TE", None, "receiving_yards", True),
            (None, "LB", "receiving_yards", False),
            ("QB", None, "rushing_yards", True),
            (None, "K", "rushing_yards", False),
            # Both unknown: passes for every modeled market.
            (None, None, "passing_yards", True),
            (None, None, "receiving_yards", True),
            (None, None, "rushing_yards", True),
            # Unmodeled market / missing market cannot disprove a known side.
            ("K", None, "receptions", True),
            ("QB", None, None, True),
        ],
    )
    def test_full_truth_table(self, proj, odds, market, expected) -> None:
        assert positions_compatible(proj, odds, market) is expected

    def test_non_scalar_position_does_not_crash_and_fails_equality(self) -> None:
        # pd.isna on a multi-element sequence raises internally; the guard
        # must swallow that and fall through to string comparison, which
        # cannot equal a real position code.
        assert positions_compatible(["QB", "RB"], "QB", "passing_yards") is False


# ======================================================================
# Name variants (suffix-preserving vs stripped)
# ======================================================================


class TestNameVariants:
    def test_david_long_suffix_forms_stay_distinct(self) -> None:
        # TEN LB "David Long Jr." vs IND DB "David Long": the with-suffix
        # forms differ, so a suffix-preserving tier never merges them...
        ten = name_variants("David Long Jr.")
        ind = name_variants("David Long")
        assert ten.with_suffix == "david long jr"
        assert ind.with_suffix == "david long"
        assert ten.with_suffix != ind.with_suffix
        # ...while the stripped forms collide — the exact merge the old
        # destructive normalization performed.
        assert ten.stripped == ind.stripped == "david long"

    def test_michael_carter_suffix_forms_stay_distinct(self) -> None:
        # NYJ DB "Michael Carter II" vs ARI RB "Michael Carter".
        nyj = name_variants("Michael Carter II")
        ari = name_variants("Michael Carter")
        assert nyj.with_suffix == "michael carter ii"
        assert ari.with_suffix == "michael carter"
        assert nyj.with_suffix != ari.with_suffix
        assert nyj.stripped == ari.stripped == "michael carter"

    def test_same_player_with_and_without_suffix_meet_at_stripped_form(self) -> None:
        # Books often drop the suffix for the SAME player; the stripped form
        # is the legitimate fallback for that case.
        proj = name_variants("Michael Pittman Jr.")
        odds = name_variants("Michael Pittman")
        assert proj.stripped == odds.stripped == "michael pittman"

    def test_normalize_name_preserves_suffix_and_cleans(self) -> None:
        assert normalize_name("Ja'Marr Chase Jr.") == "ja marr chase jr"
        assert normalize_name("  D.K.   Metcalf ") == "dk metcalf"
        assert normalize_name(None) == ""
        assert normalize_name("") == ""

    def test_strip_name_suffix_removes_all_generational_tokens(self) -> None:
        assert strip_name_suffix("david long jr") == "david long"
        assert strip_name_suffix("michael carter ii") == "michael carter"
        assert strip_name_suffix("plain name") == "plain name"


class TestSuffixConflict:
    def test_david_long_jr_vs_david_long_is_a_conflict(self) -> None:
        # TEN LB "David Long Jr." vs IND DB "David Long": both roster names
        # known, stripped forms collide, suffix forms differ → different
        # players, a stripped-name merge must be blocked.
        assert suffix_conflict("David Long Jr.", "David Long") is True

    def test_michael_carter_ii_vs_michael_carter_is_a_conflict(self) -> None:
        # NYJ DB "Michael Carter II" vs ARI RB "Michael Carter".
        assert suffix_conflict("Michael Carter II", "Michael Carter") is True

    def test_identical_names_are_not_a_conflict(self) -> None:
        assert suffix_conflict("Michael Pittman Jr.", "Michael Pittman Jr") is False
        assert suffix_conflict("David Long", "David Long") is False

    def test_unknown_side_is_never_a_conflict(self) -> None:
        # A name derived from an already-stripped player_id carries no suffix
        # information; treating its absence as evidence would block legitimate
        # traded-player matches.
        assert suffix_conflict("David Long Jr.", "") is False
        assert suffix_conflict("", "David Long") is False
        assert suffix_conflict(None, None) is False

    def test_different_base_names_are_not_a_suffix_conflict(self) -> None:
        # Entirely different names are not this guard's business.
        assert suffix_conflict("Justin Jefferson", "Lamar Jackson") is False
        assert suffix_conflict("David Long Jr.", "Michael Carter II") is False

    def test_suffix_vs_different_suffix_is_a_conflict(self) -> None:
        # Jr vs Sr: stripped forms collide, suffix forms differ.
        assert suffix_conflict("David Long Jr.", "David Long Sr.") is True
        assert suffix_conflict("Michael Carter II", "Michael Carter III") is True

    def test_case_and_punctuation_variants_of_same_name_not_a_conflict(self) -> None:
        assert suffix_conflict("DAVID LONG JR.", "david long jr") is False

    def test_case_variant_with_missing_suffix_still_conflicts(self) -> None:
        assert suffix_conflict("DAVID LONG JR.", "David Long") is True


# ======================================================================
# latest_snapshot_per_key
# ======================================================================


def _odds_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _snapshot(as_of: str, line: float, **overrides) -> dict:
    row = {
        "event_id": "game1",
        "player_id": "BAL_lamar_jackson",
        "market": "passing_yards",
        "sportsbook": "DraftKings",
        "line": line,
        "price": -110,
        "as_of": as_of,
    }
    row.update(overrides)
    return row


class TestLatestSnapshotPerKey:
    def test_five_duplicate_snapshots_reduce_to_one_latest(self) -> None:
        frame = _odds_frame(
            [
                _snapshot("2025-11-01T10:00:00+00:00", 210.5),
                _snapshot("2025-11-02T10:00:00+00:00", 214.5),
                _snapshot("2025-11-05T09:00:00+00:00", 219.5),  # latest
                _snapshot("2025-11-03T10:00:00+00:00", 216.0),
                _snapshot("2025-11-04T10:00:00+00:00", 217.5),
            ]
        )
        result = latest_snapshot_per_key(frame)
        assert len(result) == 1
        assert result.iloc[0]["line"] == 219.5
        assert result.iloc[0]["as_of"] == "2025-11-05T09:00:00+00:00"

    def test_latest_is_chronological_not_lexical(self) -> None:
        # Lexically "2025-11-04T23:00:00-05:00" > "2025-11-05T01:00:00+00:00",
        # but chronologically it is 04:00 UTC on the 5th — the true latest.
        frame = _odds_frame(
            [
                _snapshot("2025-11-05T01:00:00+00:00", 210.5),
                _snapshot("2025-11-04T23:00:00-05:00", 225.5),
            ]
        )
        result = latest_snapshot_per_key(frame)
        assert len(result) == 1
        assert result.iloc[0]["line"] == 225.5

    def test_multiple_keys_each_keep_their_own_latest(self) -> None:
        frame = _odds_frame(
            [
                _snapshot("2025-11-01T10:00:00+00:00", 210.5),
                _snapshot("2025-11-02T10:00:00+00:00", 212.5),
                _snapshot("2025-11-01T10:00:00+00:00", 60.5, sportsbook="FanDuel"),
                _snapshot(
                    "2025-11-03T10:00:00+00:00",
                    75.5,
                    player_id="MIN_justin_jefferson",
                    market="receiving_yards",
                ),
            ]
        )
        result = latest_snapshot_per_key(frame)
        assert len(result) == 3
        dk = result[result["sportsbook"] == "DraftKings"]
        assert set(dk["line"]) == {212.5, 75.5}
        assert not result.duplicated(subset=SNAPSHOT_KEY).any()

    def test_duplicate_index_labels_still_yield_one_row_per_key(self) -> None:
        # idxmax over duplicate index labels would be ambiguous; the function
        # must reindex positionally rather than trust the caller's index.
        frame = _odds_frame(
            [
                _snapshot("2025-11-01T10:00:00+00:00", 210.5),
                _snapshot("2025-11-02T10:00:00+00:00", 214.5),
                _snapshot("2025-11-03T10:00:00+00:00", 219.5),  # latest
            ]
        )
        frame.index = [0, 0, 0]
        result = latest_snapshot_per_key(frame)
        assert len(result) == 1
        assert result.iloc[0]["line"] == 219.5

    def test_unparseable_as_of_raises(self) -> None:
        frame = _odds_frame(
            [
                _snapshot("2025-11-01T10:00:00+00:00", 210.5),
                _snapshot("not-a-timestamp", 214.5),
            ]
        )
        with pytest.raises(ValueError, match="not parseable"):
            latest_snapshot_per_key(frame)

    def test_missing_key_column_raises(self) -> None:
        frame = _odds_frame([_snapshot("2025-11-01T10:00:00+00:00", 210.5)])
        with pytest.raises(ValueError, match="missing required columns"):
            latest_snapshot_per_key(frame.drop(columns=["sportsbook"]))

    def test_empty_frame_passes_through(self) -> None:
        empty = pd.DataFrame(columns=[*SNAPSHOT_KEY, "as_of", "line"])
        assert latest_snapshot_per_key(empty).empty

    def test_none_input_returns_empty_frame_with_key_columns(self) -> None:
        result = latest_snapshot_per_key(None)
        assert result.empty
        assert list(result.columns) == [*SNAPSHOT_KEY, "as_of"]

    def test_tie_on_as_of_keeps_exactly_one_row_deterministically(self) -> None:
        # Same key, identical timestamps: idxmax keeps the FIRST occurrence.
        frame = _odds_frame(
            [
                _snapshot("2025-11-05T09:00:00+00:00", 210.5),
                _snapshot("2025-11-05T09:00:00+00:00", 219.5),
                _snapshot("2025-11-05T09:00:00+00:00", 214.0),
            ]
        )
        result = latest_snapshot_per_key(frame)
        assert len(result) == 1
        assert result.iloc[0]["line"] == 210.5

    def test_tie_across_spellings_of_same_instant(self) -> None:
        # 09:00+00:00 and 09:00Z parse to the same instant — a tie, resolved
        # to the first tied row, never two rows for one key.
        frame = _odds_frame(
            [
                _snapshot("2025-11-01T00:00:00+00:00", 200.0),
                _snapshot("2025-11-05T09:00:00+00:00", 210.5),
                _snapshot("2025-11-05T09:00:00Z", 219.5),
            ]
        )
        result = latest_snapshot_per_key(frame)
        assert len(result) == 1
        assert result.iloc[0]["line"] == 210.5


# ======================================================================
# filter_stale_snapshots
# ======================================================================


class TestFilterStaleSnapshots:
    KICKOFF = "2025-11-09T18:00:00+00:00"

    def _kickoffs(self) -> pd.DataFrame:
        return pd.DataFrame([{"event_id": "game1", "kickoff_utc": self.KICKOFF}])

    def test_fresh_snapshot_kept(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T12:00:00+00:00", 210.5)])
        result = filter_stale_snapshots(odds, self._kickoffs())
        assert isinstance(result, StaleFilterResult)
        assert len(result.frame) == 1
        assert result.unjoinable_count == 0

    def test_pre_kickoff_stale_snapshot_dropped(self) -> None:
        # 60h before an 18:00 kickoff with a 48h window: stale.
        odds = _odds_frame([_snapshot("2025-11-07T06:00:00+00:00", 210.5)])
        result = filter_stale_snapshots(odds, self._kickoffs(), max_age_hours=48)
        assert result.frame.empty
        assert result.unjoinable_count == 0

    def test_post_kickoff_snapshot_dropped(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T19:00:00+00:00", 210.5)])
        result = filter_stale_snapshots(odds, self._kickoffs())
        assert result.frame.empty

    def test_unjoinable_rows_pass_through_and_are_counted(self) -> None:
        odds = _odds_frame(
            [
                _snapshot("2025-11-09T12:00:00+00:00", 210.5),  # fresh, joins
                _snapshot("2025-11-01T00:00:00+00:00", 60.5, event_id="mystery"),
                _snapshot("2025-11-02T00:00:00+00:00", 61.5, event_id="mystery"),
            ]
        )
        result = filter_stale_snapshots(odds, self._kickoffs())
        # The two mystery rows would be "stale" by age but have no kickoff to
        # judge against — they pass through untouched and are counted.
        assert len(result.frame) == 3
        assert result.unjoinable_count == 2

    def test_mixed_frame_filters_only_joinable_stale_rows(self) -> None:
        odds = _odds_frame(
            [
                _snapshot("2025-11-09T17:59:00+00:00", 212.5),  # kept
                _snapshot("2025-11-07T06:00:00+00:00", 210.5),  # stale, dropped
                _snapshot("2025-11-10T00:00:00+00:00", 214.5),  # post-kick, dropped
                _snapshot("2025-11-05T00:00:00+00:00", 99.5, event_id="mystery"),
            ]
        )
        result = filter_stale_snapshots(odds, self._kickoffs(), max_age_hours=48)
        assert set(result.frame["line"]) == {212.5, 99.5}
        assert result.unjoinable_count == 1

    def test_empty_kickoffs_passes_everything_with_full_count(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T12:00:00+00:00", 210.5)])
        result = filter_stale_snapshots(odds, pd.DataFrame())
        assert len(result.frame) == 1
        assert result.unjoinable_count == 1

    def test_conflicting_kickoffs_raise(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T12:00:00+00:00", 210.5)])
        kicks = pd.DataFrame(
            [
                {"event_id": "game1", "kickoff_utc": self.KICKOFF},
                {"event_id": "game1", "kickoff_utc": "2025-11-10T18:00:00+00:00"},
            ]
        )
        with pytest.raises(ValueError, match="conflicting kickoff"):
            filter_stale_snapshots(odds, kicks)

    def test_equivalent_kickoff_spellings_are_not_a_conflict(self) -> None:
        # Same instant, three spellings: 18:00Z == 18:00+00:00 == 13:00-05:00.
        odds = _odds_frame(
            [
                _snapshot("2025-11-09T12:00:00+00:00", 210.5),  # fresh, kept
                _snapshot("2025-11-09T19:00:00+00:00", 214.5),  # post-kick
            ]
        )
        kicks = pd.DataFrame(
            [
                {"event_id": "game1", "kickoff_utc": self.KICKOFF},
                {"event_id": "game1", "kickoff_utc": "2025-11-09T18:00:00Z"},
                {"event_id": "game1", "kickoff_utc": "2025-11-09T13:00:00-05:00"},
            ]
        )
        result = filter_stale_snapshots(odds, kicks)
        assert set(result.frame["line"]) == {210.5}
        assert result.unjoinable_count == 0

    def test_non_positive_window_raises(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T12:00:00+00:00", 210.5)])
        with pytest.raises(ValueError, match="max_age_hours"):
            filter_stale_snapshots(odds, self._kickoffs(), max_age_hours=0)

    def test_unparseable_kickoff_raises(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T12:00:00+00:00", 210.5)])
        kicks = pd.DataFrame([{"event_id": "game1", "kickoff_utc": "TBD"}])
        with pytest.raises(ValueError, match="kickoff_utc"):
            filter_stale_snapshots(odds, kicks)

    # -- exact boundaries: staleness is strict (<, >), the boundary is fresh --

    def test_exactly_at_max_age_boundary_is_kept(self) -> None:
        # Kickoff 18:00 on the 9th, 48h window: 18:00 on the 7th is NOT stale.
        odds = _odds_frame([_snapshot("2025-11-07T18:00:00+00:00", 210.5)])
        result = filter_stale_snapshots(odds, self._kickoffs(), max_age_hours=48)
        assert len(result.frame) == 1

    def test_one_second_older_than_max_age_is_dropped(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-07T17:59:59+00:00", 210.5)])
        result = filter_stale_snapshots(odds, self._kickoffs(), max_age_hours=48)
        assert result.frame.empty

    def test_exactly_at_kickoff_is_kept(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T18:00:00+00:00", 210.5)])
        result = filter_stale_snapshots(odds, self._kickoffs())
        assert len(result.frame) == 1

    def test_one_second_after_kickoff_is_dropped(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T18:00:01+00:00", 210.5)])
        result = filter_stale_snapshots(odds, self._kickoffs())
        assert result.frame.empty

    # -- degenerate inputs --

    def test_none_odds_returns_empty_result(self) -> None:
        result = filter_stale_snapshots(None, self._kickoffs())
        assert result.frame.empty
        assert list(result.frame.columns) == ["event_id", "as_of"]
        assert result.unjoinable_count == 0

    def test_missing_as_of_column_raises(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T12:00:00+00:00", 210.5)]).drop(
            columns=["as_of"]
        )
        with pytest.raises(ValueError, match="missing required columns"):
            filter_stale_snapshots(odds, self._kickoffs())

    def test_kickoffs_missing_column_raises(self) -> None:
        odds = _odds_frame([_snapshot("2025-11-09T12:00:00+00:00", 210.5)])
        kicks = pd.DataFrame([{"event_id": "game1"}])
        with pytest.raises(ValueError, match="kickoffs_df missing required columns"):
            filter_stale_snapshots(odds, kicks)
