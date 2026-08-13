"""Behavior of the odds -> game key resolver."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.event_keys import (
    UnresolvableEventError,
    canonical_event_id,
    event_id_for_team,
    is_canonical_event_id,
    matchups_by_team,
    resolve_event_id,
)


class TestCanonicalEventId:
    def test_matches_nflverse_game_id_format(self):
        assert canonical_event_id(2025, 1, "DAL", "PHI") == "2025_01_DAL_PHI"

    def test_pads_week_so_keys_sort_and_join(self):
        assert canonical_event_id(2026, 9, "KC", "BUF") == "2026_09_KC_BUF"

    def test_postseason_weeks_are_representable(self):
        assert canonical_event_id(2025, 22, "KC", "SF") == "2025_22_KC_SF"

    def test_team_aliases_are_canonicalized(self):
        # Whatever alias a feed uses, one game gets one key.
        assert canonical_event_id(2025, 1, "dal", "phi") == "2025_01_DAL_PHI"

    def test_away_and_home_are_not_interchangeable(self):
        assert canonical_event_id(2025, 1, "DAL", "PHI") != canonical_event_id(
            2025, 1, "PHI", "DAL"
        )

    @pytest.mark.parametrize("week", [0, 23, -1])
    def test_week_outside_the_season_is_rejected(self, week):
        with pytest.raises(UnresolvableEventError, match="week out of range"):
            canonical_event_id(2025, week, "DAL", "PHI")

    def test_unresolvable_team_is_rejected(self):
        with pytest.raises(UnresolvableEventError, match="unresolvable team codes"):
            canonical_event_id(2025, 1, "NOT_A_TEAM_AT_ALL", "PHI")

    def test_team_playing_itself_is_rejected(self):
        with pytest.raises(UnresolvableEventError, match="cannot play itself"):
            canonical_event_id(2025, 1, "DAL", "DAL")

    def test_non_numeric_season_is_rejected(self):
        with pytest.raises(UnresolvableEventError, match="season is not an integer"):
            canonical_event_id("twenty-five", 1, "DAL", "PHI")


class TestIsCanonicalEventId:
    def test_accepts_a_real_nflverse_game_id(self):
        assert is_canonical_event_id("2025_01_DAL_PHI")

    @pytest.mark.parametrize(
        "value",
        [
            "2025_W10_alpha_receiver",  # legacy data_pipeline per-player key
            "2025_W22_NE_a_hooper",  # legacy, observed in the live DB
            "2025_W1_KC_synthetic",  # legacy scraper synthetic key
            "2025_1_DAL_PHI",  # week not zero-padded
            "0bd2ab1f5a8e3c",  # Odds API opaque id
            "",
            None,
            42,
        ],
    )
    def test_rejects_everything_that_does_not_join_to_games(self, value):
        assert not is_canonical_event_id(value)


class TestResolveEventId:
    def test_canonical_existing_id_is_preserved(self):
        assert (
            resolve_event_id(2025, 1, home_team="PHI", away_team="DAL", existing="2025_01_KC_LAC")
            == "2025_01_KC_LAC"
        )

    def test_opaque_provider_id_is_replaced_by_the_matchup(self):
        # The Odds API id is meaningful to the provider but joins to nothing here.
        assert (
            resolve_event_id(2025, 1, home_team="PHI", away_team="DAL", existing="0bd2ab1f5a8e3c")
            == "2025_01_DAL_PHI"
        )

    def test_legacy_player_keyed_id_is_replaced_by_the_matchup(self):
        assert (
            resolve_event_id(
                2025, 22, home_team="NE", away_team="BUF", existing="2025_W22_NE_d_maye"
            )
            == "2025_22_BUF_NE"
        )

    def test_missing_matchup_fails_loud_rather_than_inventing_a_key(self):
        with pytest.raises(UnresolvableEventError, match="no matchup supplied"):
            resolve_event_id(2025, 1, existing="2025_W10_alpha_receiver")

    def test_missing_matchup_with_no_existing_id_fails_loud(self):
        with pytest.raises(UnresolvableEventError):
            resolve_event_id(2025, 1)

    def test_placeholder_team_is_rejected_rather_than_stored(self):
        # The scraper's synthetic path used "TBD" as an away team.
        with pytest.raises(UnresolvableEventError):
            resolve_event_id(2025, 1, home_team="KC", away_team="TBD")


class TestMatchupsByTeam:
    """Recovering a full matchup from just the player's club."""

    @staticmethod
    def _schedule(pairs):
        return pd.DataFrame(pairs, columns=["home_team", "away_team"])

    def test_both_clubs_index_to_the_same_matchup(self):
        matchups = matchups_by_team(self._schedule([("PHI", "DAL")]))
        assert matchups["PHI"] == ("PHI", "DAL")
        assert matchups["DAL"] == ("PHI", "DAL")

    def test_full_club_names_are_canonicalized(self):
        matchups = matchups_by_team(
            self._schedule([("Kansas City Chiefs", "Los Angeles Chargers")])
        )
        assert matchups["KC"] == ("KC", "LAC")

    def test_row_with_an_unresolvable_club_is_skipped_entirely(self):
        # Half-resolving would key a real club to a garbage opponent.
        matchups = matchups_by_team(self._schedule([("KC", "Toronto Argonauts"), ("PHI", "DAL")]))
        assert "KC" not in matchups
        assert matchups["PHI"] == ("PHI", "DAL")

    def test_empty_schedule_yields_no_matchups(self):
        assert matchups_by_team(self._schedule([])) == {}


class TestEventIdForTeam:
    """Resolving a game key when the caller holds only a player's team."""

    MATCHUPS = {"PHI": ("PHI", "DAL"), "DAL": ("PHI", "DAL")}

    def test_home_club_resolves_to_the_away_first_key(self):
        assert event_id_for_team(2025, 1, "PHI", self.MATCHUPS) == "2025_01_DAL_PHI"

    def test_away_club_resolves_to_the_same_key(self):
        assert event_id_for_team(2025, 1, "DAL", self.MATCHUPS) == "2025_01_DAL_PHI"

    def test_club_on_bye_fails_loud(self):
        with pytest.raises(UnresolvableEventError, match="no scheduled game"):
            event_id_for_team(2025, 1, "KC", self.MATCHUPS)

    def test_unresolvable_club_fails_loud(self):
        with pytest.raises(UnresolvableEventError, match="unresolvable team code"):
            event_id_for_team(2025, 1, "not a team", self.MATCHUPS)

    def test_missing_team_fails_loud_rather_than_defaulting(self):
        with pytest.raises(UnresolvableEventError):
            event_id_for_team(2025, 1, None, self.MATCHUPS)

    def test_empty_schedule_makes_every_club_unresolvable(self):
        # A failed schedule load must not silently produce keys.
        with pytest.raises(UnresolvableEventError):
            event_id_for_team(2025, 1, "PHI", {})
