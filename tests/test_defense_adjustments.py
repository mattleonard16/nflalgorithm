"""Tests for utils/defense_adjustments.py — pure multiplier math, no DB.

Covers the 2026-prep repairs:
- normalization invariant: each (position, stat_type) group averages ~1.0
- shrinkage: thin samples land closer to 1.0 than large samples with the
  same raw signal
- passing_yards is a first-class supported matchup (was silently 1.0)
- unknown stat_type fails loud instead of silently returning 1.0
- clamp bounds still enforced as guardrails
"""

import pandas as pd
import pytest

from utils import defense_adjustments as da
from utils.defense_adjustments import (
    CLAMP_BOUNDS,
    COMPUTED_MATCHUPS,
    NEUTRAL_MATCHUPS,
    SUPPORTED_MATCHUPS,
    compute_multipliers_from_game_stats,
    get_defense_multiplier,
)

STAT_COLS = ("rushing_yards", "receiving_yards", "passing_yards")


def make_stats(rows):
    """Build a player-game DataFrame; unmentioned stat columns default to 0."""
    df = pd.DataFrame(rows)
    for col in STAT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    return df


def balanced_group_rows(position, stat_col, factors, n_players=16):
    """Every player faces every defense once; stat = 100 * defense factor.

    Balanced schedules make every player's baseline identical, so the raw
    per-defense relative performance is factor / mean(factors).
    """
    rows = []
    for p in range(n_players):
        for d, factor in enumerate(factors):
            rows.append(
                {
                    "player_id": f"{position}_{p}",
                    "position": position,
                    "opponent": f"D{d}",
                    stat_col: 100.0 * factor,
                }
            )
    return rows


class TestNormalization:
    def test_group_mean_is_one_for_skewed_factors(self):
        # Right-skewed factors — the old ratio-of-means + one-sided outlier
        # filter produced group means well below 1.0 on shapes like this.
        factors = [0.85, 0.9, 0.95, 1.0, 1.0, 1.05, 1.1, 1.6]
        stats = make_stats(balanced_group_rows("RB", "rushing_yards", factors))

        mults = compute_multipliers_from_game_stats(stats)
        group = [v for (_, pos, st), v in mults.items()
                 if pos == "RB" and st == "rushing_yards"]

        assert len(group) == len(factors)
        mean = sum(group) / len(group)
        assert mean == pytest.approx(1.0, abs=1e-9)

    def test_group_mean_close_to_one_for_every_emitted_group(self):
        factors = [0.8, 0.9, 1.0, 1.0, 1.1, 1.4]
        rows = (
            balanced_group_rows("RB", "rushing_yards", factors)
            + balanced_group_rows("WR", "receiving_yards", factors)
            + balanced_group_rows("QB", "passing_yards", factors, n_players=8)
        )
        mults = compute_multipliers_from_game_stats(make_stats(rows))

        seen_groups = {(pos, st) for (_, pos, st) in mults}
        assert ("QB", "passing_yards") in seen_groups
        for pos, st in seen_groups:
            group = [v for (_, p, s), v in mults.items() if (p, s) == (pos, st)]
            mean = sum(group) / len(group)
            assert mean == pytest.approx(1.0, abs=0.01), (pos, st)


class TestShrinkage:
    def test_small_sample_shrinks_closer_to_one_than_large_sample(self):
        # Two defenses with the SAME raw relative signal (players gain ~38%
        # vs baseline) but different sample sizes: AAA sees 3 player-games,
        # BBB sees 16. Neutral defenses fill out the schedules.
        def player_rows(pid, special):
            rows = [
                {
                    "player_id": pid,
                    "position": "RB",
                    "opponent": f"N{g}",
                    "rushing_yards": 100.0,
                }
                for g in range(5)
            ]
            rows.append(
                {
                    "player_id": pid,
                    "position": "RB",
                    "opponent": special,
                    "rushing_yards": 150.0,
                }
            )
            return rows

        rows = []
        for p in range(3):
            rows += player_rows(f"a{p}", "AAA")
        for p in range(16):
            rows += player_rows(f"b{p}", "BBB")

        mults = compute_multipliers_from_game_stats(make_stats(rows))
        m_small = mults[("AAA", "RB", "rushing_yards")]
        m_large = mults[("BBB", "RB", "rushing_yards")]

        assert m_small > 1.0
        assert m_large > 1.0
        assert abs(m_small - 1.0) < abs(m_large - 1.0)

    def test_min_games_still_excludes_ultra_thin_defenses(self):
        factors = [0.9, 1.0, 1.1, 1.0]
        rows = balanced_group_rows("RB", "rushing_yards", factors, n_players=4)
        # One defense seen only twice (below min_games=3)
        rows += [
            {"player_id": "RB_0", "position": "RB", "opponent": "THIN",
             "rushing_yards": 500.0},
            {"player_id": "RB_1", "position": "RB", "opponent": "THIN",
             "rushing_yards": 500.0},
        ]
        mults = compute_multipliers_from_game_stats(make_stats(rows))
        assert ("THIN", "RB", "rushing_yards") not in mults


class TestPassingYardsSupported:
    def test_passing_yards_is_a_supported_matchup(self):
        assert ("QB", "passing_yards") in SUPPORTED_MATCHUPS
        assert ("QB", "passing_yards") in COMPUTED_MATCHUPS

    def test_passing_multipliers_vary_by_defense(self):
        # Regression for bug (a): stat_configs previously omitted
        # passing_yards, so every team silently got 1.0.
        factors = [0.8, 0.9, 1.0, 1.1, 1.2]
        stats = make_stats(
            balanced_group_rows("QB", "passing_yards", factors, n_players=10)
        )
        mults = compute_multipliers_from_game_stats(stats)
        group = {opp: v for (opp, pos, st), v in mults.items()
                 if pos == "QB" and st == "passing_yards"}

        assert len(group) == len(factors)
        assert len({round(v, 6) for v in group.values()}) > 1
        assert group["D0"] < 1.0 < group["D4"]


class TestFailLoud:
    def test_unknown_stat_type_raises_never_returns_one(self):
        with pytest.raises(ValueError, match="Unsupported defense matchup"):
            get_defense_multiplier(
                opponent="CIN",
                position="QB",
                stat_type="passing_tds",
                season=2025,
                through_week=12,
            )

    def test_unknown_position_raises(self):
        with pytest.raises(ValueError, match="Unsupported defense matchup"):
            get_defense_multiplier(
                opponent="CIN",
                position="K",
                stat_type="rushing_yards",
                season=2025,
                through_week=12,
            )

    def test_missing_stat_columns_raise(self):
        df = pd.DataFrame(
            {"player_id": ["a"], "position": ["RB"], "opponent": ["CIN"],
             "rushing_yards": [50.0]}
        )
        with pytest.raises(ValueError, match="missing required columns"):
            compute_multipliers_from_game_stats(df)


class TestNeutralMatchups:
    def test_neutral_matchups_return_one_without_touching_db(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("neutral matchup must not hit the DB")

        monkeypatch.setattr(da, "compute_defense_vs_position_multipliers", boom)
        for position, stat_type in sorted(NEUTRAL_MATCHUPS):
            assert get_defense_multiplier("CIN", position, stat_type, 2025, 12) == 1.0

    def test_pure_computation_emits_no_neutral_group_keys(self):
        factors = [0.9, 1.0, 1.1]
        rows = balanced_group_rows("WR", "rushing_yards", factors, n_players=8)
        mults = compute_multipliers_from_game_stats(make_stats(rows))
        assert not any(
            (pos, st) in NEUTRAL_MATCHUPS for (_, pos, st) in mults
        )


class TestGetMultiplierFallbacks:
    """Fallback chain for supported matchups, with the DB layer stubbed out."""

    def _stub(self, monkeypatch, table):
        monkeypatch.setattr(
            da, "compute_defense_vs_position_multipliers",
            lambda season, through_week, min_games=3: table,
        )

    def test_exact_key_wins(self, monkeypatch):
        self._stub(monkeypatch, {("CIN", "RB", "rushing_yards"): 1.12})
        assert get_defense_multiplier("CIN", "RB", "rushing_yards", 2025, 12) == 1.12

    def test_unseen_defense_falls_back_to_group_mean(self, monkeypatch):
        self._stub(
            monkeypatch,
            {
                ("CIN", "RB", "rushing_yards"): 1.2,
                ("DEN", "RB", "rushing_yards"): 0.8,
                ("CIN", "WR", "receiving_yards"): 1.3,
            },
        )
        result = get_defense_multiplier("SEA", "RB", "rushing_yards", 2025, 12)
        assert result == pytest.approx(1.0)

    def test_empty_group_defaults_to_one(self, monkeypatch):
        self._stub(monkeypatch, {})
        assert get_defense_multiplier("CIN", "QB", "passing_yards", 2025, 12) == 1.0


class TestClampBounds:
    def test_extreme_signals_are_clamped_to_bounds(self):
        lo, hi = CLAMP_BOUNDS

        def player_rows(pid):
            rows = [
                {"player_id": pid, "position": "WR", "opponent": f"N{g}",
                 "receiving_yards": 100.0}
                for g in range(8)
            ]
            # Blowup game vs XXX, shutout vs ZZZ
            rows.append({"player_id": pid, "position": "WR", "opponent": "XXX",
                         "receiving_yards": 1000.0})
            rows.append({"player_id": pid, "position": "WR", "opponent": "ZZZ",
                         "receiving_yards": 0.0})
            return rows

        rows = []
        for p in range(30):
            rows += player_rows(f"wr{p}")

        mults = compute_multipliers_from_game_stats(make_stats(rows))
        group = {opp: v for (opp, pos, st), v in mults.items()
                 if pos == "WR" and st == "receiving_yards"}

        assert all(lo <= v <= hi for v in group.values())
        assert group["XXX"] == hi
        assert group["ZZZ"] == lo


class TestExactPipelineValues:
    """Pin normalize→shrink to exact hand-computed values.

    Balanced design (every player faces every defense once, identical stat
    shape) makes each player's baseline 100 * mean(factors), so the raw
    per-defense relative performance is factor / mean(factors). Expected
    values below were computed BY HAND from the documented contract
    (shrink weight n/(n+25) with n=16 player-games → 16/41) and are
    hard-coded: if they drift, money-path multipliers changed.
    """

    def test_balanced_group_matches_hand_computed_values(self):
        # factors [0.8, 1.0, 1.2], mean 1.0, shrink 16/41:
        #   D0: 1 - 0.2 * 16/41 = 0.9219512195121951
        #   D1: 1.0
        #   D2: 1 + 0.2 * 16/41 = 1.0780487804878049
        stats = make_stats(
            balanced_group_rows("RB", "rushing_yards", [0.8, 1.0, 1.2], n_players=16)
        )
        mults = compute_multipliers_from_game_stats(stats)

        assert mults[("D0", "RB", "rushing_yards")] == pytest.approx(
            0.9219512195121951, abs=1e-12
        )
        assert mults[("D1", "RB", "rushing_yards")] == pytest.approx(1.0, abs=1e-12)
        assert mults[("D2", "RB", "rushing_yards")] == pytest.approx(
            1.0780487804878049, abs=1e-12
        )

    def test_clamped_member_pins_to_bound_and_group_mean_may_drift(self):
        """What the code actually guarantees when clamps bind: every member
        stays inside CLAMP_BOUNDS, unclamped members keep their exact
        values, and the post-clamp group mean is ALLOWED to drift off 1.0
        (renormalization happens before clamping, not after).

        factors [0.5, 0.5, 0.5, 4.0], mean 1.375, shrink 16/41:
          low  = 1 + (0.5/1.375 - 1) * 16/41 = 0.7516629711751663 (unclamped)
          high = 1 + (4.0/1.375 - 1) * 16/41 = 1.7450110864745011 → clamps to 1.3
        """
        stats = make_stats(
            balanced_group_rows("RB", "rushing_yards", [0.5, 0.5, 0.5, 4.0], n_players=16)
        )
        mults = compute_multipliers_from_game_stats(stats)
        group = {opp: v for (opp, pos, st), v in mults.items()
                 if (pos, st) == ("RB", "rushing_yards")}
        assert len(group) == 4

        lo, hi = CLAMP_BOUNDS
        for d in range(3):
            assert group[f"D{d}"] == pytest.approx(0.7516629711751663, abs=1e-12)
        assert group["D3"] == hi
        assert all(lo <= v <= hi for v in group.values())

        # The drift the clamp introduces — asserting mean == 1.0 here would
        # be asserting something the code does not promise.
        mean_after = sum(group.values()) / len(group)
        assert mean_after < 1.0

    def test_shrinkage_strictly_monotonic_in_sample_size(self):
        """Same raw signal at n=3, n=8, n=16: distance from 1.0 must grow
        strictly with evidence."""
        def player_rows(pid, special):
            rows = [
                {"player_id": pid, "position": "RB", "opponent": f"N{g}",
                 "rushing_yards": 100.0}
                for g in range(5)
            ]
            rows.append({"player_id": pid, "position": "RB", "opponent": special,
                         "rushing_yards": 150.0})
            return rows

        rows = []
        for p in range(3):
            rows += player_rows(f"a{p}", "AAA")
        for p in range(8):
            rows += player_rows(f"b{p}", "BBB")
        for p in range(16):
            rows += player_rows(f"c{p}", "CCC")

        mults = compute_multipliers_from_game_stats(make_stats(rows))
        d3 = abs(mults[("AAA", "RB", "rushing_yards")] - 1.0)
        d8 = abs(mults[("BBB", "RB", "rushing_yards")] - 1.0)
        d16 = abs(mults[("CCC", "RB", "rushing_yards")] - 1.0)
        assert d3 < d8 < d16


class TestDegenerateGroup:
    def test_all_zero_relative_performance_raises(self):
        """A group whose every surviving defense has zero relative signal must
        fail loud (group mean 0), never divide by zero or emit garbage."""
        rows = []
        for pid in ("a", "b", "c"):
            # One big game vs a defense seen only once (dropped by min_games)
            # keeps the player's baseline above min_baseline...
            rows.append({"player_id": pid, "position": "RB",
                         "opponent": f"RARE_{pid}", "rushing_yards": 100.0})
            # ...while every surviving defense sees only zeros.
            for d in range(3):
                rows.append({"player_id": pid, "position": "RB",
                             "opponent": f"D{d}", "rushing_yards": 0.0})
        with pytest.raises(ValueError, match="Degenerate multiplier group"):
            compute_multipliers_from_game_stats(make_stats(rows))


class TestErrorMessagesCarryOffendingNames:
    def test_unknown_stat_type_message_names_the_pair(self):
        with pytest.raises(ValueError) as exc:
            get_defense_multiplier("CIN", "QB", "passing_tds", 2025, 12)
        msg = str(exc.value)
        assert "passing_tds" in msg
        assert "QB" in msg

    def test_missing_columns_message_names_the_columns(self):
        df = pd.DataFrame(
            {"player_id": ["a"], "position": ["RB"], "opponent": ["CIN"],
             "rushing_yards": [50.0], "receiving_yards": [0.0]}
        )
        with pytest.raises(ValueError, match="passing_yards"):
            compute_multipliers_from_game_stats(df)


class TestScheduleLoading:
    """_load_schedule fallbacks; cache cleared so stubs never leak."""

    def test_no_nflreadpy_returns_empty(self, monkeypatch):
        da._load_schedule.cache_clear()
        monkeypatch.setattr(da, "nfl", None)
        try:
            assert da._load_schedule(1901).empty
        finally:
            da._load_schedule.cache_clear()

    def test_loader_error_returns_empty(self, monkeypatch):
        class Boom:
            def load_schedules(self, seasons):
                raise RuntimeError("feed down")

        da._load_schedule.cache_clear()
        monkeypatch.setattr(da, "nfl", Boom())
        try:
            assert da._load_schedule(1902).empty
        finally:
            da._load_schedule.cache_clear()

    def test_schedule_columns_selected(self, monkeypatch):
        class Frame:
            def to_pandas(self):
                return pd.DataFrame(
                    {"week": [1], "home_team": ["AAA"], "away_team": ["BBB"],
                     "extra": [9]}
                )

        class Stub:
            def load_schedules(self, seasons):
                return Frame()

        da._load_schedule.cache_clear()
        monkeypatch.setattr(da, "nfl", Stub())
        try:
            out = da._load_schedule(1903)
            assert list(out.columns) == ["week", "home_team", "away_team"]
        finally:
            da._load_schedule.cache_clear()


class TestDbEntryPoint:
    """compute_defense_vs_position_multipliers with schedule + DB stubbed."""

    @staticmethod
    def _schedule():
        return pd.DataFrame(
            [{"week": w, "home_team": "AAA", "away_team": "BBB"}
             for w in range(1, 5)]
        )

    @staticmethod
    def _stats_rows(team_a="AAA", team_b="BBB"):
        rows = []
        for p in range(4):
            for w in range(1, 5):
                for team, yards in ((team_a, 80.0), (team_b, 120.0)):
                    rows.append({
                        "season": 2025, "week": w,
                        "player_id": f"{team}_{p}", "name": f"{team}_{p}",
                        "team": team, "position": "RB",
                        "rushing_yards": yards, "receiving_yards": 0.0,
                        "passing_yards": 0.0,
                    })
        return rows

    def test_opponent_mapping_and_delegation(self, monkeypatch):
        monkeypatch.setattr(da, "_load_schedule", lambda season: self._schedule())
        monkeypatch.setattr(
            da, "read_dataframe", lambda query: pd.DataFrame(self._stats_rows())
        )
        mults = da.compute_defense_vs_position_multipliers(2025, 4)
        # Constant per-player output → relative performance exactly 1.0 both ways.
        assert mults[("AAA", "RB", "rushing_yards")] == pytest.approx(1.0)
        assert mults[("BBB", "RB", "rushing_yards")] == pytest.approx(1.0)
        # Zero-baseline receiving rows must not produce receiving keys.
        assert not any(st == "receiving_yards" for (_, _, st) in mults)

    def test_empty_stats_returns_empty(self, monkeypatch):
        monkeypatch.setattr(da, "_load_schedule", lambda season: self._schedule())
        monkeypatch.setattr(da, "read_dataframe", lambda query: pd.DataFrame())
        assert da.compute_defense_vs_position_multipliers(2025, 4) == {}

    def test_teams_absent_from_schedule_return_empty(self, monkeypatch):
        monkeypatch.setattr(da, "_load_schedule", lambda season: self._schedule())
        monkeypatch.setattr(
            da,
            "read_dataframe",
            lambda query: pd.DataFrame(self._stats_rows("XXX", "YYY")),
        )
        assert da.compute_defense_vs_position_multipliers(2025, 4) == {}

    def test_missing_schedule_returns_empty(self, monkeypatch):
        monkeypatch.setattr(da, "_load_schedule", lambda season: pd.DataFrame())
        assert da.compute_defense_vs_position_multipliers(2025, 4) == {}

