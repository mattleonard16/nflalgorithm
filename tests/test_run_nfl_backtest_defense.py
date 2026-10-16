"""The backtest runner keeps target-week games out of defense multipliers.

``utils.defense_adjustments`` reads ``week <= through_week``. A walk-forward
replay has every week of the season in the database, so a lookup at
``through_week == target week`` would leak that week's games. The runner
clamps every lookup to ``target week - 1`` for the duration of each week,
including the copy the weekly model binds by name at import. The weekly
model here is a stub, so this runs in CI.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_nfl_backtest
from utils import defense_adjustments

SEASON = 2025


def _stub_weekly(calls: list[tuple[str, int, int]]) -> SimpleNamespace:
    """A weekly module whose by-name binding records what it is asked for."""

    def get_defense_multiplier(opponent, position, stat_type, season, through_week):
        calls.append(("weekly", season, through_week))
        return 1.0

    def train_weekly_models(tuples):
        # Training resolves history through the module-level lookup.
        defense_adjustments.compute_defense_vs_position_multipliers(SEASON, through_week=99)

    def predict_week(season, week, roster_backed=True):
        # Production passes the target week positionally or by keyword.
        weekly.get_defense_multiplier("CIN", "WR", "receiving_yards", season, week)
        weekly.get_defense_multiplier(
            "CIN", "WR", "receiving_yards", season=season, through_week=week
        )
        # A prior season is history in full; it must not be clamped.
        weekly.get_defense_multiplier("CIN", "WR", "receiving_yards", season - 1, 18)
        # An earlier cutoff is left alone.
        weekly.get_defense_multiplier("CIN", "WR", "receiving_yards", season, week - 2)
        return pd.DataFrame([{"player_id": "p", "market": "receiving_yards", "mu": 1.0}])

    weekly = SimpleNamespace(
        get_defense_multiplier=get_defense_multiplier,
        train_weekly_models=train_weekly_models,
        predict_week=predict_week,
    )
    return weekly


def test_target_week_never_reaches_the_defense_lookup(monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []

    def recording_compute(season, through_week, min_games=3):
        calls.append(("module", season, through_week))
        return {}

    monkeypatch.setattr(
        defense_adjustments, "compute_defense_vs_position_multipliers", recording_compute
    )
    monkeypatch.setattr(run_nfl_backtest, "_training_tuples", lambda s, w, h: [(SEASON - 1, 1)])
    weekly = _stub_weekly(calls)
    original_binding = weekly.get_defense_multiplier
    original_module_lookup = defense_adjustments.get_defense_multiplier

    predict_fn = run_nfl_backtest._make_predict_fn(weekly, history_seasons=1)
    predict_fn(SEASON, 7)

    assert calls == [
        ("module", SEASON, 6),
        ("weekly", SEASON, 6),
        ("weekly", SEASON, 6),
        ("weekly", SEASON - 1, 18),
        ("weekly", SEASON, 5),
    ]
    # Restored after the week: live code paths see the unwrapped functions.
    assert weekly.get_defense_multiplier is original_binding
    assert defense_adjustments.get_defense_multiplier is original_module_lookup
    assert defense_adjustments.compute_defense_vs_position_multipliers is recording_compute


def test_module_lookup_is_clamped_through_the_real_entry_point(monkeypatch) -> None:
    """get_defense_multiplier resolves compute_... through its module; the
    clamp must reach that inner call too."""
    seen: list[int] = []

    def recording_compute(season, through_week, min_games=3):
        seen.append(through_week)
        return {}

    monkeypatch.setattr(
        defense_adjustments, "compute_defense_vs_position_multipliers", recording_compute
    )

    with run_nfl_backtest.clamp_defense_lookahead(SimpleNamespace(), SEASON, 1):
        value = defense_adjustments.get_defense_multiplier(
            "CIN", "WR", "receiving_yards", SEASON, 1
        )

    assert value == 1.0
    assert seen == [0]


def test_originals_are_restored_when_the_week_fails(monkeypatch) -> None:
    weekly = _stub_weekly([])
    original = weekly.get_defense_multiplier

    with pytest.raises(RuntimeError):
        with run_nfl_backtest.clamp_defense_lookahead(weekly, SEASON, 3):
            assert weekly.get_defense_multiplier is not original
            raise RuntimeError("model blew up")

    assert weekly.get_defense_multiplier is original


def test_a_lookup_without_through_week_fails_loud() -> None:
    weekly = SimpleNamespace(get_defense_multiplier=lambda opponent, position: 1.0)

    with pytest.raises(SystemExit, match="through_week"):
        with run_nfl_backtest.clamp_defense_lookahead(weekly, SEASON, 3):
            pass
