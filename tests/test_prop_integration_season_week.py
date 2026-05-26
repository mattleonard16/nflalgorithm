"""Tests for PropIntegration explicit season/week args (T0 #5)."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from prop_integration import PropIntegration


def test_init_has_no_current_season_or_week():
    """PropIntegration must not carry hardcoded season/week state."""
    integ = PropIntegration()
    assert not hasattr(integ, "current_season")
    assert not hasattr(integ, "current_week")


def test_get_best_value_opportunities_requires_season_and_week():
    integ = PropIntegration()
    with pytest.raises(TypeError):
        integ.get_best_value_opportunities()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        integ.get_best_value_opportunities(season=2025)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        integ.get_best_value_opportunities(week=13)  # type: ignore[call-arg]


def test_generate_value_report_requires_season_and_week():
    integ = PropIntegration()
    with pytest.raises(TypeError):
        integ.generate_value_report()  # type: ignore[call-arg]


def test_update_real_time_value_finder_requires_season_and_week():
    integ = PropIntegration()
    with pytest.raises(TypeError):
        integ.update_real_time_value_finder()  # type: ignore[call-arg]


def test_get_best_value_opportunities_passes_args_to_join():
    """Explicit season/week must flow through to join_odds_projections."""
    integ = PropIntegration()
    with patch("prop_integration.join_odds_projections", return_value=pd.DataFrame()) as m:
        integ.get_best_value_opportunities(season=2026, week=7)
    m.assert_called_once_with(2026, 7)


def test_generate_value_report_header_includes_season_and_week():
    integ = PropIntegration()
    with patch("prop_integration.join_odds_projections", return_value=pd.DataFrame()):
        report = integ.generate_value_report(season=2026, week=4)
    assert "season=2026" in report
    assert "week=4" in report
