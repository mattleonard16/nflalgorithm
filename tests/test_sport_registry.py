"""Tests for shared sport and market contracts."""

from __future__ import annotations

import pytest

from sports import market_to_stat
from sports.markets import SPORT_REGISTRY, get_sport


def test_registry_exposes_supported_sports() -> None:
    assert set(SPORT_REGISTRY) == {"nfl", "nba"}


def test_market_mappings_are_sport_scoped() -> None:
    assert market_to_stat("nfl", "receiving_yards") == "receiving_yards"
    assert market_to_stat("nba", "pts") == "pts"

    with pytest.raises(KeyError, match="pts"):
        market_to_stat("nfl", "pts")


def test_registry_rejects_unknown_sports() -> None:
    with pytest.raises(KeyError, match="Unsupported sport"):
        get_sport("mlb")


def test_market_definitions_are_immutable() -> None:
    nfl = get_sport("nfl")

    with pytest.raises(TypeError):
        nfl.markets["new_market"] = nfl.markets["targets"]  # type: ignore[index]


def test_period_names_allow_shared_pipeline_language() -> None:
    assert get_sport("nfl").period_name == "week"
    assert get_sport("nba").period_name == "game_date"


def test_nba_model_capabilities_are_registered_without_following_registry_growth() -> None:
    from models.nba.stat_model import MODELED_MARKETS, VALID_MARKETS

    assert MODELED_MARKETS == ("pts", "reb", "ast", "fg3m")
    assert VALID_MARKETS == frozenset(MODELED_MARKETS)
    assert VALID_MARKETS <= set(get_sport("nba").markets)
