"""Canonical sport and player-prop market definitions.

These definitions contain stable domain metadata only. Model features,
book-specific identifiers, and pipeline behavior remain in sport adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class MarketSpec:
    """Stable identity and grading metadata for one player-prop market."""

    key: str
    stat_column: str
    unit: str
    positions: frozenset[str] = frozenset()
    sides: frozenset[str] = frozenset({"over", "under"})


@dataclass(frozen=True, slots=True)
class SportSpec:
    """Stable metadata required by shared multi-sport infrastructure."""

    key: str
    display_name: str
    period_name: str
    markets: Mapping[str, MarketSpec]


def _market(
    key: str,
    *,
    unit: str,
    positions: tuple[str, ...] = (),
    stat_column: str | None = None,
) -> MarketSpec:
    return MarketSpec(
        key=key,
        stat_column=stat_column or key,
        unit=unit,
        positions=frozenset(positions),
    )


def _sport(
    key: str,
    display_name: str,
    period_name: str,
    markets: tuple[MarketSpec, ...],
) -> SportSpec:
    market_map = {market.key: market for market in markets}
    if len(market_map) != len(markets):
        raise ValueError(f"Duplicate market key configured for {key}")
    return SportSpec(
        key=key,
        display_name=display_name,
        period_name=period_name,
        markets=MappingProxyType(market_map),
    )


NFL = _sport(
    "nfl",
    "NFL",
    "week",
    (
        _market("rushing_yards", unit="yards", positions=("RB", "QB", "WR", "TE")),
        _market("receiving_yards", unit="yards", positions=("WR", "TE", "RB")),
        _market("passing_yards", unit="yards", positions=("QB",)),
        _market("receptions", unit="receptions", positions=("WR", "TE", "RB")),
        _market("targets", unit="targets", positions=("WR", "TE", "RB")),
    ),
)

NBA = _sport(
    "nba",
    "NBA",
    "game_date",
    (
        _market("pts", unit="points"),
        _market("reb", unit="rebounds"),
        _market("ast", unit="assists"),
        _market("fg3m", unit="made_threes"),
    ),
)

NCAAB = _sport("ncaab", "NCAAB", "tournament", ())

SPORT_REGISTRY: Mapping[str, SportSpec] = MappingProxyType(
    {sport.key: sport for sport in (NFL, NBA, NCAAB)}
)
MARKET_REGISTRY: Mapping[str, Mapping[str, MarketSpec]] = MappingProxyType(
    {sport_key: sport.markets for sport_key, sport in SPORT_REGISTRY.items()}
)


def get_sport(sport: str) -> SportSpec:
    """Return a registered sport using a case-insensitive key."""
    key = sport.strip().lower()
    try:
        return SPORT_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(SPORT_REGISTRY))
        raise KeyError(f"Unsupported sport '{sport}'. Supported: {supported}") from exc


def market_to_stat(sport: str, market: str) -> str:
    """Resolve a sport-scoped market to its actual-stat column."""
    sport_spec = get_sport(sport)
    try:
        return sport_spec.markets[market].stat_column
    except KeyError as exc:
        raise KeyError(f"Unsupported market '{market}' for sport '{sport_spec.key}'") from exc
