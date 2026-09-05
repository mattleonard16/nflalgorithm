"""Public NFL domain constants shared by data and model adapters."""

from __future__ import annotations

INACTIVE_ROSTER_STATUSES = frozenset({"CUT", "DEV", "INA", "IR", "PUP", "RES", "RET", "SUS"})

# Pregame expected volume required before a player is trained or projected in
# that market. Values are in the same units as the role estimate
# (attempts or targets). 2025 team-weeks: the top 5 skill players took 86.3% of
# carries+targets+attempts and the top 8 took 97.1%. Floors keep RB2 / WR3 /
# starting-QB volume. A 5-attempt rush floor was rejected: it drops Lamar
# (3.5), Hurts (4.7), and Conner (4.4).
MARKETS = (
    "rushing_yards",
    "receiving_yards",
    "passing_yards",
    "receptions",
    "anytime_touchdown",
)

MARKET_MIN_EXPECTED_VOLUME = {
    "rushing_yards": 3.0,
    "receiving_yards": 2.0,
    "passing_yards": 12.0,
    "receptions": 1.5,
    "anytime_touchdown": 0.5,
}

__all__ = ["INACTIVE_ROSTER_STATUSES", "MARKETS", "MARKET_MIN_EXPECTED_VOLUME"]
