"""Resolve odds snapshots to the game they belong to.

Pure functions only — no database access — so the logic stays testable in CI
even though both ``weekly_odds`` writers (``data_pipeline.py`` and
``scripts/prop_line_scraper.py``) are gitignored or partially untracked.

``weekly_odds.event_id`` is supposed to identify a *game*, so that a snapshot
can be joined to ``games`` for its kickoff. Two historical writers broke that
contract by minting per-player strings instead:

- ``data_pipeline._odds_row``      -> ``"{season}_W{week}_{player_id}"``
- ``prop_line_scraper`` synthetic  -> ``"{season}_W{week}_{team}_synthetic"``

Neither joins to ``games.game_id``, so every kickoff-dependent feature silently
degrades: closing-line-before-kickoff CLV cannot be expressed, and
:func:`utils.matching.filter_stale_snapshots` routes every row to its
"unjoinable" branch rather than actually filtering.

The Odds API supplies its own opaque event id on the real path. That id is
meaningful to the provider but is not a ``games.game_id``, so it is mapped
through :func:`resolve_event_id` as well, keyed on the matchup it carries.

Canonical form mirrors nflverse: ``{season}_{week:02d}_{away}_{home}``.
"""

from __future__ import annotations

import re

from utils.player_id_utils import canonicalize_team

# nflverse game_id: 2025_01_DAL_PHI (week zero-padded, away then home).
_CANONICAL_RE = re.compile(
    r"^(?P<season>\d{4})_(?P<week>\d{2})_(?P<away>[A-Z]{2,3})_(?P<home>[A-Z]{2,3})$"
)


class UnresolvableEventError(ValueError):
    """Raised when a snapshot cannot be tied to a specific game.

    Callers must not invent a placeholder key: an event_id that joins to
    nothing is worse than an absent one, because it looks joinable.
    """


def canonical_event_id(season: int, week: int, away_team: str, home_team: str) -> str:
    """Build the canonical game key for a matchup.

    Args:
        season: Four-digit season.
        week: Week number, 1-22 (postseason weeks included).
        away_team: Away club code, any casing/alias accepted.
        home_team: Home club code, any casing/alias accepted.

    Raises:
        UnresolvableEventError: on an unusable season, week, or team code,
            including a team playing itself.
    """
    season_int = _coerce_int(season, "season")
    week_int = _coerce_int(week, "week")
    if not 1 <= week_int <= 22:
        raise UnresolvableEventError(f"week out of range 1-22: {week_int}")

    away = canonicalize_team(away_team)
    home = canonicalize_team(home_team)
    if not away or not home:
        raise UnresolvableEventError(
            f"unresolvable team codes: away={away_team!r} home={home_team!r}"
        )
    if away == home:
        raise UnresolvableEventError(f"team cannot play itself: {away}")

    return f"{season_int}_{week_int:02d}_{away}_{home}"


def is_canonical_event_id(value: object) -> bool:
    """Return whether ``value`` already has the canonical game-key shape."""
    return bool(_CANONICAL_RE.match(value)) if isinstance(value, str) else False


def resolve_event_id(
    season: int,
    week: int,
    *,
    home_team: object = None,
    away_team: object = None,
    existing: object = None,
) -> str:
    """Resolve a snapshot's game key, preferring an already-canonical value.

    ``existing`` is the id a writer already holds (the Odds API's opaque event
    id, or a legacy per-player string). It is trusted only when it is already
    canonical; otherwise the matchup is authoritative, because a provider id
    and a legacy player-keyed string are equally unjoinable to ``games``.

    Raises:
        UnresolvableEventError: when the matchup is absent or unusable and
            ``existing`` is not canonical. Fail loud rather than mint a key
            that joins to nothing.
    """
    if is_canonical_event_id(existing):
        return str(existing)

    if home_team is None or away_team is None:
        raise UnresolvableEventError(
            f"cannot resolve event_id for season={season} week={week}: "
            f"no matchup supplied and existing={existing!r} is not canonical"
        )

    return canonical_event_id(season, week, str(away_team), str(home_team))


def _coerce_int(value: object, field: str) -> int:
    # bool is an int subclass; True would otherwise become week 1.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise UnresolvableEventError(f"{field} is not an integer: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise UnresolvableEventError(f"{field} is not an integer: {value!r}") from exc
