"""Decide which games in a week are finished and safe to grade.

Why this exists
---------------
``scripts/record_outcomes.grade_bets`` graded every bet in the week and marked
any bet whose player had no row in ``player_stats_enhanced`` as a ``push`` with
zero profit. That is correct for a player who did not take a snap, and wrong for
every bet in a game that has not kicked off yet or whose stats feed has not
published. Both cases look identical from the stats table: no row.

A week does not finish at once. 2026 week 1 opened Thursday 2026-09-10 with
NE at SEA and does not close until Monday 2026-09-15 with DEN at KC, so any
grading run in between sees a mix. Without this gate, grading on Sunday night
writes 14 games' worth of real results and two games' worth of fake pushes, and
the pushes then flow into ``weekly_performance`` and the CLV average as if they
were settled bets that broke even.

How a game is judged final
--------------------------
Two conditions, both required:

1. ``kickoff_utc`` is at least ``SETTLE_AFTER_KICKOFF`` in the past. An NFL game
   runs about three hours and ten minutes; the margin here is four hours so an
   overtime game or a weather delay does not read as final while it is still
   being played.
2. At least one player in that game has a row in the actuals frame. The stats
   feed publishes per game, and it lags the final whistle. A game that is over
   on the clock but absent from the feed is *pending*, not a week of pushes.

Anything failing either test is reported as pending with a reason, so the caller
can skip those bets and say why instead of recording a result nobody checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

# An NFL game runs about 3h10m including stoppages. Four hours clears overtime
# and a short weather delay without waiting so long that a Sunday-night grading
# run misses the afternoon window.
SETTLE_AFTER_KICKOFF = timedelta(hours=4)

# Reasons a game is not gradeable yet. Strings rather than an enum because they
# are printed in the runbook output and stored in nothing.
PENDING_NOT_KICKED_OFF = "not_kicked_off"
PENDING_IN_PROGRESS = "in_progress"
PENDING_NO_STATS = "stats_not_published"
PENDING_NO_KICKOFF = "kickoff_unknown"


@dataclass(frozen=True)
class CompletionReport:
    """Which games of a week can be graded, and why the rest cannot."""

    final: tuple[str, ...] = ()
    pending: dict[str, str] = field(default_factory=dict)

    @property
    def is_week_complete(self) -> bool:
        """True when every game in the week has settled."""
        return bool(self.final) and not self.pending

    def summary(self) -> str:
        """One line per state, for a grading run's log."""
        lines = [f"{len(self.final)} game(s) final, {len(self.pending)} pending"]
        for game_id, reason in sorted(self.pending.items()):
            lines.append(f"  pending {game_id}: {reason}")
        return "\n".join(lines)


def _parse_kickoff(value) -> Optional[pd.Timestamp]:
    """Kickoff as a UTC timestamp, or ``None`` when it cannot be read.

    A schedule row can carry a null kickoff before times land, and an
    unparseable value is a data problem rather than a reason to treat the game
    as over. Both answer ``None`` and the caller reports the game as pending.
    """
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def classify_games(
    games: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    now: Optional[datetime] = None,
    settle_after: timedelta = SETTLE_AFTER_KICKOFF,
) -> CompletionReport:
    """Split a week's games into settled and pending.

    ``games`` needs ``game_id`` (or ``event_id``) and ``kickoff_utc``.
    ``actuals`` is the ``player_stats_enhanced`` slice for the week; it needs a
    ``team`` column so a game can be tied to its players, and the frame may be
    empty, which makes every game pending on stats.

    ``now`` defaults to the current UTC time and exists so tests pin it.
    """
    if games is None or games.empty:
        return CompletionReport()

    frame = games.rename(columns={"event_id": "game_id"}) if "event_id" in games.columns else games
    for column in ("game_id", "kickoff_utc"):
        if column not in frame.columns:
            raise ValueError(f"games missing required column: {column}")
    if not {"home_team", "away_team"} <= set(frame.columns):
        raise ValueError("games missing required columns: home_team, away_team")

    moment = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC")
    teams_with_stats = _teams_with_stats(actuals)

    final: list[str] = []
    pending: dict[str, str] = {}
    for _, row in frame.drop_duplicates("game_id").iterrows():
        game_id = str(row["game_id"])
        kickoff = _parse_kickoff(row["kickoff_utc"])
        if kickoff is None:
            pending[game_id] = PENDING_NO_KICKOFF
        elif moment < kickoff:
            pending[game_id] = PENDING_NOT_KICKED_OFF
        elif moment < kickoff + settle_after:
            pending[game_id] = PENDING_IN_PROGRESS
        elif not _has_stats(row, teams_with_stats):
            pending[game_id] = PENDING_NO_STATS
        else:
            final.append(game_id)

    return CompletionReport(final=tuple(final), pending=pending)


def _teams_with_stats(actuals: pd.DataFrame) -> set[str]:
    """Teams that have at least one player row in the actuals frame."""
    if actuals is None or actuals.empty or "team" not in actuals.columns:
        return set()
    return {str(team) for team in actuals["team"].dropna().unique()}


def _has_stats(game_row: pd.Series, teams_with_stats: set[str]) -> bool:
    """Whether the stats feed has published either side of this game.

    Either team is enough. A feed that published one club and not its opponent
    is a partial write, but the game is plainly in the feed, and holding the
    whole game back on that would strand it forever.
    """
    return str(game_row["home_team"]) in teams_with_stats or (
        str(game_row["away_team"]) in teams_with_stats
    )


def gradeable_event_ids(
    games: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    now: Optional[datetime] = None,
    settle_after: timedelta = SETTLE_AFTER_KICKOFF,
) -> set[str]:
    """The ``event_id`` values a grading run may settle. Convenience wrapper."""
    report = classify_games(games, actuals, now=now, settle_after=settle_after)
    return set(report.final)
