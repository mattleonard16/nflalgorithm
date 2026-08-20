"""Per-player weekly context adjustment factors.

The schedule's pregame market numbers (``spread_line``, ``total_line``) have
been persisted on ``games`` by :mod:`utils.game_context` for a while, and
nothing reads them. This module turns them — plus a player's own history —
into small, bounded multipliers that a projection can be scaled by:

``game_script_factor``
    Trailing teams throw and leading teams run. Derived from the spread (which
    side is favored, by how much) and the market's implied team total (how many
    points this offense is priced to score).

``matchup_history_factor``
    How this player has done against this specific defense, relative to his own
    baseline, shrunk hard toward 1.0 by how few meetings there have been.

``usage_trend_factor``
    Whether his opportunity share (targets / carries / attempts) is rising or
    falling, measured over *played* games so a bye or an injury absence does not
    read as a collapse in role.

Every component degrades to exactly 1.0 when its inputs are missing, and names
itself in the ``neutral_components`` diagnostic when it does. Nothing here ever
returns NaN — a NaN multiplier silently zeroes a projection.

DEFENSE IS DELIBERATELY ABSENT
------------------------------
``utils.defense_adjustments.get_defense_multiplier`` already grades the
opponent for the same (position, stat) pair, and the projection path already
applies it exactly once. Adding a second opponent-strength term here would
double-apply the same signal — the single most likely way this module makes
projections worse rather than better. So there is no ``defense_factor``.

``matchup_history_factor`` is the one component that brushes against it: a
player's history versus one defense partly reflects that defense's quality.
It is kept because it is *player*-specific (his relationship with a scheme)
where the defense multiplier is a position-group average, and it is held to
+/-5% with a hard shrinkage so the overlapping part cannot amount to much.
If you ever add a defense term here, remove the one at the mu site — not both.

Nothing in this module reads or writes ``player_stats_enhanced.game_script``:
that column is hardcoded 0.0 on every row.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import pandas as pd

from utils.db import DBConnection, read_dataframe
from utils.game_context import home_favored_by, implied_team_totals

logger = logging.getLogger(__name__)

# The opportunity column whose trajectory defines "role" for each market.
MARKET_OPPORTUNITY_COLUMN: Mapping[str, str] = {
    "receiving_yards": "targets",
    "rushing_yards": "rushing_attempts",
    "passing_yards": "passing_attempts",
}

SUPPORTED_MARKETS = frozenset(MARKET_OPPORTUNITY_COLUMN)

# Markets that live on the passing side of the game-script trade.
_PASS_MARKETS = frozenset({"receiving_yards", "passing_yards"})

# How much of the spread's script effect each position actually absorbs.
# A trailing offense throws more, so its pass catchers gain; a leading offense
# runs more, so its feature back gains. QB rushing is scramble-driven and moves
# in both directions at once, and WR/TE carries are gadget plays — both get a
# zero weight rather than a guessed sign.
_RUSH_SCRIPT_POSITION_WEIGHT: Mapping[str, float] = {"RB": 1.0, "QB": 0.0, "WR": 0.0, "TE": 0.0}
_PASS_SCRIPT_POSITION_WEIGHT: Mapping[str, float] = {"WR": 1.0, "TE": 1.0, "RB": 1.0, "QB": 1.0}

# Component names as they appear in the `neutral_components` diagnostic.
COMPONENT_GAME_SCRIPT = "game_script"
COMPONENT_MATCHUP_HISTORY = "matchup_history"
COMPONENT_USAGE_TREND = "usage_trend"

OUTPUT_COLUMNS: Tuple[str, ...] = (
    "player_id",
    "name",
    "season",
    "week",
    "market",
    "position",
    "team",
    "opponent",
    "game_script_factor",
    "matchup_history_factor",
    "usage_trend_factor",
    "composite_factor",
    "context_n_games",
    "matchup_n_games",
    "trend_recent_n",
    "trend_baseline_n",
    "has_game_row",
    "neutral_components",
)


@dataclass(frozen=True)
class ContextParams:
    """Tuning surface for the factors. Every default is deliberately timid.

    These multipliers sit on top of a model that is already fit to the data;
    they exist to nudge, not to re-price. The bounds are the contract — a
    caller can widen them, but the defaults keep any single week's context
    adjustment smaller than the model's own error bar.
    """

    script_bounds: Tuple[float, float] = (0.92, 1.08)
    matchup_bounds: Tuple[float, float] = (0.95, 1.05)
    trend_bounds: Tuple[float, float] = (0.90, 1.10)
    composite_bounds: Tuple[float, float] = (0.85, 1.15)

    # Fraction of volume moved per point of spread, before position weighting.
    pass_spread_sensitivity: float = 0.004
    rush_spread_sensitivity: float = 0.006
    # Fraction moved per 100% deviation of the implied team total from average.
    total_sensitivity: float = 0.10
    # ~22.5 points a side is a 45-point game, the league's rough center.
    baseline_team_total: float = 22.5

    # Meetings with one defense at which its signal gets half weight.
    matchup_shrinkage_k: float = 6.0
    # Below this season-long average the ratio is noise over a rounding error.
    matchup_min_baseline: float = 1.0

    trend_recent_games: int = 3
    trend_baseline_games: int = 5
    trend_shrinkage_k: float = 4.0
    trend_min_baseline_opportunities: float = 1.0
    # Clamp the raw ratio before shrinking so one 0-opportunity baseline week
    # cannot hand a 10x signal to the shrinkage step.
    trend_raw_ratio_bounds: Tuple[float, float] = (0.5, 2.0)

    # Target season plus this many prior ones feed history and trend.
    history_seasons: int = 3

    def __post_init__(self) -> None:
        for name in ("script_bounds", "matchup_bounds", "trend_bounds", "composite_bounds"):
            lo, hi = getattr(self, name)
            if not (lo <= 1.0 <= hi):
                raise ValueError(f"{name}={lo, hi} must bracket the neutral value 1.0")
        lo, hi = self.trend_raw_ratio_bounds
        if not (0.0 < lo <= 1.0 <= hi):
            raise ValueError(f"trend_raw_ratio_bounds={lo, hi} must be positive and bracket 1.0")
        if self.baseline_team_total <= 0:
            raise ValueError("baseline_team_total must be positive")
        if self.matchup_shrinkage_k <= 0 or self.trend_shrinkage_k <= 0:
            raise ValueError("shrinkage constants must be positive")
        if self.trend_recent_games < 1 or self.trend_baseline_games < 1:
            raise ValueError("trend windows must each hold at least one game")
        if self.history_seasons < 1:
            raise ValueError("history_seasons must be at least 1")


DEFAULT_PARAMS = ContextParams()


class GameView(NamedTuple):
    """One team's view of one scheduled game."""

    opponent: Optional[str]
    team_favored_by: Optional[float]
    implied_team_total: Optional[float]


_NEUTRAL_GAME = GameView(opponent=None, team_favored_by=None, implied_team_total=None)


def _clip(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return float(min(hi, max(lo, value)))


def _shrink(raw: float, n: float, k: float) -> float:
    """Pull ``raw`` toward 1.0 with weight ``n / (n + k)``.

    Same form as the shrinkage in ``utils.defense_adjustments`` so the two read
    alike: n observations of evidence buy n/(n+k) of the signal.
    """
    return 1.0 + (raw - 1.0) * (n / (n + k))


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def game_script_factor(
    *,
    market: str,
    position: str,
    team_favored_by: Optional[float],
    implied_team_total: Optional[float],
    params: ContextParams = DEFAULT_PARAMS,
) -> Tuple[float, bool]:
    """Volume nudge from the game's expected shape.

    Args:
        team_favored_by: Points this player's team is favored by; negative when
            it is the underdog. ``None`` when the spread is unknown.
        implied_team_total: Points the market prices this offense to score.

    Returns:
        ``(factor, is_neutral)``. ``is_neutral`` is True only when neither the
        spread nor the total was usable — a position that carries a zero spread
        weight still gets the total's effect, which is not a fallback.
    """
    if team_favored_by is None and implied_team_total is None:
        return 1.0, True

    if market in _PASS_MARKETS:
        weight = _PASS_SCRIPT_POSITION_WEIGHT.get(position, 0.0)
        # Favored (positive) suppresses pass volume, so the sign is negative.
        sensitivity = -params.pass_spread_sensitivity * weight
    else:
        weight = _RUSH_SCRIPT_POSITION_WEIGHT.get(position, 0.0)
        sensitivity = params.rush_spread_sensitivity * weight

    spread_part = 1.0 if team_favored_by is None else 1.0 + team_favored_by * sensitivity

    if implied_team_total is None:
        total_part = 1.0
    else:
        deviation = (implied_team_total - params.baseline_team_total) / params.baseline_team_total
        total_part = 1.0 + deviation * params.total_sensitivity

    return _clip(spread_part * total_part, params.script_bounds), False


def matchup_history_factor(
    player_history: Optional[pd.DataFrame],
    *,
    opponent: Optional[str],
    stat_column: str,
    params: ContextParams = DEFAULT_PARAMS,
) -> Tuple[float, bool, int]:
    """How this player has fared against this defense, relative to himself.

    Args:
        player_history: This player's completed games, already restricted to
            weeks before the target week, carrying ``opponent`` and
            ``stat_column``.

    Returns:
        ``(factor, is_neutral, n_meetings)``. Zero prior meetings returns
        exactly 1.0 — never a half-informed guess.
    """
    if player_history is None or player_history.empty or not opponent:
        return 1.0, True, 0

    values = pd.to_numeric(player_history[stat_column], errors="coerce").dropna()
    if values.empty:
        return 1.0, True, 0

    baseline = float(values.mean())
    if baseline < params.matchup_min_baseline:
        # A player who averages a rounding error in this stat carries no
        # matchup signal; his ratios would be noise divided by noise.
        return 1.0, True, 0

    versus = player_history[player_history["opponent"] == opponent]
    versus_values = pd.to_numeric(versus[stat_column], errors="coerce").dropna()
    n = int(len(versus_values))
    if n == 0:
        return 1.0, True, 0

    relative = float(versus_values.mean()) / baseline
    shrunk = _shrink(relative, n, params.matchup_shrinkage_k)
    return _clip(shrunk, params.matchup_bounds), False, n


def usage_trend_factor(
    player_history: Optional[pd.DataFrame],
    *,
    opportunity_column: str,
    params: ContextParams = DEFAULT_PARAMS,
) -> Tuple[float, bool, int, int]:
    """Whether the player's role is growing or shrinking.

    Windows count *played games*, not calendar weeks: a bye or a two-week
    absence must not read as three weeks of zero usage.

    Returns:
        ``(factor, is_neutral, recent_n, baseline_n)``.
    """
    if player_history is None or player_history.empty:
        return 1.0, True, 0, 0

    opportunities = pd.to_numeric(player_history[opportunity_column], errors="coerce").dropna()
    if opportunities.empty:
        return 1.0, True, 0, 0

    ordered = opportunities.tolist()
    recent = ordered[-params.trend_recent_games :]
    baseline_end = len(ordered) - len(recent)
    baseline_start = max(0, baseline_end - params.trend_baseline_games)
    baseline = ordered[baseline_start:baseline_end]

    if not recent or not baseline:
        return 1.0, True, len(recent), len(baseline)

    baseline_mean = sum(baseline) / len(baseline)
    if baseline_mean < params.trend_min_baseline_opportunities:
        # No usable denominator: a player going from 0 to 1 target a week is
        # not a 100% role increase in any sense worth pricing.
        return 1.0, True, len(recent), len(baseline)

    recent_mean = sum(recent) / len(recent)
    raw_ratio = _clip(recent_mean / baseline_mean, params.trend_raw_ratio_bounds)
    # The shorter window is what the evidence is really worth.
    n = float(min(len(recent), len(baseline)))
    shrunk = _shrink(raw_ratio, n, params.trend_shrinkage_k)
    return _clip(shrunk, params.trend_bounds), False, len(recent), len(baseline)


def _target_week_games(games: pd.DataFrame, season: int, week: int) -> Dict[str, GameView]:
    """Per-team view of the target week's schedule."""
    week_games = games[(games["season"] == season) & (games["week"] == week)]
    views: Dict[str, GameView] = {}
    for row in week_games.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        spread = home_favored_by(row.spread_line)
        total = home_favored_by(row.total_line)
        implied = implied_team_totals(spread, total)
        home_points, away_points = implied if implied is not None else (None, None)
        sides = (
            (home, away, spread, home_points),
            (away, home, None if spread is None else -spread, away_points),
        )
        for team, opponent, favored, points in sides:
            if team in views:
                raise ValueError(
                    f"Team {team!r} appears in more than one game for season={season} "
                    f"week={week}; the schedule frame is malformed"
                )
            views[team] = GameView(
                opponent=opponent, team_favored_by=favored, implied_team_total=points
            )
    return views


def _opponent_lookup(games: pd.DataFrame) -> Dict[Tuple[int, int, str], str]:
    """(season, week, team) -> opponent, over every game supplied."""
    lookup: Dict[Tuple[int, int, str], str] = {}
    for row in games.itertuples(index=False):
        season = int(row.season)
        week = int(row.week)
        home = str(row.home_team)
        away = str(row.away_team)
        lookup[(season, week, home)] = away
        lookup[(season, week, away)] = home
    return lookup


def _normalize_games(games: pd.DataFrame) -> pd.DataFrame:
    _require_columns(games, ("season", "week", "home_team", "away_team"), "games")
    normalized = games.copy()
    for column in ("spread_line", "total_line"):
        if column not in normalized.columns:
            normalized[column] = float("nan")
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["season"] = pd.to_numeric(normalized["season"], errors="coerce").astype("Int64")
    normalized["week"] = pd.to_numeric(normalized["week"], errors="coerce").astype("Int64")
    return normalized.dropna(subset=["season", "week", "home_team", "away_team"])


def _prepare_history(
    history: pd.DataFrame,
    *,
    season: int,
    week: int,
    stat_column: str,
    opportunity_column: str,
    opponents: Mapping[Tuple[int, int, str], str],
) -> Dict[str, pd.DataFrame]:
    """Trim history to pregame-safe rows, attach opponents, group by player."""
    # An empty frame has no columns to validate, and "no history at all" is a
    # normal state (week 1 of the earliest tracked season), not a malformed
    # input. A non-empty frame of the wrong shape still fails loud below.
    if history.empty:
        return {}
    required = ("player_id", "season", "week", "team", stat_column, opportunity_column)
    _require_columns(history, required, "history")

    prepared = history.copy()
    prepared["season"] = pd.to_numeric(prepared["season"], errors="coerce").astype("Int64")
    prepared["week"] = pd.to_numeric(prepared["week"], errors="coerce").astype("Int64")
    prepared = prepared.dropna(subset=["season", "week", "player_id"])

    # Guard against leakage even when the caller over-supplies: only games that
    # had already been played when the target week kicked off may inform it.
    before = (prepared["season"] < season) | (
        (prepared["season"] == season) & (prepared["week"] < week)
    )
    dropped = int((~before).sum())
    if dropped:
        logger.debug(
            "context_factors: ignoring %d history rows at or after season=%s week=%s",
            dropped,
            season,
            week,
        )
    prepared = prepared[before]
    if prepared.empty:
        return {}

    prepared = prepared.sort_values(["season", "week"], kind="stable")
    keys = zip(
        prepared["season"].astype(int),
        prepared["week"].astype(int),
        prepared["team"].astype(str),
    )
    prepared["opponent"] = [opponents.get(key) for key in keys]

    return {str(pid): group for pid, group in prepared.groupby("player_id", sort=False)}


def _latest_names(grouped: Mapping[str, pd.DataFrame]) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for player_id, frame in grouped.items():
        if "name" not in frame.columns or frame.empty:
            continue
        value = frame["name"].iloc[-1]
        if isinstance(value, str) and value.strip():
            names[player_id] = value.strip()
    return names


def compute_context_factors(
    *,
    season: int,
    week: int,
    market: str,
    players: pd.DataFrame,
    games: pd.DataFrame,
    history: pd.DataFrame,
    params: ContextParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Bounded context multipliers for one (season, week, market).

    Pure: every input is a frame, nothing is read from or written to a database.

    Args:
        players: One row per player to score, with ``player_id``, ``team`` and
            ``position``. Duplicate ``player_id`` values are an error — the
            result is keyed (player_id, season, week, market).
        games: Schedule rows covering both the target week and the history
            window, with ``season``, ``week``, ``home_team``, ``away_team`` and
            optionally ``spread_line`` / ``total_line``.
        history: ``player_stats_enhanced``-shaped rows. Anything at or after the
            target week is discarded rather than trusted.

    Returns:
        One row per input player with every column in ``OUTPUT_COLUMNS``, in the
        input's order. Factors are always finite; a component with no usable
        input is exactly 1.0 and names itself in ``neutral_components``.

    Raises:
        ValueError: Unsupported market, missing required columns, duplicate
            players, or a schedule that puts one team in two games in a week.
    """
    if market not in SUPPORTED_MARKETS:
        raise ValueError(
            f"Unsupported market {market!r}. Supported: {sorted(SUPPORTED_MARKETS)}"
        )
    season = int(season)
    week = int(week)
    stat_column = market
    opportunity_column = MARKET_OPPORTUNITY_COLUMN[market]

    _require_columns(players, ("player_id", "team", "position"), "players")
    if players.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    duplicates = players["player_id"].astype(str).duplicated()
    if duplicates.any():
        offenders = sorted(set(players.loc[duplicates, "player_id"].astype(str)))
        raise ValueError(
            f"players holds duplicate player_id values for season={season} week={week} "
            f"market={market}: {offenders[:5]}"
        )

    normalized_games = _normalize_games(games)
    week_views = _target_week_games(normalized_games, season, week)
    grouped_history = _prepare_history(
        history,
        season=season,
        week=week,
        stat_column=stat_column,
        opportunity_column=opportunity_column,
        opponents=_opponent_lookup(normalized_games),
    )
    names = _latest_names(grouped_history)

    records: List[dict] = []
    for row in players.itertuples(index=False):
        player_id = str(row.player_id)
        team = str(row.team) if row.team is not None else ""
        position = str(row.position) if row.position is not None else ""
        view = week_views.get(team, _NEUTRAL_GAME)
        player_history = grouped_history.get(player_id)

        neutral: List[str] = []

        script, script_neutral = game_script_factor(
            market=market,
            position=position,
            team_favored_by=view.team_favored_by,
            implied_team_total=view.implied_team_total,
            params=params,
        )
        if script_neutral:
            neutral.append(COMPONENT_GAME_SCRIPT)

        matchup, matchup_neutral, matchup_n = matchup_history_factor(
            player_history,
            opponent=view.opponent,
            stat_column=stat_column,
            params=params,
        )
        if matchup_neutral:
            neutral.append(COMPONENT_MATCHUP_HISTORY)

        trend, trend_neutral, recent_n, baseline_n = usage_trend_factor(
            player_history,
            opportunity_column=opportunity_column,
            params=params,
        )
        if trend_neutral:
            neutral.append(COMPONENT_USAGE_TREND)

        composite = _clip(script * matchup * trend, params.composite_bounds)

        records.append(
            {
                "player_id": player_id,
                "name": names.get(player_id, ""),
                "season": season,
                "week": week,
                "market": market,
                "position": position,
                "team": team,
                "opponent": view.opponent or "",
                "game_script_factor": script,
                "matchup_history_factor": matchup,
                "usage_trend_factor": trend,
                "composite_factor": composite,
                "context_n_games": 0 if player_history is None else int(len(player_history)),
                "matchup_n_games": matchup_n,
                "trend_recent_n": recent_n,
                "trend_baseline_n": baseline_n,
                "has_game_row": bool(team in week_views),
                "neutral_components": ",".join(neutral),
            }
        )

    frame = pd.DataFrame(records, columns=list(OUTPUT_COLUMNS))
    return _guard_finite(frame)


def _guard_finite(frame: pd.DataFrame) -> pd.DataFrame:
    """Last line of defense: a NaN multiplier silently zeroes a projection."""
    factor_columns = (
        "game_script_factor",
        "matchup_history_factor",
        "usage_trend_factor",
        "composite_factor",
    )
    repaired = frame.copy()
    for column in factor_columns:
        values = pd.to_numeric(repaired[column], errors="coerce")
        bad = ~values.map(lambda v: isinstance(v, float) and math.isfinite(v))
        if bad.any():
            logger.error(
                "context_factors produced %d non-finite values in %s; forcing 1.0",
                int(bad.sum()),
                column,
            )
            values = values.where(~bad, 1.0)
        repaired[column] = values.astype(float)
    return repaired


def load_context_inputs(
    season: int,
    week: int,
    market: str,
    *,
    players: Optional[pd.DataFrame] = None,
    params: ContextParams = DEFAULT_PARAMS,
    conn: Optional[DBConnection] = None,
) -> Dict[str, pd.DataFrame]:
    """Read the three input frames ``compute_context_factors`` needs.

    Args:
        players: Supply the roster being projected when the caller already has
            it (the prediction path does). When omitted, the players are read
            from ``weekly_projections`` for this key, which is the offline audit
            path — it needs projections to already exist.

    Returns:
        ``{"players": ..., "games": ..., "history": ...}``.
    """
    if market not in SUPPORTED_MARKETS:
        raise ValueError(
            f"Unsupported market {market!r}. Supported: {sorted(SUPPORTED_MARKETS)}"
        )
    season = int(season)
    week = int(week)
    opportunity_column = MARKET_OPPORTUNITY_COLUMN[market]
    first_season = season - params.history_seasons + 1

    games = read_dataframe(
        "SELECT season, week, home_team, away_team, spread_line, total_line "
        "FROM games WHERE season >= ? AND season <= ?",
        (first_season, season),
        conn=conn,
    )

    history = read_dataframe(
        "SELECT player_id, name, season, week, team, position, "
        f"{market} AS {market}, {opportunity_column} AS {opportunity_column} "
        "FROM player_stats_enhanced "
        "WHERE season >= ? AND season <= ? AND (season < ? OR (season = ? AND week < ?))",
        (first_season, season, season, season, week),
        conn=conn,
    )

    if players is None:
        projected = read_dataframe(
            "SELECT player_id, team FROM weekly_projections "
            "WHERE season = ? AND week = ? AND market = ?",
            (season, week, market),
            conn=conn,
        )
        players = projected

    history = _bridge_history_ids(history, players=players, season=season, conn=conn)
    players = _attach_positions(players, history)

    return {"players": players, "games": games, "history": history}


def _bridge_history_ids(
    history: pd.DataFrame,
    *,
    players: pd.DataFrame,
    season: int,
    conn: Optional[DBConnection] = None,
) -> pd.DataFrame:
    """Remap history rows onto caller player_ids that have no direct history.

    ``weekly_projections`` ids are minted from the season's roster
    (``ARI_james_conner``) while ``player_stats_enhanced`` uses a legacy
    abbreviated form (``ARI_j_conner``); the shared key is ``gsis_id`` via
    ``nfl_roster_players``. Only caller ids with zero direct history rows are
    bridged, so seasons whose projections already use stats-style ids are
    untouched. When the bridge tables are unavailable (minimal test databases),
    history is returned unchanged with a warning — a missing bridge degrades to
    neutral factors, never to a crash.
    """
    if history.empty or players.empty or "player_id" not in players.columns:
        return history

    caller_ids = set(players["player_id"].astype(str))
    known_ids = set(history["player_id"].astype(str))
    missing = sorted(caller_ids - known_ids)
    if not missing:
        return history

    placeholders = ",".join("?" for _ in missing)
    try:
        roster = read_dataframe(
            "SELECT player_id, gsis_id, roster_week FROM nfl_roster_players "
            f"WHERE season = ? AND gsis_id IS NOT NULL AND gsis_id != '' "
            f"AND player_id IN ({placeholders})",
            (season, *missing),
            conn=conn,
        )
        stats_ids = read_dataframe(
            "SELECT DISTINCT player_id, gsis_id FROM player_stats_enhanced "
            "WHERE gsis_id IS NOT NULL AND gsis_id != ''",
            conn=conn,
        )
    except Exception as exc:  # noqa: BLE001 - backend-specific missing-table errors
        logger.warning(
            "context_factors: gsis bridge unavailable (%s); %d caller ids keep "
            "neutral history for season=%s",
            exc,
            len(missing),
            season,
        )
        return history

    if roster.empty or stats_ids.empty:
        return history

    # One roster row per gsis_id: a mid-season mover keeps his latest stint.
    roster = (
        roster.sort_values("roster_week", kind="stable")
        .drop_duplicates(subset=["gsis_id"], keep="last")
    )
    gsis_to_caller = dict(zip(roster["gsis_id"].astype(str), roster["player_id"].astype(str)))
    alias = {
        str(stats_pid): gsis_to_caller[str(gsis)]
        for stats_pid, gsis in zip(stats_ids["player_id"], stats_ids["gsis_id"])
        if str(gsis) in gsis_to_caller
    }
    if not alias:
        return history

    bridged = history.copy()
    ids = bridged["player_id"].astype(str)
    bridged["player_id"] = ids.map(alias).fillna(ids)
    logger.info(
        "context_factors: bridged %d of %d caller ids to stats history via gsis_id "
        "for season=%s",
        len(set(alias.values())),
        len(missing),
        season,
    )
    return bridged


def _attach_positions(players: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Fill ``position`` from each player's most recent completed game.

    ``weekly_projections`` does not carry a position, and the script weights
    are position-specific. A player with no history keeps an empty position,
    which resolves to a zero spread weight rather than a guessed one.
    """
    if "position" in players.columns:
        return players
    resolved = players.copy()
    if history.empty or "position" not in history.columns:
        resolved["position"] = ""
        return resolved

    latest = (
        history.sort_values(["season", "week"], kind="stable")
        .drop_duplicates(subset=["player_id"], keep="last")
        .set_index("player_id")["position"]
    )
    resolved["position"] = (
        resolved["player_id"].map(latest).fillna("").astype(str)
    )
    return resolved


def context_factors_for_week(
    season: int,
    week: int,
    market: str,
    *,
    players: Optional[pd.DataFrame] = None,
    params: ContextParams = DEFAULT_PARAMS,
    conn: Optional[DBConnection] = None,
) -> pd.DataFrame:
    """Database-reading convenience wrapper around ``compute_context_factors``."""
    inputs = load_context_inputs(
        season, week, market, players=players, params=params, conn=conn
    )
    return compute_context_factors(
        season=season,
        week=week,
        market=market,
        players=inputs["players"],
        games=inputs["games"],
        history=inputs["history"],
        params=params,
    )


def context_factor_lookup(
    season: int,
    week: int,
    market: str,
    *,
    players: Optional[pd.DataFrame] = None,
    params: ContextParams = DEFAULT_PARAMS,
    conn: Optional[DBConnection] = None,
) -> Dict[str, float]:
    """``player_id -> composite_factor`` for the projection path.

    The intended shape at the mu site: call this once per market, then look up
    each player with a 1.0 default. An unknown player is never a silent zero.
    """
    frame = context_factors_for_week(
        season, week, market, players=players, params=params, conn=conn
    )
    if frame.empty:
        return {}
    return {
        str(pid): float(value)
        for pid, value in zip(frame["player_id"], frame["composite_factor"])
    }
