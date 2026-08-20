"""Memo sections that look ahead to the week about to be projected.

Sections 3 through 5 all take the week that just COMPLETED and derive
``week + 1`` themselves, so the rollover rule sits next to the query it
constrains. A season that has no ``week + 1`` says so and moves on; that is the
normal state after the last week of the year, not an error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from scripts.research_format import (
    _fmt,
    _markdown_table,
    _max_timestamp,
    _no_data,
    _read_optional,
    _text,
)
from utils.context_factors import (
    DEFAULT_PARAMS,
    MARKET_OPPORTUNITY_COLUMN,
    ContextParams,
    _prepare_history,
    usage_trend_factor,
)
from utils.db import DBConnection, read_dataframe
from utils.game_context import home_favored_by, implied_team_totals

# The market whose opportunity column defines "role" for each position. One
# metric per player keeps the trend table readable; a WR's targets and a QB's
# attempts are not comparable numbers, but each is the right one for its man.
POSITION_PRIMARY_MARKET: Mapping[str, str] = {
    "QB": "passing_yards",
    "RB": "rushing_yards",
    "WR": "receiving_yards",
    "TE": "receiving_yards",
}

# A player whose last snap was a month ago is not a "faller", he is out. Only
# players who appeared in one of the last this-many completed weeks are ranked.
RECENT_ACTIVITY_WEEKS = 2

DEFAULT_TREND_LIMIT = 15


def usage_trends(
    season: int,
    week: int,
    *,
    conn: Optional[DBConnection] = None,
    params: ContextParams = DEFAULT_PARAMS,
    limit: int = DEFAULT_TREND_LIMIT,
) -> Tuple[str, Dict[str, Any]]:
    """Whose role is growing or shrinking going into ``week + 1``.

    The trend math is ``utils.context_factors.usage_trend_factor`` -- the same
    function the projection path uses -- so a player who shows up here as a
    riser is the same player whose composite factor is above 1.0 next week.
    History is trimmed by ``_prepare_history`` for the identical reason: its
    pregame cutoff is what guarantees nothing after the just-completed week
    leaks in.

    Only ``player_stats_enhanced`` is read. Projections are not joined, so the
    id-namespace gap between the two tables never comes up.
    """
    title = "## 3. USAGE TRENDS (into next week)"
    section = "usage_trends"
    next_week = week + 1

    markets = sorted(set(POSITION_PRIMARY_MARKET.values()))
    columns = sorted({*markets, *(MARKET_OPPORTUNITY_COLUMN[m] for m in markets)})
    first_season = season - params.history_seasons + 1

    history = read_dataframe(
        f"SELECT player_id, name, season, week, team, position, {', '.join(columns)} "
        "FROM player_stats_enhanced "
        "WHERE season >= ? AND season <= ? AND (season < ? OR (season = ? AND week < ?))",
        (first_season, season, season, season, next_week),
        conn=conn,
    )
    if history.empty:
        return _no_data(
            title,
            f"player_stats_enhanced has no rows at or before {season} W{week:02d} "
            f"across seasons {first_season}-{season}",
            section,
        )

    earliest_active_week = week - RECENT_ACTIVITY_WEEKS + 1
    records: List[Dict[str, Any]] = []
    neutral_count = 0

    for market in markets:
        opportunity = MARKET_OPPORTUNITY_COLUMN[market]
        grouped = _prepare_history(
            history,
            season=season,
            week=next_week,
            stat_column=market,
            opportunity_column=opportunity,
            opponents={},
        )
        for player_id, group in grouped.items():
            latest = group.iloc[-1]
            position = _text(latest["position"]).upper()
            if POSITION_PRIMARY_MARKET.get(position) != market:
                continue
            if int(latest["season"]) != season or int(latest["week"]) < earliest_active_week:
                continue

            factor, is_neutral, recent_n, baseline_n = usage_trend_factor(
                group, opportunity_column=opportunity, params=params
            )
            if is_neutral:
                neutral_count += 1
                continue

            recent_mean, baseline_mean = _window_means(
                group, opportunity, recent_n=recent_n, baseline_n=baseline_n
            )
            records.append(
                {
                    "player_id": player_id,
                    "name": _text(latest["name"]),
                    "team": _text(latest["team"]),
                    "position": position,
                    "market": market,
                    "opportunity_column": opportunity,
                    "trend_factor": round(float(factor), 4),
                    "recent_games": int(recent_n),
                    "baseline_games": int(baseline_n),
                    "recent_opportunities": None
                    if recent_mean is None
                    else round(recent_mean, 2),
                    "baseline_opportunities": None
                    if baseline_mean is None
                    else round(baseline_mean, 2),
                    "last_played_week": int(latest["week"]),
                }
            )

    if not records:
        return _no_data(
            title,
            f"no player has both a usable opportunity trend and a game in the last "
            f"{RECENT_ACTIVITY_WEEKS} completed weeks of {season} "
            f"({neutral_count} players had history but no readable trend)",
            section,
        )

    # Deterministic ordering: many players pile up on the +/-10% bound, so the
    # tiebreak is volume -- the biggest role among equal movers ranks first.
    risers = sorted(
        (r for r in records if r["trend_factor"] > 1.0),
        key=lambda r: (-r["trend_factor"], -(r["recent_opportunities"] or 0.0), r["player_id"]),
    )[:limit]
    fallers = sorted(
        (r for r in records if r["trend_factor"] < 1.0),
        key=lambda r: (r["trend_factor"], -(r["baseline_opportunities"] or 0.0), r["player_id"]),
    )[:limit]

    payload = {
        "section": section,
        "status": "ok",
        "into_week": next_week,
        "ranked_players": len(records),
        "neutral_players": neutral_count,
        "limit": limit,
        "trend_bounds": list(params.trend_bounds),
        "recent_window_games": params.trend_recent_games,
        "baseline_window_games": params.trend_baseline_games,
        "risers": risers,
        "fallers": fallers,
    }

    body = [
        title,
        "",
        f"Trend for {season} W{next_week:02d}, from {len(records)} players with a readable "
        f"trajectory ({neutral_count} had too little usage to read). Windows are the last "
        f"{params.trend_recent_games} played games against the "
        f"{params.trend_baseline_games} before them, shrunk by sample size and bounded to "
        f"[{params.trend_bounds[0]:.2f}, {params.trend_bounds[1]:.2f}].",
        "",
        f"**Top {len(risers)} risers**",
        "",
        _trend_table(risers),
        "",
        f"**Top {len(fallers)} fallers**",
        "",
        _trend_table(fallers),
    ]
    return "\n".join(body), payload


def _window_means(
    group: pd.DataFrame, opportunity_column: str, *, recent_n: int, baseline_n: int
) -> Tuple[Optional[float], Optional[float]]:
    """Mean opportunities in each window ``usage_trend_factor`` actually used.

    The window SIZES come back from that function rather than being recomputed
    here, so the memo can never report means over a different span than the
    factor was built from.
    """
    values = pd.to_numeric(group[opportunity_column], errors="coerce").dropna().tolist()
    if not values or recent_n <= 0:
        return None, None
    recent = values[len(values) - recent_n :]
    baseline_end = len(values) - recent_n
    baseline = values[max(0, baseline_end - baseline_n) : baseline_end]
    recent_mean = sum(recent) / len(recent) if recent else None
    baseline_mean = sum(baseline) / len(baseline) if baseline else None
    return recent_mean, baseline_mean


def _trend_table(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return "_none_"
    return _markdown_table(
        ["player", "team", "pos", "metric", "recent", "baseline", "factor"],
        [
            [
                _text(row["name"]),
                _text(row["team"]),
                _text(row["position"]),
                _text(row["opportunity_column"]),
                f"{_fmt(row['recent_opportunities'], 2)} ({row['recent_games']}g)",
                f"{_fmt(row['baseline_opportunities'], 2)} ({row['baseline_games']}g)",
                _fmt(row["trend_factor"], 4),
            ]
            for row in records
        ],
    )


# ---------------------------------------------------------------------------
# section 4: next week game scripts
# ---------------------------------------------------------------------------


def next_week_game_scripts(
    season: int, week: int, *, conn: Optional[DBConnection] = None
) -> Tuple[str, Dict[str, Any]]:
    """The matchup watchlist for ``week + 1``: spreads, totals, divisional games.

    ``spread_line`` is quoted from the home team's side, and the implied team
    totals are derived through ``utils.game_context`` rather than re-deriving
    the arithmetic here.
    """
    title = "## 4. NEXT WEEK GAME SCRIPTS"
    section = "next_week_game_scripts"
    next_week = week + 1

    games = read_dataframe(
        "SELECT game_id, home_team, away_team, kickoff_utc, spread_line, total_line, div_game "
        "FROM games WHERE season = ? AND week = ?",
        (season, next_week),
        conn=conn,
    )
    if games.empty:
        return _no_data(
            title,
            f"games has no rows for {season} W{next_week:02d}; there is no next week to "
            "preview (season over, or the schedule for it is not loaded)",
            section,
        )

    records: List[Dict[str, Any]] = []
    for row in games.itertuples(index=False):
        spread = home_favored_by(row.spread_line)
        total = home_favored_by(row.total_line)
        implied = implied_team_totals(row.spread_line, row.total_line)
        home_points, away_points = implied if implied is not None else (None, None)
        if spread is None:
            favorite, favored_by = None, None
        elif spread >= 0:
            favorite, favored_by = str(row.home_team), spread
        else:
            favorite, favored_by = str(row.away_team), -spread
        records.append(
            {
                "game_id": _text(row.game_id),
                "matchup": f"{row.away_team} @ {row.home_team}",
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "kickoff_utc": _text(row.kickoff_utc),
                "spread_line": spread,
                "total_line": total,
                "favorite": favorite,
                "favored_by": favored_by,
                "implied_home_total": home_points,
                "implied_away_total": away_points,
                "divisional": bool(row.div_game) if not pd.isna(row.div_game) else False,
            }
        )

    priced = [r for r in records if r["favored_by"] is not None]
    totalled = [r for r in records if r["total_line"] is not None]
    divisional = [r for r in records if r["divisional"]]

    # Sort by spread size, unpriced games last, game_id breaking every tie.
    ordered = sorted(
        records,
        key=lambda r: (r["favored_by"] is None, -(r["favored_by"] or 0.0), r["game_id"]),
    )

    payload: Dict[str, Any] = {
        "section": section,
        "status": "ok",
        "week": next_week,
        "game_count": len(records),
        "priced_game_count": len(priced),
        "divisional_count": len(divisional),
        "games": ordered,
        "biggest_spread": ordered[0] if priced else None,
        "highest_total": max(totalled, key=lambda r: (r["total_line"], r["game_id"]))
        if totalled
        else None,
        "lowest_total": min(totalled, key=lambda r: (r["total_line"], r["game_id"]))
        if totalled
        else None,
    }

    body = [
        title,
        "",
        f"{len(records)} games in {season} W{next_week:02d}; {len(priced)} carry a spread, "
        f"{len(divisional)} are divisional.",
        "",
    ]

    highlights: List[str] = []
    if payload["biggest_spread"] is not None:
        game = payload["biggest_spread"]
        highlights.append(
            f"- **Biggest spread**: {game['matchup']}, {game['favorite']} by "
            f"{game['favored_by']:.1f}. The favorite's back and the underdog's pass catchers "
            "are where the script shows up."
        )
    if payload["highest_total"] is not None:
        game = payload["highest_total"]
        highlights.append(
            f"- **Highest total**: {game['matchup']} at {game['total_line']:.1f}."
        )
    if payload["lowest_total"] is not None:
        game = payload["lowest_total"]
        highlights.append(f"- **Lowest total**: {game['matchup']} at {game['total_line']:.1f}.")
    if divisional:
        highlights.append(
            "- **Divisional**: " + ", ".join(sorted(g["matchup"] for g in divisional)) + "."
        )
    if not priced:
        highlights.append(
            "- No game in this week carries a spread or total yet, so nothing here is "
            "script-informed."
        )
    body.extend(highlights)
    body.append("")
    body.append(
        _markdown_table(
            ["matchup", "favorite", "spread", "total", "impl. home", "impl. away", "div"],
            [
                [
                    row["matchup"],
                    _text(row["favorite"]),
                    _fmt(row["favored_by"]),
                    _fmt(row["total_line"]),
                    _fmt(row["implied_home_total"]),
                    _fmt(row["implied_away_total"]),
                    "yes" if row["divisional"] else "no",
                ]
                for row in ordered
            ],
        )
    )
    return "\n".join(body), payload


# ---------------------------------------------------------------------------
# section 5: data freshness
# ---------------------------------------------------------------------------


def data_freshness(
    season: int, week: int, *, conn: Optional[DBConnection] = None
) -> Tuple[str, Dict[str, Any]]:
    """Row counts and newest timestamps for the feeds next week's run depends on."""
    title = "## 5. DATA FRESHNESS"
    section = "data_freshness"
    next_week = week + 1
    now = pd.Timestamp.now(tz="UTC")

    feeds = [
        _feed_status(
            label=f"player_stats_enhanced (season {season})",
            table="player_stats_enhanced",
            timestamp_column="updated_at",
            sql="SELECT updated_at FROM player_stats_enhanced WHERE season = ?",
            params=(season,),
            now=now,
            conn=conn,
        ),
        _feed_status(
            label=f"weekly_odds ({season} W{next_week:02d})",
            table="weekly_odds",
            timestamp_column="as_of",
            sql="SELECT as_of FROM weekly_odds WHERE season = ? AND week = ?",
            params=(season, next_week),
            now=now,
            conn=conn,
        ),
        _feed_status(
            label=f"games ({season} W{next_week:02d})",
            table="games",
            timestamp_column="created_at",
            sql="SELECT created_at FROM games WHERE season = ? AND week = ?",
            params=(season, next_week),
            now=now,
            conn=conn,
        ),
    ]

    empty = [feed["label"] for feed in feeds if feed["rows"] == 0]
    payload = {
        "section": section,
        "status": "ok",
        "checked_at": now.isoformat(),
        "feeds": feeds,
        "empty_feeds": empty,
    }

    body = [
        title,
        "",
        _markdown_table(
            ["feed", "rows", "newest", "age (days)", "unparseable", "flag"],
            [
                [
                    feed["label"],
                    str(feed["rows"]),
                    _text(feed["newest"]),
                    _fmt(feed["age_days"], 1),
                    str(feed["unparseable_timestamps"]),
                    feed["flag"],
                ]
                for feed in feeds
            ],
        ),
        "",
    ]
    if empty:
        body.append(
            "**EMPTY**: " + ", ".join(empty) + ". Next week's run has nothing to read there."
        )
    else:
        body.append("All three feeds have rows.")
    return "\n".join(body), payload


def _feed_status(
    *,
    label: str,
    table: str,
    timestamp_column: str,
    sql: str,
    params: Tuple[Any, ...],
    now: pd.Timestamp,
    conn: Optional[DBConnection] = None,
) -> Dict[str, Any]:
    frame = _read_optional(sql, params, table=table, conn=conn)
    if frame.empty:
        return {
            "label": label,
            "table": table,
            "timestamp_column": timestamp_column,
            "rows": 0,
            "newest": None,
            "age_days": None,
            "unparseable_timestamps": 0,
            "flag": "EMPTY",
        }
    newest, unparseable = _max_timestamp(frame[timestamp_column])
    age_days = None if newest is None else float((now - newest).total_seconds() / 86400.0)
    if newest is None:
        flag = "NO PARSEABLE TIMESTAMP"
    elif age_days is not None and age_days > 7.0:
        flag = "STALE (>7d)"
    else:
        flag = "ok"
    return {
        "label": label,
        "table": table,
        "timestamp_column": timestamp_column,
        "rows": int(len(frame)),
        "newest": None if newest is None else newest.isoformat(),
        "age_days": age_days,
        "unparseable_timestamps": unparseable,
        "flag": flag,
    }
