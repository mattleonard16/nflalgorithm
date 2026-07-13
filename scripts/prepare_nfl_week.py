#!/usr/bin/env python3
"""Refresh NFL context and generate predictions for one season/week."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from config import config
from models.position_specific import predict_week
from models.position_specific.weekly import INACTIVE_ROSTER_STATUSES
from schema_migrations import MigrationManager
from scripts.ingest_real_nfl_data import NFL_TEAM_COUNT, default_history_seasons, ingest_seasons
from scripts.populate_player_dim import populate_player_dim
from utils.db import fetchall, fetchone

MIN_GSIS_HISTORY_COVERAGE = 0.95
MIN_ROSTER_HISTORY_COVERAGE = 0.50
WEEK_ONE_GAME_COUNT = NFL_TEAM_COUNT // 2


def run_migrations() -> None:
    """Ensure the configured database supports the current refresh workflow."""
    MigrationManager(config.database.path).run()


def _count_rows(query: str, params: tuple[object, ...]) -> int:
    row = fetchone(query, params=params)
    return int(row[0]) if row else 0


def _has_gsis_history(history_seasons: list[int]) -> bool:
    if not history_seasons:
        return False
    placeholders = ", ".join(["?"] * len(history_seasons))
    rows = fetchall(
        f"""
        SELECT season,
               COUNT(DISTINCT player_id) AS total_players,
               COUNT(DISTINCT CASE WHEN gsis_id IS NOT NULL THEN player_id END)
                   AS linked_players
        FROM player_stats_enhanced
        WHERE season IN ({placeholders})
        GROUP BY season
        """,
        params=tuple(history_seasons),
    )
    coverage_by_season = {
        int(season): int(linked_players) / int(total_players)
        for season, total_players, linked_players in rows
        if int(total_players) > 0
    }
    return all(
        coverage_by_season.get(season, 0) >= MIN_GSIS_HISTORY_COVERAGE for season in history_seasons
    )


def _count_roster_players(season: int) -> int:
    return _count_rows(
        "SELECT COUNT(*) AS n FROM nfl_roster_players WHERE season = ?",
        (season,),
    )


def _count_games(season: int, week: int) -> int:
    return _count_rows(
        "SELECT COUNT(*) AS n FROM games WHERE season = ? AND week = ?",
        (season, week),
    )


def _earliest_kickoff(season: int, week: int) -> Optional[datetime]:
    row = fetchone(
        "SELECT MIN(kickoff_utc) FROM games WHERE season = ? AND week = ?",
        params=(season, week),
    )
    if not row or not row[0]:
        return None
    return datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")).astimezone(timezone.utc)


def _count_scheduled_teams(season: int, week: int) -> int:
    return _count_rows(
        """
        SELECT COUNT(DISTINCT team) FROM (
            SELECT home_team AS team FROM games WHERE season = ? AND week = ?
            UNION
            SELECT away_team AS team FROM games WHERE season = ? AND week = ?
        ) scheduled_teams
        """,
        (season, week, season, week),
    )


def _count_roster_teams(season: int) -> int:
    return _count_rows(
        "SELECT COUNT(DISTINCT team) FROM nfl_roster_players WHERE season = ?",
        (season,),
    )


def _active_roster_status_clause(alias: str = "roster") -> str:
    statuses = ", ".join(f"'{status}'" for status in INACTIVE_ROSTER_STATUSES)
    return (
        f"({alias}.roster_status IS NULL OR TRIM({alias}.roster_status) = '' "
        f"OR UPPER({alias}.roster_status) NOT IN ({statuses}))"
    )


def _count_prediction_eligible_roster(season: int) -> int:
    return _count_rows(
        f"""
        SELECT COUNT(*) AS n
        FROM nfl_roster_players roster
        WHERE roster.season = ?
          AND roster.position IN ('QB', 'RB', 'WR', 'TE', 'FB')
          AND {_active_roster_status_clause()}
        """,
        (season,),
    )


def _count_players_with_history(season: int) -> int:
    return _count_rows(
        f"""
        SELECT COUNT(*) AS n
        FROM nfl_roster_players roster
        WHERE roster.season = ?
          AND roster.position IN ('QB', 'RB', 'WR', 'TE', 'FB')
          AND {_active_roster_status_clause()}
          AND EXISTS (
              SELECT 1 FROM player_stats_enhanced stats
              WHERE stats.gsis_id = roster.gsis_id AND stats.season < ?
              LIMIT 1
          )
        """,
        (season, season),
    )


def prepare_week(
    season: int,
    week: int,
    *,
    history_seasons: Optional[list[int]] = None,
    refresh_history: Optional[bool] = None,
) -> dict[str, Any]:
    """Refresh source data and produce a validated weekly prediction slate."""
    if season < 2000:
        raise ValueError("season must be a four-digit NFL season")
    if not 1 <= week <= 22:
        raise ValueError("week must be between 1 and 22")

    if history_seasons is None:
        history_seasons = default_history_seasons(season)
    history_seasons = sorted({candidate for candidate in history_seasons if candidate < season})

    run_migrations()
    historical_player_weeks = 0
    history_refreshed = False
    if refresh_history is None and history_seasons:
        refresh_history = not _has_gsis_history(history_seasons)
    if refresh_history and history_seasons:
        historical_player_weeks = ingest_seasons(history_seasons, through_week=22)
        history_refreshed = True

    current_player_weeks = ingest_seasons([season], through_week=week, stats_through_week=week - 1)
    roster_players = _count_roster_players(season)
    roster_teams = _count_roster_teams(season)
    prediction_eligible_roster_players = _count_prediction_eligible_roster(season)
    games = _count_games(season, week)
    scheduled_teams = _count_scheduled_teams(season, week)
    players_with_history = _count_players_with_history(season)
    if roster_players == 0:
        raise RuntimeError(f"No roster players are available for season {season}")
    if games == 0:
        raise RuntimeError(f"No schedule games are available for season {season} week {week}")
    if roster_teams != NFL_TEAM_COUNT:
        raise RuntimeError(
            f"Roster snapshot covers {roster_teams} of {NFL_TEAM_COUNT} NFL teams for {season}"
        )
    if week == 1 and (games != WEEK_ONE_GAME_COUNT or scheduled_teams != NFL_TEAM_COUNT):
        raise RuntimeError(
            f"Week 1 schedule is incomplete: {games} games and {scheduled_teams} teams"
        )
    if prediction_eligible_roster_players == 0:
        raise RuntimeError(f"No prediction-eligible roster players are available for {season}")
    history_coverage = players_with_history / prediction_eligible_roster_players
    if history_coverage < MIN_ROSTER_HISTORY_COVERAGE:
        raise RuntimeError(
            f"Roster history coverage is only {history_coverage:.1%} for season {season}; "
            "refresh historical seasons before generating predictions"
        )
    earliest_kickoff = _earliest_kickoff(season, week)
    if earliest_kickoff is not None and datetime.now(timezone.utc) >= earliest_kickoff:
        raise RuntimeError(
            f"Season {season} week {week} has already kicked off; "
            "refusing to overwrite pregame projections"
        )

    player_dim_updates = populate_player_dim()
    predictions = predict_week(season, week, roster_backed=True)
    if predictions.empty:
        raise RuntimeError(
            f"No predictions were generated for season {season} week {week}; "
            "verify historical GSIS coverage and trained models"
        )
    predicted_players = int(predictions["player_id"].nunique())

    return {
        "season": season,
        "week": week,
        "history_seasons": history_seasons,
        "history_refreshed": history_refreshed,
        "historical_player_weeks": historical_player_weeks,
        "current_player_weeks": current_player_weeks,
        "roster_players": roster_players,
        "roster_teams": roster_teams,
        "prediction_eligible_roster_players": prediction_eligible_roster_players,
        "games": games,
        "scheduled_teams": scheduled_teams,
        "players_with_history": players_with_history,
        "history_coverage": history_coverage,
        "player_dim_updates": player_dim_updates,
        "predictions": len(predictions),
        "predicted_players": predicted_players,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh NFL roster/schedule context and generate weekly predictions"
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--history-seasons",
        help="Comma-separated historical seasons (default: most recent configured number before target)",
    )
    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument(
        "--refresh-history",
        action="store_true",
        help="Force a historical re-ingest even when GSIS-linked history already exists",
    )
    history_group.add_argument(
        "--skip-history",
        action="store_true",
        help="Reuse locally ingested historical stats and only refresh the target season",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    history_seasons = None
    if args.history_seasons is not None:
        history_seasons = [
            int(value.strip()) for value in args.history_seasons.split(",") if value.strip()
        ]
    refresh_history: Optional[bool] = None
    if args.refresh_history:
        refresh_history = True
    elif args.skip_history:
        refresh_history = False

    result = prepare_week(
        args.season,
        args.week,
        history_seasons=history_seasons,
        refresh_history=refresh_history,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
