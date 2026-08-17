"""Algorithm-slate reads from weekly_projections.

These rows are model output, not a published betting card. Value bets stay
behind ``value_visibility_scope``; the Board can show this slate before
live odds exist. Who is on the slate is public; the projected numbers
(mu/sigma) are the model's asset and require a signed-in session.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from api.auth import validate_session
from utils.db import read_dataframe, table_exists
from utils.slate_eligibility import likely_to_play

router = APIRouter(tags=["projections"])

MARKET_LABELS = {
    "passing_yards": "Pass",
    "receiving_yards": "Rec",
    "rushing_yards": "Rush",
}


def _has_values_access(request: Request) -> bool:
    """Any valid signed-in session may see projected numbers; visitors may not."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.removeprefix("Bearer ")
    return bool(token) and validate_session(token) is not None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return number


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    number = _as_int(value)
    if number is not None:
        return number != 0
    return bool(_clean_str(value))


def _display_name(player_id: str, player_name: Any) -> str:
    cleaned = _clean_str(player_name)
    if cleaned:
        return cleaned
    raw = str(player_id or "")
    if "_" in raw:
        raw = raw.split("_", 1)[1]
    return " ".join(part.capitalize() for part in raw.replace("_", " ").split()) or "Unknown"


def _projection_payload(row: dict[str, Any], *, include_values: bool) -> dict[str, Any]:
    market = str(row.get("market") or "")
    team = _clean_str(row.get("team"))
    home_team = _clean_str(row.get("home_team"))
    away_team = _clean_str(row.get("away_team"))
    is_home = team == home_team if team and home_team else None
    return {
        "player_id": str(row.get("player_id") or ""),
        "player_name": _display_name(str(row.get("player_id") or ""), row.get("player_name")),
        "position": (_clean_str(row.get("position")) or "").upper() or None,
        "team": team,
        "opponent": _clean_str(row.get("opponent")),
        "home_team": home_team,
        "away_team": away_team,
        "is_home": is_home,
        "market": market,
        "market_label": MARKET_LABELS.get(market, market),
        "mu": _as_float(row.get("mu")) if include_values else None,
        "sigma": _as_float(row.get("sigma")) if include_values else None,
        "model_version": _clean_str(row.get("model_version")) or "",
        "generated_at": _clean_str(row.get("generated_at")) or "",
        "depth_rank": _as_int(row.get("depth_rank")),
        "is_starter": _as_bool(row.get("is_starter")),
        "injury_status": _clean_str(row.get("injury_status")),
        "roster_status": _clean_str(row.get("roster_status")),
    }


def _eligibility_row(row: dict[str, Any], pick: dict[str, Any]) -> dict[str, Any]:
    """The subset of the joined row that decides week-of participation."""
    return {
        "position": pick["position"],
        "roster_status": pick["roster_status"],
        "depth_rank": pick["depth_rank"],
        "is_starter": 1 if pick["is_starter"] else 0,
        "injury_status": pick["injury_status"],
        "expected_rushing_attempts": row.get("expected_rushing_attempts"),
        "expected_targets": row.get("expected_targets"),
        "expected_passing_attempts": row.get("expected_passing_attempts"),
    }


@router.get("/api/projections/weeks")
def list_projection_weeks() -> dict[str, list[dict[str, int]]]:
    if not table_exists("weekly_projections"):
        return {"available_weeks": []}
    frame = read_dataframe("""
        SELECT DISTINCT season, week
        FROM weekly_projections
        ORDER BY season DESC, week DESC
        """)
    if frame.empty:
        return {"available_weeks": []}
    return {
        "available_weeks": [
            {"season": int(season), "week": int(week)}
            for season, week in frame.itertuples(index=False)
        ]
    }


@router.get("/api/projections")
def list_projections(
    request: Request,
    season: int = Query(...),
    week: int = Query(...),
    market: str | None = Query(default=None),
    position: str | None = Query(default=None),
    likely: bool = Query(default=True),
) -> dict[str, Any]:
    include_values = _has_values_access(request)
    if not table_exists("weekly_projections"):
        return {
            "season": season,
            "week": week,
            "total_count": 0,
            "values_visible": include_values,
            "picks": [],
        }

    name_expr = "NULL AS player_name"
    position_expr = "NULL AS position"
    joins: list[str] = []
    has_roster = table_exists("nfl_roster_players")
    has_snapshots = table_exists("nfl_player_context_snapshots")
    if table_exists("player_dim"):
        joins.append("LEFT JOIN player_dim d ON d.player_id = p.player_id")
        name_expr = "d.player_name AS player_name"
        position_expr = "d.position AS position"
    if has_roster:
        joins.append("""
            LEFT JOIN (
                SELECT season, player_id, MAX(player_name) AS player_name,
                       MAX(position) AS position, MAX(roster_status) AS roster_status
                FROM nfl_roster_players
                GROUP BY season, player_id
            ) r ON r.season = p.season AND r.player_id = p.player_id
            """)
        if table_exists("player_dim"):
            name_expr = "COALESCE(d.player_name, r.player_name) AS player_name"
            position_expr = "COALESCE(d.position, r.position) AS position"
        else:
            name_expr = "r.player_name AS player_name"
            position_expr = "r.position AS position"

    snapshot_expr = """
            NULL AS depth_rank,
            NULL AS is_starter,
            NULL AS injury_status,
            NULL AS snapshot_roster_status,
            NULL AS expected_rushing_attempts,
            NULL AS expected_targets,
            NULL AS expected_passing_attempts"""
    if has_snapshots:
        # The snapshot PK is (season, week, gsis_id), not player_id, so a
        # player_id can map to more than one row. Pin the join to the latest
        # capture instead of fanning the pick out into duplicates.
        joins.append("""
            LEFT JOIN nfl_player_context_snapshots s
              ON s.season = p.season AND s.week = p.week AND s.player_id = p.player_id
             AND s.gsis_id = (
                 SELECT s2.gsis_id
                 FROM nfl_player_context_snapshots s2
                 WHERE s2.season = p.season AND s2.week = p.week
                   AND s2.player_id = p.player_id
                 ORDER BY s2.captured_at DESC, s2.gsis_id DESC
                 LIMIT 1
             )
            """)
        snapshot_expr = """
            s.depth_rank AS depth_rank,
            s.is_starter AS is_starter,
            s.injury_status AS injury_status,
            s.roster_status AS snapshot_roster_status,
            s.expected_rushing_attempts AS expected_rushing_attempts,
            s.expected_targets AS expected_targets,
            s.expected_passing_attempts AS expected_passing_attempts"""

    game_expr = """
            NULL AS home_team,
            NULL AS away_team"""
    if table_exists("games"):
        joins.append("""
            LEFT JOIN games g
              ON g.season = p.season AND g.week = p.week
             AND (
               (g.home_team = p.team AND g.away_team = p.opponent)
               OR (g.away_team = p.team AND g.home_team = p.opponent)
             )
            """)
        game_expr = """
            g.home_team AS home_team,
            g.away_team AS away_team"""

    if has_snapshots and has_roster:
        roster_status_expr = "COALESCE(s.roster_status, r.roster_status) AS roster_status"
    elif has_snapshots:
        roster_status_expr = "s.roster_status AS roster_status"
    elif has_roster:
        roster_status_expr = "r.roster_status AS roster_status"
    else:
        roster_status_expr = "NULL AS roster_status"

    query = f"""
        SELECT
            p.season,
            p.week,
            p.player_id,
            p.team,
            p.opponent,
            p.market,
            p.mu,
            p.sigma,
            p.model_version,
            p.generated_at,
            {name_expr},
            {position_expr},
            {roster_status_expr},
{game_expr},
{snapshot_expr}
        FROM weekly_projections p
        {" ".join(joins)}
        WHERE p.season = ? AND p.week = ?
    """
    params: list[object] = [season, week]
    if market:
        query += " AND p.market = ?"
        params.append(market)
    query += " ORDER BY p.mu DESC, p.player_id, p.market"

    frame = read_dataframe(query, tuple(params))
    picks: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        pick = _projection_payload(row, include_values=include_values)
        if likely and not likely_to_play(_eligibility_row(row, pick), pick["market"]):
            continue
        picks.append(pick)
    if position:
        wanted = position.strip().upper()
        picks = [pick for pick in picks if (pick.get("position") or "") == wanted]
    return {
        "season": season,
        "week": week,
        "total_count": len(picks),
        "values_visible": include_values,
        "picks": picks,
    }
