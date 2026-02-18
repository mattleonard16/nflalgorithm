"""NBA API router.

All endpoints are prefixed /api/nba.  Data is served from:
  - nba_projections   (ML model outputs)
  - nba_player_game_logs  (raw game history for averages)

Live schedule data is fetched from nba_api on-demand but cached briefly.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.nba_explainability import build_why_payload, build_why_payloads_batch
from utils.db import fetchall, fetchone, read_dataframe

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/nba", tags=["nba"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class NbaProjection(BaseModel):
    model_config = {"from_attributes": True}

    player_id: int
    player_name: str
    team: str
    game_date: str
    matchup: str | None = None
    market: str
    projected_value: float
    confidence: float | None = None
    stat_last5_avg: float | None = None
    stat_last10_avg: float | None = None
    fga_last5_avg: float | None = None
    min_last5_avg: float | None = None


class NbaProjectionsResponse(BaseModel):
    projections: list[NbaProjection]
    game_date: str
    games_count: int
    total_players: int


class NbaGame(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    status: str | None = None
    home_score: int | None = None
    away_score: int | None = None


class NbaScheduleResponse(BaseModel):
    games: list[NbaGame]
    game_date: str


class NbaPlayerSummary(BaseModel):
    player_id: int
    player_name: str
    team: str
    games_played: int
    avg_pts: float
    avg_reb: float
    avg_ast: float
    avg_min: float


class NbaPlayersResponse(BaseModel):
    players: list[NbaPlayerSummary]
    season: int
    total: int


class NbaMeta(BaseModel):
    available_seasons: list[int]
    latest_game_date: str | None
    total_players: int
    total_games: int


class NbaBetOutcome(BaseModel):
    model_config = {"from_attributes": True}

    bet_id: str
    player_name: str | None = None
    market: str
    line: float
    actual_result: float | None = None
    result: str | None = None
    profit_units: float | None = None
    confidence_tier: str | None = None


class NbaDailyPerformance(BaseModel):
    model_config = {"from_attributes": True}

    season: int
    game_date: str
    total_bets: int
    wins: int
    losses: int
    pushes: int
    profit_units: float
    roi_pct: float
    avg_edge: float
    best_bet: str | None = None
    worst_bet: str | None = None


class NbaPerformanceResponse(BaseModel):
    total_bets: int
    total_wins: int
    total_losses: int
    total_profit: float
    overall_roi: float
    win_rate: float
    days: list[NbaDailyPerformance]


class NbaValueBetItem(BaseModel):
    model_config = {"from_attributes": True}

    player_id: int | None = None
    player_name: str
    team: str | None = None
    event_id: str
    market: str
    sportsbook: str
    line: float
    over_price: int
    under_price: int | None = None
    mu: float
    sigma: float
    p_win: float
    edge_percentage: float
    expected_roi: float
    kelly_fraction: float
    confidence: float | None = None
    generated_at: str | None = None
    why: dict | None = None


class NbaValueBetsResponse(BaseModel):
    bets: list[NbaValueBetItem]
    total: int
    game_date: str
    filters: dict


class NbaExplainResponse(BaseModel):
    player_id: int
    market: str
    game_date: str
    why: dict


class NbaCorrelationMember(BaseModel):
    player_id: int
    market: str
    sportsbook: str


class NbaCorrelationResponse(BaseModel):
    game_date: str
    correlation_groups: dict[str, list[NbaCorrelationMember]]
    team_stacks: dict[str, int]


class NbaRiskSummaryResponse(BaseModel):
    game_date: str
    total_assessed: int
    correlated: int
    exposure_flagged: int
    avg_risk_adjusted_kelly: float | None
    avg_drawdown: float | None
    guardrails: list[str]


# ---------------------------------------------------------------------------
# Schedule helpers (simple cache to avoid hammering NBA.com)
# ---------------------------------------------------------------------------

_schedule_cache: dict[str, Any] = {"ts": 0.0, "data": None}
_schedule_lock = threading.Lock()
_CACHE_TTL = 300  # 5 minutes

VALID_MARKETS = {"pts", "reb", "ast", "fg3m"}


def _fetch_live_schedule() -> list[dict]:
    with _schedule_lock:
        now = time.time()
        if now - _schedule_cache["ts"] < _CACHE_TTL and _schedule_cache["data"] is not None:
            return _schedule_cache["data"]

        try:
            from nba_api.live.nba.endpoints import scoreboard

            board = scoreboard.ScoreBoard()
            raw = board.get_dict()["scoreboard"]["games"]
            games = []
            for g in raw:
                status_text = g.get("gameStatusText", "")
                home = g["homeTeam"]
                away = g["awayTeam"]
                games.append(
                    {
                        "game_id": g["gameId"],
                        "game_date": g["gameEt"][:10],
                        "home_team": home["teamTricode"],
                        "away_team": away["teamTricode"],
                        "status": status_text,
                        "home_score": home.get("score"),
                        "away_score": away.get("score"),
                    }
                )
            _schedule_cache["ts"] = now
            _schedule_cache["data"] = games
            return games
        except Exception as exc:
            log.warning("Could not fetch live NBA schedule: %s", exc)
            if _schedule_cache["data"] is not None:
                return _schedule_cache["data"]
            return []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/meta", response_model=NbaMeta)
def nba_meta() -> NbaMeta:
    """Return available seasons and summary counts."""
    seasons_rows = fetchall(
        "SELECT DISTINCT season FROM nba_player_game_logs ORDER BY season"
    )
    seasons = [r[0] for r in seasons_rows]

    latest = fetchone(
        "SELECT MAX(game_date) FROM nba_player_game_logs"
    )
    latest_date = latest[0] if latest else None

    totals = fetchone(
        "SELECT COUNT(DISTINCT player_id), COUNT(DISTINCT game_id) "
        "FROM nba_player_game_logs"
    )

    return NbaMeta(
        available_seasons=seasons,
        latest_game_date=latest_date,
        total_players=totals[0] if totals else 0,
        total_games=totals[1] if totals else 0,
    )


@router.get("/schedule", response_model=NbaScheduleResponse)
def nba_schedule() -> NbaScheduleResponse:
    """Return today's NBA games from live scoreboard."""
    games_raw = _fetch_live_schedule()
    today = date.today().isoformat()

    games = [NbaGame(**g) for g in games_raw]
    return NbaScheduleResponse(games=games, game_date=today)


@router.get("/projections", response_model=NbaProjectionsResponse)
def nba_projections(
    game_date: str | None = Query(None, description="ISO date e.g. 2026-02-17"),
    market: str = Query("pts", description="Market: pts, reb, ast, fg3m"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
) -> NbaProjectionsResponse:
    """Return NBA player projections for a given date and market."""
    if market not in VALID_MARKETS:
        raise HTTPException(status_code=400, detail=f"Invalid market '{market}'. Must be one of: {', '.join(sorted(VALID_MARKETS))}")

    if game_date is None:
        game_date = date.today().isoformat()

    # Validate date format
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="game_date must be YYYY-MM-DD")

    df = read_dataframe(
        "SELECT p.player_id, p.player_name, p.team, p.game_date, p.market, "
        "p.projected_value, p.confidence "
        "FROM nba_projections p "
        "WHERE p.game_date = ? AND p.market = ? AND (p.confidence >= ? OR p.confidence IS NULL) "
        "ORDER BY p.projected_value DESC LIMIT ?",
        [game_date, market, min_confidence, limit],
    )

    # Enrich with rolling averages from game logs (last 5 and 10 games, not days)
    # Use a whitelist for the market column to avoid SQL injection
    _MARKET_COL = {"pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m"}
    col = _MARKET_COL[market]

    if not df.empty:
        placeholders = ",".join("?" * len(df))
        avgs = read_dataframe(
            "SELECT player_id, "
            f"AVG(CASE WHEN rn <= 5 THEN {col} END) as stat_last5_avg, "
            f"AVG(CASE WHEN rn <= 10 THEN {col} END) as stat_last10_avg, "
            "AVG(CASE WHEN rn <= 5 THEN fga END) as fga_last5_avg, "
            "AVG(CASE WHEN rn <= 5 THEN min END) as min_last5_avg "
            "FROM ("
            f"  SELECT player_id, {col}, fga, min, "
            "  ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) as rn "
            "  FROM nba_player_game_logs "
            "  WHERE player_id IN (" + placeholders + ") AND game_date < ?"
            ") sub WHERE rn <= 10 "
            "GROUP BY player_id",
            df["player_id"].tolist() + [game_date],
        )
        if not avgs.empty:
            df = df.merge(avgs, on="player_id", how="left")

    # Add matchup from today's schedule
    schedule = _fetch_live_schedule()
    team_matchup: dict[str, str] = {}
    for g in schedule:
        team_matchup[g["home_team"]] = f"{g['home_team']} vs. {g['away_team']}"
        team_matchup[g["away_team"]] = f"{g['away_team']} @ {g['home_team']}"

    projections = []
    for _, row in df.iterrows():
        matchup = team_matchup.get(row["team"])
        projections.append(
            NbaProjection(
                player_id=int(row["player_id"]),
                player_name=str(row["player_name"]),
                team=str(row["team"]),
                game_date=str(row["game_date"]),
                matchup=matchup,
                market=str(row["market"]),
                projected_value=float(row["projected_value"]),
                confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
                stat_last5_avg=float(row["stat_last5_avg"]) if "stat_last5_avg" in row and row["stat_last5_avg"] is not None else None,
                stat_last10_avg=float(row["stat_last10_avg"]) if "stat_last10_avg" in row and row["stat_last10_avg"] is not None else None,
                fga_last5_avg=float(row["fga_last5_avg"]) if "fga_last5_avg" in row and row["fga_last5_avg"] is not None else None,
                min_last5_avg=float(row["min_last5_avg"]) if "min_last5_avg" in row and row["min_last5_avg"] is not None else None,
            )
        )

    return NbaProjectionsResponse(
        projections=projections,
        game_date=game_date,
        games_count=len(schedule),
        total_players=len(projections),
    )


@router.get("/players", response_model=NbaPlayersResponse)
def nba_players(
    season: int = Query(2025),
    team: str | None = Query(None),
    search: str | None = Query(None, description="Partial player name search"),
    limit: int = Query(100, ge=1, le=500),
) -> NbaPlayersResponse:
    """Return players with season averages."""
    where_clauses = ["season = ?"]
    params: list[Any] = [season]

    if team:
        where_clauses.append("team_abbreviation = ?")
        params.append(team.upper())

    if search:
        where_clauses.append("LOWER(player_name) LIKE ?")
        params.append(f"%{search.lower()}%")

    where = " AND ".join(where_clauses)
    params.append(limit)

    df = read_dataframe(
        "SELECT player_id, player_name, team_abbreviation as team, "
        "COUNT(*) as games_played, "
        "ROUND(AVG(pts), 1) as avg_pts, "
        "ROUND(AVG(reb), 1) as avg_reb, "
        "ROUND(AVG(ast), 1) as avg_ast, "
        "ROUND(AVG(min), 1) as avg_min "
        "FROM nba_player_game_logs WHERE " + where + " "
        "GROUP BY player_id, player_name, team_abbreviation "
        "ORDER BY avg_pts DESC LIMIT ?",
        params,
    )

    players = [
        NbaPlayerSummary(
            player_id=int(row["player_id"]),
            player_name=str(row["player_name"]),
            team=str(row["team"]),
            games_played=int(row["games_played"]),
            avg_pts=float(row["avg_pts"] or 0),
            avg_reb=float(row["avg_reb"] or 0),
            avg_ast=float(row["avg_ast"] or 0),
            avg_min=float(row["avg_min"] or 0),
        )
        for _, row in df.iterrows()
    ]

    return NbaPlayersResponse(players=players, season=season, total=len(players))


@router.get("/value-bets", response_model=NbaValueBetsResponse)
def nba_value_bets(
    game_date: str | None = Query(None, description="ISO date e.g. 2026-02-17"),
    market: str = Query("pts", description="Market: pts, reb, ast, fg3m"),
    min_edge: float = Query(0.0, ge=0.0),
    best_line_only: bool = Query(False, description="Keep only the highest-edge line per player+market"),
    sportsbook: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    include_why: bool = Query(False, description="Include explainability payloads"),
) -> NbaValueBetsResponse:
    """Return NBA value bets from the materialized value view."""
    if market not in VALID_MARKETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid market '{market}'. Must be one of: {', '.join(sorted(VALID_MARKETS))}",
        )

    if game_date is None:
        game_date = date.today().isoformat()

    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="game_date must be YYYY-MM-DD")

    df = _query_value_bets(game_date, market, min_edge, sportsbook, limit)
    df = _apply_best_line_filter(df, best_line_only)
    bets = _build_bet_items(df)

    if include_why and bets:
        bets = _enrich_with_why(game_date, bets)

    applied_filters = _build_filters(market, min_edge, best_line_only, sportsbook)
    return NbaValueBetsResponse(
        bets=bets,
        total=len(bets),
        game_date=game_date,
        filters=applied_filters,
    )


def _query_value_bets(
    game_date: str,
    market: str,
    min_edge: float,
    sportsbook: str | None,
    limit: int,
) -> Any:
    """Build and execute the value bets query."""
    sql = (
        "SELECT player_id, player_name, team, event_id, market, sportsbook, "
        "line, over_price, under_price, mu, sigma, p_win, edge_percentage, "
        "expected_roi, kelly_fraction, confidence, generated_at "
        "FROM nba_materialized_value_view "
        "WHERE game_date = ? AND market = ? AND edge_percentage >= ?"
    )
    params: list[Any] = [game_date, market, min_edge]

    if sportsbook is not None:
        sql += " AND sportsbook = ?"
        params.append(sportsbook)

    sql += " ORDER BY edge_percentage DESC LIMIT ?"
    params.append(limit)

    return read_dataframe(sql, params)


def _apply_best_line_filter(df: Any, best_line_only: bool) -> Any:
    """Keep only the highest-edge line per player+market if requested."""
    if best_line_only and not df.empty and "player_id" in df.columns:
        return (
            df.sort_values("edge_percentage", ascending=False)
            .groupby(["player_id", "market"], dropna=False)
            .first()
            .reset_index()
            .sort_values("edge_percentage", ascending=False)
        )
    return df


def _build_bet_items(df: Any) -> list[NbaValueBetItem]:
    """Convert a dataframe of value bet rows into NbaValueBetItem objects."""
    return [
        NbaValueBetItem(
            player_id=int(row["player_id"]) if row.get("player_id") is not None else None,
            player_name=str(row["player_name"]),
            team=str(row["team"]) if row.get("team") is not None else None,
            event_id=str(row["event_id"]),
            market=str(row["market"]),
            sportsbook=str(row["sportsbook"]),
            line=float(row["line"]),
            over_price=int(row["over_price"]),
            under_price=int(row["under_price"]) if row.get("under_price") is not None else None,
            mu=float(row["mu"]),
            sigma=float(row["sigma"]),
            p_win=float(row["p_win"]),
            edge_percentage=float(row["edge_percentage"]),
            expected_roi=float(row["expected_roi"]),
            kelly_fraction=float(row["kelly_fraction"]),
            confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
            generated_at=str(row["generated_at"]) if row.get("generated_at") is not None else None,
        )
        for _, row in df.iterrows()
    ]


def _enrich_with_why(game_date: str, bets: list[NbaValueBetItem]) -> list[NbaValueBetItem]:
    """Attach explainability payloads to each bet."""
    bet_dicts = [{"player_id": b.player_id, "market": b.market} for b in bets]
    why_map = build_why_payloads_batch(game_date, bet_dicts)
    return [
        NbaValueBetItem(**{**b.model_dump(), "why": why_map.get(f"{b.player_id}:{b.market}")})
        for b in bets
    ]


def _build_filters(
    market: str,
    min_edge: float,
    best_line_only: bool,
    sportsbook: str | None,
) -> dict[str, Any]:
    """Build the applied filters dict for the response."""
    filters: dict[str, Any] = {
        "market": market,
        "min_edge": min_edge,
        "best_line_only": best_line_only,
    }
    if sportsbook is not None:
        filters["sportsbook"] = sportsbook
    return filters


@router.get("/performance", response_model=NbaPerformanceResponse)
def nba_performance(
    season: int | None = Query(None, description="Filter by season"),
) -> NbaPerformanceResponse:
    """Return NBA betting performance with daily breakdown."""
    where = "WHERE 1=1"
    params: list[Any] = []

    if season is not None:
        where += " AND season = ?"
        params.append(season)

    rows = fetchall(
        f"SELECT season, game_date, total_bets, wins, losses, pushes, "
        f"profit_units, roi_pct, avg_edge, best_bet, worst_bet "
        f"FROM nba_daily_performance {where} "
        f"ORDER BY game_date DESC",
        tuple(params),
    )

    days = [
        NbaDailyPerformance(
            season=r[0],
            game_date=r[1],
            total_bets=r[2],
            wins=r[3],
            losses=r[4],
            pushes=r[5],
            profit_units=r[6],
            roi_pct=r[7],
            avg_edge=r[8],
            best_bet=r[9],
            worst_bet=r[10],
        )
        for r in rows
    ]

    total_bets = sum(d.total_bets for d in days)
    total_wins = sum(d.wins for d in days)
    total_losses = sum(d.losses for d in days)
    total_profit = sum(d.profit_units for d in days)

    units_risked = total_wins + total_losses
    overall_roi = (total_profit / units_risked * 100) if units_risked > 0 else 0.0
    win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0.0

    return NbaPerformanceResponse(
        total_bets=total_bets,
        total_wins=total_wins,
        total_losses=total_losses,
        total_profit=round(total_profit, 2),
        overall_roi=round(overall_roi, 1),
        win_rate=round(win_rate, 1),
        days=days,
    )


@router.get("/outcomes", response_model=list[NbaBetOutcome])
def nba_outcomes(
    game_date: str = Query(..., description="Game date YYYY-MM-DD"),
) -> list[NbaBetOutcome]:
    """Return individual graded bets for a specific game date."""
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="game_date must be YYYY-MM-DD")

    rows = fetchall(
        "SELECT bet_id, player_name, market, line, actual_result, result, "
        "profit_units, confidence_tier "
        "FROM nba_bet_outcomes WHERE game_date = ? "
        "ORDER BY profit_units DESC",
        (game_date,),
    )

    return [
        NbaBetOutcome(
            bet_id=r[0],
            player_name=r[1],
            market=r[2],
            line=r[3],
            actual_result=r[4],
            result=r[5],
            profit_units=r[6],
            confidence_tier=r[7],
        )
        for r in rows
    ]


@router.get("/explain/{player_id}/{market}", response_model=NbaExplainResponse)
def nba_explain(
    player_id: int,
    market: str,
    game_date: str | None = Query(None, description="ISO date e.g. 2026-02-17"),
) -> NbaExplainResponse:
    """Return a single bet explanation for a player/market."""
    if market not in VALID_MARKETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid market '{market}'. Must be one of: {', '.join(sorted(VALID_MARKETS))}",
        )
    if game_date is None:
        game_date = date.today().isoformat()
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="game_date must be YYYY-MM-DD")

    payload = build_why_payload(game_date, player_id, market)
    return NbaExplainResponse(
        player_id=player_id, market=market, game_date=game_date, why=payload
    )


@router.get("/analytics/correlation", response_model=NbaCorrelationResponse)
def nba_correlation(
    game_date: str | None = Query(None, description="ISO date e.g. 2026-02-17"),
) -> NbaCorrelationResponse:
    """Return correlation groups and team stacks for a game date."""
    if game_date is None:
        game_date = date.today().isoformat()
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="game_date must be YYYY-MM-DD")

    rows = fetchall(
        "SELECT player_id, market, sportsbook, correlation_group "
        "FROM nba_risk_assessments "
        "WHERE game_date = ? AND correlation_group IS NOT NULL",
        (game_date,),
    )

    groups: dict[str, list[NbaCorrelationMember]] = {}
    for r in rows:
        grp = r[3]
        groups.setdefault(grp, []).append(NbaCorrelationMember(
            player_id=r[0],
            market=r[1],
            sportsbook=r[2],
        ))

    # Team stacks from the value view
    team_rows = fetchall(
        "SELECT team, COUNT(*) as bet_count "
        "FROM nba_materialized_value_view "
        "WHERE game_date = ? AND team IS NOT NULL "
        "GROUP BY team HAVING COUNT(*) > 1",
        (game_date,),
    )
    team_stacks: dict[str, int] = {r[0]: r[1] for r in team_rows}

    return NbaCorrelationResponse(
        game_date=game_date,
        correlation_groups=groups,
        team_stacks=team_stacks,
    )


@router.get("/analytics/risk-summary", response_model=NbaRiskSummaryResponse)
def nba_risk_summary(
    game_date: str | None = Query(None, description="ISO date e.g. 2026-02-17"),
) -> NbaRiskSummaryResponse:
    """Return exposure summary for a game date."""
    if game_date is None:
        game_date = date.today().isoformat()
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="game_date must be YYYY-MM-DD")

    rows = fetchall(
        "SELECT COUNT(*) as total, "
        "SUM(CASE WHEN correlation_group IS NOT NULL THEN 1 ELSE 0 END) as correlated, "
        "SUM(CASE WHEN exposure_warning IS NOT NULL THEN 1 ELSE 0 END) as exposure_flagged, "
        "AVG(risk_adjusted_kelly) as avg_risk_kelly, "
        "AVG(mean_drawdown) as avg_drawdown "
        "FROM nba_risk_assessments WHERE game_date = ?",
        (game_date,),
    )

    if not rows or rows[0][0] == 0:
        return NbaRiskSummaryResponse(
            game_date=game_date,
            total_assessed=0,
            correlated=0,
            exposure_flagged=0,
            avg_risk_adjusted_kelly=None,
            avg_drawdown=None,
            guardrails=[],
        )

    r = rows[0]
    guardrails: list[str] = []
    if r[1] and r[1] > 0:
        guardrails.append(f"{r[1]} correlated bets detected")
    if r[2] and r[2] > 0:
        guardrails.append(f"{r[2]} bets with exposure warnings")

    return NbaRiskSummaryResponse(
        game_date=game_date,
        total_assessed=r[0],
        correlated=r[1] or 0,
        exposure_flagged=r[2] or 0,
        avg_risk_adjusted_kelly=round(r[3], 6) if r[3] else None,
        avg_drawdown=round(r[4], 6) if r[4] else None,
        guardrails=guardrails,
    )


@router.get("/health")
def nba_health() -> dict:
    """Return NBA data freshness info."""
    latest = fetchone(
        "SELECT MAX(game_date), COUNT(*), COUNT(DISTINCT player_id) "
        "FROM nba_player_game_logs"
    )
    proj_latest = fetchone(
        "SELECT MAX(game_date), COUNT(*) FROM nba_projections"
    )

    return {
        "status": "ok",
        "game_logs": {
            "latest_game_date": latest[0] if latest else None,
            "total_rows": latest[1] if latest else 0,
            "total_players": latest[2] if latest else 0,
        },
        "projections": {
            "latest_date": proj_latest[0] if proj_latest else None,
            "total_rows": proj_latest[1] if proj_latest else 0,
        },
    }
