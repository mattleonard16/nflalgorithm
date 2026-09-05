"""FastAPI backend for NFL Algorithm dashboard."""

import asyncio
import csv
import io
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth import (
    UserCreate,
    UserLogin,
    UserPreferences,
    UserResponse,
    authenticate_user,
    create_user,
    generate_session_id,
    get_user_preferences,
    logout_user,
    update_bankroll,
    update_user_preferences,
    validate_session,
)
from api.pipeline_router import require_pipeline_operator, require_pipeline_reader
from api.value_visibility import value_visibility_scope
from config import config
from utils.db import execute, fetchall, fetchone, get_connection, read_dataframe

logger = logging.getLogger(__name__)

PUBLIC_VALUE_VISIBILITY_CONTRACT = "publication-safe-v1"


# ── Pydantic response models (Feature 2: Contract Hardening) ──────────


class ValueBetItem(BaseModel):
    player_id: str
    player_name: Optional[str] = None
    position: Optional[str] = None
    event_id: Optional[str] = None
    team: Optional[str] = None
    team_odds: Optional[str] = None
    market: str
    sportsbook: str
    line: float
    price: int
    side: Optional[str] = None
    mu: float
    sigma: float
    p_win: float
    edge_percentage: float
    expected_roi: float
    kelly_fraction: float
    stake: float
    generated_at: Optional[str] = None
    confidence_score: Optional[float] = None
    confidence_tier: Optional[str] = None
    why: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}


class ValueBetsResponse(BaseModel):
    bets: List[ValueBetItem]
    total: int
    total_count: Optional[int] = None
    filtered_count: Optional[int] = None
    season: Optional[int] = None
    week: Optional[int] = None
    filters: Dict[str, Any]


class BetCreate(BaseModel):
    """Boundary-validated payload for POST /api/user/bets (FR-T0-3)."""

    season: int
    week: int
    player_id: str
    market: str
    sportsbook: str
    side: Literal["over", "under"]
    line: float
    price: int
    stake_units: float
    player_name: Optional[str] = None
    model_edge: Optional[float] = None
    confidence_tier: Optional[str] = None
    stake_dollars: Optional[float] = None


class PipelineRunResponse(BaseModel):
    run_id: str
    season: int
    week: int
    status: str
    stages_requested: int = 7
    stages_completed: int = 0
    error_message: Optional[str] = None
    started_at: str
    finished_at: Optional[str] = None
    report_json: Optional[Dict[str, Any]] = None
    data_health: Optional[Dict[str, Any]] = None


app = FastAPI(
    title="NFL Algorithm API",
    description="FastAPI backend for NFL betting value analysis",
    version="1.0.0",
)

# NBA router
from api.nba_router import router as nba_router  # noqa: E402

app.include_router(nba_router)

# CORS middleware
_default_origins = ["http://localhost:3000", "http://localhost:3001"]
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
_allowed_origins = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else _default_origins
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Auth helper
async def get_current_user(request: Request) -> UserResponse:
    """Extract and validate user from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    session_id = auth_header.replace("Bearer ", "")
    user = validate_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


# Public endpoints


@app.get("/")
async def root():
    """Root endpoint."""
    return {"status": "ok", "name": "NFL Algorithm API"}


@app.get("/api/meta")
async def get_meta():
    """Get available weeks, sportsbooks, and markets from materialized data."""
    try:
        visibility_sql, visibility_params = value_visibility_scope()
        # Get available weeks
        weeks_query = f"""
            SELECT DISTINCT v.season, v.week
            FROM materialized_value_view v
            WHERE {visibility_sql}
            ORDER BY season DESC, week DESC
        """
        weeks_df = read_dataframe(weeks_query, params=list(visibility_params))
        weeks = (
            [
                {"season": int(row["season"]), "week": int(row["week"])}
                for _, row in weeks_df.iterrows()
            ]
            if not weeks_df.empty
            else []
        )

        # Get available sportsbooks
        sportsbooks_query = f"""
            SELECT DISTINCT v.sportsbook
            FROM materialized_value_view v
            WHERE {visibility_sql}
              AND v.sportsbook IS NOT NULL
            ORDER BY sportsbook
        """
        sportsbooks_df = read_dataframe(sportsbooks_query, params=list(visibility_params))
        sportsbooks = sportsbooks_df["sportsbook"].tolist() if not sportsbooks_df.empty else []

        # Get available markets
        markets_query = f"""
            SELECT DISTINCT v.market
            FROM materialized_value_view v
            WHERE {visibility_sql}
              AND v.market IS NOT NULL
            ORDER BY market
        """
        markets_df = read_dataframe(markets_query, params=list(visibility_params))
        markets = markets_df["market"].tolist() if not markets_df.empty else []

        return {
            "weeks": weeks,
            "available_weeks": weeks,
            "sportsbooks": sportsbooks,
            "markets": markets,
        }
    except Exception as e:
        logger.error(f"Error fetching meta: {e}")
        raise HTTPException(status_code=500, detail="Metadata unavailable")


@app.get("/api/value-bets", response_model=ValueBetsResponse)
async def get_value_bets(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
    min_edge: float = Query(0.0, description="Minimum edge percentage"),
    best_line_only: bool = Query(False, description="Show only best line per player/market"),
    sportsbook: Optional[str] = Query(None, description="Filter by sportsbook"),
    market: Optional[str] = Query(None, description="Filter by market"),
    position: Optional[str] = Query(None, description="Filter by position"),
    include_why: bool = Query(False, description="Include explainability payloads"),
    run_id: Optional[str] = Query(None, description="Cache key - pin to specific run"),
):
    """Get value bets with filters."""
    try:
        from api.cache import make_cache_key, value_bets_cache

        cache_key = make_cache_key(
            "value-bets",
            season=season,
            week=week,
            min_edge=min_edge,
            best_line_only=best_line_only,
            sportsbook=sportsbook,
            market=market,
            position=position,
            include_why=include_why,
            run_id=run_id,
            demo_mode=config.api.demo_mode,
        )
        cached = value_bets_cache.get(cache_key)
        if cached is not None:
            return cached

        # Build query - JOIN player_dim for canonical name/position,
        # falling back to player_stats_enhanced when player_dim lacks the player
        visibility_sql, visibility_params = value_visibility_scope()
        query = f"""
            SELECT
                v.player_id,
                COALESCE(pd.player_name, ps.name) AS player_name,
                COALESCE(pd.position, ps.position) AS position,
                v.event_id,
                v.team,
                v.team_odds,
                v.market,
                v.sportsbook,
                v.line,
                v.price,
                v.side,
                v.mu,
                v.sigma,
                v.p_win,
                v.edge_percentage,
                v.expected_roi,
                v.kelly_fraction,
                v.stake,
                v.generated_at,
                v.confidence_score,
                v.confidence_tier
            FROM materialized_value_view v
            LEFT JOIN player_dim pd ON v.player_id = pd.player_id
            LEFT JOIN (
                SELECT player_id, MAX(name) AS name, MAX(position) AS position
                FROM player_stats_enhanced
                GROUP BY player_id
            ) ps ON v.player_id = ps.player_id
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ? AND v.edge_percentage >= ?
        """
        params = [*visibility_params, season, week, min_edge]

        if sportsbook:
            query += " AND v.sportsbook = ?"
            params.append(sportsbook)

        if market:
            query += " AND v.market = ?"
            params.append(market)

        if position:
            query += " AND COALESCE(pd.position, ps.position) = ?"
            params.append(position)

        query += " ORDER BY v.edge_percentage DESC"

        df = read_dataframe(query, params=params)

        if df.empty:
            return {
                "bets": [],
                "total": 0,
                "filters": {
                    "season": season,
                    "week": week,
                    "min_edge": min_edge,
                    "best_line_only": best_line_only,
                    "sportsbook": sportsbook,
                    "market": market,
                    "position": position,
                },
            }

        # If best_line_only, group by player_id + market and keep best edge
        if best_line_only:
            df = df.sort_values("edge_percentage", ascending=False)
            df = df.groupby(["player_id", "market", "side"]).first().reset_index()

        # Convert to records
        bets = df.to_dict(orient="records")

        # Attach explainability payloads if requested
        if include_why:
            try:
                from api.explainability import build_why_payloads_batch

                why_map = build_why_payloads_batch(season, week, bets)
                for bet in bets:
                    key = f"{bet['player_id']}:{bet['market']}"
                    bet["why"] = why_map.get(key)
            except Exception as exc:
                logger.warning("Explainability payloads failed: %s", exc)

        result = {
            "bets": bets,
            "total": len(bets),
            "total_count": len(bets),
            "filtered_count": len(bets),
            "season": season,
            "week": week,
            "filters": {
                "season": season,
                "week": week,
                "min_edge": min_edge,
                "best_line_only": best_line_only,
                "sportsbook": sportsbook,
                "market": market,
                "position": position,
            },
        }

        # Cache if pinned to a run_id (immutable data)
        if run_id:
            value_bets_cache.set(cache_key, result)

        return result
    except Exception as e:
        logger.error(f"Error fetching value bets: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching value bets: {str(e)}")


@app.get("/api/performance")
async def get_performance(season: Optional[int] = Query(None, description="Filter by season")):
    """Get weekly performance metrics."""
    try:
        query = """
            SELECT
                season,
                week,
                total_bets,
                wins,
                losses,
                pushes,
                profit_units,
                roi_pct,
                avg_edge,
                clv_avg,
                best_bet,
                worst_bet,
                updated_at
            FROM weekly_performance
        """
        params = []

        if season:
            query += " WHERE season = ?"
            params.append(season)

        query += " ORDER BY season DESC, week DESC"

        df = read_dataframe(query, params=params if params else None)

        if df.empty:
            return {
                "weeks": [],
                "total_bets": 0,
                "total_wins": 0,
                "total_losses": 0,
                "total_pushes": 0,
                "total_profit": 0.0,
                "overall_roi": 0.0,
                "win_rate": 0.0,
            }

        weeks = df.to_dict(orient="records")

        # Calculate summary
        total_bets = int(df["total_bets"].sum())
        total_wins = int(df["wins"].sum())
        total_losses = int(df["losses"].sum())
        total_pushes = int(df["pushes"].sum())
        total_profit = float(df["profit_units"].sum())
        win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0.0

        return {
            "weeks": weeks,
            "total_bets": total_bets,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_pushes": total_pushes,
            "total_profit": total_profit,
            "overall_roi": float(df["roi_pct"].mean()) if not df.empty else 0.0,
            "win_rate": win_rate,
        }
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        return {
            "weeks": [],
            "total_bets": 0,
            "total_wins": 0,
            "total_losses": 0,
            "total_pushes": 0,
            "total_profit": 0.0,
            "overall_roi": 0.0,
            "win_rate": 0.0,
        }


@app.get("/api/outcomes")
async def get_outcomes(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
):
    """Get bet outcomes for a specific week."""
    try:
        query = """
            SELECT
                b.bet_id,
                b.season,
                b.week,
                b.player_id,
                COALESCE(NULLIF(b.player_name, 'Unknown'), pd.player_name, b.player_name) AS player_name,
                b.market,
                b.sportsbook,
                b.side,
                b.line,
                b.price,
                b.actual_result,
                b.result,
                b.profit_units,
                b.confidence_tier,
                b.edge_at_placement,
                b.recorded_at
            FROM bet_outcomes b
            LEFT JOIN player_dim pd ON b.player_id = pd.player_id
            WHERE b.season = ? AND b.week = ?
            ORDER BY b.recorded_at DESC
        """
        df = read_dataframe(query, params=[season, week])

        return {
            "outcomes": df.to_dict(orient="records") if not df.empty else [],
            "total": len(df),
        }
    except Exception as e:
        logger.error(f"Error fetching outcomes: {e}")
        return {
            "outcomes": [],
            "total": 0,
        }


@app.get("/api/health")
async def get_health(
    season: Optional[int] = Query(None, description="Filter by season"),
    week: Optional[int] = Query(None, description="Filter by week"),
):
    """Get feed freshness status."""
    try:
        query = """
            SELECT
                feed,
                season,
                week,
                as_of
            FROM feed_freshness
        """
        params = []

        conditions = []
        if season:
            conditions.append("season = ?")
            params.append(season)
        if week:
            conditions.append("week = ?")
            params.append(week)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY as_of DESC"

        df = read_dataframe(query, params=params if params else None)

        if df.empty:
            return {
                "feeds": [],
                "overall_status": "unknown",
            }

        feeds = df.to_dict(orient="records")

        # Determine overall status based on freshness
        now = datetime.now(timezone.utc)
        statuses = []
        for feed in feeds:
            as_of = pd.to_datetime(feed["as_of"])
            age_hours = (now - as_of).total_seconds() / 3600
            if age_hours < 24:
                statuses.append("healthy")
            elif age_hours < 48:
                statuses.append("stale")
            else:
                statuses.append("unhealthy")

        if not statuses:
            overall = "unknown"
        elif all(s == "healthy" for s in statuses):
            overall = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall = "unhealthy"
        else:
            overall = "stale"

        return {
            "feeds": feeds,
            "overall_status": overall,
        }
    except Exception as e:
        logger.error(f"Error fetching health: {e}")
        return {
            "feeds": [],
            "overall_status": "unknown",
        }


@app.get("/api/analytics/edge-distribution")
async def get_edge_distribution(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
    bins: int = Query(20, description="Number of histogram bins"),
):
    """Get edge distribution histogram data."""
    try:
        visibility_sql, visibility_params = value_visibility_scope()
        query = f"""
            SELECT v.edge_percentage
            FROM materialized_value_view v
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ?
        """
        df = read_dataframe(query, params=[*visibility_params, season, week])

        if df.empty:
            return {
                "bins": [],
                "counts": [],
            }

        # Create histogram
        counts, bin_edges = pd.cut(df["edge_percentage"], bins=bins, retbins=True)
        hist = counts.value_counts().sort_index()

        # 4 decimals so narrow bins stay distinguishable; frontend formats for display
        bin_labels = [f"{bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}" for i in range(len(bin_edges) - 1)]
        bin_counts = [int(hist.get(interval, 0)) for interval in hist.index]

        return {
            "bins": bin_labels,
            "counts": bin_counts,
        }
    except Exception as e:
        logger.error(f"Error fetching edge distribution: {e}")
        return {
            "bins": [],
            "counts": [],
        }


@app.get("/api/analytics/by-position")
async def get_by_position(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
):
    """Get stats grouped by position."""
    try:
        visibility_sql, visibility_params = value_visibility_scope()
        query = f"""
            SELECT
                COALESCE(pd.position, ps.position) as position,
                COUNT(*) as bet_count,
                AVG(v.edge_percentage) as avg_edge,
                AVG(v.expected_roi) as avg_roi,
                AVG(v.p_win) as avg_p_win
            FROM materialized_value_view v
            LEFT JOIN player_dim pd ON v.player_id = pd.player_id
            LEFT JOIN (
                SELECT player_id, MAX(name) AS name, MAX(position) AS position
                FROM player_stats_enhanced
                GROUP BY player_id
            ) ps ON v.player_id = ps.player_id
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ?
              AND COALESCE(pd.position, ps.position) IS NOT NULL
            GROUP BY COALESCE(pd.position, ps.position)
            ORDER BY bet_count DESC
        """
        df = read_dataframe(query, params=[*visibility_params, season, week])

        return {
            "by_position": df.to_dict(orient="records") if not df.empty else [],
        }
    except Exception as e:
        logger.error(f"Error fetching by-position stats: {e}")
        return {
            "by_position": [],
        }


@app.get("/api/analytics/by-market")
async def get_by_market(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
):
    """Get stats grouped by market."""
    try:
        visibility_sql, visibility_params = value_visibility_scope()
        query = f"""
            SELECT
                v.market,
                COUNT(*) as bet_count,
                AVG(v.edge_percentage) as avg_edge,
                AVG(v.expected_roi) as avg_roi,
                AVG(v.p_win) as avg_p_win
            FROM materialized_value_view v
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ?
            GROUP BY v.market
            ORDER BY bet_count DESC
        """
        df = read_dataframe(query, params=[*visibility_params, season, week])

        return {
            "by_market": df.to_dict(orient="records") if not df.empty else [],
        }
    except Exception as e:
        logger.error(f"Error fetching by-market stats: {e}")
        return {
            "by_market": [],
        }


@app.get("/api/weekly-summary")
async def get_weekly_summary(
    weeks: int = Query(4, ge=1, le=52, description="Number of recent weeks")
):
    """Get recent weekly summaries."""
    try:
        query = """
            SELECT
                season,
                week,
                total_bets,
                wins,
                losses,
                pushes,
                profit_units,
                roi_pct,
                avg_edge
            FROM weekly_performance
            ORDER BY season DESC, week DESC
            LIMIT ?
        """
        df = read_dataframe(query, params=[weeks])

        return {
            "weeks": df.to_dict(orient="records") if not df.empty else [],
        }
    except Exception as e:
        logger.error(f"Error fetching weekly summary: {e}")
        return {
            "weeks": [],
        }


# ── Pipeline run read models ────────────────────────────────────────
# The legacy thread-per-request POST /api/run lived here. It was removed so
# api.server:app cannot be served with an unqueued pipeline trigger; production
# runs go through the durable job pipeline (api.application + pipeline worker).


def _parse_pipeline_run_row(row) -> PipelineRunResponse:
    """Parse a pipeline_runs row into PipelineRunResponse."""
    report = None
    if row[9]:
        try:
            report = json.loads(row[9])
        except (json.JSONDecodeError, TypeError):
            pass

    health = None
    if len(row) > 10 and row[10]:
        try:
            health = json.loads(row[10])
        except (json.JSONDecodeError, TypeError):
            pass

    return PipelineRunResponse(
        run_id=row[0],
        season=row[1],
        week=row[2],
        status=row[3],
        stages_requested=row[4],
        stages_completed=row[5],
        error_message=row[6],
        started_at=row[7],
        finished_at=row[8],
        report_json=report,
        data_health=health,
    )


_PIPELINE_RUN_COLS = "run_id, season, week, status, stages_requested, stages_completed, error_message, started_at, finished_at, report_json, data_health_json"


@app.get("/api/run/latest", response_model=Optional[PipelineRunResponse])
async def get_latest_pipeline_run(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
):
    """Get most recent pipeline run for a season/week."""
    row = fetchone(
        f"SELECT {_PIPELINE_RUN_COLS} FROM pipeline_runs WHERE season = ? AND week = ? ORDER BY started_at DESC LIMIT 1",
        params=(season, week),
    )
    if not row:
        return None
    return _parse_pipeline_run_row(row)


@app.get("/api/run/{run_id}", response_model=PipelineRunResponse)
async def get_pipeline_run(run_id: str):
    """Poll status of a pipeline run."""
    row = fetchone(
        f"SELECT {_PIPELINE_RUN_COLS} FROM pipeline_runs WHERE run_id = ?",
        params=(run_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return _parse_pipeline_run_row(row)


# ── Explainability endpoint (Feature 4) ─────────────────────────────


@app.get("/api/explain/{player_id}/{market}")
async def get_explainability(
    player_id: str,
    market: str,
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
):
    """Get detailed explainability payload for a single bet."""
    try:
        from api.explainability import build_why_payload

        payload = build_why_payload(season, week, player_id, market)
        return {"player_id": player_id, "market": market, "why": payload}
    except Exception as e:
        logger.error(f"Error building explainability: {e}")
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")


# ── Risk & correlation endpoints (Feature 5) ─────────────────────────


@app.get("/api/analytics/correlation")
async def get_correlation_analysis(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
):
    """Get correlation groups and team stacks for current bets."""
    try:
        from risk_manager import detect_correlations, detect_team_stacks

        visibility_sql, visibility_params = value_visibility_scope()
        query = f"""
            SELECT v.player_id, pd.player_name, pd.position, v.team, v.market, v.sportsbook,
                   v.event_id, v.stake, v.edge_percentage, v.kelly_fraction, v.price, v.p_win
            FROM materialized_value_view v
            LEFT JOIN player_dim pd ON v.player_id = pd.player_id
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ?
        """
        df = read_dataframe(query, params=[*visibility_params, season, week])

        if df.empty:
            return {"correlation_groups": [], "team_stacks": []}

        corr_df = detect_correlations(df)
        stacks = detect_team_stacks(df)

        # Build correlation groups
        groups = []
        if "correlation_group" in corr_df.columns:
            for group_name, group_df in corr_df[corr_df["correlation_group"].notna()].groupby(
                "correlation_group"
            ):
                groups.append(
                    {
                        "group": group_name,
                        "type": group_name.split("_")[0] if "_" in str(group_name) else "unknown",
                        "players": [
                            {
                                "player_id": row["player_id"],
                                "player_name": row.get("player_name"),
                                "market": row["market"],
                                "team": row.get("team"),
                            }
                            for _, row in group_df.iterrows()
                        ],
                        "combined_stake": float(group_df["stake"].sum()),
                    }
                )

        # Build team stacks
        team_stack_list = []
        for team, idxs in stacks.items():
            team_rows = df.iloc[idxs] if isinstance(idxs[0], int) else df.loc[idxs]
            team_stack_list.append(
                {
                    "team": team,
                    "count": len(idxs),
                    "player_ids": team_rows["player_id"].tolist(),
                }
            )

        return {"correlation_groups": groups, "team_stacks": team_stack_list}
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Correlation analysis error: {str(e)}")


@app.get("/api/analytics/risk-summary")
async def get_risk_summary(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
):
    """Get risk exposure summary for current bets."""
    try:
        from risk_manager import compute_exposure

        visibility_sql, visibility_params = value_visibility_scope()
        query = f"""
            SELECT v.player_id, pd.player_name, v.team, v.market, v.sportsbook,
                   v.event_id, v.stake, v.kelly_fraction, v.edge_percentage, v.price, v.p_win
            FROM materialized_value_view v
            LEFT JOIN player_dim pd ON v.player_id = pd.player_id
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ?
        """
        df = read_dataframe(query, params=[*visibility_params, season, week])

        if df.empty:
            return {
                "total_stake": 0.0,
                "bankroll": config.betting.bankroll,
                "team_exposure": [],
                "game_exposure": [],
                "guardrails": {
                    "max_team_exposure": config.risk.max_team_exposure,
                    "max_game_exposure": config.risk.max_game_exposure,
                    "max_player_exposure": config.risk.max_player_exposure,
                },
                "warnings": [],
            }

        bankroll = config.betting.bankroll
        exposure_df = compute_exposure(df, bankroll)

        # Team exposure
        team_exp = df.groupby("team")["stake"].sum()
        team_exposure = [
            {"team": team, "stake": float(stake), "fraction": float(stake / bankroll)}
            for team, stake in team_exp.items()
        ]

        # Game exposure
        game_col = "event_id" if "event_id" in df.columns else "team"
        game_exp = df.groupby(game_col)["stake"].sum()
        game_exposure = [
            {"game": str(game), "stake": float(stake), "fraction": float(stake / bankroll)}
            for game, stake in game_exp.items()
        ]

        # Collect warnings
        warnings = []
        if "exposure_warning" in exposure_df.columns:
            for _, row in exposure_df[exposure_df["exposure_warning"].notna()].iterrows():
                warnings.append(
                    {
                        "player_id": row["player_id"],
                        "player_name": row.get("player_name"),
                        "warning": row["exposure_warning"],
                    }
                )

        return {
            "total_stake": float(df["stake"].sum()),
            "bankroll": bankroll,
            "team_exposure": team_exposure,
            "game_exposure": game_exposure,
            "guardrails": {
                "max_team_exposure": config.risk.max_team_exposure,
                "max_game_exposure": config.risk.max_game_exposure,
                "max_player_exposure": config.risk.max_player_exposure,
            },
            "warnings": warnings,
        }
    except Exception as e:
        logger.error(f"Error in risk summary: {e}")
        raise HTTPException(status_code=500, detail=f"Risk summary error: {str(e)}")


# ── Export endpoints (P1: UX polish) ─────────────────────────────


@app.get("/api/export/csv")
async def export_csv(
    season: int = Query(..., ge=2020, le=2030, description="NFL season year"),
    week: int = Query(..., ge=1, le=22, description="Week number"),
    min_edge: float = Query(0.0, ge=0.0, le=1.0, description="Minimum edge percentage"),
):
    """Export current table view as CSV."""
    try:
        visibility_sql, visibility_params = value_visibility_scope()
        query = f"""
            SELECT
                v.player_id,
                pd.player_name,
                pd.position,
                v.team,
                v.market,
                v.sportsbook,
                v.line,
                v.price,
                v.side,
                v.mu,
                v.sigma,
                v.p_win,
                v.edge_percentage,
                v.expected_roi,
                v.kelly_fraction,
                v.stake,
                v.confidence_score,
                v.confidence_tier,
                v.generated_at
            FROM materialized_value_view v
            LEFT JOIN player_dim pd ON v.player_id = pd.player_id
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ? AND v.edge_percentage >= ?
            ORDER BY v.edge_percentage DESC
        """
        df = read_dataframe(
            query,
            params=[*visibility_params, season, week, min_edge],
        )

        output = io.StringIO()
        if df.empty:
            output.write("No data for the given filters\n")
        else:
            df.to_csv(output, index=False)

        output.seek(0)
        filename = f"value_bets_s{season}_w{week}.csv"
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@app.get("/api/export/bundle")
async def export_bundle(
    season: int = Query(..., ge=2020, le=2030, description="NFL season year"),
    week: int = Query(..., ge=1, le=22, description="Week number"),
):
    """Export full JSON run bundle with bets, correlations, and risk summary."""
    try:
        # Fetch bets
        visibility_sql, visibility_params = value_visibility_scope()
        bets_query = f"""
            SELECT
                v.player_id, pd.player_name, pd.position, v.team,
                v.market, v.sportsbook, v.line, v.price, v.side, v.mu, v.sigma,
                v.p_win, v.edge_percentage, v.expected_roi,
                v.kelly_fraction, v.stake, v.confidence_score,
                v.confidence_tier, v.generated_at, v.published_run_id
            FROM materialized_value_view v
            LEFT JOIN player_dim pd ON v.player_id = pd.player_id
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ?
            ORDER BY v.edge_percentage DESC
        """
        bets_df = read_dataframe(
            bets_query,
            params=[*visibility_params, season, week],
        )
        bets_list = bets_df.to_dict(orient="records") if not bets_df.empty else []

        if config.api.demo_mode:
            run_row = fetchone(
                f"SELECT {_PIPELINE_RUN_COLS} FROM pipeline_runs WHERE season = ? AND week = ? "
                "ORDER BY started_at DESC LIMIT 1",
                params=(season, week),
            )
        else:
            published_run_id = next(
                (str(row["published_run_id"]) for row in bets_list if row.get("published_run_id")),
                None,
            )
            run_row = (
                fetchone(
                    f"SELECT {_PIPELINE_RUN_COLS} FROM pipeline_runs WHERE run_id = ?",
                    params=(published_run_id,),
                )
                if published_run_id
                else None
            )
        for bet in bets_list:
            bet.pop("published_run_id", None)
        run_meta = None
        if run_row:
            parsed = _parse_pipeline_run_row(run_row)
            run_meta = parsed.model_dump()

        bundle = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "season": season,
            "week": week,
            "total_bets": len(bets_list),
            "bets": bets_list,
            "pipeline_run": run_meta,
        }

        # Try to include correlation and risk data
        try:
            from risk_manager import compute_exposure, detect_correlations, detect_team_stacks

            risk_query = f"""
                SELECT v.player_id, pd.player_name, pd.position, v.team, v.market,
                       v.sportsbook, v.event_id, v.stake, v.edge_percentage,
                       v.kelly_fraction, v.price, v.p_win
                FROM materialized_value_view v
                LEFT JOIN player_dim pd ON v.player_id = pd.player_id
                WHERE {visibility_sql}
                  AND v.season = ? AND v.week = ?
            """
            risk_df = read_dataframe(
                risk_query,
                params=[*visibility_params, season, week],
            )
            if not risk_df.empty:
                stacks = detect_team_stacks(risk_df)
                bundle["team_stacks"] = {team: len(idxs) for team, idxs in stacks.items()}
                bundle["total_stake"] = float(risk_df["stake"].sum())
        except Exception:
            pass

        content = json.dumps(bundle, indent=2, default=str)
        filename = f"run_bundle_s{season}_w{week}.json"
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        logger.error(f"Error exporting bundle: {e}")
        raise HTTPException(status_code=500, detail=f"Export bundle error: {str(e)}")


# ── Agent review endpoint (P2: Agent teamwork) ──────────────────


@app.post("/api/run/{run_id}/review")
async def request_agent_review(
    run_id: str,
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
    operator_id: str = Depends(require_pipeline_operator),
):
    """Request on-demand agent review for a pipeline run's bets."""
    del operator_id
    # Verify run exists
    row = fetchone(
        "SELECT run_id, status FROM pipeline_runs WHERE run_id = ?",
        params=(run_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    visibility_sql, visibility_params = value_visibility_scope()
    run_predicate = "" if config.api.demo_mode else "AND v.published_run_id = ?"
    run_params = [] if config.api.demo_mode else [run_id]
    visible_bets = fetchone(
        f"""
        SELECT 1
        FROM materialized_value_view v
        WHERE {visibility_sql}
          AND v.season = ? AND v.week = ?
          {run_predicate}
        LIMIT 1
        """,
        params=(*visibility_params, season, week, *run_params),
    )
    if not visible_bets:
        raise HTTPException(status_code=409, detail="Run has no reviewable published bets")

    with _review_threads_lock:
        if run_id in _review_runs_in_flight:
            raise HTTPException(
                status_code=409, detail="Agent review already running for this run"
            )
        _review_runs_in_flight.add(run_id)

    def _review_worker() -> None:
        # Cleanup lives here, not in _run_agent_review_background, so the
        # in-flight entry is released even when tests monkeypatch that function.
        try:
            _run_agent_review_background(run_id, season, week)
        finally:
            with _review_threads_lock:
                _review_runs_in_flight.discard(run_id)

    thread = threading.Thread(target=_review_worker, daemon=True)
    thread.start()

    return {
        "run_id": run_id,
        "review_status": "started",
        "message": "Agent review started in background",
    }


# Operator-gated but stackable: without dedup, repeated POSTs spawn one review
# thread each for the same run. The set is per-process; the spawning route's
# worker removes entries when the review finishes.
_review_threads_lock = threading.Lock()
_review_runs_in_flight: set[str] = set()


def _run_agent_review_background(run_id: str, season: int, week: int):
    """Execute agent review in background and store results in agent_decisions."""
    try:
        from agents.coordinator import AgentCoordinator

        coordinator = AgentCoordinator()
        coordinator.run_review(season, week)

        # Update run metadata to record review timestamp
        execute(
            """
            UPDATE pipeline_runs
            SET report_json = CASE
                WHEN report_json IS NOT NULL
                THEN json_set(report_json, '$.agent_review_at', ?)
                ELSE json_object('agent_review_at', ?)
            END
            WHERE run_id = ?
            """,
            params=(
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
                run_id,
            ),
        )
    except ImportError:
        logger.warning("Agent coordinator not available; storing stub review")
        _store_stub_agent_review(season, week)
    except Exception as exc:
        logger.error("Agent review failed: %s", exc)
        _store_stub_agent_review(season, week)


def _store_stub_agent_review(season: int, week: int):
    """Store a stub agent review when the full coordinator isn't available."""
    visibility_sql, visibility_params = value_visibility_scope()
    bets_query = f"""
        SELECT v.player_id, v.market FROM materialized_value_view v
        WHERE {visibility_sql}
          AND v.season = ? AND v.week = ?
    """
    rows = fetchall(bets_query, params=(*visibility_params, season, week))
    for row in rows:
        player_id, market_name = row[0], row[1]
        existing = fetchone(
            "SELECT 1 FROM agent_decisions WHERE season = ? AND week = ? AND player_id = ? AND market = ?",
            params=(season, week, player_id, market_name),
        )
        if not existing:
            execute(
                """
                INSERT INTO agent_decisions
                    (season, week, player_id, market, decision, merged_confidence,
                     votes, rationale, decided_at)
                VALUES (?, ?, ?, ?, 'APPROVED', 0.70,
                        '{"model": "APPROVE", "risk": "APPROVE", "market": "PENDING", "diagnostics": "PENDING"}',
                        'Auto-approved pending full agent review', ?)
                """,
                params=(
                    season,
                    week,
                    player_id,
                    market_name,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


@app.get("/api/run/{run_id}/review-status")
async def get_agent_review_status(
    run_id: str,
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
    reader_id: str = Depends(require_pipeline_reader),
):
    """Check if agent review has been completed for this run."""
    del reader_id
    row = fetchone(
        "SELECT report_json FROM pipeline_runs WHERE run_id = ?",
        params=(run_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    review_at = None
    if row[0]:
        try:
            report = json.loads(row[0])
            review_at = report.get("agent_review_at")
        except (json.JSONDecodeError, TypeError):
            pass

    # Count agent decisions for this season/week
    count_row = fetchone(
        "SELECT COUNT(*) FROM agent_decisions WHERE season = ? AND week = ?",
        params=(season, week),
    )
    decision_count = count_row[0] if count_row else 0

    return {
        "run_id": run_id,
        "reviewed": review_at is not None,
        "reviewed_at": review_at,
        "decision_count": decision_count,
    }


# Auth endpoints


@app.post("/api/auth/register")
async def register(user_data: UserCreate):
    """Register a new user.

    T0 #2: pass the validated UserCreate model directly to create_user — the
    previous kwargs unpacking didn't match the function signature and raised
    TypeError on every call. authenticate_user creates the session itself so
    register reuses that path instead of duplicating the INSERT.
    """
    try:
        # bcrypt at rounds=12 spends ~300–500ms on a real machine. Running
        # it inline would block the event loop and stall every other
        # in-flight request. Hand off to a worker thread.
        user = await asyncio.to_thread(create_user, user_data)
        if user is None:
            raise HTTPException(status_code=500, detail="Registration failed")

        session = await asyncio.to_thread(
            authenticate_user, UserLogin(email=user_data.email, password=user_data.password)
        )
        if session is None:
            raise HTTPException(status_code=500, detail="Session creation failed")

        return {
            "user": user.model_dump(),
            "session_id": session["session_id"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error registering user: %s", e)
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    """Authenticate user and create session.

    T0 #2: pass UserLogin model directly. authenticate_user returns
    {"session_id", "expires_at", "user"} — server returns the public subset.
    """
    try:
        # See register() — bcrypt verify must not block the loop.
        session = await asyncio.to_thread(authenticate_user, credentials)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {
            "user": session["user"].model_dump(),
            "session_id": session["session_id"],
            "expires_at": session["expires_at"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error logging in: %s", e)
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/api/auth/logout")
async def logout(request: Request, current_user: UserResponse = Depends(get_current_user)):
    """Logout user and delete the current session only.

    Bug W2: previously deleted every session_id for this user_id, which
    forced sign-out across all the user's other devices.
    """
    auth_header = request.headers.get("Authorization", "")
    session_id = auth_header.replace("Bearer ", "", 1) if auth_header.startswith("Bearer ") else ""
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        logout_user(session_id)
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.error("Error logging out: %s", e)
        raise HTTPException(status_code=500, detail="Logout failed")


@app.get("/api/auth/me")
async def get_me(current_user: UserResponse = Depends(get_current_user)):
    """Get current user."""
    return current_user


# User endpoints (require auth)


@app.get("/api/user/preferences")
async def get_preferences(current_user: UserResponse = Depends(get_current_user)):
    """Get user preferences."""
    try:
        prefs = get_user_preferences(current_user.user_id)
        return prefs
    except Exception as e:
        logger.error(f"Error fetching preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch preferences")


@app.put("/api/user/preferences")
async def update_preferences(
    preferences: UserPreferences,
    current_user: UserResponse = Depends(get_current_user),
):
    """Update user preferences."""
    try:
        updated = update_user_preferences(current_user.user_id, preferences)
        return updated
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@app.put("/api/user/bankroll")
async def update_user_bankroll(
    bankroll: float = Query(..., description="New bankroll amount"),
    current_user: UserResponse = Depends(get_current_user),
):
    """Update user bankroll."""
    try:
        update_bankroll(current_user.user_id, bankroll)
        return {"message": "Bankroll updated", "bankroll": bankroll}
    except Exception as e:
        logger.error(f"Error updating bankroll: {e}")
        raise HTTPException(status_code=500, detail="Failed to update bankroll")


@app.post("/api/user/bets")
async def record_bet(
    bet: BetCreate,
    current_user: UserResponse = Depends(get_current_user),
):
    """Record a user bet (FR-T0-3).

    Validates payload at the trust boundary via the BetCreate Pydantic model,
    rejects malformed/under-side-missing bodies with 422 before touching the
    database. The INSERT now matches the user_bets schema column-for-column
    (id, side, stake_units, model_edge) — the prior version listed 10 columns
    for an 11-column NOT NULL schema and silently failed.
    """
    side = bet.side
    bet_id = str(uuid.uuid4())
    placed_at = datetime.now(timezone.utc).isoformat()
    try:
        execute(
            """
            INSERT INTO user_bets (
                id, user_id, season, week, player_id, player_name, market,
                sportsbook, side, line, price, stake_units, stake_dollars,
                model_edge, confidence_tier, placed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params=[
                bet_id,
                current_user.user_id,
                bet.season,
                bet.week,
                bet.player_id,
                bet.player_name,
                bet.market,
                bet.sportsbook,
                side,
                bet.line,
                bet.price,
                bet.stake_units,
                bet.stake_dollars,
                bet.model_edge,
                bet.confidence_tier,
                placed_at,
            ],
        )
        return {"bet_id": bet_id, "message": "Bet recorded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error recording bet: %s", e, extra={"user_id": current_user.user_id})
        raise HTTPException(status_code=500, detail="Failed to record bet")


@app.get("/api/user/bets")
async def get_user_bets(
    season: Optional[int] = Query(None, description="Filter by season"),
    week: Optional[int] = Query(None, description="Filter by week"),
    current_user: UserResponse = Depends(get_current_user),
):
    """Get user bets."""
    try:
        query = """
            SELECT
                id AS bet_id, season, week, player_id, player_name, market,
                sportsbook, side, line, price, stake_units, stake_dollars,
                model_edge, confidence_tier, placed_at,
                actual_result, outcome, profit_units, profit_dollars, graded_at
            FROM user_bets
            WHERE user_id = ?
        """
        params = [current_user.user_id]

        if season:
            query += " AND season = ?"
            params.append(season)
        if week:
            query += " AND week = ?"
            params.append(week)

        query += " ORDER BY placed_at DESC"

        df = read_dataframe(query, params=params)

        return {
            "bets": df.to_dict(orient="records") if not df.empty else [],
            "total": len(df),
        }
    except Exception as e:
        logger.error(f"Error fetching user bets: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bets")


@app.get("/api/user/stats")
async def get_user_stats(current_user: UserResponse = Depends(get_current_user)):
    """Get user betting statistics."""
    try:
        query = """
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
                SUM(COALESCE(profit_units, 0)) as total_profit,
                AVG(model_edge) as avg_edge
            FROM user_bets
            WHERE user_id = ?
        """
        result = fetchone(query, params=[current_user.user_id])

        if not result:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "total_profit": 0.0,
                "avg_edge": 0.0,
                "roi": 0.0,
            }

        total_bets = result[0] or 0
        wins = result[1] or 0
        losses = result[2] or 0
        pushes = result[3] or 0
        total_profit = result[4] or 0.0
        avg_edge = result[5] or 0.0

        roi = (total_profit / total_bets * 100) if total_bets > 0 else 0.0

        return {
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total_profit": total_profit,
            "avg_edge": avg_edge,
            "roi": roi,
        }
    except Exception as e:
        logger.error(f"Error fetching user stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")


@app.get("/api/backtest/summary")
async def backtest_summary(
    sport: str = Query("nfl", description="Sport: nfl or nba"),
    season: Optional[int] = Query(None, description="Filter by season"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD, NBA only)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD, NBA only)"),
):
    """Return aggregate backtest metrics and recent individual picks with outcomes."""
    try:
        if sport == "nba":
            conditions = []
            params: list = []

            if season is not None:
                conditions.append("n.season = ?")
                params.append(season)
            if start_date:
                conditions.append("n.game_date >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("n.game_date <= ?")
                params.append(end_date)

            where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

            summary_query = f"""
                SELECT
                    AVG(model_abs_error) AS avg_model_mae,
                    AVG(line_abs_error) AS avg_line_mae,
                    AVG(CAST(model_beats_line AS REAL)) AS model_beats_line_pct,
                    COUNT(*) AS total_bets
                FROM nba_line_accuracy_history n
                {where_clause}
            """
            summary_row = fetchone(summary_query, params=params if params else None)

            picks_query = f"""
                SELECT
                    n.game_date,
                    p.player_name,
                    n.market,
                    n.line,
                    n.actual,
                    n.mu,
                    n.model_abs_error,
                    n.line_abs_error,
                    n.model_beats_line
                FROM nba_line_accuracy_history n
                LEFT JOIN nba_player_game_logs p
                    ON n.player_id = p.player_id
                    AND n.game_date = p.game_date
                {where_clause}
                ORDER BY n.game_date DESC, n.model_abs_error ASC
                LIMIT 100
            """
            picks_df = read_dataframe(picks_query, params=params if params else None)

        else:
            # NFL
            conditions = []
            params = []

            if season is not None:
                conditions.append("l.season = ?")
                params.append(season)

            where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

            summary_query = f"""
                SELECT
                    AVG(model_abs_error) AS avg_model_mae,
                    AVG(line_abs_error) AS avg_line_mae,
                    AVG(CAST(model_beats_line AS REAL)) AS model_beats_line_pct,
                    COUNT(*) AS total_bets
                FROM line_accuracy_history l
                {where_clause}
            """
            summary_row = fetchone(summary_query, params=params if params else None)

            picks_query = f"""
                SELECT
                    l.season || '-W' || printf('%02d', l.week) AS game_date,
                    pd.player_name,
                    l.market,
                    l.line,
                    l.actual,
                    l.mu,
                    l.model_abs_error,
                    l.line_abs_error,
                    l.model_beats_line
                FROM line_accuracy_history l
                LEFT JOIN player_dim pd ON l.player_id = pd.player_id
                {where_clause}
                ORDER BY l.season DESC, l.week DESC, l.model_abs_error ASC
                LIMIT 100
            """
            picks_df = read_dataframe(picks_query, params=params if params else None)

        if summary_row:
            avg_model_mae = summary_row[0]
            avg_line_mae = summary_row[1]
            model_beats_line_pct = summary_row[2]
            total_bets = summary_row[3]
        else:
            avg_model_mae = None
            avg_line_mae = None
            model_beats_line_pct = None
            total_bets = 0

        recent_picks = picks_df.to_dict(orient="records") if not picks_df.empty else []

        return {
            "summary": {
                "avg_model_mae": round(avg_model_mae, 3) if avg_model_mae is not None else None,
                "avg_line_mae": round(avg_line_mae, 3) if avg_line_mae is not None else None,
                "model_beats_line_pct": (
                    round(model_beats_line_pct * 100, 1)
                    if model_beats_line_pct is not None
                    else None
                ),
                "total_bets": int(total_bets) if total_bets is not None else 0,
            },
            "recent_picks": recent_picks,
        }
    except Exception as e:
        logger.error(f"Error fetching backtest summary: {e}")
        return {
            "summary": {
                "avg_model_mae": None,
                "avg_line_mae": None,
                "model_beats_line_pct": None,
                "total_bets": 0,
            },
            "recent_picks": [],
        }
