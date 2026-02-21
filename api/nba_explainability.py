"""NBA explainability payloads for value bet recommendations.

Assembles a 'why' object per bet with model inputs, recency stats,
variance, confidence breakdown, risk flags, and agent verdicts.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from utils.db import fetchone, read_dataframe
from utils.nba_sigma import get_sigma_or_default

logger = logging.getLogger(__name__)

VALID_MARKETS = frozenset({"pts", "reb", "ast", "fg3m"})
_MARKET_COL: Dict[str, str] = {"pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m"}


def build_why_payload(
    game_date: str, player_id: int, market: str
) -> Dict[str, Any]:
    """Build a structured explainability payload for a single NBA bet.

    Returns dict with keys: model, recency, variance, confidence, risk, agents.
    """
    return {
        "model": _get_model_section(game_date, player_id, market),
        "recency": _get_recency_section(game_date, player_id, market),
        "variance": _get_variance_section(game_date, player_id, market),
        "confidence": _get_confidence_section(game_date, player_id, market),
        "risk": _get_risk_section(game_date, player_id, market),
        "agents": _get_agents_section(game_date, player_id, market),
    }


def build_why_payloads_batch(
    game_date: str, bets: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Build explainability payloads for a batch of bets.

    Uses 6 bulk queries for game_date then assembles per-bet payloads
    from the preloaded data to avoid N+1 query patterns.

    Returns dict keyed by 'player_id:market'.
    """
    if not bets:
        return {}

    player_ids = [
        int(b["player_id"]) for b in bets if b.get("player_id") is not None
    ]

    bulk = _load_bulk_data(game_date, player_ids)

    result: Dict[str, Dict[str, Any]] = {}
    for bet in bets:
        pid = bet.get("player_id")
        mkt = bet.get("market", "")
        key = f"{pid}:{mkt}"
        try:
            pid_int = int(pid) if pid is not None else 0
            result[key] = _assemble_payload(game_date, pid_int, mkt, bulk)
        except Exception as exc:
            logger.warning("Failed to build NBA why for %s: %s", key, exc)
            result[key] = {"error": str(exc)}
    return result


def _load_bulk_data(
    game_date: str, player_ids: List[int]
) -> Dict[str, Any]:
    """Execute 6 bulk queries for the given game_date and player list.

    Returns a dict with keys: projections, game_logs, value_view,
    risk, agents — each keyed by (player_id, market) or player_id.
    """
    if not player_ids:
        placeholders = "(-1)"
    else:
        placeholders = "(" + ",".join("?" * len(player_ids)) + ")"

    pid_params = player_ids if player_ids else []

    # Query 1: projections (include sigma for explainability)
    proj_rows = read_dataframe(
        "SELECT player_id, market, projected_value, confidence, sigma "
        "FROM nba_projections "
        "WHERE game_date = ? AND player_id IN " + placeholders,
        [game_date] + pid_params,
    )
    projections: Dict[tuple, Any] = {}
    if not proj_rows.empty:
        for _, r in proj_rows.iterrows():
            projections[(int(r["player_id"]), str(r["market"]))] = r

    # Query 2: game log recency (last 10 per player, all markets in one pass)
    log_rows = read_dataframe(
        "SELECT player_id, pts, reb, ast, fg3m, rn "
        "FROM ("
        "  SELECT player_id, pts, reb, ast, fg3m, "
        "  ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) as rn "
        "  FROM nba_player_game_logs "
        "  WHERE player_id IN " + placeholders + " AND game_date < ?"
        ") sub WHERE rn <= 10",
        pid_params + [game_date],
    )
    # Build per-(player_id, market) recency stats from the raw rows
    recency_by_pid: Dict[int, Dict[str, Any]] = {}
    if not log_rows.empty:
        for pid in log_rows["player_id"].unique():
            player_logs = log_rows[log_rows["player_id"] == pid]
            pid_int = int(pid)
            recency_by_pid[pid_int] = {}
            for col_name in ("pts", "reb", "ast", "fg3m"):
                last5 = player_logs[player_logs["rn"] <= 5][col_name].mean()
                last10 = player_logs[col_name].mean()
                recency_by_pid[pid_int][col_name] = {
                    "last5_avg": float(last5) if last5 == last5 else None,
                    "last10_avg": float(last10) if last10 == last10 else None,
                }

    # Query 3: value view (confidence metrics)
    vv_rows = read_dataframe(
        "SELECT player_id, market, p_win, edge_percentage, expected_roi, kelly_fraction "
        "FROM nba_materialized_value_view "
        "WHERE game_date = ? AND player_id IN " + placeholders,
        [game_date] + pid_params,
    )
    value_view: Dict[tuple, Any] = {}
    if not vv_rows.empty:
        for _, r in vv_rows.iterrows():
            value_view[(int(r["player_id"]), str(r["market"]))] = r

    # Query 4: risk assessments
    risk_rows = read_dataframe(
        "SELECT player_id, market, correlation_group, exposure_warning, risk_adjusted_kelly "
        "FROM nba_risk_assessments "
        "WHERE game_date = ? AND player_id IN " + placeholders,
        [game_date] + pid_params,
    )
    risk: Dict[tuple, Any] = {}
    if not risk_rows.empty:
        for _, r in risk_rows.iterrows():
            risk[(int(r["player_id"]), str(r["market"]))] = r

    # Query 5: agent decisions
    agent_rows = read_dataframe(
        "SELECT player_id, market, decision, merged_confidence, votes, rationale "
        "FROM nba_agent_decisions "
        "WHERE game_date = ? AND player_id IN " + placeholders,
        [game_date] + pid_params,
    )
    agents: Dict[tuple, Any] = {}
    if not agent_rows.empty:
        for _, r in agent_rows.iterrows():
            agents[(int(r["player_id"]), str(r["market"]))] = r

    return {
        "projections": projections,
        "recency": recency_by_pid,
        "value_view": value_view,
        "risk": risk,
        "agents": agents,
    }


def _assemble_payload(
    game_date: str, player_id: int, market: str, bulk: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble a single why payload from preloaded bulk data."""
    payload: Dict[str, Any] = {}
    payload["model"] = _model_from_bulk(player_id, market, bulk["projections"])
    payload["recency"] = _recency_from_bulk(player_id, market, bulk["recency"])
    payload["variance"] = _variance_from_bulk(player_id, market, bulk["projections"])
    payload["confidence"] = _confidence_from_bulk(player_id, market, bulk["value_view"])
    payload["risk"] = _risk_from_bulk(player_id, market, bulk["risk"])
    payload["agents"] = _agents_from_bulk(player_id, market, bulk["agents"])
    return payload


def _model_from_bulk(
    player_id: int, market: str, projections: Dict[tuple, Any]
) -> Dict[str, Any]:
    row = projections.get((player_id, market))
    if row is None:
        return {"projected_value": None, "sigma": None, "confidence": None}
    projected_value = row["projected_value"]
    sigma_raw = row.get("sigma") if "sigma" in row.index else None
    sigma = get_sigma_or_default(sigma_raw, float(projected_value), market) if projected_value else None
    return {
        "projected_value": projected_value,
        "sigma": sigma,
        "confidence": row["confidence"],
    }


def _recency_from_bulk(
    player_id: int, market: str, recency: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    col = _MARKET_COL.get(market)
    if col is None:
        return {"last5_avg": None, "last10_avg": None, "trend": None}
    pid_data = recency.get(player_id, {})
    market_data = pid_data.get(col, {})
    last5 = market_data.get("last5_avg")
    last10 = market_data.get("last10_avg")
    if last5 is None or last10 is None:
        return {"last5_avg": last5, "last10_avg": last10, "trend": None}

    if last5 > last10 * 1.05:
        trend = "up"
    elif last5 < last10 * 0.95:
        trend = "down"
    else:
        trend = "stable"

    return {
        "last5_avg": round(last5, 1),
        "last10_avg": round(last10, 1),
        "trend": trend,
    }


def _variance_from_bulk(
    player_id: int, market: str, projections: Dict[tuple, Any]
) -> Dict[str, Any]:
    row = projections.get((player_id, market))
    if row is None or row["projected_value"] is None:
        return {"sigma": None, "cv": None}
    projected = float(row["projected_value"])
    sigma_raw = row.get("sigma") if "sigma" in row.index else None
    sigma = get_sigma_or_default(sigma_raw, projected, market)
    cv = sigma / projected if projected > 0 else None
    return {
        "sigma": round(sigma, 2),
        "cv": round(cv, 3) if cv is not None else None,
    }


def _confidence_from_bulk(
    player_id: int, market: str, value_view: Dict[tuple, Any]
) -> Dict[str, Any]:
    row = value_view.get((player_id, market))
    if row is None:
        return {
            "p_win": None,
            "edge_percentage": None,
            "expected_roi": None,
            "kelly_fraction": None,
        }
    return {
        "p_win": row["p_win"],
        "edge_percentage": row["edge_percentage"],
        "expected_roi": row["expected_roi"],
        "kelly_fraction": row["kelly_fraction"],
    }


def _risk_from_bulk(
    player_id: int, market: str, risk: Dict[tuple, Any]
) -> Dict[str, Any]:
    row = risk.get((player_id, market))
    if row is None:
        return {
            "correlation_group": None,
            "exposure_warning": None,
            "risk_adjusted_kelly": None,
        }
    return {
        "correlation_group": row["correlation_group"],
        "exposure_warning": row["exposure_warning"],
        "risk_adjusted_kelly": row["risk_adjusted_kelly"],
    }


def _agents_from_bulk(
    player_id: int, market: str, agents: Dict[tuple, Any]
) -> Dict[str, Any]:
    row = agents.get((player_id, market))
    if row is None:
        return {
            "decision": None,
            "merged_confidence": None,
            "votes": None,
            "top_rationale": None,
        }
    votes = row["votes"]
    if isinstance(votes, str):
        try:
            votes = json.loads(votes)
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "decision": row["decision"],
        "merged_confidence": row["merged_confidence"],
        "votes": votes,
        "top_rationale": row["rationale"],
    }


def _get_model_section(
    game_date: str, player_id: int, market: str
) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT projected_value, confidence, sigma
        FROM nba_projections
        WHERE game_date = ? AND player_id = ? AND market = ?
        """,
        params=(game_date, player_id, market),
    )
    if not row:
        return {"projected_value": None, "sigma": None, "confidence": None}

    projected_value = row[0]
    sigma_raw = row[2] if len(row) > 2 else None
    sigma = get_sigma_or_default(sigma_raw, float(projected_value), market) if projected_value else None
    return {
        "projected_value": row[0],
        "sigma": sigma,
        "confidence": row[1],
    }


def _get_recency_section(
    game_date: str, player_id: int, market: str
) -> Dict[str, Any]:
    """Get last-5 and last-10 game averages for the relevant stat."""
    col = _MARKET_COL.get(market)
    if col is None:
        return {"last5_avg": None, "last10_avg": None, "trend": None}

    row = fetchone(
        "SELECT "
        f"AVG(CASE WHEN rn <= 5 THEN {col} END) as last5, "
        f"AVG(CASE WHEN rn <= 10 THEN {col} END) as last10 "
        "FROM ("
        f"  SELECT {col}, "
        "  ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) as rn "
        "  FROM nba_player_game_logs "
        "  WHERE player_id = ? AND game_date < ?"
        ") sub WHERE rn <= 10",
        params=(player_id, game_date),
    )
    if not row or row[0] is None:
        return {"last5_avg": None, "last10_avg": None, "trend": None}

    last5 = float(row[0])
    last10 = float(row[1]) if row[1] else last5

    if last5 > last10 * 1.05:
        trend = "up"
    elif last5 < last10 * 0.95:
        trend = "down"
    else:
        trend = "stable"

    return {
        "last5_avg": round(last5, 1),
        "last10_avg": round(last10, 1),
        "trend": trend,
    }


def _get_variance_section(
    game_date: str, player_id: int, market: str
) -> Dict[str, Any]:
    """Compute sigma and coefficient of variation from stored projections."""
    row = fetchone(
        """
        SELECT projected_value, sigma
        FROM nba_projections
        WHERE game_date = ? AND player_id = ? AND market = ?
        """,
        params=(game_date, player_id, market),
    )
    if not row or row[0] is None:
        return {"sigma": None, "cv": None}

    projected = float(row[0])
    sigma_raw = row[1] if len(row) > 1 else None
    sigma = get_sigma_or_default(sigma_raw, projected, market)
    cv = sigma / projected if projected > 0 else None

    return {
        "sigma": round(sigma, 2),
        "cv": round(cv, 3) if cv is not None else None,
    }


def _get_confidence_section(
    game_date: str, player_id: int, market: str
) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT p_win, edge_percentage, expected_roi, kelly_fraction
        FROM nba_materialized_value_view
        WHERE game_date = ? AND player_id = ? AND market = ?
        LIMIT 1
        """,
        params=(game_date, player_id, market),
    )
    if not row:
        return {
            "p_win": None,
            "edge_percentage": None,
            "expected_roi": None,
            "kelly_fraction": None,
        }
    return {
        "p_win": row[0],
        "edge_percentage": row[1],
        "expected_roi": row[2],
        "kelly_fraction": row[3],
    }


def _get_risk_section(
    game_date: str, player_id: int, market: str
) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT correlation_group, exposure_warning, risk_adjusted_kelly
        FROM nba_risk_assessments
        WHERE game_date = ? AND player_id = ? AND market = ?
        LIMIT 1
        """,
        params=(game_date, player_id, market),
    )
    if not row:
        return {
            "correlation_group": None,
            "exposure_warning": None,
            "risk_adjusted_kelly": None,
        }
    return {
        "correlation_group": row[0],
        "exposure_warning": row[1],
        "risk_adjusted_kelly": row[2],
    }


def _get_agents_section(
    game_date: str, player_id: int, market: str
) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT decision, merged_confidence, votes, rationale
        FROM nba_agent_decisions
        WHERE game_date = ? AND player_id = ? AND market = ?
        """,
        params=(game_date, player_id, market),
    )
    if not row:
        return {
            "decision": None,
            "merged_confidence": None,
            "votes": None,
            "top_rationale": None,
        }
    votes = row[2]
    if isinstance(votes, str):
        try:
            votes = json.loads(votes)
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "decision": row[0],
        "merged_confidence": row[1],
        "votes": votes,
        "top_rationale": row[3],
    }
