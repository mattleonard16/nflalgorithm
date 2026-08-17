"""Explainability payloads for value bet recommendations.

Assembles a 'why' object per bet with model inputs, confidence breakdown,
risk flags, and agent verdicts.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from api.value_visibility import value_visibility_scope
from utils.db import fetchone

logger = logging.getLogger(__name__)


def build_why_payload(season: int, week: int, player_id: str, market: str) -> Dict[str, Any]:
    """Build a structured explainability payload for a single bet.

    Returns dict with keys: model, volume, volatility, confidence, risk, agents.
    """
    if not _is_public_value_row(season, week, player_id, market):
        return _empty_why_payload()

    payload: Dict[str, Any] = {}

    # Model section from weekly_projections
    payload["model"] = _get_model_section(season, week, player_id, market)

    # Volume section from weekly_projections
    payload["volume"] = _get_volume_section(season, week, player_id, market)

    # Volatility section
    payload["volatility"] = _get_volatility_section(season, week, player_id, market)

    # Confidence breakdown from materialized_value_view
    payload["confidence"] = _get_confidence_section(season, week, player_id, market)

    # Risk section from risk_assessments
    payload["risk"] = _get_risk_section(season, week, player_id, market)

    # Agent verdicts from agent_decisions
    payload["agents"] = _get_agents_section(season, week, player_id, market)

    return payload


def _is_public_value_row(season: int, week: int, player_id: str, market: str) -> bool:
    visibility_sql, visibility_params = value_visibility_scope()
    return (
        fetchone(
            f"""
            SELECT 1
            FROM materialized_value_view v
            WHERE {visibility_sql}
              AND v.season = ? AND v.week = ? AND v.player_id = ? AND v.market = ?
            LIMIT 1
            """,
            params=(*visibility_params, season, week, player_id, market),
        )
        is not None
    )


def _empty_why_payload() -> Dict[str, Any]:
    return {
        "model": {"mu": None, "sigma": None, "context_sensitivity": None},
        "volume": {"target_share": None},
        "volatility": {"score": None},
        "confidence": {
            "total": None,
            "edge_score": None,
            "stability_score": None,
            "volume_score": None,
            "volatility_score": None,
        },
        "risk": {
            "correlation_group": None,
            "exposure_warning": None,
            "risk_adjusted_kelly": None,
        },
        "agents": {
            "decision": None,
            "merged_confidence": None,
            "votes": None,
            "top_rationale": None,
        },
    }


def build_why_payloads_batch(
    season: int, week: int, bets: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Build explainability payloads for a batch of bets.

    Returns dict keyed by 'player_id:market'.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for bet in bets:
        pid = bet.get("player_id", "")
        mkt = bet.get("market", "")
        key = f"{pid}:{mkt}"
        try:
            result[key] = build_why_payload(season, week, pid, mkt)
        except Exception as exc:
            logger.warning("Failed to build why for %s: %s", key, exc)
            result[key] = {"error": str(exc)}
    return result


def _get_model_section(season: int, week: int, player_id: str, market: str) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT mu, sigma, context_sensitivity
        FROM weekly_projections
        WHERE season = ? AND week = ? AND player_id = ? AND market = ?
        """,
        params=(season, week, player_id, market),
    )
    if not row:
        return {"mu": None, "sigma": None, "context_sensitivity": None}
    return {
        "mu": row[0],
        "sigma": row[1],
        "context_sensitivity": row[2],
    }


def _get_volume_section(season: int, week: int, player_id: str, market: str) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT target_share
        FROM weekly_projections
        WHERE season = ? AND week = ? AND player_id = ? AND market = ?
        """,
        params=(season, week, player_id, market),
    )
    if not row:
        return {"target_share": None}
    return {"target_share": row[0]}


def _get_volatility_section(season: int, week: int, player_id: str, market: str) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT volatility_score
        FROM weekly_projections
        WHERE season = ? AND week = ? AND player_id = ? AND market = ?
        """,
        params=(season, week, player_id, market),
    )
    if not row:
        return {"score": None}
    return {"score": row[0]}


def _get_confidence_section(season: int, week: int, player_id: str, market: str) -> Dict[str, Any]:
    visibility_sql, visibility_params = value_visibility_scope()
    row = fetchone(
        f"""
        SELECT v.confidence_score, v.confidence_tier, v.edge_percentage, v.p_win
        FROM materialized_value_view v
        WHERE {visibility_sql}
          AND v.season = ? AND v.week = ? AND v.player_id = ? AND v.market = ?
        LIMIT 1
        """,
        params=(*visibility_params, season, week, player_id, market),
    )
    if not row:
        return {
            "total": None,
            "edge_score": None,
            "stability_score": None,
            "volume_score": None,
            "volatility_score": None,
        }
    return {
        "total": row[0],
        "tier": row[1],
        "edge_pct": row[2],
        "p_win": row[3],
    }


def _get_risk_section(season: int, week: int, player_id: str, market: str) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT correlation_group, exposure_warning, risk_adjusted_kelly
        FROM risk_assessments
        WHERE season = ? AND week = ? AND player_id = ? AND market = ?
        LIMIT 1
        """,
        params=(season, week, player_id, market),
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


def _get_agents_section(season: int, week: int, player_id: str, market: str) -> Dict[str, Any]:
    row = fetchone(
        """
        SELECT decision, merged_confidence, votes, rationale
        FROM agent_decisions
        WHERE season = ? AND week = ? AND player_id = ? AND market = ?
        """,
        params=(season, week, player_id, market),
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
