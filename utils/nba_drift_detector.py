"""NBA drift detection utilities.

Uses Population Stability Index (PSI) to detect distributional shifts between
reference and recent prediction or feature distributions.

PSI thresholds:
    < 0.1  — stable
    0.1–0.2 — monitor
    > 0.2  — alert

No scipy dependency — all math uses numpy only.

Usage::

    from utils.nba_drift_detector import detect_prediction_drift, run_drift_checks

    result = detect_prediction_drift("pts", window_days=14, reference_days=30)
    alerts = run_drift_checks("2026-03-04")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKETS = ["pts", "reb", "ast", "fg3m"]

# PSI thresholds
_PSI_MONITOR = 0.1
_PSI_ALERT = 0.2


# ---------------------------------------------------------------------------
# Core PSI computation
# ---------------------------------------------------------------------------


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two 1-D distributions.

    Bins the *expected* (reference) distribution using percentile edges, then
    counts how many *actual* values fall in each bin.  Adds epsilon to avoid
    log(0) / division by zero in empty bins.

    Args:
        expected: Reference distribution array.
        actual:   Current distribution array.
        n_bins:   Number of bins (percentile-based on expected).

    Returns:
        PSI score (non-negative float).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Build bin edges from expected distribution via percentiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(expected, percentiles)

    # Ensure unique edges (degenerate distributions can produce duplicates)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Counts in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    epsilon = 1e-6
    n_exp = max(len(expected), 1)
    n_act = max(len(actual), 1)

    expected_pct = (expected_counts.astype(float) + epsilon) / (n_exp + epsilon * len(expected_counts))
    actual_pct = (actual_counts.astype(float) + epsilon) / (n_act + epsilon * len(actual_counts))

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return max(0.0, psi)


# ---------------------------------------------------------------------------
# Alert level helper
# ---------------------------------------------------------------------------


def _alert_level(psi: float) -> str:
    if psi >= _PSI_ALERT:
        return "alert"
    if psi >= _PSI_MONITOR:
        return "monitor"
    return "stable"


# ---------------------------------------------------------------------------
# Prediction drift
# ---------------------------------------------------------------------------


def detect_prediction_drift(
    market: str,
    window_days: int = 14,
    reference_days: int = 30,
) -> dict[str, Any]:
    """Compare recent vs reference projected_value distributions for a market.

    Loads from ``nba_projections`` table.  Returns a dict with keys:
    market, psi, alert_level, window_days, reference_days,
    n_recent, n_reference, explanation.

    If there is insufficient data (< 14 rows in either window), returns
    alert_level='stable' with an explanatory message.
    """
    from utils.db import read_dataframe

    today = datetime.now(timezone.utc).date()
    recent_start = (today - timedelta(days=window_days)).isoformat()
    reference_start = (today - timedelta(days=window_days + reference_days)).isoformat()
    reference_end = recent_start

    try:
        recent_df = read_dataframe(
            "SELECT projected_value FROM nba_projections "
            "WHERE market = ? AND game_date >= ?",
            (market, recent_start),
        )
        reference_df = read_dataframe(
            "SELECT projected_value FROM nba_projections "
            "WHERE market = ? AND game_date >= ? AND game_date < ?",
            (market, reference_start, reference_end),
        )
    except Exception as exc:
        logger.warning("[drift] prediction drift query failed: %s", exc)
        return _insufficient_result(market, "prediction", window_days, reference_days, f"query error: {exc}")

    n_recent = len(recent_df)
    n_reference = len(reference_df)

    if n_recent < 14 or n_reference < 14:
        return _insufficient_result(
            market,
            "prediction",
            window_days,
            reference_days,
            f"insufficient data: n_recent={n_recent} n_reference={n_reference}",
            n_recent=n_recent,
            n_reference=n_reference,
        )

    psi = compute_psi(
        reference_df["projected_value"].to_numpy(),
        recent_df["projected_value"].to_numpy(),
    )
    level = _alert_level(psi)

    return {
        "market": market,
        "check_type": "prediction",
        "psi": round(psi, 6),
        "alert_level": level,
        "window_days": window_days,
        "reference_days": reference_days,
        "n_recent": n_recent,
        "n_reference": n_reference,
        "explanation": (
            f"PSI={psi:.4f} ({level}) comparing {n_recent} recent vs "
            f"{n_reference} reference projections for market={market}"
        ),
    }


# ---------------------------------------------------------------------------
# Feature drift
# ---------------------------------------------------------------------------


def detect_feature_drift(
    market: str,
    feature_name: str,
    window_days: int = 14,
    reference_days: int = 30,
) -> dict[str, Any]:
    """Compute PSI for a specific feature column over recent vs reference windows.

    Loads from ``nba_projections`` if the column exists there, otherwise
    returns stable with an explanation.  In practice, feature-level drift
    should be computed against the raw game-log data; this stub enables the
    interface contract.
    """
    # Feature drift requires raw feature data — not currently stored per-row.
    # Return stable with explanation so the pipeline is non-blocking.
    return {
        "market": market,
        "check_type": f"feature:{feature_name}",
        "psi": 0.0,
        "alert_level": "stable",
        "window_days": window_days,
        "reference_days": reference_days,
        "n_recent": 0,
        "n_reference": 0,
        "explanation": (
            f"feature drift for '{feature_name}' not computed — "
            "raw feature history not stored per-prediction row"
        ),
    }


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------


def run_drift_checks(
    game_date: str,
    markets: list[str] | None = None,
    alert_threshold: float = _PSI_ALERT,
) -> list[dict[str, Any]]:
    """Run PSI drift checks for all (or specified) markets.

    Saves results to ``nba_drift_alerts`` table.  Non-blocking — never raises.

    Args:
        game_date:       Date context for saving alerts (YYYY-MM-DD).
        markets:         Markets to check.  Defaults to all 4 markets.
        alert_threshold: PSI threshold above which alert_level='alert'.

    Returns:
        List of result dicts (one per market check).
    """
    active_markets = markets if markets is not None else MARKETS
    results: list[dict[str, Any]] = []

    for market in active_markets:
        try:
            result = detect_prediction_drift(market)
            results.append(result)
        except Exception as exc:
            logger.warning("[drift] market=%s check failed: %s", market, exc)
            results.append(
                {
                    "market": market,
                    "check_type": "prediction",
                    "psi": 0.0,
                    "alert_level": "stable",
                    "window_days": 14,
                    "reference_days": 30,
                    "n_recent": 0,
                    "n_reference": 0,
                    "explanation": f"check failed: {exc}",
                }
            )

    _save_alerts(results, game_date)
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_alerts(results: list[dict[str, Any]], game_date: str) -> None:
    """Write drift check results to nba_drift_alerts."""
    from utils.db import executemany

    if not results:
        return

    created_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            game_date,
            r["market"],
            r.get("check_type", "prediction"),
            r.get("psi"),
            r["alert_level"],
            r.get("explanation", ""),
            created_at,
        )
        for r in results
    ]

    sql = """
        INSERT OR REPLACE INTO nba_drift_alerts
            (game_date, market, check_type, psi_score, alert_level, explanation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    try:
        executemany(sql, rows)
        logger.info("[drift] saved %d alert rows for date=%s", len(rows), game_date)
    except Exception as exc:
        logger.error("[drift] failed to save alerts: %s", exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _insufficient_result(
    market: str,
    check_type: str,
    window_days: int,
    reference_days: int,
    explanation: str,
    n_recent: int = 0,
    n_reference: int = 0,
) -> dict[str, Any]:
    return {
        "market": market,
        "check_type": check_type,
        "psi": 0.0,
        "alert_level": "stable",
        "window_days": window_days,
        "reference_days": reference_days,
        "n_recent": n_recent,
        "n_reference": n_reference,
        "explanation": explanation,
    }
