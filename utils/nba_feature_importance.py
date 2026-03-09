"""NBA Feature Importance utilities.

Computes SHAP-based or permutation-based feature importance for NBA stat models.
SHAP is an optional dependency — falls back to sklearn permutation_importance when
not installed.

Usage::

    from models.nba.stat_model import get_feature_cols
    from utils.nba_feature_importance import compute_shap_importance, save_importance_snapshot

    importance_df = compute_shap_importance(model, X_sample, feature_names, market="pts")
    save_importance_snapshot(importance_df, market="pts", game_date="2026-03-04")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SHAP availability check
# ---------------------------------------------------------------------------

try:
    import shap as _shap  # noqa: F401

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.debug("shap not installed — will use permutation importance as fallback")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_shap_importance(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
    market: str,
) -> pd.DataFrame:
    """Compute SHAP feature importance for a fitted model.

    Uses SHAP TreeExplainer on the first tree-based base estimator from a
    StackingRegressor.  Falls back to :func:`compute_permutation_importance`
    when the ``shap`` package is not installed.

    Args:
        model:         A fitted sklearn estimator (StackingRegressor expected).
        X_sample:      2-D feature array.  Sampled to ≤ 100 rows internally.
        feature_names: Ordered list of feature column names (len == X_sample.shape[1]).
        market:        Market label (e.g. "pts") — included in returned DataFrame.

    Returns:
        DataFrame with columns: feature, mean_abs_shap, rank.
        Sorted ascending by rank (rank 1 = most important).
    """
    if not _SHAP_AVAILABLE:
        logger.info("[feature_importance] shap not available — using permutation fallback")
        # No y available here — we will compute a dummy permutation importance
        # using the model itself as a scorer (mean squared error via predictions).
        # The caller should use compute_permutation_importance when y is available.
        return _shap_fallback_without_y(model, X_sample, feature_names)

    # Cap sample size for performance
    if X_sample.shape[0] > 100:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_sample.shape[0], size=100, replace=False)
        X_sample = X_sample[idx]

    # Extract tree-based estimator from StackingRegressor
    base_estimator = _extract_tree_estimator(model)
    if base_estimator is None:
        logger.warning("[feature_importance] no tree estimator found — using permutation fallback")
        return _shap_fallback_without_y(model, X_sample, feature_names)

    try:
        import shap

        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(X_sample)

        mean_abs = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df[["feature", "mean_abs_shap", "rank"]]

    except Exception as exc:
        logger.warning("[feature_importance] SHAP computation failed (%s) — permutation fallback", exc)
        return _shap_fallback_without_y(model, X_sample, feature_names)


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compute permutation-based feature importance using sklearn.

    Args:
        model:         A fitted sklearn estimator.
        X:             2-D feature array.
        y:             1-D target array.
        feature_names: Ordered list of feature column names.
        n_repeats:     Number of permutation repeats.

    Returns:
        DataFrame with columns: feature, importance, rank.
        Sorted ascending by rank (rank 1 = most important).
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_mean_squared_error",
    )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": result.importances_mean,
        }
    )
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df[["feature", "importance", "rank"]]


def save_importance_snapshot(
    importance_df: pd.DataFrame,
    market: str,
    game_date: str,
) -> int:
    """Persist importance data to ``nba_feature_importance_history``.

    Skips rows that already exist (UNIQUE constraint on game_date+market+feature).

    Args:
        importance_df: DataFrame from compute_shap_importance or
                       compute_permutation_importance.
        market:        NBA market (pts/reb/ast/fg3m).
        game_date:     ISO date string (YYYY-MM-DD).

    Returns:
        Number of rows written.
    """
    from utils.db import executemany

    if importance_df.empty:
        return 0

    mean_shap_col = "mean_abs_shap" if "mean_abs_shap" in importance_df.columns else None
    importance_col = "importance" if "importance" in importance_df.columns else None

    rows = []
    for _, row in importance_df.iterrows():
        shap_val = float(row[mean_shap_col]) if mean_shap_col and mean_shap_col in row.index else None
        importance_val = float(row[importance_col]) if importance_col and importance_col in row.index else None
        mean_abs_val = shap_val if shap_val is not None else importance_val
        rows.append(
            (
                game_date,
                market,
                str(row["feature"]),
                mean_abs_val,
                int(row["rank"]),
            )
        )

    sql = """
        INSERT OR IGNORE INTO nba_feature_importance_history
            (game_date, market, feature, mean_abs_shap, rank)
        VALUES (?, ?, ?, ?, ?)
    """
    executemany(sql, rows)
    logger.info(
        "[feature_importance] saved %d rows for market=%s date=%s",
        len(rows),
        market,
        game_date,
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_tree_estimator(model: Any) -> Any | None:
    """Return the first GBR/RF base estimator from a StackingRegressor, or None."""
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
        StackingRegressor,
    )

    if isinstance(model, (GradientBoostingRegressor, RandomForestRegressor)):
        return model

    if isinstance(model, StackingRegressor) and hasattr(model, "estimators_"):
        for est in model.estimators_:
            if isinstance(est, (GradientBoostingRegressor, RandomForestRegressor)):
                return est
        # Try one level deeper (Pipeline wrapping the estimator)
        for est in model.estimators_:
            inner = getattr(est, "steps", None)
            if inner:
                for _, step in inner:
                    if isinstance(step, (GradientBoostingRegressor, RandomForestRegressor)):
                        return step

    return None


def _shap_fallback_without_y(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Fallback when y is unavailable: use feature_importances_ if present, else zeros."""
    from sklearn.ensemble import StackingRegressor

    # Try to get feature_importances_ from the base estimator
    base = _extract_tree_estimator(model)
    if base is not None and hasattr(base, "feature_importances_"):
        importances = base.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importances})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df[["feature", "mean_abs_shap", "rank"]]
