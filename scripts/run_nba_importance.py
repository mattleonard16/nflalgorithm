"""Compute NBA feature importance for all trained markets.

Loads each trained model from models/nba/{market}_model.joblib, computes
feature importance (SHAP if available, else feature_importances_ fallback),
prints ranked features, and saves to nba_feature_importance_history.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/run_nba_importance.py
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/run_nba_importance.py --market pts
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.nba.stat_model import get_feature_cols
from utils.db import read_dataframe
from utils.nba_feature_importance import (
    compute_shap_importance,
    save_importance_snapshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models" / "nba"
ALL_MARKETS = ["pts", "reb", "ast", "fg3m"]


def _load_sample_features(
    market: str, max_rows: int = 200, feature_cols_override: list[str] | None = None,
) -> np.ndarray | None:
    """Load a sample of feature data from nba_player_game_logs for importance computation."""
    feature_cols = feature_cols_override or get_feature_cols(market)

    # Build rolling averages from recent game logs
    df = read_dataframe(
        "SELECT * FROM nba_player_game_logs ORDER BY game_date DESC LIMIT ?",
        (max_rows * 5,),
    )
    if df.empty:
        return None

    # Engineer features the same way stat_model does
    from models.nba.stat_model import _engineer_features

    df = _engineer_features(df, market)

    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        logger.warning("Missing feature columns: %s — filling with 0", missing)
        for col in missing:
            df[col] = 0.0

    X = df[feature_cols].dropna()
    if X.empty:
        return None

    if len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=42)

    return X.values


def _get_model_feature_names(model, market: str) -> list[str]:
    """Get the feature names the model was actually trained on.

    Falls back to get_feature_cols() truncated to n_features_in_ if the model
    doesn't store feature_names_in_.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    n = getattr(model, "n_features_in_", None)
    all_cols = get_feature_cols(market)
    if n is not None and n != len(all_cols):
        logger.info("Model trained on %d features, get_feature_cols returns %d — truncating", n, len(all_cols))
        return all_cols[:n]
    return all_cols


def run_importance(market: str, game_date: str) -> None:
    """Compute and display feature importance for a single market."""
    model_path = MODEL_DIR / f"{market}_model.joblib"
    if not model_path.exists():
        logger.warning("No trained model at %s — skipping %s", model_path, market)
        return

    logger.info("Loading model: %s", model_path)
    model = joblib.load(model_path)

    feature_names = _get_model_feature_names(model, market)
    X_sample = _load_sample_features(market, feature_cols_override=feature_names)

    if X_sample is None:
        logger.warning("No feature data available for %s — skipping", market)
        return

    logger.info("Computing feature importance for %s (%d samples, %d features)",
                market, X_sample.shape[0], X_sample.shape[1])

    importance_df = compute_shap_importance(model, X_sample, feature_names, market)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Feature Importance: {market.upper()}")
    print(f"{'='*60}")
    shap_col = "mean_abs_shap" if "mean_abs_shap" in importance_df.columns else "importance"
    for _, row in importance_df.iterrows():
        rank = int(row["rank"])
        feature = row["feature"]
        score = float(row[shap_col])
        bar = "#" * max(1, int(score / importance_df[shap_col].max() * 30))
        print(f"  {rank:>2}. {feature:<30s} {score:.6f}  {bar}")

    # Save to DB
    n_saved = save_importance_snapshot(importance_df, market, game_date)
    logger.info("Saved %d importance rows for %s (date=%s)", n_saved, market, game_date)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute NBA feature importance for trained models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--market",
        default=None,
        help="Single market to analyze (default: all trained markets)",
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Date label for the snapshot (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    markets = [args.market] if args.market else ALL_MARKETS
    game_date = args.date

    for market in markets:
        try:
            run_importance(market, game_date)
        except Exception as exc:
            logger.error("Feature importance failed for %s: %s", market, exc)

    print(f"\nDone. Results saved to nba_feature_importance_history (date={game_date})")


if __name__ == "__main__":
    main()
