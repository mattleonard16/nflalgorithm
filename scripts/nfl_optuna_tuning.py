"""Optuna hyperparameter tuning script for NFL stat projection models.

Runs a 50-trial study per market using 4-fold TimeSeriesSplit CV MAE as the
objective. Best parameters are written to models/weekly/best_params_{market}.json
and are picked up automatically by _build_nfl_model() on the next training run.

Usage:
    # Tune all markets (default)
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/nfl_optuna_tuning.py

    # Tune a single market with a custom trial count
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/nfl_optuna_tuning.py \
        --market rushing_yards --n-trials 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from models.position_specific.weekly import (
    MARKET_CONFIGS,
    _MARKET_STATS,
    _engineer_rolling_features,
    _load_training_data,
    get_nfl_feature_cols,
)
from utils.db import read_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models" / "weekly"
VALID_MARKETS = set(MARKET_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_market_data(market: str) -> pd.DataFrame:
    """Load and feature-engineer player stats for the given market."""
    config_dict = MARKET_CONFIGS[market]
    filter_col = config_dict["filter_col"]
    target_col = config_dict["target_col"]

    # Load all available season-week pairs
    try:
        available = read_dataframe(
            "SELECT DISTINCT season, week FROM player_stats_enhanced ORDER BY season, week"
        )
    except Exception as e:
        log.error("Failed to load available seasons: %s", e)
        return pd.DataFrame()

    if available.empty:
        return pd.DataFrame()

    tuples = list(available.itertuples(index=False, name=None))
    df = _load_training_data(tuples)

    if df.empty:
        return df

    df = _engineer_rolling_features(df, market)

    if filter_col in df.columns:
        df = df[df[filter_col] > config_dict["min_value"]].copy()

    feature_cols = get_nfl_feature_cols(market)
    df = df.dropna(subset=feature_cols + [target_col])

    return df


# ---------------------------------------------------------------------------
# Model builder for a trial
# ---------------------------------------------------------------------------


def _build_trial_model(trial: object, market: str) -> StackingRegressor:
    """Construct a StackingRegressor using Optuna trial hyperparameters."""
    gbr_n = trial.suggest_int("gbr_n", 200, 600)
    gbr_lr = trial.suggest_float("gbr_lr", 0.01, 0.1, log=True)
    gbr_depth = trial.suggest_int("gbr_depth", 3, 7)
    gbr_sub = trial.suggest_float("gbr_sub", 0.6, 1.0)
    xgb_n = trial.suggest_int("xgb_n", 200, 600)
    xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.1, log=True)
    xgb_depth = trial.suggest_int("xgb_depth", 3, 7)

    gbr = GradientBoostingRegressor(
        n_estimators=gbr_n,
        learning_rate=gbr_lr,
        max_depth=gbr_depth,
        subsample=gbr_sub,
        min_samples_leaf=5,
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=gbr_depth + 2,
        min_samples_leaf=5,
        max_features=0.7,
        n_jobs=-1,
        random_state=42,
    )

    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_n,
            max_depth=xgb_depth,
            learning_rate=xgb_lr,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        estimators = [("gbr", gbr), ("rf", rf), ("xgb", xgb_model)]
    except ImportError:
        log.warning("xgboost not installed; using GBR + RF only for this trial.")
        estimators = [("gbr", gbr), ("rf", rf)]

    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=4,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _make_objective(X: np.ndarray, y: np.ndarray, market: str):
    """Return an Optuna objective function closed over the data arrays."""

    def objective(trial) -> float:
        model = _build_trial_model(trial, market)
        tscv = TimeSeriesSplit(n_splits=4)
        fold_maes: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            fold_maes.append(float(np.mean(np.abs(preds - y[val_idx]))))
        return float(np.mean(fold_maes))

    return objective


# ---------------------------------------------------------------------------
# Per-market tuning
# ---------------------------------------------------------------------------


def tune_market(market: str, n_trials: int) -> None:
    """Run an Optuna study for one NFL market and persist the best params."""
    try:
        import optuna
    except ImportError:
        log.error("optuna is not installed. Run: uv add optuna")
        return

    log.info("[%s] Loading and engineering features ...", market)
    df = _load_market_data(market)
    if df.empty:
        log.error(
            "[%s] No data found in player_stats_enhanced. Run ingest first.", market
        )
        return

    feature_cols = get_nfl_feature_cols(market)
    target_col = MARKET_CONFIGS[market]["target_col"]

    df = df.dropna(subset=feature_cols + [target_col])
    if len(df) < 100:
        log.error(
            "[%s] Insufficient data after dropping NaN rows (%d rows).", market, len(df)
        )
        return

    log.info("[%s] Tuning on %d rows ...", market, len(df))
    X = df[feature_cols].values
    y = df[target_col].values

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        _make_objective(X, y, market),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_mae = study.best_value

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    params_path = MODEL_DIR / f"best_params_{market}.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    log.info(
        "[%s] Best MAE: %.4f  |  params: %s  |  saved to %s",
        market,
        best_mae,
        best_params,
        params_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for NFL stat projection models"
    )
    parser.add_argument(
        "--market",
        default="all",
        help=(
            f"Market to tune: one of {sorted(VALID_MARKETS)} or 'all' (default: all)"
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per market (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.market == "all":
        markets = sorted(VALID_MARKETS)
    elif args.market in VALID_MARKETS:
        markets = [args.market]
    else:
        valid = sorted(VALID_MARKETS)
        raise ValueError(
            f"Unknown market '{args.market}'. Choose from {valid} or 'all'."
        )

    log.info(
        "Starting NFL Optuna tuning: markets=%s, n_trials=%d", markets, args.n_trials
    )
    for market in markets:
        tune_market(market, args.n_trials)
    log.info("Tuning complete. Re-train models to apply new params: make nfl-train")


if __name__ == "__main__":
    main()
