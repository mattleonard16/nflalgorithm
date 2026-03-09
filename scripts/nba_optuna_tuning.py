"""Optuna hyperparameter tuning script for NBA stat projection models.

Runs a 50-trial study per market using 4-fold TimeSeriesSplit CV MAE as the
objective.  Best parameters are written to models/nba/best_params_{market}.json
and are picked up automatically by _build_model() on the next training run.

Usage:
    # Tune all markets (default)
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/nba_optuna_tuning.py

    # Tune a single market with a custom trial count
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/nba_optuna_tuning.py \
        --market pts --n-trials 30
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

from models.nba.stat_model import (
    VALID_MARKETS,
    _MARKET_STATS,
    _engineer_features,
    _lookup_opponent_defense,
    get_feature_cols,
)
from models.nba.minutes_model import (
    FEATURE_COLS as MINUTES_FEATURE_COLS,
    _build_model as _build_minutes_model,
    _engineer_features as _engineer_minutes_features,
    _load_game_logs as _load_minutes_game_logs,
    _lookup_opp_pace,
    BEST_PARAMS_PATH as MINUTES_BEST_PARAMS_PATH,
)
from utils.db import read_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models" / "nba"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_game_logs(market: str) -> pd.DataFrame:
    """Load and feature-engineer game logs for the given market.

    Defense lookup is applied per game_date using only stats available on that
    date (as_of_date <= game_date) to prevent lookahead bias in CV folds.
    """
    stats_needed = _MARKET_STATS[market]
    base_cols = (
        "player_id, player_name, team_abbreviation, season, "
        "game_id, game_date, matchup, min, fga"
    )
    extra_cols = ", ".join(c for c in stats_needed if c not in {"min", "fga"})
    select_cols = f"{base_cols}, {extra_cols}" if extra_cols else base_cols

    df = read_dataframe(
        f"SELECT {select_cols} FROM nba_player_game_logs "
        f"WHERE {market} IS NOT NULL ORDER BY player_id, game_date"
    )
    if df.empty:
        return df

    df = _engineer_features(df, market)

    # Apply defense lookup per unique game_date with as_of_date guard to
    # prevent future defensive ratings leaking into historical training folds.
    unique_dates = sorted(df["game_date"].astype(str).unique())
    date_slices: list[pd.DataFrame] = []
    for udate in unique_dates:
        date_slice = df[df["game_date"].astype(str) == udate]
        date_slices.append(_lookup_opponent_defense(date_slice, market, as_of_date=udate))
    df = pd.concat(date_slices, ignore_index=True)
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

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
    rf_n = trial.suggest_int("rf_n", 100, 500)
    rf_max_features = trial.suggest_float("rf_max_features", 0.3, 1.0)
    rf_max_depth = trial.suggest_int("rf_max_depth", 4, 12)

    gbr = GradientBoostingRegressor(
        n_estimators=gbr_n,
        learning_rate=gbr_lr,
        max_depth=gbr_depth,
        subsample=gbr_sub,
        min_samples_leaf=5,
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=rf_n,
        max_depth=rf_max_depth,
        min_samples_leaf=5,
        max_features=rf_max_features,
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

    try:
        import lightgbm as lgb

        lgb_n = trial.suggest_int("lgb_n", 200, 600)
        lgb_lr = trial.suggest_float("lgb_lr", 0.01, 0.1, log=True)
        lgb_depth = trial.suggest_int("lgb_depth", 3, 7)
        lgb_model = lgb.LGBMRegressor(
            n_estimators=lgb_n,
            max_depth=lgb_depth,
            learning_rate=lgb_lr,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        estimators = [*estimators, ("lgb", lgb_model)]
    except (ImportError, OSError):
        log.warning("lightgbm not available; skipping LGB estimator for this trial.")

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
    """Run an Optuna study for one market and persist the best params."""
    try:
        import optuna
    except ImportError:
        log.error("optuna is not installed. Run: uv add optuna")
        return

    log.info("[%s] Loading and engineering features …", market)
    df = _load_game_logs(market)
    if df.empty:
        log.error("[%s] No data found in nba_player_game_logs. Run ingest first.", market)
        return

    feature_cols = get_feature_cols(market)
    df = df.dropna(subset=feature_cols + [market])
    if len(df) < 100:
        log.error("[%s] Insufficient data after dropping NaN rows (%d rows).", market, len(df))
        return

    log.info("[%s] Tuning on %d rows …", market, len(df))
    X = df[feature_cols].values
    y = df[market].values

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        _make_objective(X, y, market),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_mae = study.best_value

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
# Minutes model tuning
# ---------------------------------------------------------------------------


def tune_minutes(n_trials: int) -> None:
    """Run an Optuna study for the MinutesModel and persist the best params."""
    try:
        import optuna
    except ImportError:
        log.error("optuna is not installed. Run: uv add optuna")
        return

    import os
    db_path = os.environ.get("SQLITE_DB_PATH", "nfl_data.db")

    log.info("[minutes] Loading and engineering features …")
    df = _load_minutes_game_logs(db_path)
    if df.empty:
        log.error("[minutes] No data found. Run NBA ingest first.")
        return

    df = _engineer_minutes_features(df)
    df = _lookup_opp_pace(df)
    df = df.dropna(subset=MINUTES_FEATURE_COLS + ["min"])
    if len(df) < 100:
        log.error("[minutes] Insufficient data after dropping NaN rows (%d rows).", len(df))
        return

    log.info("[minutes] Tuning on %d rows …", len(df))
    X = df[MINUTES_FEATURE_COLS].values
    y = df["min"].values

    def objective(trial) -> float:
        # Mirror the same hyperparameter space as stat model trials
        gbr_n = trial.suggest_int("gbr_n", 200, 600)
        gbr_lr = trial.suggest_float("gbr_lr", 0.01, 0.1, log=True)
        gbr_depth = trial.suggest_int("gbr_depth", 3, 7)
        gbr_sub = trial.suggest_float("gbr_sub", 0.6, 1.0)
        rf_n = trial.suggest_int("rf_n", 100, 500)
        rf_max_features = trial.suggest_float("rf_max_features", 0.3, 1.0)
        rf_max_depth = trial.suggest_int("rf_max_depth", 4, 12)
        xgb_n = trial.suggest_int("xgb_n", 200, 600)
        xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.1, log=True)
        xgb_depth = trial.suggest_int("xgb_depth", 3, 7)

        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
        from sklearn.linear_model import Ridge

        gbr = GradientBoostingRegressor(
            n_estimators=gbr_n,
            learning_rate=gbr_lr,
            max_depth=gbr_depth,
            subsample=gbr_sub,
            min_samples_leaf=5,
            random_state=42,
        )
        rf = RandomForestRegressor(
            n_estimators=rf_n,
            max_depth=rf_max_depth,
            min_samples_leaf=5,
            max_features=rf_max_features,
            n_jobs=-1,
            random_state=42,
        )
        estimators = [("gbr", gbr), ("rf", rf)]

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
            estimators = [*estimators, ("xgb", xgb_model)]
        except ImportError:
            pass

        try:
            import lightgbm as lgb
            lgb_n = trial.suggest_int("lgb_n", 200, 600)
            lgb_lr = trial.suggest_float("lgb_lr", 0.01, 0.1, log=True)
            lgb_depth = trial.suggest_int("lgb_depth", 3, 7)
            lgb_model = lgb.LGBMRegressor(
                n_estimators=lgb_n,
                max_depth=lgb_depth,
                learning_rate=lgb_lr,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            estimators = [*estimators, ("lgb", lgb_model)]
        except (ImportError, OSError):
            pass

        model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=4,
            n_jobs=-1,
        )

        tscv = TimeSeriesSplit(n_splits=4)
        fold_maes: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            fold_maes.append(float(np.mean(np.abs(preds - y[val_idx]))))
        return float(np.mean(fold_maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_mae = study.best_value

    with open(MINUTES_BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)

    log.info(
        "[minutes] Best MAE: %.4f  |  params: %s  |  saved to %s",
        best_mae,
        best_params,
        MINUTES_BEST_PARAMS_PATH,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_TUNABLE_MARKETS = sorted(VALID_MARKETS) + ["minutes"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for NBA stat projection models"
    )
    parser.add_argument(
        "--market",
        default="all",
        help=(
            f"Market to tune: one of {ALL_TUNABLE_MARKETS} or 'all' (default: all)"
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
        include_minutes = True
    elif args.market == "minutes":
        markets = []
        include_minutes = True
    elif args.market in VALID_MARKETS:
        markets = [args.market]
        include_minutes = False
    else:
        raise ValueError(
            f"Unknown market '{args.market}'. Choose from {ALL_TUNABLE_MARKETS} or 'all'."
        )

    log.info("Starting NBA Optuna tuning: markets=%s, include_minutes=%s, n_trials=%d", markets, include_minutes, args.n_trials)
    if include_minutes:
        tune_minutes(args.n_trials)
    for market in markets:
        tune_market(market, args.n_trials)
    log.info("Tuning complete. Re-train models to apply new params: make nba-train")


if __name__ == "__main__":
    main()
