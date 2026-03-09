"""NBA player minutes prediction model.

Trains a StackingRegressor to predict minutes played per game.
Enables minutes × rate decomposition for improved stat projections.

Usage:
    model = MinutesModel()
    model.train("nfl_data.db")
    model.save()

    model2 = MinutesModel()
    model2.load()
    predictions = model2.predict("nfl_data.db", target_date="2025-03-15")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from utils.db import read_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = MODEL_DIR / "minutes_model.joblib"
BEST_PARAMS_PATH = MODEL_DIR / "best_params_minutes.json"

MIN_MINUTES_FILTER = 5
MIN_GAMES_FOR_PREDICTION = 5

FEATURE_COLS = [
    "min_last5_avg",
    "min_last10_avg",
    "min_last20_avg",
    "days_rest",
    "b2b",
    "home_game",
    "opp_pace_normalized",
    "starter_flag",
    "games_last_7_days",
]


def _load_best_params() -> dict[str, Any]:
    """Load Optuna-tuned params if available."""
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            return json.load(f)
    return {}


def _build_model() -> StackingRegressor:
    """Build a StackingRegressor ensemble for minutes prediction."""
    params = _load_best_params()
    depth = params.get("gbr_depth", 4)

    gbr = GradientBoostingRegressor(
        n_estimators=params.get("gbr_n", 400),
        learning_rate=params.get("gbr_lr", 0.04),
        max_depth=depth,
        subsample=params.get("gbr_sub", 0.8),
        min_samples_leaf=5,
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=params.get("rf_n", 300),
        max_depth=params.get("rf_max_depth", depth + 2),
        min_samples_leaf=5,
        max_features=params.get("rf_max_features", 0.7),
        n_jobs=-1,
        random_state=42,
    )

    estimators: list[tuple[str, Any]] = [("gbr", gbr), ("rf", rf)]

    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=params.get("xgb_n", 400),
            max_depth=params.get("xgb_depth", depth),
            learning_rate=params.get("xgb_lr", 0.04),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        estimators = [*estimators, ("xgb", xgb_model)]
    except (ImportError, Exception):
        # ImportError: xgboost not installed
        # XGBoostError / OSError: native library (libxgboost.dylib) missing
        pass

    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(
            n_estimators=params.get("lgb_n", 400),
            max_depth=params.get("lgb_depth", depth),
            learning_rate=params.get("lgb_lr", 0.04),
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

    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=4,
        n_jobs=-1,
    )


def _load_game_logs(db_path: str, seasons: list[int] | None = None) -> pd.DataFrame:
    """Load player game logs filtered to qualifying minute entries."""
    base_sql = (
        "SELECT player_id, player_name, team_abbreviation, season, "
        "game_id, game_date, matchup, min "
        "FROM nba_player_game_logs "
        "WHERE min >= ? "
    )
    params: list[Any] = [float(MIN_MINUTES_FILTER)]

    if seasons:
        placeholders = ", ".join("?" for _ in seasons)
        base_sql += f"AND season IN ({placeholders}) "
        params.extend(seasons)

    base_sql += "ORDER BY player_id, game_date"

    import os
    os.environ.setdefault("DB_BACKEND", "sqlite")
    os.environ.setdefault("SQLITE_DB_PATH", db_path)

    return read_dataframe(base_sql, params)


def _lookup_opp_pace(df: pd.DataFrame) -> pd.DataFrame:
    """Merge opponent pace (normalized) from nba_team_defensive_stats."""
    df = df.copy()

    def_stats = read_dataframe(
        "SELECT team_abbreviation, season, opp_pace "
        "FROM nba_team_defensive_stats "
        "ORDER BY team_abbreviation, season, as_of_date DESC"
    )

    if def_stats.empty:
        df["opp_pace_normalized"] = 0.0
        return df

    def_stats = def_stats.drop_duplicates(subset=["team_abbreviation", "season"], keep="first")

    pace_vals = def_stats["opp_pace"].dropna()
    if len(pace_vals) > 1:
        pace_mean = float(pace_vals.mean())
        pace_std = float(pace_vals.std())
        if pace_std > 0:
            def_stats = def_stats.copy()
            def_stats["opp_pace_normalized"] = (def_stats["opp_pace"] - pace_mean) / pace_std
        else:
            def_stats = def_stats.copy()
            def_stats["opp_pace_normalized"] = 0.0
    else:
        def_stats = def_stats.copy()
        def_stats["opp_pace_normalized"] = 0.0

    if "opponent" not in df.columns:
        df["opponent"] = df["matchup"].str.strip().str[-3:]

    merge_df = def_stats[["team_abbreviation", "season", "opp_pace_normalized"]].rename(
        columns={"team_abbreviation": "opponent"}
    )

    if "season" in df.columns:
        df = df.merge(merge_df, on=["opponent", "season"], how="left")
    else:
        merge_df_no_season = (
            merge_df.drop(columns=["season"])
            .drop_duplicates(subset=["opponent"])
        )
        df = df.merge(merge_df_no_season, on=["opponent"], how="left")

    df["opp_pace_normalized"] = df["opp_pace_normalized"].fillna(0.0)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and contextual features for minutes prediction.

    All rolling features use shift(1) to prevent lookahead bias.
    Input DataFrame must be sorted by (player_id, game_date) ascending.

    Features added:
        min_last5_avg    : EWMA(span=5) of minutes, shift(1)
        min_last10_avg   : EWMA(span=10) of minutes, shift(1)
        min_last20_avg   : EWMA(span=20) of minutes, shift(1)
        days_rest        : days since last game, clipped [1, 7], fillna=3
        b2b              : 1 if days_rest <= 1 else 0
        home_game        : 1 if "vs." in matchup else 0
        opp_pace_normalized : z-score normalized opponent pace
        starter_flag     : 1 if rolling min_share (last 5 games) > 0.30
        games_last_7_days: count of games per player in past 7 calendar days
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # EWMA rolling minutes — shift(1) ensures no lookahead
    for span in [5, 10, 20]:
        col = f"min_last{span}_avg"
        df[col] = df.groupby("player_id")["min"].transform(
            lambda s, sp=span: s.shift(1).ewm(span=sp, min_periods=1).mean()
        )

    # Home game flag
    df["home_game"] = df["matchup"].str.contains("vs.", na=False).astype(int)

    # Opponent abbreviation (last 3 chars of matchup)
    df["opponent"] = df["matchup"].str.strip().str[-3:]

    # Player rest days
    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["days_rest"] = (
        (df["game_date"] - df["prev_game_date"]).dt.days
        .fillna(3)
        .clip(lower=1, upper=7)
    )
    df["b2b"] = (df["days_rest"] <= 1).astype(int)

    # Starter flag: rolling min_share (last 5 games, shift(1)) > 0.30
    # min_share = player_min / team_min; approximate team_min as ~240 per game
    df["approx_min_share"] = df["min"] / 240.0
    df["min_share_l5"] = df.groupby("player_id")["approx_min_share"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["starter_flag"] = (df["min_share_l5"] > 0.30).astype(int)

    # Games in last 7 calendar days (shift logic: only count past games)
    # Vectorized: for each row, count rows from same player in (game_date-7d, game_date)
    # The shift(1) semantic is implicit since we use strict inequality game_date < current
    def _count_games_last_7(group: pd.Series) -> pd.Series:
        """Count prior games within 7-day window for each date in group."""
        dates_arr = group.values  # already sorted
        counts = np.zeros(len(dates_arr), dtype=int)
        for i in range(1, len(dates_arr)):
            window_start = dates_arr[i] - np.timedelta64(7, "D")
            counts[i] = int(np.sum((dates_arr[:i] >= window_start)))
        return pd.Series(counts, index=group.index, name="games_last_7_days")

    df["games_last_7_days"] = (
        df.groupby("player_id")["game_date"]
        .transform(_count_games_last_7)
    )

    return df


class MinutesModel:
    """Dedicated model for predicting NBA player minutes per game.

    Enables the minutes × rate decomposition used by the rate-based
    stat projection pipeline (Task #4).
    """

    def __init__(self) -> None:
        self._model: StackingRegressor | None = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = str(DEFAULT_MODEL_PATH)) -> None:
        """Serialize trained model to disk."""
        if self._model is None:
            raise RuntimeError("No trained model to save. Run train() first.")
        joblib.dump(self._model, path)
        log.info("[minutes] Model saved to %s", path)

    def load(self, path: str = str(DEFAULT_MODEL_PATH)) -> None:
        """Load a previously saved model from disk."""
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Minutes model not found at {path}")
        self._model = joblib.load(model_path)
        log.info("[minutes] Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, db_path: str, seasons: list[int] | None = None) -> dict[str, Any]:
        """Train the minutes model and store it in self._model.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database (or ignored if DB_BACKEND=mysql).
        seasons : list[int] or None
            Filter to specific seasons. None uses all available data.

        Returns
        -------
        dict
            {"cv_mae": float, "n_samples": int}
        """
        log.info("[minutes] Loading game logs …")
        df = _load_game_logs(db_path, seasons=seasons)

        if df.empty:
            log.error("[minutes] No data found. Run NBA ingest first.")
            return {"cv_mae": float("nan"), "n_samples": 0}

        log.info("[minutes] Engineering features on %d rows …", len(df))
        df = _engineer_features(df)
        df = _lookup_opp_pace(df)

        df = df.dropna(subset=FEATURE_COLS + ["min"])
        n_samples = len(df)
        log.info("[minutes] Training on %d rows after dropping NaN rows", n_samples)

        if n_samples < 10:
            log.error("[minutes] Insufficient training data (%d rows).", n_samples)
            return {"cv_mae": float("nan"), "n_samples": n_samples}

        X = df[FEATURE_COLS].values
        y = df["min"].values

        tscv = TimeSeriesSplit(n_splits=4)
        fold_maes: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            fold_model = _build_model()
            fold_model.fit(X[train_idx], y[train_idx])
            fold_preds = fold_model.predict(X[val_idx])
            fold_mae = float(np.mean(np.abs(fold_preds - y[val_idx])))
            fold_maes.append(fold_mae)
            log.info("[minutes] Fold %d MAE: %.3f", fold_idx + 1, fold_mae)

        cv_mae = float(np.mean(fold_maes))
        log.info("[minutes] CV MAE: %.3f +/- %.3f", cv_mae, float(np.std(fold_maes)))

        # Final model on all data
        self._model = _build_model()
        self._model.fit(X, y)

        train_mae = float(np.mean(np.abs(self._model.predict(X) - y)))
        log.info("[minutes] Training MAE: %.3f (full dataset)", train_mae)

        return {"cv_mae": cv_mae, "n_samples": n_samples}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        db_path: str,
        target_date: str,
        players: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Predict minutes for players scheduled on target_date.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database.
        target_date : str
            Date string in YYYY-MM-DD format.
        players : list[dict] or None
            Optional explicit player list. Each dict must contain at minimum:
            {"player_id": ..., "team": ...}. If None, all players from teams
            with historical logs before target_date are used.

        Returns
        -------
        list[dict]
            Each entry: {
                "player_id": str,
                "player_name": str,
                "team": str,
                "predicted_minutes": float,
                "minutes_sigma": float,
            }
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() or train() first.")

        import os
        os.environ.setdefault("DB_BACKEND", "sqlite")
        os.environ.setdefault("SQLITE_DB_PATH", db_path)

        # Load all historical logs before target_date
        df_all = read_dataframe(
            "SELECT player_id, player_name, team_abbreviation, season, "
            "game_id, game_date, matchup, min "
            "FROM nba_player_game_logs "
            "WHERE min >= ? AND game_date < ? "
            "ORDER BY player_id, game_date",
            [float(MIN_MINUTES_FILTER), target_date],
        )

        if df_all.empty:
            log.warning("[minutes] No historical logs found before %s.", target_date)
            return []

        df_all = _engineer_features(df_all)
        df_all = _lookup_opp_pace(df_all)

        # Latest row per player = most recent rolling features
        latest = (
            df_all.sort_values("game_date")
            .groupby("player_id")
            .last()
            .reset_index()
        )

        # Filter by provided player list if given
        if players is not None:
            requested_ids = {str(p["player_id"]) for p in players}
            latest = latest[latest["player_id"].astype(str).isin(requested_ids)].copy()

        # Require minimum game history
        game_counts = df_all.groupby("player_id").size().rename("game_count")
        latest = latest.join(game_counts, on="player_id")
        latest = latest[latest["game_count"] >= MIN_GAMES_FOR_PREDICTION].copy()

        if latest.empty:
            log.info("[minutes] No qualifying players found for %s.", target_date)
            return []

        X = latest[FEATURE_COLS].fillna(0).values
        raw_preds = self._model.predict(X)
        # Clip predictions to realistic NBA range
        preds = np.clip(raw_preds, 0.0, 48.0)

        # Per-player minutes sigma from historical variance (EWMA std)
        player_min_history = df_all.groupby("player_id")["min"].apply(list)

        results: list[dict[str, Any]] = []
        for (_, row), pred in zip(latest.iterrows(), preds):
            pid = str(row["player_id"])
            history = player_min_history.get(row["player_id"], [])
            sigma = _compute_minutes_sigma(history)

            results.append({
                "player_id": pid,
                "player_name": str(row["player_name"]),
                "team": str(row["team_abbreviation"]),
                "predicted_minutes": float(round(float(pred), 2)),
                "minutes_sigma": float(round(sigma, 4)),
            })

        log.info(
            "[minutes] Generated %d minute predictions for %s",
            len(results),
            target_date,
        )
        return results


# ---------------------------------------------------------------------------
# Sigma helper
# ---------------------------------------------------------------------------


def _compute_minutes_sigma(game_minutes: list[float], decay: float = 0.65) -> float:
    """Compute EWMA-weighted standard deviation for a player's minutes history.

    Uses the same approach as utils/nba_sigma.py but specialized for minutes.
    Floor of 2.0 minutes regardless of history.
    """
    arr = np.asarray(game_minutes, dtype=float)
    arr = arr[~np.isnan(arr)]

    floor = 2.0
    default = 4.0  # fallback when fewer than 8 game samples

    if len(arr) < 8:
        return default

    n = len(arr)
    raw_weights = np.array([decay ** i for i in range(n - 1, -1, -1)])
    weights = raw_weights / raw_weights.sum()

    weighted_mean = float(np.dot(weights, arr))
    squared_diffs = (arr - weighted_mean) ** 2
    weighted_var = float(np.dot(weights, squared_diffs))

    sum_w2 = float(np.dot(weights, weights))
    correction = 1.0 / (1.0 - sum_w2) if sum_w2 < 1.0 else 1.0
    sigma = float(np.sqrt(weighted_var * correction))

    return max(sigma, floor)
