"""NBA multi-market stat projection model.

Trains a GradientBoostingRegressor for each supported market (pts, reb, ast, fg3m)
on historical player game logs and writes per-player per-game projections to
nba_projections.

Usage:
    # Train a single market
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python models/nba/stat_model.py --train --market pts

    # Train all markets
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python models/nba/stat_model.py --train --market all

    # Predict for a single market
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python models/nba/stat_model.py --predict --market pts

    # Predict for all markets
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python models/nba/stat_model.py --predict --market all
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from utils.db import executemany, read_dataframe
from utils.nba_defense import get_nba_defense_multiplier
from utils.nba_sigma import compute_player_sigma
from utils.nba_usage import compute_usage_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent

VALID_MARKETS = {"pts", "reb", "ast", "fg3m"}
ROLLING_WINDOWS = [5, 10]
MIN_GAMES_FOR_PREDICTION = 5

# Market-specific primary stat columns used for rolling features
_MARKET_STATS: dict[str, list[str]] = {
    "pts": ["pts", "min", "fga"],
    "reb": ["reb", "min"],
    "ast": ["ast", "min"],
    "fg3m": ["fg3m", "fga"],
}

# All stat columns needed across all markets (for DB query)
_ALL_STAT_COLS = sorted(
    {"player_id", "player_name", "team_abbreviation", "season", "game_id",
     "game_date", "matchup", "wl", "min", "pts", "reb", "ast", "fg3m", "fga"}
)


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def get_feature_cols(market: str) -> list[str]:
    """Return the ordered feature column list for a given market."""
    if market not in VALID_MARKETS:
        raise ValueError(f"Unknown market '{market}'. Valid: {VALID_MARKETS}")
    stats = _MARKET_STATS[market]
    rolling_cols = [
        f"{stat}_last{w}_avg"
        for stat in stats
        for w in ROLLING_WINDOWS
    ]
    contextual_cols = ["home_game", "b2b", "days_rest", "opp_def_rating_normalized"]
    usage_cols = ["fga_share", "min_share", "usage_delta"]
    return rolling_cols + contextual_cols + usage_cols


def _engineer_features(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Add rolling and contextual features to a sorted game-log DataFrame.

    The DataFrame must contain all columns referenced by the given market.
    Rows are sorted by (player_id, game_date) ascending internally.
    """
    if market not in VALID_MARKETS:
        raise ValueError(f"Unknown market '{market}'. Valid: {VALID_MARKETS}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    stats = _MARKET_STATS[market]
    for stat in stats:
        for window in ROLLING_WINDOWS:
            df[f"{stat}_last{window}_avg"] = (
                df.groupby("player_id")[stat]
                .transform(lambda s, w=window: s.shift(1).ewm(span=w, min_periods=1).mean())
            )

    # Home game flag: "TEAM vs. OPP" (home) or "TEAM @ OPP" (away)
    df["home_game"] = df["matchup"].str.contains("vs.", na=False).astype(int)

    # Opponent team abbreviation (last 3 chars of matchup)
    df["opponent"] = df["matchup"].str.strip().str[-3:]

    # Back-to-back flag
    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(3)
    df["days_rest"] = df["days_rest"].clip(upper=7)
    df["b2b"] = (df["days_rest"] <= 1).astype(int)

    # Usage features (fga_share, min_share, usage_delta)
    if "fga" in df.columns and "min" in df.columns:
        df = compute_usage_features(df)
    else:
        df["fga_share"] = 0.0
        df["min_share"] = 0.0
        df["usage_delta"] = 0.0

    return df


# Market -> defensive column mapping
_MARKET_DEF_COL: dict[str, str] = {
    "pts": "opp_pts_per100",
    "reb": "opp_reb_per100",
    "ast": "opp_ast_per100",
    "fg3m": "opp_fg3m_per100",
}


def _lookup_opponent_defense(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Merge opponent defensive ratings into the feature DataFrame."""
    df = df.copy()
    def_col = _MARKET_DEF_COL.get(market, "def_rating")

    # Load defensive stats
    def_stats = read_dataframe(
        f"SELECT team_abbreviation, season, {def_col}, def_rating "
        "FROM nba_team_defensive_stats "
        "ORDER BY team_abbreviation, season, as_of_date DESC"
    )

    if def_stats.empty:
        df["opp_def_rating_normalized"] = 0.0
        return df

    # Keep most recent per team+season
    def_stats = def_stats.drop_duplicates(subset=["team_abbreviation", "season"], keep="first")

    # Z-score normalize the defensive column
    col_vals = def_stats[def_col].dropna()
    if len(col_vals) > 1:
        mean_val = float(col_vals.mean())
        std_val = float(col_vals.std())
        if std_val > 0:
            def_stats = def_stats.copy()
            def_stats["opp_def_norm"] = (def_stats[def_col] - mean_val) / std_val
        else:
            def_stats = def_stats.copy()
            def_stats["opp_def_norm"] = 0.0
    else:
        def_stats = def_stats.copy()
        def_stats["opp_def_norm"] = 0.0

    # Ensure opponent column exists
    if "opponent" not in df.columns:
        df["opponent"] = df["matchup"].str.strip().str[-3:]

    merge_df = def_stats[["team_abbreviation", "season", "opp_def_norm"]].rename(
        columns={"team_abbreviation": "opponent", "opp_def_norm": "opp_def_rating_normalized"}
    )

    # Merge on opponent + season if season column exists
    if "season" in df.columns:
        df = df.merge(merge_df, on=["opponent", "season"], how="left")
    else:
        merge_df_no_season = (
            merge_df.drop(columns=["season"])
            .drop_duplicates(subset=["opponent"])
        )
        df = df.merge(merge_df_no_season, on=["opponent"], how="left")

    df["opp_def_rating_normalized"] = df["opp_def_rating_normalized"].fillna(0.0)
    return df


def _load_best_params(market: str) -> dict | None:
    """Load Optuna-tuned params if available."""
    params_path = MODEL_DIR / f"best_params_{market}.json"
    if params_path.exists():
        with open(params_path) as f:
            return json.load(f)
    return None


def _build_model(market: str) -> StackingRegressor:
    """Build a StackingRegressor ensemble for the given market."""
    params = _load_best_params(market) or {}
    default_depth = {"pts": 5, "reb": 4, "ast": 4, "fg3m": 3}.get(market, 4)
    depth = params.get("gbr_depth", default_depth)

    gbr = GradientBoostingRegressor(
        n_estimators=params.get("gbr_n", 400),
        learning_rate=params.get("gbr_lr", 0.04),
        max_depth=depth,
        subsample=params.get("gbr_sub", 0.8),
        min_samples_leaf=5,
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=depth + 2,
        min_samples_leaf=5,
        max_features=0.7,
        n_jobs=-1,
        random_state=42,
    )

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
        estimators = [("gbr", gbr), ("rf", rf), ("xgb", xgb_model)]
    except ImportError:
        estimators = [("gbr", gbr), ("rf", rf)]

    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=4,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(market: str) -> None:
    """Train a GBR model for the given market and save to disk."""
    if market not in VALID_MARKETS:
        raise ValueError(f"Unknown market '{market}'. Valid: {VALID_MARKETS}")

    model_path = MODEL_DIR / f"{market}_model.joblib"
    feature_cols = get_feature_cols(market)
    stats_needed = _MARKET_STATS[market]

    # Build the column list for the query (always include fga for usage features)
    base_cols = "player_id, player_name, team_abbreviation, season, game_id, game_date, matchup, min, fga"
    extra_cols = ", ".join(
        c for c in stats_needed if c not in {"min", "fga"}
    )
    select_cols = f"{base_cols}, {extra_cols}" if extra_cols else base_cols

    log.info("[%s] Loading game logs for training …", market)
    df = read_dataframe(
        f"SELECT {select_cols} FROM nba_player_game_logs "
        f"WHERE {market} IS NOT NULL ORDER BY player_id, game_date"
    )

    if df.empty:
        log.error("[%s] No data found in nba_player_game_logs. Run ingest first.", market)
        return

    log.info("[%s] Engineering features on %d rows …", market, len(df))
    df = _engineer_features(df, market)
    df = _lookup_opponent_defense(df, market)

    df = df.dropna(subset=feature_cols + [market])
    log.info("[%s] Training on %d rows after dropping NaN rows", market, len(df))

    X = df[feature_cols].values
    y = df[market].values

    # TimeSeriesSplit cross-validation for honest MAE tracking
    tscv = TimeSeriesSplit(n_splits=4)
    fold_maes = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        fold_model = _build_model(market)
        fold_model.fit(X[train_idx], y[train_idx])
        fold_preds = fold_model.predict(X[val_idx])
        fold_mae = float(np.mean(np.abs(fold_preds - y[val_idx])))
        fold_maes.append(fold_mae)
        log.info("[%s] Fold %d MAE: %.3f", market, fold_idx + 1, fold_mae)
    log.info(
        "[%s] CV MAE: %.3f +/- %.3f", market, float(np.mean(fold_maes)), float(np.std(fold_maes))
    )

    # Final model on full dataset
    model = _build_model(market)
    model.fit(X, y)

    train_preds = model.predict(X)
    train_mae = float(np.mean(np.abs(train_preds - y)))
    log.info("[%s] Training MAE: %.3f (full training set)", market, train_mae)

    joblib.dump(model, model_path)
    log.info("[%s] Model saved to %s", market, model_path)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def _get_todays_games() -> list[dict[str, Any]]:
    """Return today's NBA games from nba_api live scoreboard."""
    try:
        from nba_api.live.nba.endpoints import scoreboard

        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()["scoreboard"]["games"]
        games = []
        for g in games_data:
            games.append(
                {
                    "game_id": g["gameId"],
                    "game_date": g["gameEt"][:10],
                    "home_team": g["homeTeam"]["teamTricode"],
                    "away_team": g["awayTeam"]["teamTricode"],
                }
            )
        return games
    except Exception as exc:
        log.warning("Could not fetch live scoreboard: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _get_games_for_date(game_date: str) -> list[dict[str, Any]]:
    """Return games for a specific date from historical game logs."""
    df = read_dataframe(
        "SELECT DISTINCT game_id, game_date, "
        "       CASE WHEN matchup LIKE '%vs.%' THEN team_abbreviation END AS home_team, "
        "       CASE WHEN matchup LIKE '%@%' THEN team_abbreviation END AS away_team "
        "FROM nba_player_game_logs WHERE game_date = ?",
        [game_date],
    )
    if df.empty:
        return []
    # Pivot: each game_id has a home row and an away row
    games: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        gid = str(row["game_id"])
        if gid not in games:
            games[gid] = {"game_id": gid, "game_date": game_date, "home_team": "", "away_team": ""}
        if row["home_team"] and str(row["home_team"]) != "None":
            games[gid]["home_team"] = str(row["home_team"])
        if row["away_team"] and str(row["away_team"]) != "None":
            games[gid]["away_team"] = str(row["away_team"])
    return [g for g in games.values() if g["home_team"] and g["away_team"]]


def predict(market: str, game_date: str | None = None) -> None:
    """Generate projections for the given market. Uses today if game_date is None."""
    if market not in VALID_MARKETS:
        raise ValueError(f"Unknown market '{market}'. Valid: {VALID_MARKETS}")

    model_path = MODEL_DIR / f"{market}_model.joblib"
    if not model_path.exists():
        log.error(
            "[%s] Model not found at %s. Run --train --market %s first.",
            market, model_path, market,
        )
        return

    model = joblib.load(model_path)
    feature_cols = get_feature_cols(market)

    if game_date:
        games = _get_games_for_date(game_date)
    else:
        games = _get_todays_games()
    if not games:
        log.info("[%s] No games for %s. Nothing to project.", market, game_date or "today")
        return

    today = games[0]["game_date"]
    log.info("[%s] Generating projections for %s (%d games) …", market, today, len(games))

    # Build team -> game context mapping
    team_game: dict[str, dict] = {}
    for g in games:
        team_game[g["home_team"]] = {**g, "home_game": 1}
        team_game[g["away_team"]] = {**g, "home_game": 0}

    # All stat cols needed for feature engineering for this market (always include fga for usage)
    stats_needed = _MARKET_STATS[market]
    base_cols = "player_id, player_name, team_abbreviation, season, game_id, game_date, matchup, min, fga"
    extra_cols = ", ".join(c for c in stats_needed if c not in {"min", "fga"})
    select_cols = f"{base_cols}, {extra_cols}" if extra_cols else base_cols

    df_all = read_dataframe(
        f"SELECT {select_cols} FROM nba_player_game_logs "
        f"WHERE game_date < ? ORDER BY player_id, game_date",
        [today],
    )

    if df_all.empty:
        log.error("[%s] No historical game logs found. Run ingest first.", market)
        return

    df_all = _engineer_features(df_all, market)
    df_all = _lookup_opponent_defense(df_all, market)

    # For each player, grab their latest row (most recent rolling features)
    latest = df_all.sort_values("game_date").groupby("player_id").last().reset_index()

    playing_teams = set(team_game.keys())
    players_today = latest[latest["team_abbreviation"].isin(playing_teams)].copy()

    if players_today.empty:
        log.info("[%s] No players matched today's teams. Check team abbreviations.", market)
        return

    # Filter to players with enough game history
    game_counts = df_all.groupby("player_id").size().rename("game_count")
    players_today = players_today.join(game_counts, on="player_id")
    players_today = players_today[
        players_today["game_count"] >= MIN_GAMES_FOR_PREDICTION
    ].copy()

    log.info("[%s] Projecting %d players …", market, len(players_today))

    # Override home_game context for today's matchup
    players_today["home_game"] = players_today["team_abbreviation"].map(
        lambda t: team_game.get(t, {}).get("home_game", 0)
    )

    def _get_opponent(team: str) -> str:
        g = team_game.get(team, {})
        return g["away_team"] if team == g.get("home_team") else g.get("home_team", "UNK")

    players_today["opponent"] = players_today["team_abbreviation"].map(_get_opponent)
    players_today = _lookup_opponent_defense(players_today, market)

    X = players_today[feature_cols].fillna(0).values
    preds = model.predict(X)

    # Confidence: inverse of rolling std (capped [0.01, 0.99])
    rolling_std = df_all.groupby("player_id")[market].apply(
        lambda s: s.rolling(10, min_periods=3).std().iloc[-1] if len(s) >= 3 else np.nan
    )
    _scale = {"pts": 30.0, "reb": 15.0, "ast": 12.0, "fg3m": 6.0}
    scale = _scale.get(market, 20.0)
    players_today = players_today.copy()
    players_today["confidence"] = players_today["player_id"].map(
        lambda pid: max(0.01, min(0.99, 1 - (rolling_std.get(pid, scale * 0.2) / scale)))
    )

    # Pre-compute player history for sigma (Phase 1)
    player_history = df_all.groupby("player_id")[market].apply(list)

    rows = []
    for (_, row), proj in zip(players_today.iterrows(), preds):
        g = team_game.get(row["team_abbreviation"], {})
        pid = int(row["player_id"])
        season_val = int(row.get("season", date.today().year))
        opponent = str(row.get("opponent", "UNK"))

        # Phase 3: Apply defense multiplier to base projection
        defense_mult = get_nba_defense_multiplier(
            opponent=opponent, market=market,
            season=season_val, through_date=today,
        )
        adjusted_proj = float(proj) * defense_mult

        # Phase 1: Compute player-specific sigma from historical variance
        history = player_history.get(pid, [])
        sigma = compute_player_sigma(history, market=market)

        rows.append(
            (
                pid,
                str(row["player_name"]),
                str(row["team_abbreviation"]),
                season_val,
                today,
                str(g.get("game_id", "unknown")),
                market,
                float(round(adjusted_proj, 2)),
                float(round(row["confidence"], 4)),
                float(round(sigma, 4)),
            )
        )

    sql = """
        INSERT OR REPLACE INTO nba_projections (
            player_id, player_name, team, season, game_date,
            game_id, market, projected_value, confidence, sigma
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    executemany(sql, rows)
    log.info("[%s] Wrote %d projections to nba_projections for %s", market, len(rows), today)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA multi-market stat projection model")
    parser.add_argument("--train", action="store_true", help="Train and save the model(s)")
    parser.add_argument("--predict", action="store_true", help="Generate projections")
    parser.add_argument(
        "--market",
        default="pts",
        help=f"Market to train/predict: one of {sorted(VALID_MARKETS)} or 'all' (default: pts)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Game date for predictions (YYYY-MM-DD). Uses live scoreboard if omitted.",
    )
    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.print_help()
        return

    markets: list[str]
    if args.market == "all":
        markets = sorted(VALID_MARKETS)
    elif args.market in VALID_MARKETS:
        markets = [args.market]
    else:
        parser.error(
            f"Unknown market '{args.market}'. Choose from {sorted(VALID_MARKETS)} or 'all'."
        )
        return

    for m in markets:
        if args.train:
            train(m)
        if args.predict:
            predict(m, game_date=args.date)


if __name__ == "__main__":
    main()
