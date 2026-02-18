"""NBA points (PTS) projection model.

Trains a GradientBoostingRegressor on historical player game logs and
writes per-player per-game projections to nba_projections.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python models/nba/points_model.py --train
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python models/nba/points_model.py --predict
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python models/nba/points_model.py --train --predict
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from utils.db import executemany, read_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "points_model.joblib"
ENCODER_PATH = MODEL_DIR / "team_encoder.joblib"

MARKET = "pts"
MIN_GAMES_FOR_PREDICTION = 5  # Skip players with fewer games (too sparse)
ROLLING_WINDOWS = [5, 10]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and contextual features to a sorted game-log DataFrame.

    The DataFrame must be sorted by (player_id, game_date) ascending.
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Rolling means (shift(1) to avoid leaking same-game result)
    for stat in ["pts", "min", "fga"]:
        for window in ROLLING_WINDOWS:
            df[f"{stat}_last{window}_avg"] = (
                df.groupby("player_id")[stat]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    # Home game flag: NBA matchup format is "TEAM vs. OPP" (home) or "TEAM @ OPP" (away)
    df["home_game"] = df["matchup"].str.contains("vs.", na=False).astype(int)

    # Opponent team (last 3 chars of matchup)
    df["opponent"] = df["matchup"].str.strip().str[-3:]

    # Back-to-back flag
    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(3)
    df["b2b"] = (df["days_rest"] <= 1).astype(int)

    return df


def _encode_opponents(df: pd.DataFrame, encoder: LabelEncoder | None = None):
    """Label-encode the opponent column. Returns (df, encoder)."""
    if encoder is None:
        encoder = LabelEncoder()
        df["opponent_enc"] = encoder.fit_transform(df["opponent"].fillna("UNK"))
    else:
        known = set(encoder.classes_)
        df["opponent_enc"] = df["opponent"].apply(
            lambda t: encoder.transform([t])[0] if t in known else -1
        )
    return df, encoder


FEATURE_COLS = [
    "pts_last5_avg",
    "pts_last10_avg",
    "min_last5_avg",
    "min_last10_avg",
    "fga_last5_avg",
    "fga_last10_avg",
    "home_game",
    "b2b",
    "opponent_enc",
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train() -> None:
    log.info("Loading game logs for training …")
    df = read_dataframe(
        "SELECT player_id, player_name, team_abbreviation, season, game_id, "
        "game_date, matchup, pts, min, fga FROM nba_player_game_logs "
        "WHERE pts IS NOT NULL ORDER BY player_id, game_date"
    )

    if df.empty:
        log.error("No data found in nba_player_game_logs. Run ingest first.")
        return

    log.info("Engineering features on %d rows …", len(df))
    df = _engineer_features(df)
    df, encoder = _encode_opponents(df)

    # Drop rows where we don't have enough history for rolling features
    df = df.dropna(subset=FEATURE_COLS + ["pts"])
    log.info("Training on %d rows after dropping NaN rows", len(df))

    X = df[FEATURE_COLS].values
    y = df["pts"].values

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    # Chronological 80/20 split for held-out MAE evaluation
    split = int(len(X) * 0.8)
    if split > 0 and split < len(X):
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        val_mae = float(np.mean(np.abs(val_preds - y_val)))
        log.info("Validation MAE: %.2f pts (held-out 20%%)", val_mae)
        # Refit on full data for production model
        model.fit(X, y)

    train_preds = model.predict(X)
    train_mae = float(np.mean(np.abs(train_preds - y)))
    log.info("Training MAE: %.2f pts (full training set)", train_mae)
    log.info("Model saved to %s", MODEL_PATH)


# ---------------------------------------------------------------------------
# Prediction
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
                    "game_date": g["gameEt"][:10],  # ISO date
                    "home_team": g["homeTeam"]["teamTricode"],
                    "away_team": g["awayTeam"]["teamTricode"],
                }
            )
        return games
    except Exception as exc:
        log.warning("Could not fetch live scoreboard: %s", exc)
        return []


def predict() -> None:
    if not MODEL_PATH.exists():
        log.error("Model not found at %s. Run --train first.", MODEL_PATH)
        return

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    games = _get_todays_games()
    if not games:
        log.info("No games today. Nothing to project.")
        return

    today = games[0]["game_date"]
    log.info("Generating NBA PTS projections for %s (%d games) …", today, len(games))

    # Build team -> game mapping
    team_game: dict[str, dict] = {}
    for g in games:
        team_game[g["home_team"]] = {**g, "home_game": 1}
        team_game[g["away_team"]] = {**g, "home_game": 0}

    # Load all game logs up to (but not including) today
    df_all = read_dataframe(
        "SELECT player_id, player_name, team_abbreviation, game_id, game_date, "
        "matchup, pts, min, fga FROM nba_player_game_logs "
        "WHERE game_date < ? ORDER BY player_id, game_date",
        [today],
    )

    if df_all.empty:
        log.error("No historical game logs found. Run ingest first.")
        return

    df_all = _engineer_features(df_all)
    df_all, _ = _encode_opponents(df_all, encoder=encoder)

    # For each player, grab their latest row (most recent rolling features)
    latest = df_all.sort_values("game_date").groupby("player_id").last().reset_index()

    # Filter to players on teams playing today
    playing_teams = set(team_game.keys())
    players_today = latest[latest["team_abbreviation"].isin(playing_teams)].copy()

    if players_today.empty:
        log.info("No players matched today's teams. Check team abbreviations.")
        return

    # Filter to players with enough games
    game_counts = df_all.groupby("player_id").size().rename("game_count")
    players_today = players_today.join(game_counts, on="player_id")
    players_today = players_today[players_today["game_count"] >= MIN_GAMES_FOR_PREDICTION]

    log.info("Projecting %d players …", len(players_today))

    # Override home_game and opponent for today's context
    players_today = players_today.copy()
    players_today["home_game"] = players_today["team_abbreviation"].map(
        lambda t: team_game.get(t, {}).get("home_game", 0)
    )
    # Opponent is the other team in today's matchup
    def _get_opponent(team: str) -> str:
        g = team_game.get(team, {})
        return g["away_team"] if team == g.get("home_team") else g.get("home_team", "UNK")

    players_today["opponent"] = players_today["team_abbreviation"].map(_get_opponent)
    players_today, _ = _encode_opponents(players_today, encoder=encoder)

    X = players_today[FEATURE_COLS].fillna(0).values
    preds = model.predict(X)

    # Confidence: inverse of rolling variance (capped at 0.99)
    rolling_std = df_all.groupby("player_id")["pts"].apply(
        lambda s: s.rolling(10, min_periods=3).std().iloc[-1] if len(s) >= 3 else np.nan
    )
    players_today["confidence"] = players_today["player_id"].map(
        lambda pid: max(0.01, min(0.99, 1 - rolling_std.get(pid, 5) / 30))
    )

    rows = []
    for (_, row), proj in zip(players_today.iterrows(), preds):
        g = team_game.get(row["team_abbreviation"], {})
        rows.append(
            (
                int(row["player_id"]),
                str(row["player_name"]),
                str(row["team_abbreviation"]),
                int(row.get("season", date.today().year)),
                today,
                str(g.get("game_id", "unknown")),
                MARKET,
                float(round(proj, 2)),
                float(round(row["confidence"], 4)),
            )
        )

    sql = """
        INSERT OR REPLACE INTO nba_projections (
            player_id, player_name, team, season, game_date,
            game_id, market, projected_value, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    executemany(sql, rows)
    log.info("Wrote %d projections to nba_projections for %s", len(rows), today)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA points projection model")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--predict", action="store_true", help="Generate today's projections")
    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.print_help()
        return

    if args.train:
        train()
    if args.predict:
        predict()


if __name__ == "__main__":
    main()
