"""Public baseline projector for NFL weekly props.

A clone without the deployment-supplied ``weekly.py`` projects with this, so the
pipeline runs end to end and ``make nfl-backtest`` can score a change. It is
deliberately plain: each player's last six games per market, weighted toward the
recent ones. No trained model, no defense adjustment, no role priors.

It keeps the contract the tracked callers use (``scripts/prepare_nfl_week.py``
and ``scripts/run_nfl_backtest.py``): ``predict_week``, ``train_weekly_models``,
``_write_predictions`` and ``MODEL_DIR``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from sports.nfl import INACTIVE_ROSTER_STATUSES, MARKET_MIN_EXPECTED_VOLUME
from utils.db import execute, executemany, get_backend, get_connection, read_dataframe
from utils.nfl_markets import synthesize_anytime_td
from utils.nfl_sigma import compute_player_sigma
from utils.slate_eligibility import backup_qb, ruled_out
from utils.volatility_scoring import volatility_score_or_none

logger = logging.getLogger(__name__)

MODEL_VERSION = "public_baseline_ewma_v1"
FEATURESET_HASH = "public_baseline"
# Nothing is trained, so nothing is saved here. The backtest harness still
# redirects it before every run, so it stays part of the contract.
MODEL_DIR = Path(__file__).parent.parent / "weekly"

RECENT_GAMES = 6
EWMA_SPAN = 3

# market -> (stat projected, volume that decides eligibility, positions)
MARKETS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "rushing_yards": ("rushing_yards", "rushing_attempts", ("RB", "QB", "WR", "TE")),
    "receiving_yards": ("receiving_yards", "targets", ("WR", "TE", "RB")),
    "passing_yards": ("passing_yards", "passing_attempts", ("QB",)),
    "receptions": ("receptions", "targets", ("WR", "TE", "RB")),
    "anytime_touchdown": ("anytime_td", "red_zone_touches", ("RB", "WR", "TE", "QB")),
}
SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "FB")
_HISTORY_COLUMNS = (
    "rushing_yards",
    "rushing_attempts",
    "receiving_yards",
    "receptions",
    "targets",
    "passing_yards",
    "passing_attempts",
    "red_zone_touches",
    "rushing_tds",
    "receiving_tds",
)
OUTPUT_COLUMNS = [
    "player_id",
    "position",
    "market",
    "mu",
    "sigma",
    "volatility_score",
    "model_version",
    "featureset_hash",
    "generated_at",
]


def train_weekly_models(season_week_tuples: Iterable[tuple[int, int]]) -> dict[str, str]:
    """Nothing to fit: the baseline reads each player's recent games directly."""
    return {}


def predict_week(
    season: int,
    week: int,
    *,
    roster_backed: bool = False,
    exclude_teams: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Project every eligible player for the week and store the rows.

    ``roster_backed`` takes players from the current roster, as a pregame run
    must. Without it, players come from the week's stat rows, which a backtest
    uses. Only their identities are read, never the week's results.
    ``exclude_teams`` are teams already playing: they get no new rows, and their
    stored pregame rows survive.
    """
    players = _roster_players(season, week) if roster_backed else _stat_players(season, week)
    if exclude_teams and not players.empty:
        players = players[~players["team"].isin(exclude_teams)]
    if players.empty:
        logger.warning("No players to project for season %d week %d", season, week)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    predictions = _project(players, _load_history(players, season, week))
    if predictions.empty:
        logger.warning("No player cleared a market's volume floor for %d week %d", season, week)
        return predictions
    logger.info(
        "Baseline projected %d rows for %d players, season %d week %d",
        len(predictions),
        predictions["player_id"].nunique(),
        season,
        week,
    )
    _write_predictions(season, week, predictions, players, keep_teams=exclude_teams)
    return predictions


def _opponents(season: int, week: int) -> dict[str, str]:
    games = read_dataframe(
        "SELECT home_team, away_team FROM games WHERE season = ? AND week = ?",
        params=(season, week),
    )
    opponents: dict[str, str] = {}
    for game in games.itertuples(index=False):
        opponents[str(game.home_team)] = str(game.away_team)
        opponents[str(game.away_team)] = str(game.home_team)
    return opponents


def _stat_players(season: int, week: int) -> pd.DataFrame:
    players = read_dataframe(
        """
        SELECT DISTINCT player_id, gsis_id, team, position
        FROM player_stats_enhanced
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )
    players = players[players["position"].isin(SKILL_POSITIONS)].copy()
    players["opponent"] = players["team"].map(_opponents(season, week)).fillna("")
    return players


def _roster_players(season: int, week: int) -> pd.DataFrame:
    placeholders = ", ".join("?" for _ in SKILL_POSITIONS)
    roster = read_dataframe(
        f"""
        SELECT gsis_id, player_id, team, position, roster_status
        FROM nfl_roster_players
        WHERE season = ? AND position IN ({placeholders})
        """,
        params=(season, *SKILL_POSITIONS),
    )
    opponents = _opponents(season, week)
    status = roster["roster_status"].fillna("").astype(str).str.upper().str.strip()
    roster = roster[~status.isin(INACTIVE_ROSTER_STATUSES) & roster["team"].isin(opponents)]
    context = read_dataframe(
        """
        SELECT gsis_id, depth_rank, is_starter, injury_status, injury_report_week
        FROM nfl_player_context_snapshots
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )
    if not roster.empty and not context.empty:
        roster = roster.merge(context.drop_duplicates("gsis_id", keep="last"), on="gsis_id", how="left")
        # Ruled out on this week's report, or a QB behind the starter: his past
        # starts would project him like one.
        sits = roster.apply(lambda row: ruled_out(row, week) or backup_qb(row), axis=1)
        if sits.any():
            logger.info("Dropping %d players ruled out or behind the starting QB", int(sits.sum()))
        roster = roster[~sits]
    roster = roster.assign(opponent=roster["team"].map(opponents))
    return roster[["player_id", "gsis_id", "team", "opponent", "position"]]


def _load_history(players: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Every game strictly before the target week, keyed to the players' current ids."""
    player_ids = players["player_id"].dropna().astype(str).unique().tolist()
    gsis_ids = players["gsis_id"].dropna().astype(str).unique().tolist()
    if not player_ids:
        return pd.DataFrame()
    identity = [f"player_id IN ({', '.join('?' for _ in player_ids)})"]
    if gsis_ids:
        identity.append(f"gsis_id IN ({', '.join('?' for _ in gsis_ids)})")
    history = read_dataframe(
        f"""
        SELECT player_id, gsis_id, season, week, {', '.join(_HISTORY_COLUMNS)}
        FROM player_stats_enhanced
        WHERE ({' OR '.join(identity)})
          AND (season < ? OR (season = ? AND week < ?))
        """,
        params=(*player_ids, *gsis_ids, season, season, week),
    )
    if history.empty:
        return history
    # A player's id can differ across seasons; the GSIS id does not.
    current_ids = (
        players.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["player_id"]
    )
    mapped = history["gsis_id"].map(current_ids)
    history["player_id"] = mapped.fillna(history["player_id"]).astype(str)
    return synthesize_anytime_td(history).sort_values(["player_id", "season", "week"])


def _recent_mean(values: pd.Series) -> float:
    numbers = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float(numbers.ewm(span=EWMA_SPAN).mean().iloc[-1])


def _project(players: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    position_of = players.drop_duplicates("player_id").set_index("player_id")["position"]
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for player_id, games in history.groupby("player_id", sort=False):
        position = position_of.get(player_id)
        if position is None:
            continue
        recent = games.tail(RECENT_GAMES)
        for market, (stat, volume, positions) in MARKETS.items():
            if position not in positions:
                continue
            if _recent_mean(recent[volume]) < MARKET_MIN_EXPECTED_VOLUME[market]:
                continue
            values = pd.to_numeric(games[stat], errors="coerce").dropna().tolist()
            rows.append(
                {
                    "player_id": player_id,
                    "position": position,
                    "market": market,
                    "mu": _recent_mean(recent[stat]),
                    "sigma": compute_player_sigma(values, market=market, position=position),
                    "volatility_score": volatility_score_or_none(values),
                    "model_version": MODEL_VERSION,
                    "featureset_hash": FEATURESET_HASH,
                    "generated_at": generated_at,
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _optional_float(value: Any) -> float | None:
    # NULL means "not measured". sqlite3 stores NaN as NULL but MySQL rejects it.
    if value is None or pd.isna(value):
        return None
    return float(value)


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _write_predictions(
    season: int,
    week: int,
    predictions_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    keep_teams: frozenset[str] = frozenset(),
) -> None:
    """Replace the week's weekly_projections rows, except those for ``keep_teams``."""
    teams = source_df[["player_id", "team", "opponent"]].drop_duplicates("player_id")
    rows = predictions_df.merge(teams, on="player_id", how="left")
    records = [
        (
            season,
            week,
            str(row.player_id),
            _text(row.team),
            _text(row.opponent),
            str(row.market),
            float(row.mu),
            float(row.sigma),
            _optional_float(row.volatility_score),
            str(row.model_version),
            str(row.featureset_hash),
            str(row.generated_at),
        )
        for row in rows.itertuples(index=False)
    ]
    updated = (
        "team", "opponent", "mu", "sigma", "volatility_score",
        "model_version", "featureset_hash", "generated_at",
    )
    if get_backend() == "mysql":
        conflict = "ON DUPLICATE KEY UPDATE " + ", ".join(f"{c} = VALUES({c})" for c in updated)
    else:
        conflict = "ON CONFLICT (season, week, player_id, market) DO UPDATE SET " + ", ".join(
            f"{c} = excluded.{c}" for c in updated
        )
    upsert = f"""
        INSERT INTO weekly_projections
        (season, week, player_id, team, opponent, market, mu, sigma, volatility_score,
         model_version, featureset_hash, generated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        {conflict}
    """
    delete = "DELETE FROM weekly_projections WHERE season = ? AND week = ?"
    if keep_teams:
        delete += f" AND team NOT IN ({', '.join('?' for _ in keep_teams)})"
    with get_connection() as conn:
        execute(delete, (season, week, *sorted(keep_teams)), conn=conn)
        executemany(upsert, records, conn=conn)
        conn.commit()
