"""Season-level volume priors for early-week NFL projections.

Short EWMs (span 3/6) forget last season after a full prior year. Week 1
needs an explicit last-season / two-seasons-ago blend: 70/30 when last
season looks healthy, and a crash kicker that lifts the year before when
last season's games or volume collapsed.
"""

from __future__ import annotations

import pandas as pd

REGULAR_SEASON_MAX_WEEK = 18
EARLY_SEASON_GAME_THRESHOLD = 4
HEALTHY_LAST_WEIGHT = 0.70
HEALTHY_PRIOR_WEIGHT = 0.30
CRASH_LAST_WEIGHT = 0.45
CRASH_PRIOR_WEIGHT = 0.55
CRASH_MIN_GAMES = 8
CRASH_VOLUME_RATIO = 0.50

__all__ = [
    "CRASH_LAST_WEIGHT",
    "CRASH_MIN_GAMES",
    "CRASH_PRIOR_WEIGHT",
    "CRASH_VOLUME_RATIO",
    "DEFAULT_WR_ROLE_PRIORS",
    "EARLY_SEASON_GAME_THRESHOLD",
    "HEALTHY_LAST_WEIGHT",
    "HEALTHY_PRIOR_WEIGHT",
    "REGULAR_SEASON_MAX_WEEK",
    "apply_early_season_role_prior",
    "attach_season_volume_features",
    "blend_per_game_volume",
    "calibrate_wr_role_priors",
    "is_volume_crash",
    "regular_season_training_weeks",
    "season_feature_cols",
    "season_prior_weights",
]

DEFAULT_WR_ROLE_PRIORS: dict[str, float] = {
    "alpha": 58.0,
    "secondary": 43.0,
    "slot": 30.0,
    "fringe": 10.0,
}


def calibrate_wr_role_priors(history_df: pd.DataFrame | None = None) -> dict[str, float]:
    """Calibrate empirical WR receiving yards role priors across snap tiers.

    Tiers:
      - alpha: snap_percentage >= 80% (empirical ~57.6)
      - secondary: 60% <= snap_percentage < 80% (empirical ~42.8)
      - slot: 40% <= snap_percentage < 60% (empirical ~30.1)
      - fringe: snap_percentage < 40% (empirical ~9.8)
    """
    if history_df is None:
        try:
            from utils.db import read_dataframe

            history_df = read_dataframe(
                "SELECT position, snap_percentage, receiving_yards, week "
                "FROM player_stats_enhanced WHERE position = 'WR' AND week <= 18"
            )
        except Exception:
            history_df = pd.DataFrame()

    if history_df is None or history_df.empty or "receiving_yards" not in history_df.columns:
        return dict(DEFAULT_WR_ROLE_PRIORS)

    df = history_df.copy()
    if "position" in df.columns:
        df = df[df["position"].astype(str).str.upper() == "WR"]
    if df.empty:
        return dict(DEFAULT_WR_ROLE_PRIORS)

    snaps = pd.to_numeric(df.get("snap_percentage", 0.0), errors="coerce").fillna(0.0)
    if snaps.max() > 1.0:
        snaps = snaps / 100.0

    yards = pd.to_numeric(df["receiving_yards"], errors="coerce").fillna(0.0)

    alpha_mask = snaps >= 0.80
    sec_mask = (snaps >= 0.60) & (snaps < 0.80)
    slot_mask = (snaps >= 0.40) & (snaps < 0.60)
    fringe_mask = snaps < 0.40

    result = dict(DEFAULT_WR_ROLE_PRIORS)
    if alpha_mask.sum() >= 5:
        result["alpha"] = round(float(yards[alpha_mask].mean()), 1)
    if sec_mask.sum() >= 5:
        result["secondary"] = round(float(yards[sec_mask].mean()), 1)
    if slot_mask.sum() >= 5:
        result["slot"] = round(float(yards[slot_mask].mean()), 1)
    if fringe_mask.sum() >= 5:
        result["fringe"] = round(float(yards[fringe_mask].mean()), 1)

    return result


def is_volume_crash(last_games: float, last_volume: float, prior_volume: float) -> bool:
    if prior_volume <= 0:
        return False
    if last_games < CRASH_MIN_GAMES:
        return True
    return last_volume < CRASH_VOLUME_RATIO * prior_volume


def season_prior_weights(
    *,
    last_games: float,
    last_volume: float,
    prior_volume: float,
) -> tuple[float, float]:
    """Return (last_season_weight, prior_season_weight)."""
    if prior_volume <= 0:
        return 1.0, 0.0
    if is_volume_crash(last_games, last_volume, prior_volume):
        return CRASH_LAST_WEIGHT, CRASH_PRIOR_WEIGHT
    return HEALTHY_LAST_WEIGHT, HEALTHY_PRIOR_WEIGHT


def blend_per_game_volume(
    last_pg: float,
    prior_pg: float,
    *,
    last_games: float,
    last_volume: float,
    prior_volume: float,
) -> float | None:
    """Blend last and prior per-game volume. None means keep the EWM estimate."""
    if last_games <= 0:
        return None
    last_w, prior_w = season_prior_weights(
        last_games=last_games,
        last_volume=last_volume,
        prior_volume=prior_volume,
    )
    return float(last_w * last_pg + prior_w * prior_pg)


def regular_season_training_weeks(available: pd.DataFrame) -> list[tuple[int, int]]:
    """Last two ingested seasons, weeks 1–18, chronological."""
    if available is None or available.empty:
        return []
    if not {"season", "week"}.issubset(available.columns):
        raise ValueError("available must include season and week")
    seasons = sorted({int(season) for season in available["season"].tolist()})
    keep = set(seasons[-2:])
    weeks = available.loc[
        available["season"].map(int).isin(keep)
        & available["week"].map(int).between(1, REGULAR_SEASON_MAX_WEEK),
        ["season", "week"],
    ]
    pairs = {(int(season), int(week)) for season, week in weeks.itertuples(index=False)}
    return sorted(pairs)


def _identity_keys(df: pd.DataFrame) -> pd.Series:
    if "player_id" not in df.columns:
        raise ValueError("frame missing required column: player_id")
    player_id = df["player_id"].astype(str)
    if "gsis_id" not in df.columns:
        return player_id
    gsis = df["gsis_id"].astype(str).str.strip()
    missing = gsis.isin({"", "nan", "None"}) | df["gsis_id"].isna()
    return gsis.mask(missing, player_id)


def season_volume_table(history: pd.DataFrame, volume_col: str) -> pd.DataFrame:
    if history.empty or volume_col not in history.columns:
        return pd.DataFrame(columns=["_pid", "season", "volume", "games", "per_game"])
    work = pd.DataFrame(
        {
            "_pid": _identity_keys(history),
            "season": pd.to_numeric(history["season"], errors="coerce"),
            "week": pd.to_numeric(history["week"], errors="coerce"),
            "volume": pd.to_numeric(history[volume_col], errors="coerce").fillna(0.0),
        }
    )
    work = work[work["week"].between(1, REGULAR_SEASON_MAX_WEEK)]
    work = work.dropna(subset=["_pid", "season"])
    if work.empty:
        return pd.DataFrame(columns=["_pid", "season", "volume", "games", "per_game"])
    grouped = work.groupby(["_pid", "season"], as_index=False).agg(
        volume=("volume", "sum"),
        games=("week", "nunique"),
    )
    grouped["per_game"] = grouped["volume"] / grouped["games"].clip(lower=1)
    return grouped


def attach_season_volume_features(df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
    """Add last_season_{col}_pg and prior_season_{col}_pg from completed seasons."""
    out = df.copy()
    last_col = f"last_season_{volume_col}_pg"
    prior_col = f"prior_season_{volume_col}_pg"
    out[last_col] = 0.0
    out[prior_col] = 0.0
    if out.empty or "season" not in out.columns:
        return out
    table = season_volume_table(out, volume_col)
    if table.empty:
        return out
    lookup = table.set_index(["_pid", "season"])["per_game"]
    keys = _identity_keys(out)
    seasons = pd.to_numeric(out["season"], errors="coerce")
    last_index = pd.MultiIndex.from_arrays([keys.to_numpy(), (seasons - 1).to_numpy()])
    prior_index = pd.MultiIndex.from_arrays([keys.to_numpy(), (seasons - 2).to_numpy()])
    out[last_col] = lookup.reindex(last_index).fillna(0.0).to_numpy()
    out[prior_col] = lookup.reindex(prior_index).fillna(0.0).to_numpy()
    return out


def apply_early_season_role_prior(df: pd.DataFrame, volume_col: str) -> pd.DataFrame:
    """Replace expected_{col} with the 70/30 season blend before week 4 of a season."""
    out = attach_season_volume_features(df, volume_col)
    role_col = f"expected_{volume_col}"
    last_col = f"last_season_{volume_col}_pg"
    prior_col = f"prior_season_{volume_col}_pg"
    if role_col not in out.columns:
        out[role_col] = 0.0
    if out.empty or "season" not in out.columns or "week" not in out.columns:
        return out

    keys = _identity_keys(out)
    seasons = pd.to_numeric(out["season"], errors="coerce")
    weeks = pd.to_numeric(out["week"], errors="coerce")
    left = pd.DataFrame(
        {"idx": out.index, "_pid": keys.to_numpy(), "_season": seasons, "_week": weeks}
    )
    hist = left.loc[
        left["_week"].between(1, REGULAR_SEASON_MAX_WEEK),
        ["_pid", "_season", "_week"],
    ].rename(columns={"_week": "hist_week"})
    merged = left.merge(hist, on=["_pid", "_season"], how="left")
    prior_games = (
        merged.loc[merged["hist_week"] < merged["_week"]]
        .groupby("idx")["hist_week"]
        .nunique()
        .reindex(out.index)
        .fillna(0)
    )

    table = season_volume_table(out, volume_col)
    if table.empty:
        return out
    lookup = table.set_index(["_pid", "season"])
    last_index = pd.MultiIndex.from_arrays([keys.to_numpy(), (seasons - 1).to_numpy()])
    prior_index = pd.MultiIndex.from_arrays([keys.to_numpy(), (seasons - 2).to_numpy()])
    last_games = lookup["games"].reindex(last_index).fillna(0).to_numpy()
    last_volume = lookup["volume"].reindex(last_index).fillna(0).to_numpy()
    prior_volume = lookup["volume"].reindex(prior_index).fillna(0).to_numpy()
    last_pg = pd.to_numeric(out[last_col], errors="coerce").fillna(0.0).to_numpy()
    prior_pg = pd.to_numeric(out[prior_col], errors="coerce").fillna(0.0).to_numpy()

    last_w = pd.Series(
        [
            season_prior_weights(
                last_games=float(games),
                last_volume=float(vol),
                prior_volume=float(prior),
            )[0]
            for games, vol, prior in zip(last_games, last_volume, prior_volume)
        ],
        index=out.index,
        dtype=float,
    )
    blended = last_w.to_numpy() * last_pg + (1.0 - last_w.to_numpy()) * prior_pg
    mask = (prior_games.to_numpy() < EARLY_SEASON_GAME_THRESHOLD) & (last_games > 0)
    role = pd.to_numeric(out[role_col], errors="coerce").fillna(0.0).to_numpy(copy=True)
    role[mask] = blended[mask]
    out[role_col] = role
    return out


def last_season_feature_cols(volume_col: str) -> list[str]:
    return [f"last_season_{volume_col}_pg", f"prior_season_{volume_col}_pg"]


# Keep a stable alias for callers that want the feature names without importing internals.
def season_feature_cols(volume_col: str) -> list[str]:
    return last_season_feature_cols(volume_col)
