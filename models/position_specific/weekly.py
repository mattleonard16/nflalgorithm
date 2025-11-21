"""Weekly model training and inference utilities."""

from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import config
from data_pipeline import DataPipeline, compute_week_features, update_week


MODEL_DIR = config.models_dir / "weekly"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURE_COLUMNS: List[str] = [
    'line', 'price', 'targets', 'rolling_targets', 'rolling_routes', 'rolling_air_yards',
    'usage_delta', 'offensive_rank', 'defensive_rank', 'pace_rank', 'red_zone_efficiency',
    'temperature', 'wind_speed', 'injury_indicator', 'weather_penalty'
]


def train_weekly_models(season_weeks: List[Tuple[int, int]]) -> Dict[str, str]:
    """Train position-specific weekly models using rolling history."""

    if not season_weeks:
        raise ValueError("season_weeks must contain at least one (season, week) tuple")

    pipeline = DataPipeline()
    market_frames: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    for season, week in season_weeks:
        update_week(season, week)
        features = pipeline.compute_week_feature_frame(season, week)
        if features.empty:
            continue
        for market, frame in features.groupby('market'):
            market_frames[market].append(frame)

    if not market_frames:
        raise RuntimeError("No training data available for any market")

    model_version = datetime.utcnow().strftime("weekly-%Y%m%dT%H%M%SZ")
    trained_paths: Dict[str, str] = {}

    for market, frames in market_frames.items():
        data = pd.concat(frames, ignore_index=True)
        if data.empty:
            continue

        feature_columns = [col for col in BASE_FEATURE_COLUMNS if col in data.columns]
        if not feature_columns:
            continue

        X = data[feature_columns]
        y = data['mu_prior']

        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])
        model.fit(X, y)

        preds = model.predict(X)
        residuals = y - preds
        sigma_default = float(np.maximum(residuals.std(ddof=1) if len(residuals) > 1 else 7.5, 5.0))

        artifact = {
            'model': model,
            'feature_columns': feature_columns,
            'market': market,
            'model_version': model_version,
            'sigma_default': sigma_default
        }

        path = MODEL_DIR / f"{market}_model.joblib"
        joblib.dump(artifact, path)
        trained_paths[market] = str(path)

    return trained_paths


def predict_week(season: int, week: int, model_directory: Optional[Path | str] = None) -> pd.DataFrame:
    """Produce weekly projections for the given week and persist them."""

    update_week(season, week)
    features = compute_week_features(season, week)
    if features.empty:
        return pd.DataFrame()

    model_dir = Path(model_directory) if model_directory is not None else MODEL_DIR
    projections: List[pd.DataFrame] = []
    generated_at = datetime.utcnow().isoformat()

    for market, group in features.groupby('market'):
        model_path = model_dir / f"{market}_model.joblib"
        if model_path.exists():
            artifact = joblib.load(model_path)
            model: Pipeline = artifact['model']
            feature_columns: Iterable[str] = artifact.get('feature_columns', BASE_FEATURE_COLUMNS)
            model_version = artifact.get('model_version', 'weekly-unknown')
            sigma_default = float(artifact.get('sigma_default', 7.5))

            X = group.reindex(columns=feature_columns, fill_value=np.nan)
            mu = model.predict(X)
            
            # If model predicts 0 or negative, fall back to mu_prior
            # This handles cases where features are all NaN/imputed to 0
            mu_prior = group['mu_prior'].to_numpy()
            mu = np.where((mu <= 0) | np.isnan(mu), mu_prior, mu)
            # Guardrail: QB rushing projections can explode if trained on mixed-position data.
            # Keep them within a reasonable band around mu_prior.
            if market == 'rushing_yards':
                is_qb = group['position'].str.upper().eq('QB').to_numpy()
                qb_cap = np.minimum(mu_prior * 1.5 + 10.0, 90.0)
                mu = np.where(is_qb, np.minimum(mu, qb_cap), mu)
            
            sigma = np.maximum(sigma_default, np.abs(mu) * 0.3)
            featureset_hash = _featureset_hash(feature_columns)
        else:
            mu = group['mu_prior'].to_numpy()
            sigma = np.maximum(7.5, np.abs(mu) * 0.35)
            model_version = 'weekly-baseline'
            featureset_hash = _featureset_hash(group.columns)

        projections.append(pd.DataFrame({
            'season': season,
            'week': week,
            'player_id': group['player_id'],
            'team': group['team'].fillna('UNK'),
            'opponent': group['opponent'].fillna('UNK'),
            'market': market,
            'mu': mu,
            'sigma': sigma,
            'model_version': model_version,
            'featureset_hash': featureset_hash,
            'generated_at': generated_at
        }))

    if not projections:
        return pd.DataFrame()

    result = pd.concat(projections, ignore_index=True)
    _persist_weekly_projections(result)
    return result


def _persist_weekly_projections(df: pd.DataFrame) -> None:
    if df.empty:
        return

    payload = df.copy()
    payload['mu'] = payload['mu'].astype(float)
    payload['sigma'] = payload['sigma'].astype(float)

    sql = (
        """
        INSERT INTO weekly_projections (
            season, week, player_id, team, opponent, market, mu, sigma, model_version, featureset_hash, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(season, week, player_id, market)
        DO UPDATE SET
            team=excluded.team,
            opponent=excluded.opponent,
            mu=excluded.mu,
            sigma=excluded.sigma,
            model_version=excluded.model_version,
            featureset_hash=excluded.featureset_hash,
            generated_at=excluded.generated_at
        """
    )

    with sqlite3.connect(config.database.path) as conn:
        conn.executemany(
            sql,
            payload[['season', 'week', 'player_id', 'team', 'opponent', 'market', 'mu', 'sigma', 'model_version', 'featureset_hash', 'generated_at']]
            .itertuples(index=False, name=None)
        )
        conn.commit()


def _featureset_hash(columns: Iterable[str]) -> str:
    joined = '|'.join(sorted(str(col) for col in columns))
    return hashlib.sha1(joined.encode('utf-8')).hexdigest()
