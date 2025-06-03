#!/usr/bin/env python3
"""Simple neural network model for NFL yardage predictions."""

import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(db_path: str = "nfl_data.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM player_stats ORDER BY season", conn)
    conn.close()
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features['yards_per_attempt'] = features['rushing_yards'] / (features['rushing_attempts'] + 1)
    features['receptions_per_target'] = features['receptions'] / (features['targets'] + 1)
    features['position_encoded'] = features['position'].map({'RB':1,'WR':2,'TE':3,'QB':4}).fillna(0)

    lag_cols = ['rushing_yards','receiving_yards','rushing_attempts','receptions']
    features = features.sort_values(['player_id','season']).reset_index(drop=True)
    for col in lag_cols:
        features[f'{col}_prev'] = features.groupby('player_id')[col].shift(1).fillna(0)
    for col in lag_cols:
        career_avg = features.groupby('player_id')[col].expanding().mean().reset_index(level=0,drop=True)
        features[f'{col}_career_avg'] = career_avg.shift(1).fillna(0)

    features['age_squared'] = features['age'] ** 2
    return features

def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(df: pd.DataFrame, model_path: str | None = None) -> tf.keras.Model:
    """Train the neural network and optionally save it."""
    df = prepare_features(df)
    feature_cols = [
        'age','age_squared','position_encoded','rushing_attempts','receptions',
        'yards_per_attempt','receptions_per_target','rushing_yards_prev','receiving_yards_prev',
        'rushing_attempts_prev','receptions_prev','rushing_yards_career_avg','receiving_yards_career_avg'
    ]
    X = df[feature_cols].fillna(0)
    y = df['rushing_yards']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Neural network MAE: %.2f", mae)
    if model_path:
        model.save(model_path)
        logger.info("Model saved to %s", model_path)

    return model

if __name__ == "__main__":
    data = load_data()
    if data.empty:
        logger.error("No data found. Run data_collection.py first.")
    else:
        train_model(data, "nfl_model.h5")

