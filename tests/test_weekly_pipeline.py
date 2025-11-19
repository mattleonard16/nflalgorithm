"""Weekly pipeline integration tests."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path
from config import config
from data_pipeline import compute_week_features, update_week
from materialized_value_view import materialize_week
from models.position_specific import predict_week, train_weekly_models
from models.position_specific import weekly as weekly_module
from schema_migrations import MigrationManager
from value_betting_engine import kelly_fraction, prob_over, rank_weekly_value


def with_temp_database():
    original = Path(config.database.path)
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    if original.exists():
        shutil.copy(original, tmp.name)
    else:
        Path(tmp.name).touch()
    return tmp.name


def test_prob_over_symmetry():
    assert prob_over(100, 10, 100) == 0.5
    assert prob_over(110, 10, 100) > 0.5
    assert prob_over(90, 10, 100) < 0.5


def test_kelly_fraction_bounds():
    assert kelly_fraction(0.6, -110) > 0
    assert kelly_fraction(0.4, -110) == 0


def test_weekly_roundtrip_pipeline():
    tmp_db = with_temp_database()
    original_path = config.database.path
    original_model_dir = weekly_module.MODEL_DIR
    tmp_model_dir = Path(tempfile.mkdtemp())
    weekly_module.MODEL_DIR = tmp_model_dir
    config.database.path = tmp_db
    try:
        MigrationManager(tmp_db).run()
        update_week(2023, 1)
        features = compute_week_features(2023, 1)
        assert not features.empty

        train_weekly_models([(2023, 1)])
        predictions = predict_week(2023, 1)
        assert not predictions.empty

        ranked = rank_weekly_value(2023, 1, min_edge=0.0, place=False)
        assert not ranked.empty

        materialize_week(2023, 1)
        with sqlite3.connect(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM materialized_value_view WHERE season=? AND week=?",
                (2023, 1)
            ).fetchone()[0]
        assert count > 0
    finally:
        config.database.path = original_path
        weekly_module.MODEL_DIR = original_model_dir
        Path(tmp_db).unlink(missing_ok=True)
        shutil.rmtree(tmp_model_dir, ignore_errors=True)
