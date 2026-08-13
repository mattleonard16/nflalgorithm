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


def _seed_schedule(db_path: str, season: int, weeks: range) -> None:
    """Give the seeded players a game to belong to.

    Odds snapshots are keyed by game, so a club with no scheduled game gets no
    line at all. The pipeline seeds its own players but not a schedule, so
    without this the roundtrip generates zero odds and every later assertion
    is vacuously empty.
    """
    # Every club _seed_baseline_stats uses, so no seeded player is left without
    # a game. A club missing here silently loses its odds rows.
    clubs = [
        "KC", "BUF", "PHI", "SF", "DAL", "MIA", "CIN", "DET",
        "BAL", "MIN", "ATL", "CLE", "DEN", "JAX", "NYG", "SEA",
    ]  # fmt: skip
    pairs = [(clubs[i], clubs[i + 1]) for i in range(0, len(clubs), 2)]
    # Plain INSERT, not INSERT OR IGNORE: a constraint failure here means the
    # fixture is wrong, and silently seeding nothing would make every later
    # assertion pass vacuously.
    with sqlite3.connect(db_path) as conn:
        for week in weeks:
            for home, away in pairs:
                conn.execute(
                    "INSERT INTO games "
                    "(game_id, season, week, home_team, away_team, game_date) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        f"{season}_{week:02d}_{away}_{home}",
                        season,
                        week,
                        home,
                        away,
                        f"{season}-09-{min(week, 28):02d}",
                    ),
                )
        conn.commit()


def test_weekly_roundtrip_pipeline():
    tmp_db = with_temp_database()
    original_path = config.database.path
    original_model_dir = weekly_module.MODEL_DIR
    tmp_model_dir = Path(tempfile.mkdtemp())
    weekly_module.MODEL_DIR = tmp_model_dir
    config.database.path = tmp_db
    try:
        MigrationManager(tmp_db).run()
        _seed_schedule(tmp_db, 2023, range(1, 14))
        # The causal model needs prior weeks to derive pregame role estimates;
        # a single realized week is intentionally insufficient for training.
        for week in range(1, 14):
            update_week(2023, week)

        features = compute_week_features(2023, 13)
        assert not features.empty

        train_weekly_models([(2023, week) for week in range(2, 13)])
        predictions = predict_week(2023, 13)
        assert not predictions.empty

        ranked = rank_weekly_value(2023, 13, min_edge=0.0)
        assert not ranked.empty

        materialize_week(2023, 13)
        with sqlite3.connect(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM materialized_value_view WHERE season=? AND week=?", (2023, 13)
            ).fetchone()[0]
        assert count > 0
    finally:
        config.database.path = original_path
        weekly_module.MODEL_DIR = original_model_dir
        Path(tmp_db).unlink(missing_ok=True)
        shutil.rmtree(tmp_model_dir, ignore_errors=True)
