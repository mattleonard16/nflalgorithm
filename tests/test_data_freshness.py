"""Data freshness: each pipeline table is compared with the NFL calendar."""

from __future__ import annotations

import sqlite3

import pytest

from schema_migrations import MigrationManager
from scripts.preflight import collect_diagnostics, main
from utils.data_freshness import check_data_freshness

# Week N kicks off 2026-09-(3 + 7N) 00:00 UTC; weeks 1-4 are scheduled.
KICKOFFS = {1: "2026-09-10", 2: "2026-09-17", 3: "2026-09-24", 4: "2026-10-01"}
# Week 3 is upcoming; weeks 1 and 2 ended more than 48h ago; odds are due (from 09-19).
NOW = "2026-09-22T12:00:00+00:00"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "fresh.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", path)
    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    MigrationManager(path).run()
    for week, day in KICKOFFS.items():
        _insert(path, "games", game_id=f"g{week}", season=2026, week=week,
                kickoff_utc=f"{day}T00:00:00+00:00")
    return path


def _insert(path: str, table: str, **values) -> None:
    """Insert a row, filling any other NOT NULL column without a default."""
    with sqlite3.connect(path) as conn:
        for _, name, kind, notnull, default, _ in conn.execute(f"PRAGMA table_info({table})"):
            if name not in values and notnull and default is None:
                values[name] = 0 if kind.upper().startswith(("INT", "REAL")) else "x"
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))


def _fill_fresh(path: str) -> None:
    for week in (1, 2):
        _insert(path, "player_stats_enhanced", player_id=f"p{week}", season=2026, week=week)
        _insert(path, "materialized_value_view", player_id=f"p{week}", season=2026, week=week)
        _insert(path, "bet_outcomes", bet_id=f"b{week}", season=2026, week=week)
    _insert(path, "weekly_projections", player_id="p3", season=2026, week=3)
    _insert(path, "weekly_odds", player_id="p3", season=2026, week=3,
            as_of="2026-09-22T06:00:00+00:00")
    _insert(path, "materialized_value_view", player_id="p3", season=2026, week=3)
    _insert(path, "pipeline_runs", run_id="r1", season=2026, week=2, status="completed",
            started_at="2026-09-20T00:00:00+00:00", finished_at="2026-09-20T00:05:00+00:00")


def _by_table(now=NOW):
    return {r.table: r for r in check_data_freshness(now=now)}


def test_fresh_pipeline_passes_every_table(db):
    _fill_fresh(db)
    assert {t: r.status for t, r in _by_table().items()} == {
        t: "pass" for t in ("stats", "projections", "odds", "card", "grading", "runs")
    }


def test_stats_missing_for_a_completed_week_fails_naming_the_week(db):
    _fill_fresh(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM player_stats_enhanced WHERE week = 2")
    result = _by_table()["stats"]
    assert result.status == "fail"
    assert "week(s) 2" in result.reason


def test_projections_missing_for_upcoming_week_fails(db):
    _fill_fresh(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM weekly_projections")
    assert _by_table()["projections"].status == "fail"


def test_odds_older_than_24h_fail_with_the_age(db):
    _fill_fresh(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE weekly_odds SET as_of = '2026-09-20T06:00:00+00:00'")
    result = _by_table()["odds"]
    assert result.status == "fail"
    assert "2d 6h old" in result.reason


def test_odds_not_yet_due_skips_instead_of_failing(db):
    # Week 3 props are not expected before 2026-09-19.
    assert _by_table(now="2026-09-17T12:00:00+00:00")["odds"].status == "skip"


def test_card_missing_once_odds_exist_fails(db):
    _fill_fresh(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM materialized_value_view WHERE week = 3")
    assert _by_table()["card"].status == "fail"


def test_completed_week_with_a_card_but_no_outcomes_fails_grading(db):
    _fill_fresh(db)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM bet_outcomes WHERE week = 1")
    result = _by_table()["grading"]
    assert result.status == "fail"
    assert "week(s) 1" in result.reason


def test_latest_completed_run_older_than_8_days_fails(db):
    _fill_fresh(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE pipeline_runs SET finished_at = '2026-09-10T00:00:00+00:00'")
    assert _by_table()["runs"].status == "fail"


def test_offseason_skips_every_table(db):
    results = check_data_freshness(now="2027-03-01T00:00:00+00:00")
    assert {r.status for r in results} == {"skip"}


def test_preflight_exits_nonzero_on_a_stale_table(db, monkeypatch, capsys):
    # Empty tables with a live schedule: the real clock is past this fixture
    # schedule, so pin the clock the module reads.
    import utils.data_freshness as module

    real = module.check_data_freshness
    monkeypatch.setattr(module, "check_data_freshness", lambda: real(now=NOW))
    assert main(["--check-freshness"]) == 1
    assert "[FAIL] freshness_stats" in capsys.readouterr().out


def test_preflight_exits_zero_when_every_table_is_fresh(db, monkeypatch):
    _fill_fresh(db)
    import utils.data_freshness as module

    real = module.check_data_freshness
    monkeypatch.setattr(module, "check_data_freshness", lambda: real(now=NOW))
    diagnostics = collect_diagnostics(check_freshness=True)
    freshness = [d for d in diagnostics if d.name.startswith("freshness_")]
    assert len(freshness) == 6 and not any(d.failed for d in freshness)
