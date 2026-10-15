"""Agent and NBA ingest upserts must work on both backends.

`INSERT OR REPLACE` is SQLite-only; on the MySQL production backend it is a
syntax error. The coordinator used to swallow that error and report success,
so production `agent_decisions` stayed empty while the stage read "ok".
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

import learning_loop
from agents import AgentReport
from agents import coordinator
from config import config
from schema_migrations import MigrationManager
from scripts import ingest_nba_data
from utils.db import read_dataframe


@pytest.fixture()
def sqlite_db(tmp_path, monkeypatch) -> str:
    db_path = str(tmp_path / "agents.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setattr(config.database, "backend", "sqlite")
    monkeypatch.setattr(config.database, "path", db_path)
    MigrationManager(db_path).run()
    return db_path


def _decision(confidence: float, decision: str = "APPROVED") -> dict[str, Any]:
    return {
        "player_id": "BUF_some_player",
        "market": "receiving_yards",
        "decision": decision,
        "merged_confidence": confidence,
        "votes": {"APPROVE": 3, "REJECT": 1, "NEUTRAL": 0},
        "rationale": "test",
        "override": False,
        "agent_reports": [{"agent": "odds", "recommendation": "APPROVE", "confidence": 0.7}],
    }


# ------------------------------------------------------------- coordinator


def test_coordinator_upsert_overwrites_in_place_on_sqlite(sqlite_db) -> None:
    assert coordinator._persist_decisions([_decision(0.7)], 2026, 3) == 1
    assert coordinator._persist_decisions([_decision(0.9, "REJECTED")], 2026, 3) == 1

    rows = read_dataframe("SELECT decision, merged_confidence FROM agent_decisions")
    assert len(rows) == 1
    assert rows.iloc[0]["decision"] == "REJECTED"
    assert rows.iloc[0]["merged_confidence"] == pytest.approx(0.9)


def test_coordinator_uses_mysql_upsert_syntax(monkeypatch) -> None:
    monkeypatch.setattr(coordinator, "get_backend", lambda: "mysql")
    sql = coordinator._decision_upsert_sql()
    assert "INSERT OR" not in sql
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "decision = VALUES(decision)" in sql
    assert sql.count("?") == len(coordinator._DECISION_COLUMNS)


def test_coordinator_sqlite_sql_has_no_mysql_syntax(monkeypatch) -> None:
    monkeypatch.setattr(coordinator, "get_backend", lambda: "sqlite")
    sql = coordinator._decision_upsert_sql()
    assert "ON CONFLICT(season, week, player_id, market) DO UPDATE" in sql
    assert "DUPLICATE" not in sql


class _Conn:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    def __enter__(self) -> "_Conn":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_coordinator_persistence_failure_is_not_reported_as_success(monkeypatch) -> None:
    conn = _Conn()

    def boom(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("You have an error in your SQL syntax")

    monkeypatch.setattr(coordinator, "get_connection", lambda: conn)
    monkeypatch.setattr(coordinator, "execute", boom)

    with pytest.raises(RuntimeError, match="Failed to persist 1 agent decisions"):
        coordinator._persist_decisions([_decision(0.7)], 2026, 3)
    assert conn.rolled_back is True
    assert conn.committed is False


def _approving_agent() -> MagicMock:
    agent = MagicMock()
    agent.name = "mock"
    agent.analyze.return_value = [
        AgentReport(
            agent_name="mock",
            recommendation="APPROVE",
            confidence=0.75,
            rationale="test",
            player_id="P001",
            market="rushing_yards",
        )
    ]
    return agent


def test_agents_stage_reports_error_when_persistence_fails(monkeypatch) -> None:
    from scripts.production_runner import stage_agents

    for name in ("OddsAgent", "ModelDiagnosticsAgent", "MarketBiasAgent", "RiskAgent"):
        monkeypatch.setattr(coordinator, name, lambda: _approving_agent())

    def fail(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("Failed to persist agent decisions")

    monkeypatch.setattr(coordinator, "_persist_decisions", fail)

    result = stage_agents(2026, 3)
    assert result["status"] == "error"


# ------------------------------------------------------------ learning loop


def _seed_learning_inputs(monkeypatch) -> None:
    decisions = pd.DataFrame(
        [
            {
                "player_id": "BUF_some_player",
                "market": "receiving_yards",
                "decision": "APPROVED",
                "merged_confidence": 0.8,
                "votes": "{}",
                "agent_reports": '[{"agent": "odds", "recommendation": "APPROVE",'
                ' "confidence": 0.7}]',
            }
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "player_id": "BUF_some_player",
                "market": "receiving_yards",
                "result": "win",
                "profit_units": 0.9,
            }
        ]
    )
    monkeypatch.setattr(learning_loop, "_load_agent_decisions", lambda s, w: decisions)
    monkeypatch.setattr(learning_loop, "_load_bet_outcomes", lambda s, w: outcomes)


def test_learning_loop_upsert_runs_twice_on_sqlite(sqlite_db, monkeypatch) -> None:
    _seed_learning_inputs(monkeypatch)
    assert learning_loop.update_agent_performance(2026, 3) == 1
    assert learning_loop.update_agent_performance(2026, 3) == 1
    rows = read_dataframe("SELECT agent_name, correct FROM agent_performance")
    assert len(rows) == 1
    assert int(rows.iloc[0]["correct"]) == 1


def test_learning_loop_uses_mysql_upsert_syntax(monkeypatch) -> None:
    _seed_learning_inputs(monkeypatch)
    captured: list[str] = []
    monkeypatch.setattr(learning_loop, "get_backend", lambda: "mysql")
    monkeypatch.setattr(learning_loop, "executemany", lambda sql, rows: captured.append(sql))

    learning_loop.update_agent_performance(2026, 3)

    assert len(captured) == 1
    assert "INSERT OR" not in captured[0]
    assert "ON DUPLICATE KEY UPDATE" in captured[0]
    assert captured[0].count("?") == 12


# ------------------------------------------------------------- NBA ingest


def test_nba_upsert_sql_per_dialect(monkeypatch) -> None:
    columns = ("team_abbreviation", "season", "as_of_date", "def_rating")
    key = ("team_abbreviation", "season", "as_of_date")

    monkeypatch.setattr(ingest_nba_data, "get_backend", lambda: "mysql")
    mysql_sql = ingest_nba_data._upsert_sql("nba_team_defensive_stats", columns, key)
    assert "INSERT OR" not in mysql_sql
    assert mysql_sql.endswith("ON DUPLICATE KEY UPDATE def_rating = VALUES(def_rating)")

    monkeypatch.setattr(ingest_nba_data, "get_backend", lambda: "sqlite")
    sqlite_sql = ingest_nba_data._upsert_sql("nba_team_defensive_stats", columns, key)
    assert "ON CONFLICT(team_abbreviation, season, as_of_date) DO UPDATE" in sqlite_sql


def test_nba_game_log_upsert_runs_twice_on_sqlite(sqlite_db) -> None:
    row = {
        "player_id": 201939,
        "player_name": "Some Guard",
        "team_abbreviation": "GSW",
        "season": 2025,
        "game_id": "0022500001",
        "game_date": "2025-10-21",
        "matchup": "GSW vs. LAL",
        "wl": "W",
        "min": 34.0,
        "pts": 30,
        "reb": 5,
        "ast": 6,
        "fg3m": 5,
        "fgm": 10,
        "fga": 20,
        "ftm": 5,
        "fta": 5,
        "stl": 1,
        "blk": 0,
        "tov": 2,
        "plus_minus": 8,
    }
    assert ingest_nba_data._upsert_rows([row]) == 1
    assert ingest_nba_data._upsert_rows([{**row, "pts": 31}]) == 1
    rows = read_dataframe("SELECT pts FROM nba_player_game_logs")
    assert rows["pts"].tolist() == [31]
