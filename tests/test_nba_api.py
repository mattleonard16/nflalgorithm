"""Tests for NBA API router endpoints (api/nba_router.py).

Uses FastAPI TestClient with a fresh SQLite DB. No live NBA.com calls —
the live schedule fetch is mocked.
"""

from __future__ import annotations

import pytest

from schema_migrations import MigrationManager
from utils.db import execute, executemany


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_api.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


@pytest.fixture()
def client(db):
    from fastapi.testclient import TestClient
    from api.server import app

    return TestClient(app)


@pytest.fixture()
def mock_schedule(monkeypatch):
    """Patch the live schedule fetch to avoid hitting NBA.com."""
    fake_games = [
        {
            "game_id": "0022500001",
            "game_date": "2026-02-17",
            "home_team": "BOS",
            "away_team": "MIA",
            "status": "7:30 pm ET",
            "home_score": None,
            "away_score": None,
        },
        {
            "game_id": "0022500002",
            "game_date": "2026-02-17",
            "home_team": "LAL",
            "away_team": "GSW",
            "status": "10:00 pm ET",
            "home_score": None,
            "away_score": None,
        },
    ]
    import api.nba_router as router_mod

    monkeypatch.setattr(router_mod, "_fetch_live_schedule", lambda: fake_games)
    return fake_games


def _seed_game_logs() -> None:
    rows = [
        (1628369, "Jayson Tatum", "BOS", 2024, "002240001", "2025-01-10", "BOS vs. MIA", "W",
         36.0, 31, 8, 5, 4, 11, 20, 5, 6, 1, 0, 2, 12.0),
        (1628369, "Jayson Tatum", "BOS", 2024, "002240002", "2025-01-12", "BOS @ PHI", "L",
         34.0, 24, 6, 4, 3, 9, 18, 3, 4, 0, 1, 3, -5.0),
        (1628384, "Jaylen Brown", "BOS", 2024, "002240001", "2025-01-10", "BOS vs. MIA", "W",
         34.0, 22, 5, 3, 2, 9, 18, 2, 3, 2, 1, 1, 8.0),
    ]
    executemany(
        """INSERT OR REPLACE INTO nba_player_game_logs (
            player_id, player_name, team_abbreviation, season,
            game_id, game_date, matchup, wl, min,
            pts, reb, ast, fg3m, fgm, fga, ftm, fta,
            stl, blk, tov, plus_minus
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


def _seed_projection(game_date: str = "2026-02-17") -> None:
    execute(
        """INSERT OR REPLACE INTO nba_projections
           (player_id, player_name, team, season, game_date, game_id, market, projected_value, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        params=(1628369, "Jayson Tatum", "BOS", 2025, game_date, "0022500001", "pts", 28.5, 0.82),
    )


class TestNbaMeta:
    def test_meta_returns_ok(self, client, db):
        resp = client.get("/api/nba/meta")
        assert resp.status_code == 200
        data = resp.json()
        assert "available_seasons" in data
        assert "total_players" in data
        assert "total_games" in data

    def test_meta_counts_after_seed(self, client, db):
        _seed_game_logs()
        resp = client.get("/api/nba/meta")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_players"] == 2
        assert data["total_games"] == 2  # 2 distinct game_ids


class TestNbaSchedule:
    def test_schedule_returns_games(self, client, db, mock_schedule):
        resp = client.get("/api/nba/schedule")
        assert resp.status_code == 200
        data = resp.json()
        assert "games" in data
        assert "game_date" in data
        assert len(data["games"]) == 2

    def test_schedule_game_shape(self, client, db, mock_schedule):
        resp = client.get("/api/nba/schedule")
        game = resp.json()["games"][0]
        assert "game_id" in game
        assert "home_team" in game
        assert "away_team" in game


class TestNbaProjections:
    def test_projections_returns_empty_list_no_data(self, client, db):
        resp = client.get("/api/nba/projections", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["projections"] == []

    def test_projections_returns_seeded_data(self, client, db, mock_schedule):
        _seed_game_logs()
        _seed_projection()
        resp = client.get("/api/nba/projections", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_players"] == 1
        proj = data["projections"][0]
        assert proj["player_name"] == "Jayson Tatum"
        assert proj["projected_value"] == pytest.approx(28.5)
        assert proj["market"] == "pts"
        assert proj["team"] == "BOS"

    def test_projections_shape(self, client, db, mock_schedule):
        _seed_game_logs()
        _seed_projection()
        resp = client.get("/api/nba/projections", params={"game_date": "2026-02-17"})
        proj = resp.json()["projections"][0]
        expected_fields = {
            "player_id", "player_name", "team", "game_date",
            "market", "projected_value", "confidence",
        }
        assert expected_fields.issubset(proj.keys())

    def test_projections_bad_date_returns_400(self, client, db):
        resp = client.get("/api/nba/projections", params={"game_date": "not-a-date"})
        assert resp.status_code == 400

    def test_projections_matchup_enriched_from_schedule(self, client, db, mock_schedule):
        _seed_game_logs()
        _seed_projection()
        resp = client.get("/api/nba/projections", params={"game_date": "2026-02-17"})
        proj = resp.json()["projections"][0]
        # BOS is the home team in the mock schedule for game 0022500001
        assert proj["matchup"] is not None
        assert "BOS" in proj["matchup"]


class TestNbaPlayers:
    def test_players_returns_ok(self, client, db):
        resp = client.get("/api/nba/players")
        assert resp.status_code == 200
        data = resp.json()
        assert "players" in data
        assert "total" in data

    def test_players_after_seed(self, client, db):
        _seed_game_logs()
        resp = client.get("/api/nba/players", params={"season": 2024})
        data = resp.json()
        assert data["total"] == 2

    def test_players_search_filter(self, client, db):
        _seed_game_logs()
        resp = client.get("/api/nba/players", params={"season": 2024, "search": "tatum"})
        data = resp.json()
        assert data["total"] == 1
        assert data["players"][0]["player_name"] == "Jayson Tatum"

    def test_players_avg_pts_correct(self, client, db):
        _seed_game_logs()
        resp = client.get("/api/nba/players", params={"season": 2024, "search": "tatum"})
        tatum = resp.json()["players"][0]
        # Tatum has two games: 31 and 24 pts → avg = 27.5
        assert tatum["avg_pts"] == pytest.approx(27.5)
        assert tatum["games_played"] == 2

    def test_players_sorted_by_pts_desc(self, client, db):
        _seed_game_logs()
        resp = client.get("/api/nba/players", params={"season": 2024})
        players = resp.json()["players"]
        pts_values = [p["avg_pts"] for p in players]
        assert pts_values == sorted(pts_values, reverse=True)


class TestNbaHealth:
    def test_health_returns_ok(self, client, db):
        resp = client.get("/api/nba/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "game_logs" in data
        assert "projections" in data

    def test_health_shows_zero_before_ingest(self, client, db):
        resp = client.get("/api/nba/health")
        data = resp.json()
        assert data["game_logs"]["total_rows"] == 0
        assert data["game_logs"]["total_players"] == 0

    def test_health_after_seed(self, client, db):
        _seed_game_logs()
        resp = client.get("/api/nba/health")
        data = resp.json()
        assert data["game_logs"]["total_rows"] == 3
        assert data["game_logs"]["latest_game_date"] == "2025-01-12"


def _seed_value_bet(
    game_date: str = "2026-02-17",
    player_id: int = 1628369,
    player_name: str = "Jayson Tatum",
    team: str = "BOS",
    market: str = "pts",
    sportsbook: str = "DraftKings",
    edge_percentage: float = 0.12,
    event_id: str = "dk-tatum-pts-20260217",
) -> None:
    execute(
        """INSERT OR REPLACE INTO nba_materialized_value_view (
            season, game_date, player_id, player_name, team, event_id,
            market, sportsbook, line, over_price, under_price,
            mu, sigma, p_win, edge_percentage, expected_roi,
            kelly_fraction, confidence, generated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        params=(
            2025, game_date, player_id, player_name, team, event_id,
            market, sportsbook, 27.5, -115, -105,
            31.2, 5.1, 0.62, edge_percentage, 0.08,
            0.05, 0.82, "2026-02-17T08:00:00",
        ),
    )


def _seed_value_bet_second(game_date: str = "2026-02-17") -> None:
    """Seed a second value bet with lower edge and different sportsbook."""
    execute(
        """INSERT OR REPLACE INTO nba_materialized_value_view (
            season, game_date, player_id, player_name, team, event_id,
            market, sportsbook, line, over_price, under_price,
            mu, sigma, p_win, edge_percentage, expected_roi,
            kelly_fraction, confidence, generated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        params=(
            2025, game_date, 1628384, "Jaylen Brown", "BOS",
            "fd-brown-pts-20260217",
            "pts", "FanDuel", 22.5, -110, -110,
            24.0, 4.8, 0.58, 0.07, 0.04,
            0.03, 0.75, "2026-02-17T08:00:00",
        ),
    )


class TestNbaValueBets:
    def test_value_bets_empty(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["bets"] == []
        assert data["total"] == 0
        assert data["game_date"] == "2026-02-17"
        assert "filters" in data

    def test_value_bets_returns_seeded_data(self, client, db):
        _seed_value_bet()
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        bet = data["bets"][0]
        assert bet["player_name"] == "Jayson Tatum"
        assert bet["market"] == "pts"
        assert bet["sportsbook"] == "DraftKings"
        assert bet["edge_percentage"] == pytest.approx(0.12)
        assert bet["mu"] == pytest.approx(31.2)

    def test_value_bets_shape(self, client, db):
        _seed_value_bet()
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        expected_fields = {
            "player_id", "player_name", "team", "event_id", "market",
            "sportsbook", "line", "over_price", "under_price",
            "mu", "sigma", "p_win", "edge_percentage", "expected_roi",
            "kelly_fraction", "confidence", "generated_at",
        }
        assert expected_fields.issubset(bet.keys())

    def test_value_bets_invalid_market_returns_400(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"market": "invalid"})
        assert resp.status_code == 400

    def test_value_bets_bad_date_returns_400(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "not-a-date"})
        assert resp.status_code == 400

    def test_value_bets_min_edge_filter(self, client, db):
        _seed_value_bet(edge_percentage=0.12)
        _seed_value_bet_second()  # 0.07 edge
        resp = client.get(
            "/api/nba/value-bets",
            params={"game_date": "2026-02-17", "min_edge": 0.10},
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["bets"][0]["player_name"] == "Jayson Tatum"

    def test_value_bets_sportsbook_filter(self, client, db):
        _seed_value_bet()
        _seed_value_bet_second()
        resp = client.get(
            "/api/nba/value-bets",
            params={"game_date": "2026-02-17", "sportsbook": "FanDuel"},
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["bets"][0]["player_name"] == "Jaylen Brown"

    def test_value_bets_sorted_by_edge_desc(self, client, db):
        _seed_value_bet(edge_percentage=0.12)
        _seed_value_bet_second()  # 0.07 edge
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        edges = [b["edge_percentage"] for b in resp.json()["bets"]]
        assert edges == sorted(edges, reverse=True)

    def test_value_bets_best_line_only(self, client, db):
        # Two bets for same player+market, different sportsbooks
        _seed_value_bet(edge_percentage=0.12, sportsbook="DraftKings")
        execute(
            """INSERT OR REPLACE INTO nba_materialized_value_view (
                season, game_date, player_id, player_name, team, event_id,
                market, sportsbook, line, over_price, under_price,
                mu, sigma, p_win, edge_percentage, expected_roi,
                kelly_fraction, confidence, generated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            params=(
                2025, "2026-02-17", 1628369, "Jayson Tatum", "BOS",
                "fd-tatum-pts-20260217",
                "pts", "FanDuel", 27.5, -110, -110,
                31.2, 5.1, 0.62, 0.09, 0.06,
                0.04, 0.82, "2026-02-17T08:00:00",
            ),
        )
        resp = client.get(
            "/api/nba/value-bets",
            params={"game_date": "2026-02-17", "best_line_only": "true"},
        )
        data = resp.json()
        # Should collapse two Tatum rows to one (highest edge wins)
        assert data["total"] == 1
        assert data["bets"][0]["sportsbook"] == "DraftKings"
        assert data["bets"][0]["edge_percentage"] == pytest.approx(0.12)

    def test_value_bets_filters_echoed_in_response(self, client, db):
        resp = client.get(
            "/api/nba/value-bets",
            params={"market": "reb", "min_edge": 0.05, "best_line_only": "true"},
        )
        filters = resp.json()["filters"]
        assert filters["market"] == "reb"
        assert filters["min_edge"] == pytest.approx(0.05)
        assert filters["best_line_only"] is True

    def test_value_bets_defaults_to_today(self, client, db):
        resp = client.get("/api/nba/value-bets")
        assert resp.status_code == 200
        data = resp.json()
        from datetime import date
        assert data["game_date"] == date.today().isoformat()
