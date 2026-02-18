"""Contract tests for the NBA API router (api/nba_router.py).

Validates OpenAPI schema presence, response envelope shapes, and field-level
contracts for every NBA endpoint.  No live NBA.com calls are made — the live
schedule helper is patched out for every test that touches /projections or
/schedule.
"""

from __future__ import annotations

import pytest

from schema_migrations import MigrationManager
from utils.db import execute


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_contract.db")
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
    """Suppress live NBA.com calls by patching _fetch_live_schedule."""
    import api.nba_router as router_mod

    monkeypatch.setattr(router_mod, "_fetch_live_schedule", lambda: [])


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_value_view(db: str) -> None:
    execute(
        "INSERT INTO nba_materialized_value_view "
        "(season, game_date, player_id, player_name, team, event_id, market, sportsbook, "
        "line, over_price, under_price, mu, sigma, p_win, edge_percentage, expected_roi, "
        "kelly_fraction, confidence, generated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(
            2025, "2026-02-17", 1234, "Test Player", "LAL", "evt1", "pts", "draftkings",
            25.5, -110, 110, 28.0, 3.0, 0.65, 12.0, 0.10, 0.02, 0.85,
            "2026-02-17T00:00:00",
        ),
    )


def _seed_daily_performance(db: str) -> None:
    execute(
        "INSERT INTO nba_daily_performance "
        "(season, game_date, total_bets, wins, losses, pushes, profit_units, roi_pct, "
        "avg_edge, best_bet, worst_bet, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(
            2025, "2026-02-17", 10, 6, 3, 1, 2.5, 25.0, 10.0,
            "LeBron pts over 25.5", "AD reb over 10.5", "2026-02-17T12:00:00",
        ),
    )


def _seed_bet_outcome(db: str) -> None:
    execute(
        "INSERT INTO nba_bet_outcomes "
        "(bet_id, season, game_date, player_id, player_name, market, sportsbook, side, "
        "line, price, actual_result, result, profit_units, confidence_tier, "
        "edge_at_placement, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=(
            "bet-001", 2025, "2026-02-17", 1234, "Test Player", "pts", "draftkings",
            "over", 25.5, -110, 30.0, "win", 0.91, "MEDIUM", 12.0,
            "2026-02-17T12:00:00",
        ),
    )


# ---------------------------------------------------------------------------
# 1. TestNbaOpenAPIContract
# ---------------------------------------------------------------------------


class TestNbaOpenAPIContract:
    """Verify that every NBA endpoint path is registered in the OpenAPI schema."""

    EXPECTED_NBA_PATHS = [
        "/api/nba/meta",
        "/api/nba/schedule",
        "/api/nba/projections",
        "/api/nba/players",
        "/api/nba/value-bets",
        "/api/nba/health",
        "/api/nba/performance",
        "/api/nba/outcomes",
        "/api/nba/explain/{player_id}/{market}",
        "/api/nba/analytics/correlation",
        "/api/nba/analytics/risk-summary",
    ]

    def test_openapi_json_is_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_json_contains_all_nba_paths(self, client):
        schema = client.get("/openapi.json").json()
        registered_paths = list(schema.get("paths", {}).keys())
        for expected_path in self.EXPECTED_NBA_PATHS:
            assert expected_path in registered_paths, (
                f"NBA path '{expected_path}' not found in OpenAPI schema. "
                f"Registered paths: {registered_paths}"
            )

    def test_openapi_nba_paths_have_get_operations(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        for expected_path in self.EXPECTED_NBA_PATHS:
            assert "get" in paths.get(expected_path, {}), (
                f"Path '{expected_path}' does not expose a GET operation."
            )

    def test_openapi_nba_paths_have_tags(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        for expected_path in self.EXPECTED_NBA_PATHS:
            operation = paths.get(expected_path, {}).get("get", {})
            tags = operation.get("tags", [])
            assert "nba" in tags, (
                f"Path '{expected_path}' is missing the 'nba' tag. Got: {tags}"
            )


# ---------------------------------------------------------------------------
# 2. TestNbaValueBetsContract
# ---------------------------------------------------------------------------


class TestNbaValueBetsContract:
    """Verify the /api/nba/value-bets response envelope and field-level contract."""

    _REQUIRED_ENVELOPE_KEYS = {"bets", "total", "game_date", "filters"}
    _REQUIRED_BET_KEYS = {
        "player_name",
        "market",
        "sportsbook",
        "line",
        "over_price",
        "mu",
        "sigma",
        "p_win",
        "edge_percentage",
        "expected_roi",
        "kelly_fraction",
    }

    def test_empty_response_envelope_shape(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200
        data = resp.json()
        assert self._REQUIRED_ENVELOPE_KEYS.issubset(data.keys()), (
            f"Missing envelope keys: {self._REQUIRED_ENVELOPE_KEYS - data.keys()}"
        )

    def test_empty_response_has_correct_types(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert isinstance(data["bets"], list)
        assert isinstance(data["total"], int)
        assert isinstance(data["game_date"], str)
        assert isinstance(data["filters"], dict)

    def test_empty_response_bets_is_empty_list(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert data["bets"] == []
        assert data["total"] == 0

    def test_empty_response_game_date_echoed(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert data["game_date"] == "2026-02-17"

    def test_seeded_response_envelope_shape(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200
        data = resp.json()
        assert self._REQUIRED_ENVELOPE_KEYS.issubset(data.keys())

    def test_seeded_response_total_matches_bets_length(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert data["total"] == len(data["bets"])
        assert data["total"] == 1

    def test_seeded_bet_has_required_fields(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        missing = self._REQUIRED_BET_KEYS - bet.keys()
        assert not missing, f"Bet is missing required fields: {missing}"

    def test_seeded_bet_player_name_is_string(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["player_name"], str)
        assert bet["player_name"] == "Test Player"

    def test_seeded_bet_market_is_string(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["market"], str)
        assert bet["market"] == "pts"

    def test_seeded_bet_sportsbook_is_string(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["sportsbook"], str)
        assert bet["sportsbook"] == "draftkings"

    def test_seeded_bet_line_is_numeric(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["line"], (int, float))
        assert bet["line"] == pytest.approx(25.5)

    def test_seeded_bet_over_price_is_int(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["over_price"], int)
        assert bet["over_price"] == -110

    def test_seeded_bet_mu_is_numeric(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["mu"], (int, float))
        assert bet["mu"] == pytest.approx(28.0)

    def test_seeded_bet_sigma_is_numeric(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["sigma"], (int, float))
        assert bet["sigma"] == pytest.approx(3.0)

    def test_seeded_bet_p_win_is_numeric(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["p_win"], (int, float))
        assert bet["p_win"] == pytest.approx(0.65)

    def test_seeded_bet_edge_percentage_is_numeric(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["edge_percentage"], (int, float))
        assert bet["edge_percentage"] == pytest.approx(12.0)

    def test_seeded_bet_expected_roi_is_numeric(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["expected_roi"], (int, float))
        assert bet["expected_roi"] == pytest.approx(0.10)

    def test_seeded_bet_kelly_fraction_is_numeric(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert isinstance(bet["kelly_fraction"], (int, float))
        assert bet["kelly_fraction"] == pytest.approx(0.02)

    def test_filters_dict_contains_market(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17", "market": "pts"})
        filters = resp.json()["filters"]
        assert "market" in filters
        assert filters["market"] == "pts"

    def test_filters_dict_contains_min_edge(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17", "min_edge": 0.05})
        filters = resp.json()["filters"]
        assert "min_edge" in filters
        assert filters["min_edge"] == pytest.approx(0.05)

    def test_filters_dict_contains_best_line_only(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17", "best_line_only": "true"})
        filters = resp.json()["filters"]
        assert "best_line_only" in filters
        assert filters["best_line_only"] is True


# ---------------------------------------------------------------------------
# 3. TestNbaPerformanceContract
# ---------------------------------------------------------------------------


class TestNbaPerformanceContract:
    """Verify /api/nba/performance and /api/nba/outcomes response contracts."""

    _REQUIRED_PERFORMANCE_KEYS = {
        "total_bets",
        "total_wins",
        "total_losses",
        "total_profit",
        "overall_roi",
        "win_rate",
        "days",
    }

    _REQUIRED_DAY_KEYS = {
        "season",
        "game_date",
        "total_bets",
        "wins",
        "losses",
        "pushes",
        "profit_units",
        "roi_pct",
        "avg_edge",
    }

    _REQUIRED_OUTCOME_KEYS = {
        "bet_id",
        "market",
        "line",
    }

    # ------------------------------------------------------------------
    # /api/nba/performance — empty DB
    # ------------------------------------------------------------------

    def test_performance_returns_200_when_empty(self, client, db):
        resp = client.get("/api/nba/performance")
        assert resp.status_code == 200

    def test_performance_empty_has_required_keys(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        missing = self._REQUIRED_PERFORMANCE_KEYS - data.keys()
        assert not missing, f"Performance response missing keys: {missing}"

    def test_performance_empty_total_bets_is_zero(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert data["total_bets"] == 0

    def test_performance_empty_days_is_empty_list(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert data["days"] == []

    def test_performance_empty_totals_are_zero(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert data["total_wins"] == 0
        assert data["total_losses"] == 0
        assert data["total_profit"] == pytest.approx(0.0)
        assert data["overall_roi"] == pytest.approx(0.0)
        assert data["win_rate"] == pytest.approx(0.0)

    def test_performance_total_bets_is_int(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert isinstance(data["total_bets"], int)

    def test_performance_total_wins_is_int(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert isinstance(data["total_wins"], int)

    def test_performance_total_losses_is_int(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert isinstance(data["total_losses"], int)

    def test_performance_total_profit_is_numeric(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert isinstance(data["total_profit"], (int, float))

    def test_performance_overall_roi_is_numeric(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert isinstance(data["overall_roi"], (int, float))

    def test_performance_win_rate_is_numeric(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert isinstance(data["win_rate"], (int, float))

    def test_performance_days_is_list(self, client, db):
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert isinstance(data["days"], list)

    # ------------------------------------------------------------------
    # /api/nba/performance — seeded DB
    # ------------------------------------------------------------------

    def test_performance_seeded_returns_correct_totals(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_bets"] == 10
        assert data["total_wins"] == 6
        assert data["total_losses"] == 3

    def test_performance_seeded_total_profit_correct(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert data["total_profit"] == pytest.approx(2.5)

    def test_performance_seeded_roi_is_nonzero(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        data = resp.json()
        # 6 wins, 3 losses → units_risked = 9, profit = 2.5 → roi = 2.5/9 * 100
        assert data["overall_roi"] == pytest.approx(2.5 / 9 * 100, rel=1e-2)

    def test_performance_seeded_win_rate_correct(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        data = resp.json()
        # 6 wins, 3 losses → win_rate = 6/9 * 100
        assert data["win_rate"] == pytest.approx(6 / 9 * 100, rel=1e-2)

    def test_performance_seeded_days_has_one_entry(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        data = resp.json()
        assert len(data["days"]) == 1

    def test_performance_seeded_day_has_required_keys(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        day = resp.json()["days"][0]
        missing = self._REQUIRED_DAY_KEYS - day.keys()
        assert not missing, f"Day entry missing keys: {missing}"

    def test_performance_seeded_day_values_correct(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        day = resp.json()["days"][0]
        assert day["season"] == 2025
        assert day["game_date"] == "2026-02-17"
        assert day["total_bets"] == 10
        assert day["wins"] == 6
        assert day["losses"] == 3
        assert day["pushes"] == 1
        assert day["profit_units"] == pytest.approx(2.5)
        assert day["roi_pct"] == pytest.approx(25.0)
        assert day["avg_edge"] == pytest.approx(10.0)

    def test_performance_seeded_day_best_bet_present(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        day = resp.json()["days"][0]
        assert day.get("best_bet") == "LeBron pts over 25.5"

    def test_performance_seeded_day_worst_bet_present(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance")
        day = resp.json()["days"][0]
        assert day.get("worst_bet") == "AD reb over 10.5"

    def test_performance_season_filter_returns_matching_only(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance", params={"season": 2025})
        data = resp.json()
        assert data["total_bets"] == 10

    def test_performance_season_filter_no_match_returns_empty(self, client, db):
        _seed_daily_performance(db)
        resp = client.get("/api/nba/performance", params={"season": 2099})
        data = resp.json()
        assert data["total_bets"] == 0
        assert data["days"] == []

    # ------------------------------------------------------------------
    # /api/nba/outcomes — seeded DB
    # ------------------------------------------------------------------

    def test_outcomes_empty_returns_list(self, client, db):
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_outcomes_missing_game_date_returns_422(self, client, db):
        resp = client.get("/api/nba/outcomes")
        assert resp.status_code == 422

    def test_outcomes_bad_date_format_returns_400(self, client, db):
        resp = client.get("/api/nba/outcomes", params={"game_date": "not-a-date"})
        assert resp.status_code == 400

    def test_outcomes_seeded_returns_one_entry(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_outcomes_seeded_has_required_fields(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        missing = self._REQUIRED_OUTCOME_KEYS - outcome.keys()
        assert not missing, f"Outcome entry missing keys: {missing}"

    def test_outcomes_seeded_bet_id_correct(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        assert outcome["bet_id"] == "bet-001"

    def test_outcomes_seeded_player_name_correct(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        assert outcome["player_name"] == "Test Player"

    def test_outcomes_seeded_market_correct(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        assert outcome["market"] == "pts"

    def test_outcomes_seeded_line_correct(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        assert outcome["line"] == pytest.approx(25.5)

    def test_outcomes_seeded_actual_result_correct(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        assert outcome["actual_result"] == pytest.approx(30.0)

    def test_outcomes_seeded_result_correct(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        assert outcome["result"] == "win"

    def test_outcomes_seeded_profit_units_correct(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        assert outcome["profit_units"] == pytest.approx(0.91)

    def test_outcomes_seeded_confidence_tier_correct(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-17"})
        outcome = resp.json()[0]
        assert outcome["confidence_tier"] == "MEDIUM"

    def test_outcomes_date_isolation(self, client, db):
        _seed_bet_outcome(db)
        resp = client.get("/api/nba/outcomes", params={"game_date": "2026-02-18"})
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# 4. TestNbaValueBetsIncludeWhy
# ---------------------------------------------------------------------------


class TestNbaValueBetsIncludeWhy:
    """Verify the include_why parameter on /api/nba/value-bets."""

    def test_default_no_why(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17"})
        bet = resp.json()["bets"][0]
        assert bet.get("why") is None

    def test_include_why_false_no_why(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17", "include_why": "false"})
        bet = resp.json()["bets"][0]
        assert bet.get("why") is None

    def test_include_why_true_returns_payload(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17", "include_why": "true"})
        assert resp.status_code == 200
        bet = resp.json()["bets"][0]
        assert bet["why"] is not None
        assert isinstance(bet["why"], dict)

    def test_include_why_payload_has_six_sections(self, client, db):
        _seed_value_view(db)
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17", "include_why": "true"})
        why = resp.json()["bets"][0]["why"]
        expected = {"model", "recency", "variance", "confidence", "risk", "agents"}
        assert expected.issubset(why.keys()), f"Missing sections: {expected - why.keys()}"

    def test_include_why_empty_bets_returns_empty(self, client, db):
        resp = client.get("/api/nba/value-bets", params={"game_date": "2026-02-17", "include_why": "true"})
        assert resp.status_code == 200
        assert resp.json()["bets"] == []


# ---------------------------------------------------------------------------
# 5. TestNbaExplainEndpoint
# ---------------------------------------------------------------------------


class TestNbaExplainEndpoint:
    """Verify /api/nba/explain/{player_id}/{market}."""

    def test_returns_200(self, client, db):
        resp = client.get("/api/nba/explain/1234/pts", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200

    def test_response_envelope_shape(self, client, db):
        resp = client.get("/api/nba/explain/1234/pts", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert data["player_id"] == 1234
        assert data["market"] == "pts"
        assert data["game_date"] == "2026-02-17"
        assert "why" in data

    def test_why_has_six_sections(self, client, db):
        resp = client.get("/api/nba/explain/1234/pts", params={"game_date": "2026-02-17"})
        why = resp.json()["why"]
        expected = {"model", "recency", "variance", "confidence", "risk", "agents"}
        assert expected == set(why.keys())

    def test_invalid_market_returns_400(self, client, db):
        resp = client.get("/api/nba/explain/1234/invalid_market", params={"game_date": "2026-02-17"})
        assert resp.status_code == 400

    def test_invalid_date_returns_400(self, client, db):
        resp = client.get("/api/nba/explain/1234/pts", params={"game_date": "not-a-date"})
        assert resp.status_code == 400

    def test_defaults_to_today_when_no_date(self, client, db):
        resp = client.get("/api/nba/explain/1234/pts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["game_date"] is not None


# ---------------------------------------------------------------------------
# 6. TestNbaCorrelationEndpoint
# ---------------------------------------------------------------------------


def _seed_risk_assessment(db: str) -> None:
    execute(
        "INSERT INTO nba_risk_assessments "
        "(game_date, player_id, market, sportsbook, event_id, "
        "correlation_group, exposure_warning, risk_adjusted_kelly, "
        "mean_drawdown, max_drawdown, p95_drawdown, assessed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params=("2026-02-17", 1234, "pts", "draftkings", "evt1",
                "LAL_pts_fg3m", None, 0.04, 0.05, 0.12, 0.09,
                "2026-02-17T00:00:00"),
    )


class TestNbaCorrelationEndpoint:
    """Verify /api/nba/analytics/correlation."""

    def test_returns_200(self, client, db):
        resp = client.get("/api/nba/analytics/correlation", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200

    def test_empty_response_shape(self, client, db):
        resp = client.get("/api/nba/analytics/correlation", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert data["game_date"] == "2026-02-17"
        assert isinstance(data["correlation_groups"], dict)
        assert isinstance(data["team_stacks"], dict)

    def test_with_correlation_data(self, client, db):
        _seed_risk_assessment(db)
        resp = client.get("/api/nba/analytics/correlation", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert "LAL_pts_fg3m" in data["correlation_groups"]
        members = data["correlation_groups"]["LAL_pts_fg3m"]
        assert len(members) == 1
        assert members[0]["player_id"] == 1234
        assert members[0]["market"] == "pts"

    def test_team_stacks_with_multiple_bets(self, client, db):
        # Seed two bets for same team
        _seed_value_view(db)
        execute(
            "INSERT INTO nba_materialized_value_view "
            "(season, game_date, player_id, player_name, team, event_id, market, sportsbook, "
            "line, over_price, under_price, mu, sigma, p_win, edge_percentage, expected_roi, "
            "kelly_fraction, confidence, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params=(
                2025, "2026-02-17", 5678, "Other LAL", "LAL", "evt2", "reb", "draftkings",
                10.5, -110, 110, 12.0, 2.0, 0.60, 8.0, 0.06, 0.01, 0.75,
                "2026-02-17T00:00:00",
            ),
        )
        resp = client.get("/api/nba/analytics/correlation", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert "LAL" in data["team_stacks"]
        assert data["team_stacks"]["LAL"] == 2

    def test_invalid_date_returns_400(self, client, db):
        resp = client.get("/api/nba/analytics/correlation", params={"game_date": "bad"})
        assert resp.status_code == 400

    def test_defaults_to_today_when_no_date(self, client, db):
        resp = client.get("/api/nba/analytics/correlation")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 7. TestNbaRiskSummaryEndpoint
# ---------------------------------------------------------------------------


class TestNbaRiskSummaryEndpoint:
    """Verify /api/nba/analytics/risk-summary."""

    def test_returns_200(self, client, db):
        resp = client.get("/api/nba/analytics/risk-summary", params={"game_date": "2026-02-17"})
        assert resp.status_code == 200

    def test_empty_response_shape(self, client, db):
        resp = client.get("/api/nba/analytics/risk-summary", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert data["game_date"] == "2026-02-17"
        assert data["total_assessed"] == 0
        assert data["correlated"] == 0
        assert data["exposure_flagged"] == 0
        assert data["avg_risk_adjusted_kelly"] is None
        assert data["avg_drawdown"] is None
        assert data["guardrails"] == []

    def test_with_risk_data_returns_counts(self, client, db):
        _seed_risk_assessment(db)
        resp = client.get("/api/nba/analytics/risk-summary", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert data["total_assessed"] == 1
        assert data["correlated"] == 1
        assert data["exposure_flagged"] == 0
        assert data["avg_risk_adjusted_kelly"] == pytest.approx(0.04)
        assert data["avg_drawdown"] == pytest.approx(0.05)

    def test_guardrails_populated_when_warnings(self, client, db):
        _seed_risk_assessment(db)
        resp = client.get("/api/nba/analytics/risk-summary", params={"game_date": "2026-02-17"})
        data = resp.json()
        assert len(data["guardrails"]) >= 1
        assert any("correlated" in g for g in data["guardrails"])

    def test_invalid_date_returns_400(self, client, db):
        resp = client.get("/api/nba/analytics/risk-summary", params={"game_date": "bad"})
        assert resp.status_code == 400

    def test_defaults_to_today_when_no_date(self, client, db):
        resp = client.get("/api/nba/analytics/risk-summary")
        assert resp.status_code == 200
