"""Contract tests for API shape enforcement (Feature 2).

Ensures the API never breaks its contract with the frontend by
validating response schemas, field presence, and enum values.
"""

import pytest

from schema_migrations import MigrationManager
from utils.db import execute


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Provision a temporary SQLite database with schema."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")
    monkeypatch.setattr(cfg.config.api, "demo_mode", True)

    MigrationManager(db_path).run()
    return db_path


@pytest.fixture()
def client(db):
    from fastapi.testclient import TestClient

    from api.application import app
    from api.pipeline_router import require_pipeline_operator

    app.dependency_overrides[require_pipeline_operator] = lambda: "test-operator"
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _seed_value_bet(
    db,
    player_id="P001",
    season=2025,
    week=22,
    market="receiving_yards",
    sportsbook="draftkings",
    edge=0.15,
    confidence_tier="Premium",
):
    """Seed a materialized_value_view row and player_dim row."""
    execute(
        """
        INSERT INTO player_dim (player_id, player_name, position, team, last_season, last_week, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        params=(player_id, "Test Player", "WR", "KC", season, week),
    )
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at, confidence_score, confidence_tier)
        VALUES (?, ?, ?, 'evt1', 'KC', ?, ?,
                75.5, -110, 85.0, 8.0, 0.65, ?, 0.12,
                0.02, 20.0, datetime('now'), 0.82, ?)
        """,
        params=(season, week, player_id, market, sportsbook, edge, confidence_tier),
    )


class TestOpenAPIContract:
    def test_openapi_contains_expected_paths(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        openapi = resp.json()
        paths = openapi.get("paths", {})

        expected_paths = [
            "/api/value-bets",
            "/api/meta",
            "/api/performance",
            "/api/run/{run_id}",
            "/api/run/{run_id}/cancel",
            "/api/run/{run_id}/retry",
            "/api/system/architecture",
            "/api/explain/{player_id}/{market}",
            "/api/analytics/correlation",
            "/api/analytics/risk-summary",
            "/api/export/csv",
            "/api/export/bundle",
            "/api/run/{run_id}/review",
            "/api/run/{run_id}/review-status",
            "/api/projections",
            "/api/projections/weeks",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing path: {path}"


class TestValueBetsContract:
    def test_response_has_required_fields(self, client, db):
        _seed_value_bet(db)
        resp = client.get("/api/value-bets?season=2025&week=22")
        assert resp.status_code == 200
        data = resp.json()

        assert "bets" in data
        assert "total" in data
        assert "filters" in data
        assert isinstance(data["bets"], list)
        assert len(data["bets"]) > 0

        bet = data["bets"][0]
        required_fields = [
            "player_id",
            "player_name",
            "position",
            "market",
            "sportsbook",
            "line",
            "price",
            "mu",
            "sigma",
            "p_win",
            "edge_percentage",
            "expected_roi",
            "kelly_fraction",
            "stake",
        ]
        for field in required_fields:
            assert field in bet, f"Missing field: {field}"

    def test_player_name_and_position_present(self, client, db):
        _seed_value_bet(db)
        resp = client.get("/api/value-bets?season=2025&week=22")
        bet = resp.json()["bets"][0]
        assert bet["player_name"] == "Test Player"
        assert bet["position"] == "WR"

    def test_confidence_tier_enum_values(self, client, db):
        valid_tiers = {"Premium", "Strong", "Marginal", "Pass"}

        for tier in valid_tiers:
            _seed_value_bet(db, player_id=f"P_{tier}", confidence_tier=tier)

        resp = client.get("/api/value-bets?season=2025&week=22&min_edge=0")
        data = resp.json()

        tiers_found = {b["confidence_tier"] for b in data["bets"] if b["confidence_tier"]}
        assert tiers_found.issubset(valid_tiers)

    def test_empty_response_shape(self, client, db):
        resp = client.get("/api/value-bets?season=9999&week=99")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bets"] == []
        assert data["total"] == 0
        assert "filters" in data

    def test_best_line_only_returns_one_row_per_bet_ordered_by_edge(self, client, db):
        _seed_value_bet(db, player_id="P001", week=21, sportsbook="draftkings", edge=0.10)
        _seed_value_bet(db, player_id="P002", week=21, sportsbook="draftkings", edge=0.20)
        execute("""
            INSERT INTO materialized_value_view
                (season, week, player_id, event_id, team, market, sportsbook,
                 line, price, mu, sigma, p_win, edge_percentage, expected_roi,
                 kelly_fraction, stake, generated_at)
            VALUES (2025, 21, 'P001', 'evt1', 'KC', 'receiving_yards', 'fanduel',
                    74.5, -105, 85.0, 8.0, 0.66, 0.15, 0.13, 0.02, 20.0, datetime('now'))
            """)

        resp = client.get("/api/value-bets?season=2025&week=21&best_line_only=true")

        bets = [(b["player_id"], b["sportsbook"], b["line"]) for b in resp.json()["bets"]]
        assert bets == [("P002", "draftkings", 75.5), ("P001", "fanduel", 74.5)]

    def test_a_row_missing_a_text_field_does_not_fail_the_week(self, client, db):
        # pandas reads NULL text as NaN when other rows have a value, and the
        # response model rejects NaN, so one such row used to 500 the request.
        _seed_value_bet(db, player_id="P001", week=20)
        _seed_value_bet(db, player_id="P002", week=20, confidence_tier=None)

        resp = client.get("/api/value-bets?season=2025&week=20")

        assert resp.status_code == 200
        tiers = {b["player_id"]: b["confidence_tier"] for b in resp.json()["bets"]}
        assert tiers == {"P001": "Premium", "P002": None}

    def test_outcomes_with_an_ungraded_push_still_load(self, client, db):
        # A push for a player with no stats row stores NULL actual_result next
        # to real numbers. That NaN is not valid JSON and failed the request.
        for bet_id, actual in (("graded", 71.0), ("no-stats", None)):
            execute(
                """
                INSERT INTO bet_outcomes (
                    bet_id, season, week, player_id, market, sportsbook, side,
                    line, price, actual_result, result, profit_units, recorded_at
                ) VALUES (?, 2025, 19, 'P001', 'receiving_yards', 'draftkings', 'over',
                          64.5, -110, ?, 'win', 0.91, '2026-01-20T12:00:00Z')
                """,
                (bet_id, actual),
            )

        resp = client.get("/api/outcomes?season=2025&week=19")

        assert resp.status_code == 200
        results = {o["bet_id"]: o["actual_result"] for o in resp.json()["outcomes"]}
        assert results == {"graded": 71.0, "no-stats": None}

    def test_include_why_param(self, client, db):
        _seed_value_bet(db)
        resp = client.get("/api/value-bets?season=2025&week=22&include_why=true")
        assert resp.status_code == 200
        data = resp.json()
        # why should be present (may be null or dict)
        bet = data["bets"][0]
        assert "why" in bet


def _seed_other_book(player_id, week, sportsbook, edge):
    """Price an already seeded bet again at another book."""
    execute(
        """
        INSERT INTO materialized_value_view
            (season, week, player_id, event_id, team, market, sportsbook,
             line, price, mu, sigma, p_win, edge_percentage, expected_roi,
             kelly_fraction, stake, generated_at)
        VALUES (2025, ?, ?, 'evt1', 'KC', 'receiving_yards', ?,
                74.5, -105, 85.0, 8.0, 0.66, ?, 0.13, 0.02, 20.0, datetime('now'))
        """,
        (week, player_id, sportsbook, edge),
    )


class TestAnalyticsContract:
    """Chart panels describe bets, and a bet priced at two books is one bet."""

    @pytest.fixture()
    def two_bets_three_rows(self, db):
        _seed_value_bet(db, player_id="P001", week=18, edge=0.10)
        _seed_value_bet(db, player_id="P002", week=18, edge=0.20)
        _seed_other_book("P001", 18, "fanduel", 0.15)

    def test_by_market_counts_each_bet_once(self, client, two_bets_three_rows):
        data = client.get("/api/analytics/by-market?season=2025&week=18").json()

        assert [(m["market"], m["bet_count"]) for m in data["by_market"]] == [
            ("receiving_yards", 2)
        ]

    def test_by_position_counts_each_bet_once(self, client, two_bets_three_rows):
        data = client.get("/api/analytics/by-position?season=2025&week=18").json()

        assert [(p["position"], p["bet_count"]) for p in data["by_position"]] == [("WR", 2)]

    def test_edge_distribution_counts_each_bet_once(self, client, two_bets_three_rows):
        data = client.get("/api/analytics/edge-distribution?season=2025&week=18").json()

        assert sum(data["counts"]) == 2


class TestPerformanceContract:
    """Season totals weigh every settled bet equally, as each week's own ROI does."""

    @pytest.fixture()
    def a_big_week_and_a_small_one(self, db):
        rows = (
            # week, total_bets, wins, losses, pushes, profit_units, roi_pct
            (1, 100, 60, 40, 0, 10.0, 10.0),
            (2, 20, 0, 10, 10, -10.0, -100.0),
        )
        for week, total, wins, losses, pushes, profit, roi in rows:
            execute(
                "INSERT INTO weekly_performance (season, week, total_bets, wins, losses, "
                "pushes, profit_units, roi_pct, updated_at) "
                "VALUES (2025, ?, ?, ?, ?, ?, ?, ?, '2025-12-01T00:00:00Z')",
                (week, total, wins, losses, pushes, profit, roi),
            )

    def test_overall_roi_is_profit_over_settled_bets(self, client, a_big_week_and_a_small_one):
        data = client.get("/api/performance?season=2025").json()

        assert data["overall_roi"] == pytest.approx(0.0)

    def test_win_rate_leaves_out_pushes(self, client, a_big_week_and_a_small_one):
        data = client.get("/api/performance?season=2025").json()

        assert data["win_rate"] == pytest.approx(60 / 110 * 100)


def _leaky_failure(*args, **kwargs):
    raise RuntimeError("connect failed for db password hunter2")


@pytest.mark.parametrize(
    ("path", "failing_call"),
    [
        ("/api/value-bets?season=2025&week=22", "api.server.read_dataframe"),
        (
            "/api/explain/P001/receiving_yards?season=2025&week=22",
            "api.explainability.build_why_payload",
        ),
        ("/api/analytics/correlation?season=2025&week=22", "api.server.read_dataframe"),
        ("/api/analytics/risk-summary?season=2025&week=22", "api.server.read_dataframe"),
        ("/api/export/csv?season=2025&week=22", "api.server.read_dataframe"),
        ("/api/export/bundle?season=2025&week=22", "api.server.read_dataframe"),
    ],
)
def test_a_server_error_never_shows_the_exception_text(client, monkeypatch, path, failing_call):
    monkeypatch.setattr(failing_call, _leaky_failure)

    resp = client.get(path)

    assert resp.status_code == 500
    assert "hunter2" not in resp.text


class TestPipelineRunContract:
    def test_post_returns_run_fields(self, client):
        resp = client.post("/api/run?season=2025&week=22&skip_ingest=true&skip_odds=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "status" in data
        assert "started_at" in data
        assert "season" in data
        assert "week" in data
