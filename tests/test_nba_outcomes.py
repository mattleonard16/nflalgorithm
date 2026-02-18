"""Tests for NBA grading pipeline (scripts/record_nba_outcomes.py).

Covers:
- TestGradeNbaBets: win/loss/push outcomes based on actual vs line
- TestSaveNbaOutcomes: persistence to nba_bet_outcomes and nba_daily_performance,
  idempotency (no duplicates on repeated calls)
- TestNbaMarketMapping: all 4 markets (pts, reb, ast, fg3m) grade correctly
- TestEdgeCases: player with no actuals (push), no bets found (empty list),
  unknown market (skipped), missing game logs for date
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_migrations import MigrationManager
from utils.db import execute, executemany, read_dataframe


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_nba_outcomes.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)

    import config as cfg

    monkeypatch.setattr(cfg.config.database, "path", db_path)
    monkeypatch.setattr(cfg.config.database, "backend", "sqlite")

    MigrationManager(db_path).run()
    return db_path


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

GAME_DATE = "2026-02-17"
SEASON = 2025
PLAYER_ID = 1628369
PLAYER_NAME = "Jayson Tatum"
TEAM = "BOS"
SPORTSBOOK = "FanDuel"
EVENT_ID = "evt_tatum_pts_001"


def _seed_value_view(
    player_id: int = PLAYER_ID,
    player_name: str = PLAYER_NAME,
    market: str = "pts",
    line: float = 25.5,
    over_price: int = -115,
    under_price: int = -105,
    mu: float = 28.5,
    sigma: float = 5.7,
    p_win: float = 0.65,
    edge_percentage: float = 12.0,
    expected_roi: float = 0.08,
    kelly_fraction: float = 0.05,
    confidence: float = 0.85,
    event_id: str = EVENT_ID,
    sportsbook: str = SPORTSBOOK,
    game_date: str = GAME_DATE,
    season: int = SEASON,
) -> None:
    executemany(
        """
        INSERT OR REPLACE INTO nba_materialized_value_view (
            season, game_date, player_id, player_name, team, event_id, market,
            sportsbook, line, over_price, under_price, mu, sigma, p_win,
            edge_percentage, expected_roi, kelly_fraction, confidence, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                season,
                game_date,
                player_id,
                player_name,
                TEAM,
                event_id,
                market,
                sportsbook,
                line,
                over_price,
                under_price,
                mu,
                sigma,
                p_win,
                edge_percentage,
                expected_roi,
                kelly_fraction,
                confidence,
                "2026-02-17T08:00:00+00:00",
            )
        ],
    )


def _seed_game_log(
    player_id: int = PLAYER_ID,
    player_name: str = PLAYER_NAME,
    pts: int = 31,
    reb: int = 8,
    ast: int = 5,
    fg3m: int = 4,
    game_date: str = GAME_DATE,
    game_id: str = "0022500101",
    season: int = SEASON,
) -> None:
    executemany(
        """
        INSERT OR REPLACE INTO nba_player_game_logs (
            player_id, player_name, team_abbreviation, season, game_id, game_date,
            matchup, wl, min, pts, reb, ast, fg3m, fgm, fga, ftm, fta, stl, blk,
            tov, plus_minus
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                player_id,
                player_name,
                TEAM,
                season,
                game_id,
                game_date,
                "BOS vs. MIA",
                "W",
                36.5,
                pts,
                reb,
                ast,
                fg3m,
                11,
                20,
                5,
                6,
                1,
                0,
                2,
                12.0,
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. TestGradeNbaBets
# ---------------------------------------------------------------------------


class TestGradeNbaBets:
    def test_win_when_actual_exceeds_line(self, db):
        """Player scores 31 pts vs line 25.5 -> over wins."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        assert outcomes[0]["result"] == "win"
        assert outcomes[0]["actual_result"] == 31.0

    def test_loss_when_actual_below_line(self, db):
        """Player scores 20 pts vs line 25.5 -> over loses."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=20)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        assert outcomes[0]["result"] == "loss"
        assert outcomes[0]["actual_result"] == 20.0

    def test_push_when_actual_equals_line(self, db):
        """Player scores exactly 25 pts vs integer line 25.0 -> push."""
        _seed_value_view(market="pts", line=25.0)
        _seed_game_log(pts=25)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        assert outcomes[0]["result"] == "push"
        assert outcomes[0]["profit_units"] == 0.0

    def test_profit_units_positive_for_win(self, db):
        """Win at -115 should return 100/115 units profit."""
        _seed_value_view(market="pts", line=25.5, over_price=-115)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        expected_profit = 100.0 / 115
        assert abs(outcomes[0]["profit_units"] - expected_profit) < 0.001

    def test_profit_units_negative_for_loss(self, db):
        """Loss always returns -1.0 units."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=20)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert outcomes[0]["profit_units"] == -1.0

    def test_outcome_keys_present(self, db):
        """Every outcome dict must contain all required keys."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)
        assert len(outcomes) == 1

        required_keys = {
            "bet_id",
            "season",
            "game_date",
            "player_id",
            "player_name",
            "market",
            "sportsbook",
            "side",
            "line",
            "price",
            "actual_result",
            "result",
            "profit_units",
            "confidence_tier",
            "edge_at_placement",
            "recorded_at",
        }
        missing = required_keys - set(outcomes[0].keys())
        assert not missing, f"Outcome is missing keys: {missing}"

    def test_bet_side_is_always_over(self, db):
        """The grading pipeline always grades the over side."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)
        assert outcomes[0]["side"] == "over"

    def test_correct_player_metadata(self, db):
        """Outcome records player_id, player_name, market, line from the value view."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)
        assert outcomes[0]["player_id"] == PLAYER_ID
        assert outcomes[0]["player_name"] == PLAYER_NAME
        assert outcomes[0]["market"] == "pts"
        assert outcomes[0]["line"] == 25.5
        assert outcomes[0]["sportsbook"] == SPORTSBOOK

    def test_confidence_tier_high_for_large_edge(self, db):
        """edge_percentage >= 15.0 should yield HIGH confidence tier."""
        _seed_value_view(market="pts", line=25.5, edge_percentage=18.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)
        assert outcomes[0]["confidence_tier"] == "HIGH"

    def test_confidence_tier_medium_for_mid_edge(self, db):
        """edge_percentage between 8 and 15 should yield MEDIUM tier."""
        _seed_value_view(market="pts", line=25.5, edge_percentage=10.0)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)
        assert outcomes[0]["confidence_tier"] == "MEDIUM"

    def test_multiple_players_multiple_outcomes(self, db):
        """Two players seeded -> two outcomes returned."""
        _seed_value_view(
            player_id=PLAYER_ID,
            player_name=PLAYER_NAME,
            market="pts",
            line=25.5,
            event_id="evt001",
        )
        _seed_value_view(
            player_id=1628384,
            player_name="Jaylen Brown",
            market="pts",
            line=22.5,
            event_id="evt002",
        )
        _seed_game_log(player_id=PLAYER_ID, player_name=PLAYER_NAME, pts=31, game_id="g001")
        _seed_game_log(player_id=1628384, player_name="Jaylen Brown", pts=28, game_id="g002")

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)
        assert len(outcomes) == 2

    def test_each_outcome_has_unique_bet_id(self, db):
        """Each outcome must have a distinct UUID bet_id."""
        _seed_value_view(
            player_id=PLAYER_ID,
            player_name=PLAYER_NAME,
            market="pts",
            line=25.5,
            event_id="evt001",
        )
        _seed_value_view(
            player_id=1628384,
            player_name="Jaylen Brown",
            market="pts",
            line=22.5,
            event_id="evt002",
        )
        _seed_game_log(player_id=PLAYER_ID, player_name=PLAYER_NAME, pts=31, game_id="g001")
        _seed_game_log(player_id=1628384, player_name="Jaylen Brown", pts=28, game_id="g002")

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)
        bet_ids = [o["bet_id"] for o in outcomes]
        assert len(bet_ids) == len(set(bet_ids)), "Duplicate bet_id values found"


# ---------------------------------------------------------------------------
# 2. TestSaveNbaOutcomes
# ---------------------------------------------------------------------------


class TestSaveNbaOutcomes:
    def test_outcomes_persisted_to_nba_bet_outcomes(self, db):
        """save_nba_outcomes must write rows to nba_bet_outcomes."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)

        rows = read_dataframe(
            "SELECT * FROM nba_bet_outcomes WHERE game_date = ?", (GAME_DATE,)
        )
        assert len(rows) == 1

    def test_bet_outcome_result_field_correct(self, db):
        """Stored result field must match graded outcome."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)

        rows = read_dataframe(
            "SELECT result FROM nba_bet_outcomes WHERE game_date = ?", (GAME_DATE,)
        )
        assert rows.iloc[0]["result"] == "win"

    def test_daily_performance_row_created(self, db):
        """save_nba_outcomes must create a row in nba_daily_performance."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)

        rows = read_dataframe(
            "SELECT * FROM nba_daily_performance WHERE game_date = ?", (GAME_DATE,)
        )
        assert len(rows) == 1

    def test_daily_performance_counts_correct(self, db):
        """nba_daily_performance wins/losses/pushes must sum to total_bets."""
        _seed_value_view(
            player_id=PLAYER_ID,
            player_name=PLAYER_NAME,
            market="pts",
            line=25.5,
            event_id="evt001",
        )
        _seed_value_view(
            player_id=1628384,
            player_name="Jaylen Brown",
            market="pts",
            line=22.5,
            event_id="evt002",
        )
        _seed_game_log(player_id=PLAYER_ID, player_name=PLAYER_NAME, pts=31, game_id="g001")
        _seed_game_log(player_id=1628384, player_name="Jaylen Brown", pts=20, game_id="g002")

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)

        row = read_dataframe(
            "SELECT * FROM nba_daily_performance WHERE game_date = ?", (GAME_DATE,)
        ).iloc[0]

        assert row["total_bets"] == 2
        assert row["wins"] == 1
        assert row["losses"] == 1
        assert row["pushes"] == 0
        assert row["wins"] + row["losses"] + row["pushes"] == row["total_bets"]

    def test_idempotency_no_duplicate_bet_outcomes(self, db):
        """Calling save_nba_outcomes twice must not create duplicate rows
        in nba_bet_outcomes because bet_ids are unique UUIDs and the table
        uses INSERT OR REPLACE with bet_id as PK."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)
        save_nba_outcomes(outcomes)

        rows = read_dataframe(
            "SELECT * FROM nba_bet_outcomes WHERE game_date = ?", (GAME_DATE,)
        )
        assert len(rows) == len(outcomes), "Duplicate rows found after second save"

    def test_idempotency_no_duplicate_daily_performance(self, db):
        """Calling save_nba_outcomes twice must keep exactly one row in
        nba_daily_performance (INSERT OR REPLACE on PRIMARY KEY season+game_date)."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)
        save_nba_outcomes(outcomes)

        rows = read_dataframe(
            "SELECT * FROM nba_daily_performance WHERE game_date = ?", (GAME_DATE,)
        )
        assert len(rows) == 1, "Multiple daily_performance rows found after second save"

    def test_save_empty_outcomes_is_noop(self, db):
        """save_nba_outcomes([]) must not raise and must not write anything."""
        from scripts.record_nba_outcomes import save_nba_outcomes

        save_nba_outcomes([])

        bets = read_dataframe("SELECT * FROM nba_bet_outcomes")
        perf = read_dataframe("SELECT * FROM nba_daily_performance")
        assert len(bets) == 0
        assert len(perf) == 0

    def test_profit_units_stored_correctly(self, db):
        """Stored profit_units must match graded value."""
        _seed_value_view(market="pts", line=25.5, over_price=-115)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)

        rows = read_dataframe(
            "SELECT profit_units FROM nba_bet_outcomes WHERE game_date = ?",
            (GAME_DATE,),
        )
        expected = 100.0 / 115
        assert abs(rows.iloc[0]["profit_units"] - expected) < 0.001

    def test_daily_performance_roi_computed(self, db):
        """roi_pct in nba_daily_performance must be non-zero for winning bets."""
        _seed_value_view(market="pts", line=25.5, over_price=-115)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets, save_nba_outcomes

        outcomes = grade_nba_bets(GAME_DATE)
        save_nba_outcomes(outcomes)

        row = read_dataframe(
            "SELECT roi_pct FROM nba_daily_performance WHERE game_date = ?",
            (GAME_DATE,),
        ).iloc[0]
        assert row["roi_pct"] > 0.0


# ---------------------------------------------------------------------------
# 3. TestNbaMarketMapping
# ---------------------------------------------------------------------------


class TestNbaMarketMapping:
    """Verify that each market (pts/reb/ast/fg3m) maps to the correct stat column."""

    def _grade_market(self, market: str, line: float, actual_stat: dict) -> str:
        """Seed value view + game log for the given market, run grading, return result."""
        _seed_value_view(market=market, line=line, event_id=f"evt_{market}")
        _seed_game_log(**actual_stat)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)
        assert len(outcomes) == 1, f"Expected 1 outcome for market={market}"
        return outcomes[0]["result"]

    def test_pts_market_win(self, db):
        """pts market: actual 31 > line 25.5 -> win."""
        result = self._grade_market(
            "pts",
            line=25.5,
            actual_stat={"pts": 31, "reb": 5, "ast": 3, "fg3m": 2},
        )
        assert result == "win"

    def test_pts_market_loss(self, db):
        """pts market: actual 20 < line 25.5 -> loss."""
        result = self._grade_market(
            "pts",
            line=25.5,
            actual_stat={"pts": 20, "reb": 5, "ast": 3, "fg3m": 2},
        )
        assert result == "loss"

    def test_reb_market_win(self, db):
        """reb market: actual 10 > line 7.5 -> win."""
        result = self._grade_market(
            "reb",
            line=7.5,
            actual_stat={"pts": 20, "reb": 10, "ast": 3, "fg3m": 2},
        )
        assert result == "win"

    def test_reb_market_loss(self, db):
        """reb market: actual 5 < line 7.5 -> loss."""
        result = self._grade_market(
            "reb",
            line=7.5,
            actual_stat={"pts": 20, "reb": 5, "ast": 3, "fg3m": 2},
        )
        assert result == "loss"

    def test_ast_market_win(self, db):
        """ast market: actual 9 > line 6.5 -> win."""
        result = self._grade_market(
            "ast",
            line=6.5,
            actual_stat={"pts": 20, "reb": 5, "ast": 9, "fg3m": 2},
        )
        assert result == "win"

    def test_ast_market_loss(self, db):
        """ast market: actual 4 < line 6.5 -> loss."""
        result = self._grade_market(
            "ast",
            line=6.5,
            actual_stat={"pts": 20, "reb": 5, "ast": 4, "fg3m": 2},
        )
        assert result == "loss"

    def test_fg3m_market_win(self, db):
        """fg3m market: actual 5 > line 3.5 -> win."""
        result = self._grade_market(
            "fg3m",
            line=3.5,
            actual_stat={"pts": 20, "reb": 5, "ast": 4, "fg3m": 5},
        )
        assert result == "win"

    def test_fg3m_market_loss(self, db):
        """fg3m market: actual 2 < line 3.5 -> loss."""
        result = self._grade_market(
            "fg3m",
            line=3.5,
            actual_stat={"pts": 20, "reb": 5, "ast": 4, "fg3m": 2},
        )
        assert result == "loss"

    def test_nba_market_to_stat_mapping_exported(self, db):
        """NBA_MARKET_TO_STAT must contain exactly the 4 canonical markets."""
        from scripts.record_nba_outcomes import NBA_MARKET_TO_STAT

        assert set(NBA_MARKET_TO_STAT.keys()) == {"pts", "reb", "ast", "fg3m"}

    def test_each_market_maps_to_itself(self, db):
        """All four markets map 1-to-1 to the same stat column name."""
        from scripts.record_nba_outcomes import NBA_MARKET_TO_STAT

        for market in ("pts", "reb", "ast", "fg3m"):
            assert NBA_MARKET_TO_STAT[market] == market, (
                f"Expected market '{market}' to map to column '{market}', "
                f"got '{NBA_MARKET_TO_STAT[market]}'"
            )


# ---------------------------------------------------------------------------
# 4. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_actuals_for_player_gives_push(self, db):
        """When nba_player_game_logs has no row for the player on the game date,
        the bet is graded as a push with profit_units = 0.0."""
        _seed_value_view(market="pts", line=25.5)
        # Intentionally no game log seeded

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        assert outcomes[0]["result"] == "push"
        assert outcomes[0]["actual_result"] is None
        assert outcomes[0]["profit_units"] == 0.0

    def test_no_bets_found_returns_empty_list(self, db):
        """When nba_materialized_value_view has no rows for the date,
        grade_nba_bets must return an empty list."""
        _seed_game_log(pts=31)
        # Intentionally no value view row seeded

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert outcomes == []

    def test_unknown_market_is_skipped(self, db):
        """A row with an unrecognised market value must be silently skipped."""
        _seed_value_view(market="fantasy_pts", line=50.0, event_id="evt_unknown")
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert outcomes == [], (
            "Expected unknown market to be skipped, but outcomes were returned"
        )

    def test_unknown_market_mixed_with_known_market(self, db):
        """Unknown market row is skipped while known market row is still graded."""
        _seed_value_view(market="pts", line=25.5, event_id="evt_pts")
        _seed_value_view(market="fantasy_pts", line=50.0, event_id="evt_unknown")
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        assert outcomes[0]["market"] == "pts"

    def test_no_game_logs_for_date_all_push(self, db):
        """When game logs exist for a different date but not the target date,
        all bets on the target date should be graded as pushes."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31, game_date="2026-02-16")  # different date

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        assert outcomes[0]["result"] == "push"

    def test_wrong_player_in_game_logs_gives_push(self, db):
        """When game logs contain a different player_id for the same date,
        the target player is not found and bet is a push."""
        _seed_value_view(player_id=PLAYER_ID, player_name=PLAYER_NAME, market="pts", line=25.5)
        _seed_game_log(
            player_id=9999999,
            player_name="Unknown Player",
            pts=31,
            game_id="g_other",
        )

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        assert outcomes[0]["result"] == "push"
        assert outcomes[0]["actual_result"] is None

    def test_grade_different_date_returns_empty(self, db):
        """Grading a date with no predictions must return an empty list."""
        _seed_value_view(market="pts", line=25.5, game_date="2026-02-16")
        _seed_game_log(pts=31, game_date="2026-02-16")

        from scripts.record_nba_outcomes import grade_nba_bets

        # Grade for tomorrow -- nothing seeded for that date
        outcomes = grade_nba_bets("2026-02-18")
        assert outcomes == []

    def test_season_field_correct_in_outcome(self, db):
        """Outcome season must match the season in the value view."""
        _seed_value_view(market="pts", line=25.5, season=SEASON)
        _seed_game_log(pts=31, season=SEASON)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert len(outcomes) == 1
        assert outcomes[0]["season"] == SEASON

    def test_game_date_field_correct_in_outcome(self, db):
        """Outcome game_date must match the queried game_date."""
        _seed_value_view(market="pts", line=25.5)
        _seed_game_log(pts=31)

        from scripts.record_nba_outcomes import grade_nba_bets

        outcomes = grade_nba_bets(GAME_DATE)

        assert outcomes[0]["game_date"] == GAME_DATE
