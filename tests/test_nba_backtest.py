"""Tests for utils/nba_backtest.py — NBA walk-forward backtest.

TDD: these tests were written before the implementation.

Covers:
1.  Monotone bankroll → max_drawdown == 0
2.  Loss sequence → correct drawdown calculation
3.  Mixed series → correct drawdown calculation
4.  Win increases bankroll by kelly × payout
5.  Loss decreases bankroll by kelly × stake
6.  Push leaves bankroll unchanged
7.  Empty date range → zero bets result
8.  Walk-forward: day N projections never used for day N-1 grading
9.  Per-market breakdown sums to total
10. roi_pct == total_profit_units / total_bets * 100
"""

from __future__ import annotations

import pytest

from schema_migrations import MigrationManager
from utils.db import executemany


# ---------------------------------------------------------------------------
# Shared DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_backtest.db")
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

_SEASON = 2025
_PLAYER_ID = 1628369


def _seed_value_view(
    game_date: str,
    player_id: int,
    market: str,
    line: float,
    over_price: int,
    p_win: float,
    kelly: float,
    edge_pct: float,
    confidence_tier: str = "B",
    side: str = "over",
    sportsbook: str = "FanDuel",
    event_id: str = "EVT001",
    player_name: str = "Test Player",
) -> None:
    executemany(
        """
        INSERT OR REPLACE INTO nba_materialized_value_view
        (season, game_date, player_id, player_name, team, event_id, market, sportsbook,
         line, over_price, under_price, mu, sigma, p_win, edge_percentage, expected_roi,
         kelly_fraction, confidence, confidence_score, confidence_tier, generated_at, side)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                _SEASON,
                game_date,
                player_id,
                player_name,
                "BOS",
                event_id,
                market,
                sportsbook,
                line,
                over_price,
                -110,
                line + 2.0,  # mu slightly above line
                5.0,
                p_win,
                edge_pct,
                0.05,
                kelly,
                0.8,
                70.0,
                confidence_tier,
                f"{game_date}T10:00:00",
                side,
            )
        ],
    )


def _seed_game_log(
    game_date: str,
    player_id: int,
    pts: float = 0,
    reb: float = 0,
    ast: float = 0,
    fg3m: float = 0,
    player_name: str = "Test Player",
    season: int = _SEASON,
) -> None:
    executemany(
        """
        INSERT OR REPLACE INTO nba_player_game_logs
        (player_id, player_name, team_abbreviation, season, game_id, game_date,
         matchup, wl, min, pts, reb, ast, fg3m, fgm, fga, ftm, fta, stl, blk, tov, plus_minus)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                player_id,
                player_name,
                "BOS",
                season,
                f"GAME_{game_date}_{player_id}",
                game_date,
                "BOS vs MIA",
                "W",
                32.0,
                pts,
                reb,
                ast,
                fg3m,
                8,
                16,
                3,
                4,
                1,
                0,
                2,
                5.0,
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Monotone bankroll → max_drawdown == 0
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_monotone_increasing_gives_zero_drawdown(self):
        from utils.nba_backtest import max_drawdown_from_bankroll

        series = [1.0, 1.1, 1.2, 1.3]
        assert max_drawdown_from_bankroll(series) == pytest.approx(0.0)

    def test_loss_sequence_gives_correct_drawdown(self):
        """[1.0, 0.9, 0.8, 0.7] → peak=1.0, trough=0.7 → drawdown=0.3."""
        from utils.nba_backtest import max_drawdown_from_bankroll

        series = [1.0, 0.9, 0.8, 0.7]
        result = max_drawdown_from_bankroll(series)
        assert result == pytest.approx(0.3, abs=1e-9)

    def test_mixed_series_gives_correct_drawdown(self):
        """[1.0, 0.9, 0.8, 1.1] → peak=1.0, trough=0.8 → drawdown=0.2."""
        from utils.nba_backtest import max_drawdown_from_bankroll

        series = [1.0, 0.9, 0.8, 1.1]
        result = max_drawdown_from_bankroll(series)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_single_element_gives_zero(self):
        from utils.nba_backtest import max_drawdown_from_bankroll

        assert max_drawdown_from_bankroll([1.0]) == pytest.approx(0.0)

    def test_empty_gives_zero(self):
        from utils.nba_backtest import max_drawdown_from_bankroll

        assert max_drawdown_from_bankroll([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4-6. Bankroll mechanics: win / loss / push
# ---------------------------------------------------------------------------


class TestBacktestBankrollMechanics:
    """Tests 4-6: win increases bankroll, loss decreases, push unchanged."""

    def test_win_increases_bankroll(self, db):
        """Winning bet: bankroll should increase by kelly × (decimal_odds - 1)."""
        from utils.nba_backtest import BacktestConfig, run_backtest

        # p_win=0.7, price=-110 → kelly = 0.25 * (1.909*0.7 - 1)/(1.909-1) ≈ 0.0374
        # We plant a bet that is an OVER with actual > line (win)
        game_date = "2026-01-10"
        _seed_value_view(
            game_date=game_date,
            player_id=_PLAYER_ID,
            market="pts",
            line=20.0,
            over_price=-110,
            p_win=0.70,
            kelly=0.05,  # known kelly fraction
            edge_pct=0.10,
            side="over",
        )
        # Actual pts = 30 > line=20 → WIN for over
        _seed_game_log(game_date=game_date, player_id=_PLAYER_ID, pts=30)

        config = BacktestConfig(start_date=game_date, end_date=game_date)
        result = run_backtest(config)

        assert result.total_bets == 1
        assert result.wins == 1
        assert result.losses == 0
        assert result.pushes == 0
        assert result.total_profit_units > 0.0

    def test_loss_decreases_bankroll(self, db):
        """Losing bet: bankroll should decrease by kelly fraction."""
        from utils.nba_backtest import BacktestConfig, run_backtest

        game_date = "2026-01-11"
        _seed_value_view(
            game_date=game_date,
            player_id=_PLAYER_ID,
            market="pts",
            line=30.0,
            over_price=-110,
            p_win=0.65,
            kelly=0.05,
            edge_pct=0.10,
            side="over",
        )
        # Actual pts = 20 < line=30 → LOSS for over
        _seed_game_log(game_date=game_date, player_id=_PLAYER_ID, pts=20)

        config = BacktestConfig(start_date=game_date, end_date=game_date)
        result = run_backtest(config)

        assert result.total_bets == 1
        assert result.wins == 0
        assert result.losses == 1
        assert result.total_profit_units < 0.0

    def test_push_leaves_bankroll_unchanged(self, db):
        """Push: actual == line → profit = 0."""
        from utils.nba_backtest import BacktestConfig, run_backtest

        game_date = "2026-01-12"
        _seed_value_view(
            game_date=game_date,
            player_id=_PLAYER_ID,
            market="pts",
            line=25.0,
            over_price=-110,
            p_win=0.60,
            kelly=0.05,
            edge_pct=0.10,
            side="over",
        )
        # Actual pts = 25.0 == line → PUSH
        _seed_game_log(game_date=game_date, player_id=_PLAYER_ID, pts=25)

        config = BacktestConfig(start_date=game_date, end_date=game_date)
        result = run_backtest(config)

        assert result.total_bets == 1
        assert result.pushes == 1
        assert result.total_profit_units == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 7. Empty date range → zero bets result
# ---------------------------------------------------------------------------


class TestEmptyDateRange:
    def test_no_value_bets_in_range(self, db):
        """When no rows exist in the materialized view for the date range, backtest
        should return a result with total_bets=0."""
        from utils.nba_backtest import BacktestConfig, run_backtest

        config = BacktestConfig(start_date="2020-01-01", end_date="2020-01-01")
        result = run_backtest(config)

        assert result.total_bets == 0
        assert result.wins == 0
        assert result.losses == 0
        assert result.pushes == 0
        assert result.roi_pct == pytest.approx(0.0)
        assert result.total_profit_units == pytest.approx(0.0)

    def test_start_equals_end_with_no_data(self, db):
        from utils.nba_backtest import BacktestConfig, run_backtest

        config = BacktestConfig(start_date="2020-06-15", end_date="2020-06-15")
        result = run_backtest(config)

        assert result.total_bets == 0


# ---------------------------------------------------------------------------
# 8. Walk-forward: day N projections never used for day N-1 grading
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_day1_bets_graded_only_against_day1_actuals(self, db):
        """Set up 2 days of bets and actuals.
        Day 1: over bet wins (actual > line)
        Day 2: over bet loses (actual < line)
        Verify day 1 shows win, day 2 shows loss — no cross-day contamination.
        """
        from utils.nba_backtest import BacktestConfig, run_backtest

        date1 = "2026-01-20"
        date2 = "2026-01-21"
        player1 = 1001
        player2 = 1002

        # Day 1: over bet @ line=20, actual=30 → WIN
        _seed_value_view(
            game_date=date1,
            player_id=player1,
            market="pts",
            line=20.0,
            over_price=-110,
            p_win=0.65,
            kelly=0.05,
            edge_pct=0.10,
            event_id="EVT_D1",
            player_name="Player One",
        )
        _seed_game_log(game_date=date1, player_id=player1, pts=30, player_name="Player One")

        # Day 2: over bet @ line=25, actual=15 → LOSS
        _seed_value_view(
            game_date=date2,
            player_id=player2,
            market="pts",
            line=25.0,
            over_price=-110,
            p_win=0.65,
            kelly=0.05,
            edge_pct=0.10,
            event_id="EVT_D2",
            player_name="Player Two",
        )
        _seed_game_log(game_date=date2, player_id=player2, pts=15, player_name="Player Two")

        config = BacktestConfig(start_date=date1, end_date=date2)
        result = run_backtest(config)

        assert result.total_bets == 2
        assert result.wins == 1
        assert result.losses == 1

        # Verify daily PnL has separate rows
        assert len(result.daily_pnl) == 2
        day1_pnl = result.daily_pnl[result.daily_pnl["date"] == date1]["pnl"].iloc[0]
        day2_pnl = result.daily_pnl[result.daily_pnl["date"] == date2]["pnl"].iloc[0]
        assert day1_pnl > 0  # win on day 1
        assert day2_pnl < 0  # loss on day 2


# ---------------------------------------------------------------------------
# 9. Per-market breakdown sums to total
# ---------------------------------------------------------------------------


class TestPerMarketBreakdown:
    def test_per_market_bets_sum_to_total(self, db):
        """Sum of per_market bets should equal total_bets."""
        from utils.nba_backtest import BacktestConfig, run_backtest

        game_date = "2026-01-25"
        markets = ["pts", "reb", "ast"]
        actuals = {"pts": 30, "reb": 10, "ast": 8}
        lines = {"pts": 20.0, "reb": 6.0, "ast": 5.0}

        for i, market in enumerate(markets):
            _seed_value_view(
                game_date=game_date,
                player_id=2000 + i,
                market=market,
                line=lines[market],
                over_price=-110,
                p_win=0.65,
                kelly=0.05,
                edge_pct=0.10,
                event_id=f"EVT_M{i}",
                player_name=f"Player {market}",
            )
            _seed_game_log(
                game_date=game_date,
                player_id=2000 + i,
                pts=actuals.get("pts", 0) if market == "pts" else 0,
                reb=actuals.get("reb", 0) if market == "reb" else 0,
                ast=actuals.get("ast", 0) if market == "ast" else 0,
                player_name=f"Player {market}",
            )

        config = BacktestConfig(start_date=game_date, end_date=game_date, markets=markets)
        result = run_backtest(config)

        total_from_markets = result.per_market["bets"].sum()
        assert total_from_markets == result.total_bets


# ---------------------------------------------------------------------------
# 10. roi_pct == total_profit_units / total_bets * 100
# ---------------------------------------------------------------------------


class TestRoiFormula:
    def test_roi_formula_is_correct(self, db):
        """roi_pct should equal total_profit_units / total_bets * 100."""
        from utils.nba_backtest import BacktestConfig, run_backtest

        game_date = "2026-01-30"
        # Two bets: one win, one loss
        _seed_value_view(
            game_date=game_date,
            player_id=3001,
            market="pts",
            line=20.0,
            over_price=-110,
            p_win=0.65,
            kelly=0.05,
            edge_pct=0.10,
            event_id="EVT_R1",
            player_name="ROI Win Player",
        )
        _seed_game_log(game_date=game_date, player_id=3001, pts=30, player_name="ROI Win Player")

        _seed_value_view(
            game_date=game_date,
            player_id=3002,
            market="reb",
            line=10.0,
            over_price=-110,
            p_win=0.65,
            kelly=0.05,
            edge_pct=0.10,
            event_id="EVT_R2",
            player_name="ROI Loss Player",
        )
        _seed_game_log(game_date=game_date, player_id=3002, reb=5, player_name="ROI Loss Player")

        config = BacktestConfig(start_date=game_date, end_date=game_date)
        result = run_backtest(config)

        assert result.total_bets >= 1
        expected_roi = (result.total_profit_units / result.total_bets) * 100
        assert result.roi_pct == pytest.approx(expected_roi, abs=1e-6)
