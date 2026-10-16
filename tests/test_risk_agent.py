"""RiskAgent recommendations after the correlation fix.

Exposure-cap breaches are REJECTed; membership in a correlation group alone
is reported as NEUTRAL. Before, any labeled row was REJECTed, which vetoed
nearly every multi-book core-player bet.
"""

from __future__ import annotations

import pandas as pd

from agents.risk_agent import RiskAgent


def _value_rows(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "season": 2025,
        "week": 5,
        "event_id": "G1",
        "team": "KC",
        "sportsbook": "DraftKings",
        "line": 60.5,
        "price": -110,
        "p_win": 0.58,
        "kelly_fraction": 0.02,
        "stake": 20.0,
        "side": "over",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _reports(
    monkeypatch, frame: pd.DataFrame, bankroll: float = 10_000.0
) -> dict[tuple[str, str, str], str]:
    agent = RiskAgent()
    monkeypatch.setattr(agent, "_load_value_view", lambda season, week, player_id=None: frame)
    monkeypatch.setattr("risk_manager.config.betting.bankroll", bankroll)
    # Drawdown sizing is not under test; keep Kelly unchanged.
    monkeypatch.setattr(
        "risk_manager.risk_adjusted_kelly", lambda kelly, stats: kelly
    )
    reports = agent.analyze(2025, 5)
    return {
        (r.player_id, r.market, str(i)): r.recommendation for i, r in enumerate(reports)
    }


def test_multi_book_qb_and_opposing_players_are_not_rejected(monkeypatch) -> None:
    frame = _value_rows([
        {"player_id": "KC_qb", "market": "passing_yards", "sportsbook": "DraftKings"},
        {"player_id": "KC_qb", "market": "passing_yards", "sportsbook": "FanDuel"},
        {"player_id": "BUF_wr", "market": "receiving_yards", "team": "BUF", "side": "under"},
        {"player_id": "BUF_rb", "market": "rushing_yards", "team": "BUF"},
    ])

    recommendations = _reports(monkeypatch, frame)

    assert "REJECT" not in recommendations.values()
    assert recommendations[("KC_qb", "passing_yards", "0")] == "APPROVE"
    assert recommendations[("KC_qb", "passing_yards", "1")] == "APPROVE"
    # Teammates on one card: reported as correlated, not vetoed.
    assert recommendations[("BUF_wr", "receiving_yards", "2")] == "NEUTRAL"
    assert recommendations[("BUF_rb", "rushing_yards", "3")] == "NEUTRAL"


def test_correlated_stack_past_the_team_cap_is_rejected(monkeypatch) -> None:
    # 2 x 200 on KC = 40% of the 1,000 bankroll, past the 30% team cap.
    frame = _value_rows([
        {"player_id": "KC_qb", "market": "passing_yards", "stake": 200.0},
        {"player_id": "KC_wr", "market": "receiving_yards", "stake": 200.0},
    ])
    recommendations = _reports(monkeypatch, frame, bankroll=1000.0)

    assert set(recommendations.values()) == {"REJECT"}


def test_correlation_is_named_in_the_neutral_rationale(monkeypatch) -> None:
    frame = _value_rows([
        {"player_id": "KC_qb", "market": "passing_yards"},
        {"player_id": "KC_wr", "market": "receiving_yards"},
    ])
    agent = RiskAgent()
    monkeypatch.setattr(agent, "_load_value_view", lambda season, week, player_id=None: frame)
    monkeypatch.setattr("risk_manager.risk_adjusted_kelly", lambda kelly, stats: kelly)
    monkeypatch.setattr("risk_manager.config.betting.bankroll", 10_000.0)

    reports = agent.analyze(2025, 5)

    assert [r.recommendation for r in reports] == ["NEUTRAL", "NEUTRAL"]
    assert all("correlated(pos_corr_1)" in r.rationale for r in reports)
    assert all(r.data["correlation_group"] == "pos_corr_1" for r in reports)
