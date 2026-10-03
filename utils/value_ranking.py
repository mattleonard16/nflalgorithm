"""Rank a week's projections against live odds, preferring the deployment's engine.

A deployment supplies ``value_betting_engine.py`` and its ``rank_weekly_value``
wins. A public clone has no such file, so ``rank_weekly_value_baseline`` builds
the card instead of the import crashing. The baseline is textbook pricing over
tracked math: live quotes only, gamma or Poisson win probability, no-vig fair
odds, both sides of every line, and a capped Kelly stake.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, Callable

import pandas as pd

from config import config
from utils.db import read_dataframe
from utils.live_odds import kickoffs_from_games, select_live_odds
from utils.nfl_markets import prob_over
from utils.odds_math import (
    expected_roi,
    implied_probability,
    implied_probability_no_vig,
    kelly_fraction,
)
from utils.volatility_scoring import apply_volatility_widening

logger = logging.getLogger(__name__)

RANKED_COLUMNS = [
    "season",
    "week",
    "player_id",
    "event_id",
    "team",
    "team_odds",
    "market",
    "sportsbook",
    "line",
    "price",
    "side",
    "mu",
    "sigma",
    "volatility_score",
    "target_share",
    "p_win",
    "implied_prob",
    "implied_prob_under",
    "edge_percentage",
    "expected_roi",
    "kelly_fraction",
    "stake",
    "generated_at",
]

Ranker = Callable[[int, int, float], pd.DataFrame]


def rank_weekly_value(season: int, week: int, min_edge: float = 0.05) -> pd.DataFrame:
    """Rank the week's bets with the deployment's engine, or the public baseline."""
    return _ranker()(season, week, min_edge)


def _ranker() -> Ranker:
    try:
        engine: Any = import_module("value_betting_engine")
    except ModuleNotFoundError as exc:
        # Only the engine itself may be absent. A broken dependency inside an
        # installed engine must fail, not quietly swap in the baseline.
        if exc.name != "value_betting_engine":
            raise
        if config.features.require_private_models:
            raise RuntimeError(
                "NFL_REQUIRE_PRIVATE_MODELS is set but value_betting_engine.py is not installed"
            ) from exc
        logger.warning(
            "value_betting_engine.py is not installed; ranking with the public baseline"
        )
        return rank_weekly_value_baseline
    ranker: Ranker = engine.rank_weekly_value
    return ranker


def rank_weekly_value_baseline(
    season: int, week: int, min_edge: float = 0.05
) -> pd.DataFrame:
    """Price every live quote on both sides and keep the bets whose edge clears ``min_edge``."""
    odds = read_dataframe(
        """
        SELECT season, week, event_id, player_id, market, sportsbook,
               line, price, under_price, as_of
        FROM weekly_odds
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )
    projections = read_dataframe(
        """
        SELECT season, week, player_id, team, market, mu, sigma, volatility_score,
               target_share, generated_at
        FROM weekly_projections
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )
    games = read_dataframe(
        "SELECT game_id, kickoff_utc FROM games WHERE season = ? AND week = ?",
        params=(season, week),
    )
    if odds.empty or projections.empty:
        return pd.DataFrame(columns=RANKED_COLUMNS)

    # weekly_odds is append-only snapshot history. The stale filter must run
    # before picking the newest row, or a post-kickoff scrape wins.
    live = select_live_odds(odds, kickoffs_from_games(games))
    quotes = projections.merge(
        live.drop(columns=["season", "week"], errors="ignore"),
        on=["player_id", "market"],
        how="inner",
    )
    if quotes.empty:
        return pd.DataFrame(columns=RANKED_COLUMNS)

    quotes["team_odds"] = None
    quotes["sigma"], unscored = apply_volatility_widening(
        quotes["sigma"], quotes.get("volatility_score")
    )
    if unscored:
        logger.warning(
            "volatility_score missing on %d of %d rows; sigma left unwidened there",
            unscored,
            len(quotes),
        )

    p_over = pd.Series(
        [
            prob_over(float(r.mu), float(r.sigma), float(r.line), market=str(r.market))
            for r in quotes.itertuples(index=False)
        ],
        index=quotes.index,
    )
    fair = [_fair_probabilities(r.price, r.under_price) for r in quotes.itertuples(index=False)]
    quotes["implied_prob"] = [over for over, _ in fair]
    quotes["implied_prob_under"] = [under for _, under in fair]

    over = quotes.assign(side="over", p_win=p_over)
    over["edge_percentage"] = over["p_win"] - over["implied_prob"]
    # The under is bet at under_price. Without one, assume the over's price.
    under = quotes.assign(
        side="under",
        p_win=1.0 - p_over,
        price=[_under_price(r.price, r.under_price) for r in quotes.itertuples(index=False)],
    )
    under["edge_percentage"] = under["p_win"] - under["implied_prob_under"]

    bets = pd.concat([over, under], ignore_index=True)
    bets = bets[bets["edge_percentage"] >= min_edge].copy()
    if bets.empty:
        return pd.DataFrame(columns=RANKED_COLUMNS)

    bets["expected_roi"] = [
        expected_roi(float(p), int(price)) for p, price in zip(bets["p_win"], bets["price"])
    ]
    kelly = pd.Series(
        [kelly_fraction(float(p), int(price)) for p, price in zip(bets["p_win"], bets["price"])],
        index=bets.index,
    )
    if config.features.kelly_cap_enabled:
        kelly = (kelly * float(config.betting.kelly_fraction)).clip(
            lower=0.0, upper=float(config.betting.max_kelly)
        )
    bets["kelly_fraction"] = kelly
    bets["stake"] = kelly * float(config.betting.bankroll)

    bets = bets.sort_values("edge_percentage", ascending=False)
    return bets[RANKED_COLUMNS].reset_index(drop=True)


def _has_under(under_price: Any) -> bool:
    return not pd.isna(under_price) and int(under_price) != 0


def _fair_probabilities(price: Any, under_price: Any) -> tuple[float, float]:
    """(over, under) probabilities, with the margin removed when both sides are quoted."""
    if config.features.no_vig_enabled and _has_under(under_price):
        return implied_probability_no_vig(int(price), int(under_price))
    raw = implied_probability(int(price))
    return raw, 1.0 - raw


def _under_price(price: Any, under_price: Any) -> int:
    return int(under_price) if _has_under(under_price) else int(price)
