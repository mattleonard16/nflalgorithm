"""Combine the model's projected mean with the mean the book's own price implies.

Why this exists
---------------
The value engine compares the model's probability against the book's, and never
compares the model's *mean* against the book's. So the further the projection
sits from the line, the bigger the reported edge, with nothing distinguishing a
real disagreement from model error. Measured on the 2026 week-1 slate (12 games,
6 books, 915 quotes matched to a projection), that produced 579 bets at a 19%
average edge, a third of every priced side, and 196 of them pinned to the 10%
Kelly cap. A market carried by six books does not leave a third of its prices
wrong by 19%.

The correction is forecast combination: average the two estimates instead of
treating one as truth. Combination beating its own inputs is about the most
replicated result in forecasting — 12 of the top 17 methods in the M4 competition
were combinations, and M5 repeated it. The default weight is 0.5 because equal
weights are the competition baseline that fitted weights keep failing to beat,
and because the weight cannot be fitted here yet: that needs stored market lines
sitting next to actuals, and ``clv_weekly`` is still empty.

Why the book's *mean*, not its line
-----------------------------------
A line sits near the median. ``mu`` is a mean. For a right-skewed stat the mean
is above the median, so blending ``mu`` toward the raw line drags every estimate
down and pushes the whole card to the under. That is not a small effect: doing it
that way turned the 2026 week-1 card into 87% unders, and the same slate's books
implied a mean of 38.6 receiving yards against a median line of 29.5.

So the book's price is converted to a mean first, by solving for the ``mu`` that
makes ``utils.nfl_markets.prob_over`` reproduce the book's de-vigged probability
at that line and sigma. Both numbers are then means on the same curve and the
average means something. After the fix the same slate came out 27 overs to 49
unders.

Callers pass the de-vigged probability in. The de-vig itself
(``implied_probability_no_vig``) still lives in gitignored
``value_betting_engine``, and importing it here would make this module, and every
test touching it, fail to import in CI — which is the opposite of why the math
lives in a tracked file. See ``utils/clv.py`` for the same split.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from config import config
from utils.nfl_markets import GAMMA_MARKETS, prob_over

logger = logging.getLogger(__name__)

# Below this gamma shape the curve's median is measurably in the wrong place.
# Shape is ``(mu / sigma) ** 2``, so sigma >= mu puts it under 1 and piles the
# distribution up against zero. On the 2025 walk-forward rows
# (`reports/nfl_backtest_2025_sigma_v2_rows.csv`, 5,117 rows) the actual cleared
# the gamma's own median 64.4% of the time where shape < 0.5 and 55.0% where
# shape was 0.5 to 1.0, against the 50% a correct median requires. At shape 1.0
# and above it lands at 46.4% to 49.0%, within 1 to 4 points. 23.9% of those rows
# had sigma >= mu.
#
# The normal curve is no better there, just wrong the other way: its median was
# cleared 35.0% of the time in the same shape<0.5 bucket. So this is not a reason
# to switch curves, it is a reason not to bet the row. Hence a predicate the
# caller screens on, rather than a branch inside ``prob_over``.
MIN_RELIABLE_SHAPE = 1.0

# Weeks at the start of a season before the model has any current-season data to
# work from. On the 2025 walk-forward rows QB passing yards correlates 0.21 with
# the actual across weeks 1-4 and its MAE there (73.4) is *worse* than predicting
# the league mean (71.9), so the projection carries no information a book does not
# already have. The cause is visible in the spread: those projections have a
# standard deviation of 32 yards against real passing days at 94, because the
# model falls back on priors and barely moves. From week 5 the same market
# correlates 0.51 and beats the flat baseline by 12.5%.
#
# This is a quarterback problem, not a season-start problem: receiving and rushing
# yards correlate 0.45 to 0.63 in every week of 2025, week 1 included, and beat
# the flat baseline by 15% to 23% in both halves of the season. So the screen names
# the markets it distrusts instead of muting the whole card.
EARLY_SEASON_WEEKS = 4
LOW_SKILL_EARLY_MARKETS = frozenset({"passing_yards"})


def model_has_skill(market: Optional[str], week) -> bool:
    """Whether this market's projection carries information this early in a season.

    False only for a market in ``LOW_SKILL_EARLY_MARKETS`` inside the first
    ``EARLY_SEASON_WEEKS`` weeks. A missing or unparseable week cannot be screened,
    so it answers ``True`` and leaves the decision to the caller's other rules.
    """
    if market not in LOW_SKILL_EARLY_MARKETS:
        return True
    try:
        week_num = int(week)
    except (TypeError, ValueError):
        return True
    return week_num > EARLY_SEASON_WEEKS

# Root-finding bounds for ``implied_mean``. The lower bound is a hair above zero
# because a non-positive mean is not a usable projection; the upper bound grows
# by doubling until it brackets, so no market's scale is hardcoded.
_MEAN_SEARCH_FLOOR = 1e-6
_MEAN_SEARCH_CEILING = 1e7
_SOLVE_TOLERANCE = 1e-6


def blend_weight() -> float:
    """Configured weight on the market's mean, in ``[0, 1]``."""
    weight = float(config.betting.market_blend_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"market_blend_weight must be in [0, 1], got {weight}")
    return weight


def gamma_shape(mu: float, sigma: float) -> float:
    """Shape parameter of the gamma that ``prob_over`` fits to ``mu`` and ``sigma``.

    Method of moments, same as the pricing call: ``(mu / sigma) ** 2``. Returns
    ``nan`` when either input makes the gamma undefined, so a caller comparing
    against ``MIN_RELIABLE_SHAPE`` gets a false rather than a crash. A null or
    non-numeric cell counts as undefined: these frames come from a database read,
    where ``sigma`` can arrive as ``None``.
    """
    try:
        mu_f, sigma_f = float(mu), float(sigma)
    except (TypeError, ValueError):
        return math.nan
    if not (mu_f > 0 and sigma_f > 0):
        return math.nan
    return (mu_f / sigma_f) ** 2


def skew_is_reliable(mu: float, sigma: float, market: Optional[str] = None) -> bool:
    """Whether the priced curve's median is in the range measured to be right.

    Only the gamma markets have a shape to check; everything else prices off a
    normal or Poisson curve whose spread is not parameterised this way, so this
    answers ``True`` for them and the caller's other screens apply. See
    ``MIN_RELIABLE_SHAPE`` for the measurement.
    """
    if market not in GAMMA_MARKETS:
        return True
    shape = gamma_shape(mu, sigma)
    return bool(shape >= MIN_RELIABLE_SHAPE)


def implied_mean(
    line: float, sigma: float, p_over: float, market: Optional[str] = None
) -> float:
    """The mean that makes ``prob_over`` agree with a book's de-vigged price.

    ``p_over`` must already have the vig removed, because a raw over price and a
    raw under price both carry the book's margin and sum past 1. Feeding a raw
    price in here quietly shifts the recovered mean up.

    ``prob_over`` rises monotonically with the mean for every curve it dispatches
    on, so the root is unique and bisection is enough. Returns ``nan`` when the
    price implies a mean at or below zero, which a normal-priced low-count market
    can do: that is a row to drop, not a mean to blend, and
    ``blend_quotes`` counts them.

    Raises ``ValueError`` on input that cannot have a solution at all, which is a
    caller bug rather than an odd quote: a probability outside ``(0, 1)``, a
    non-positive sigma, or a negative line.
    """
    if not 0.0 < p_over < 1.0:
        raise ValueError(f"p_over must be strictly inside (0, 1), got {p_over}")
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if line < 0:
        raise ValueError(f"line must be non-negative, got {line}")

    def excess(mean: float) -> float:
        return prob_over(mean, sigma, line, market=market) - p_over

    if excess(_MEAN_SEARCH_FLOOR) >= 0.0:
        # Even a mean of essentially zero is already over-confident at this
        # line, so no positive mean reproduces the price.
        return math.nan

    hi = max(line, sigma, 1.0)
    while excess(hi) < 0.0:
        hi *= 2.0
        if hi > _MEAN_SEARCH_CEILING:
            return math.nan

    return _bisect(excess, _MEAN_SEARCH_FLOOR, hi)


def _bisect(excess, lo: float, hi: float) -> float:
    """Bisection on a monotone increasing function bracketed by ``lo`` and ``hi``.

    Hand-rolled rather than ``scipy.optimize.brentq`` so the tolerance is in the
    same units as the mean (yards, receptions) and reads as such at the call
    site. Bisection needs no derivative and cannot diverge on a flat tail, which
    the far reaches of a low-shape gamma are.
    """
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo < _SOLVE_TOLERANCE:
            return mid
        if excess(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def blend(mu_model: float, mu_market: float, weight: Optional[float] = None) -> float:
    """Weighted average of the two means. ``weight`` is the share on the market."""
    w = blend_weight() if weight is None else float(weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {w}")
    return (1.0 - w) * float(mu_model) + w * float(mu_market)


def blend_quotes(quotes: pd.DataFrame, weight: Optional[float] = None) -> pd.DataFrame:
    """Attach market mean, blended mean, and the skew screen to a quote frame.

    Expects one row per book quote with ``player_id``, ``market``, ``line``,
    ``mu``, ``sigma`` and ``p_market_over`` (de-vigged). Returns a copy with:

    ``market_mean``
        the mean that book's own price implies at its own line.
    ``market_mean_consensus``
        the median of ``market_mean`` across every book quoting that player and
        market. Books post different lines for the same player, so pooling the
        recovered means is the only way to compare them; pooling the lines
        themselves compares different bets. The median rather than the mean so a
        single stale book cannot drag the consensus.
    ``mu_blended``
        ``blend(mu, market_mean_consensus)``.
    ``skew_reliable``
        ``skew_is_reliable(mu_blended, sigma, market)``.
    ``model_skilled``
        ``model_has_skill(market, week)``, when the frame carries a ``week``
        column. Without one the early-season screen cannot run, so the column is
        all ``True`` and a warning says so rather than the screen passing silently.

    Rows whose market mean cannot be solved keep ``nan`` in these columns and are
    logged as a count, not dropped: the caller decides, and silently shortening a
    card hides coverage loss. A row that failed contributes nothing to its
    player's consensus.
    """
    required = {"player_id", "market", "line", "mu", "sigma", "p_market_over"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"blend_quotes needs columns {sorted(missing)}")

    out = quotes.copy()
    out["market_mean"] = [
        _safe_implied_mean(line, sigma, p, market)
        for line, sigma, p, market in zip(
            out["line"], out["sigma"], out["p_market_over"], out["market"]
        )
    ]

    unsolved = int(out["market_mean"].isna().sum())
    if unsolved:
        logger.warning(
            "market mean unsolvable for %d of %d quotes; those rows carry no blend",
            unsolved,
            len(out),
        )

    consensus = (
        out.groupby(["player_id", "market"])["market_mean"].median().rename("market_mean_consensus")
    )
    out = out.join(consensus, on=["player_id", "market"])

    w = blend_weight() if weight is None else float(weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {w}")
    out["mu_blended"] = (1.0 - w) * out["mu"] + w * out["market_mean_consensus"]
    # Explicit bool dtype: a list of Python bools would come back as `object` on
    # an empty frame, and callers filter on this column.
    out["skew_reliable"] = pd.Series(
        [
            skew_is_reliable(mu, sigma, market) if pd.notna(mu) else False
            for mu, sigma, market in zip(out["mu_blended"], out["sigma"], out["market"])
        ],
        index=out.index,
        dtype=bool,
    )
    if "week" in out.columns:
        weeks = out["week"]
    else:
        logger.warning(
            "quote frame has no `week` column; the early-season skill screen is not applied"
        )
        weeks = pd.Series(None, index=out.index, dtype=object)
    out["model_skilled"] = pd.Series(
        [model_has_skill(market, week) for market, week in zip(out["market"], weeks)],
        index=out.index,
        dtype=bool,
    )
    return out


def _safe_implied_mean(line, sigma, p_over, market) -> float:
    """``implied_mean`` for one frame row, with unusable cells as ``nan``.

    A frame arrives from a database read, so a cell can be null or a string where
    a number belongs. That is a row to skip, not a reason to abandon the week's
    card, and ``blend_quotes`` reports how many were skipped.
    """
    try:
        return implied_mean(float(line), float(sigma), float(p_over), market=market)
    except (TypeError, ValueError):
        return math.nan
