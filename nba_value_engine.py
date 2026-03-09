"""NBA Value Betting Engine.

Joins NBA projections with NBA odds to compute value bets using Kelly criterion.
Modeled after the NFL value_betting_engine.py but adapted for NBA markets.

Usage:
    python nba_value_engine.py --date 2026-02-17
    python nba_value_engine.py --date 2026-02-17 --min-edge 0.08
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from nba_confidence_engine import compute_nba_confidence_score, assign_nba_tier
from utils.db import executemany, read_dataframe
from utils.nba_injury_adjustments import apply_injury_adjustments, SIGMA_INFLATION
from utils.nba_sigma import get_sigma_or_default
import logging

logger = logging.getLogger(__name__)

# Default path for pre-trained calibration model
_DEFAULT_CALIBRATION_PATH = str(
    Path(__file__).parent / "models" / "nba" / "calibration.joblib"
)

# ---------------------------------------------------------------------------
# Normal CDF approximation (avoids scipy dependency)
# Abramowitz & Stegun approximation — max absolute error < 7.5e-8
# ---------------------------------------------------------------------------
_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429
_P = 0.3275911


def _standard_norm_cdf(z: float) -> float:
    """CDF of the standard normal distribution N(0,1) at *z*."""
    sign = 1.0 if z >= 0 else -1.0
    x = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + _P * x)
    poly = t * (_A1 + t * (_A2 + t * (_A3 + t * (_A4 + t * _A5))))
    result = 1.0 - poly * math.exp(-(x * x))
    return 0.5 * (1.0 + sign * result)


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------


def implied_probability(odds: int) -> float:
    """Convert American odds to implied win probability (no vig removed)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def implied_probability_no_vig(over_odds: int, under_odds: int) -> tuple[float, float]:
    """Return (p_over, p_under) with vig removed by normalizing to sum=1.0.

    Raw book probabilities sum to > 1.0 due to the bookmaker's margin (vig).
    Dividing each raw probability by their sum removes this systematic bias,
    giving fair-market implied probabilities that sum exactly to 1.0.
    """
    raw_over = implied_probability(over_odds)
    raw_under = implied_probability(under_odds)
    total = raw_over + raw_under
    return raw_over / total, raw_under / total


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal (European) odds."""
    if odds < 0:
        return 1 + 100 / abs(odds)
    return 1 + odds / 100


def prob_over(mu: float, sigma: float, line: float) -> float:
    """Probability that a player exceeds *line* given a N(mu, sigma) projection."""
    if sigma <= 0:
        return 0.0 if line >= mu else 1.0
    z = (line - mu) / sigma
    return 1.0 - _standard_norm_cdf(z)


def prob_under(mu: float, sigma: float, line: float) -> float:
    """Probability that a player falls short of *line* given N(mu, sigma)."""
    if sigma <= 0:
        return 0.0 if line <= mu else 1.0
    z = (line - mu) / sigma
    return _standard_norm_cdf(z)


# ---------------------------------------------------------------------------
# Poisson distribution helpers for discrete markets (fg3m)
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    """Probability mass function P(X=k) for Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _poisson_cdf(k_max: int, lam: float) -> float:
    """Cumulative probability P(X <= k_max) for Poisson(lam).

    Sums PMF up to k_max capped at max(50, 5*lam) to avoid infinite loops.
    """
    if k_max < 0:
        return 0.0
    cap = max(50, int(5 * lam) + 1)
    k_max = min(k_max, cap)
    return sum(_poisson_pmf(k, lam) for k in range(k_max + 1))


def prob_over_poisson(mu: float, line: float) -> float:
    """P(X > line) for a Poisson(mu) variable.

    For half-point lines (e.g. 2.5), over means X >= ceil(line).
    For integer lines (e.g. 3.0), over means X >= line+1 (strictly more than).
    """
    if mu <= 0:
        return 0.0
    # floor(line) gives the largest integer <= line
    k_floor = int(math.floor(line))
    return 1.0 - _poisson_cdf(k_floor, mu)


def prob_under_poisson(mu: float, line: float) -> float:
    """P(X < line) for a Poisson(mu) variable.

    For half-point lines (e.g. 2.5), under means X <= floor(line).
    For integer lines (e.g. 3.0), under means X <= line-1 (strictly less than).
    """
    if mu <= 0:
        return 1.0
    k_floor = int(math.floor(line))
    # If line is an integer, "under" is strictly less than line, so X <= line-1
    if line == k_floor:
        k_floor -= 1
    return _poisson_cdf(k_floor, mu)


def kelly_fraction(win_prob: float, odds: int, fraction: float = 0.25) -> float:
    """Fractional Kelly criterion stake as a fraction of bankroll, capped at 10%."""
    if odds == 0:
        return 0.0
    decimal_odds = american_to_decimal(odds)
    if decimal_odds <= 1:
        return 0.0
    full_kelly = (decimal_odds * win_prob - 1) / (decimal_odds - 1)
    return max(0.0, min(full_kelly * fraction, 0.10))


def expected_roi(win_prob: float, odds: int) -> float:
    """Expected return on a $1 wager (positive = profitable)."""
    if odds == 0:
        return 0.0
    payout = 100 / abs(odds) if odds < 0 else odds / 100
    return win_prob * payout - (1 - win_prob)


# ---------------------------------------------------------------------------
# Player name normalisation for fallback matching
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------


def rank_nba_value(
    game_date: str,
    season: int,
    min_edge: float = 0.08,
    use_monte_carlo: bool = False,
    n_monte_carlo_sims: int = 10_000,
    calibrated: bool = False,
) -> list[dict]:
    """Return ranked NBA value bets for *game_date*.

    Algorithm:
    1. Load projections (with sigma) and odds from the database.
    2. Primary join on player_id + market for odds rows that have a player_id.
    3. Fallback name join for odds rows where player_id is NULL.
    4. Use player-specific sigma from projections (fallback to market default).
    5. Compute p_win, edge, ROI, and Kelly for each matched row.
    6. Run confidence engine to assign score/tier.
    7. Filter to rows with edge_percentage >= min_edge.
    8. Return sorted by edge_percentage descending.
    """
    projections = read_dataframe(
        "SELECT player_id, player_name, team, market, projected_value, confidence, sigma, "
        "volatility_score, usage_rate, predicted_minutes, rate_sigma "
        "FROM nba_projections WHERE game_date = ?",
        (game_date,),
    )
    odds = read_dataframe(
        "SELECT event_id, player_id, player_name, market, sportsbook, "
        "line, over_price, under_price "
        "FROM nba_odds WHERE game_date = ?",
        (game_date,),
    )

    if projections.empty or odds.empty:
        return []

    # ------------------------------------------------------------------ #
    # Injury adjustments: boost projections when teammates are OUT         #
    # Applied BEFORE join so adjusted mu flows into sigma / p_win.        #
    # ------------------------------------------------------------------ #
    proj_rows = projections.to_dict("records")
    proj_rows = apply_injury_adjustments(proj_rows, game_date)
    projections = pd.DataFrame(proj_rows)

    # ------------------------------------------------------------------ #
    # Primary join: player_id (not null) + market                         #
    # ------------------------------------------------------------------ #
    odds_with_id = odds[odds["player_id"].notna()].copy()
    if not odds_with_id.empty:
        # Coerce types to match projections so the merge key is compatible
        odds_with_id["player_id"] = odds_with_id["player_id"].astype(
            projections["player_id"].dtype
        )

    matched = pd.merge(
        projections,
        odds_with_id,
        on=["player_id", "market"],
        how="inner",
        suffixes=("_proj", "_odds"),
    )

    # ------------------------------------------------------------------ #
    # Fallback join: normalised name for odds rows without player_id       #
    # ------------------------------------------------------------------ #
    odds_no_id = odds[odds["player_id"].isna()].copy()
    if not odds_no_id.empty:
        projections_copy = projections.copy()
        projections_copy["_norm_name"] = projections_copy["player_name"].map(_normalize_name)
        odds_no_id["_norm_name"] = odds_no_id["player_name"].map(_normalize_name)

        name_matched = pd.merge(
            projections_copy,
            odds_no_id.drop(columns=["player_id"]),
            on=["_norm_name", "market"],
            how="inner",
            suffixes=("_proj", "_odds"),
        )
        name_matched = name_matched.drop(columns=["_norm_name"], errors="ignore")

        # Align to matched columns before concatenating
        for col in matched.columns:
            if col not in name_matched.columns:
                name_matched[col] = None
        name_matched = name_matched[[c for c in matched.columns if c in name_matched.columns]]

        matched = pd.concat([matched, name_matched], ignore_index=True)

    if matched.empty:
        return []

    # Resolve a single player_name column
    if "player_name_proj" in matched.columns:
        odds_name_col = "player_name_odds" if "player_name_odds" in matched.columns else "player_name"
        matched["player_name"] = matched["player_name_proj"].fillna(matched.get(odds_name_col, ""))

    # ------------------------------------------------------------------ #
    # Per-row value calculations                                           #
    # ------------------------------------------------------------------ #
    results: list[dict] = []

    # ------------------------------------------------------------------ #
    # Calibration: load calibrator once if calibrated=True                #
    # ------------------------------------------------------------------ #
    calibrator = None
    if calibrated:
        try:
            from utils.nba_calibration import NBACalibrator
            _cal = NBACalibrator()
            _cal.load(_DEFAULT_CALIBRATION_PATH)
            calibrator = _cal
        except FileNotFoundError:
            logger.warning(
                "Calibration file not found at %s; proceeding uncalibrated",
                _DEFAULT_CALIBRATION_PATH,
            )
        except Exception as exc:
            logger.warning("Failed to load calibration model: %s", exc)

    for _, row in matched.iterrows():
        try:
            projected_value = float(row["projected_value"])
            line = float(row["line"])
            over_price_raw = row.get("over_price")
            if over_price_raw is None or (
                isinstance(over_price_raw, float) and math.isnan(over_price_raw)
            ):
                continue
            over_price = int(over_price_raw)
        except (TypeError, ValueError):
            continue

        # Injury fields from apply_injury_adjustments
        base_mu = row.get("base_mu")
        if base_mu is not None and not (isinstance(base_mu, float) and math.isnan(base_mu)):
            base_mu = float(base_mu)
        else:
            base_mu = projected_value

        injury_boost = row.get("injury_boost_multiplier")
        if injury_boost is not None and not (isinstance(injury_boost, float) and math.isnan(injury_boost)):
            injury_boost = float(injury_boost)
        else:
            injury_boost = 1.0

        injury_adjusted_mu = row.get("injury_adjusted_mu")
        if injury_adjusted_mu is not None and not (isinstance(injury_adjusted_mu, float) and math.isnan(injury_adjusted_mu)):
            injury_adjusted_mu = float(injury_adjusted_mu)
        else:
            injury_adjusted_mu = projected_value

        injury_players_json = row.get("injury_boost_players")

        # Use player-specific sigma from projections (Phase 1)
        sigma_raw = row.get("sigma") if "sigma" in row.index else None
        market_str = str(row["market"])
        sigma = get_sigma_or_default(sigma_raw, projected_value, market=market_str)

        # Inflate sigma when injury boost is applied — less certainty
        if injury_boost > 1.0:
            sigma *= (1.0 + SIGMA_INFLATION)

        # --- Over / Under evaluation ---
        # Optionally use Monte Carlo simulation instead of analytic CDF.
        mc_p_win_val: float | None = None

        if use_monte_carlo:
            try:
                from utils.nba_monte_carlo import monte_carlo_prob

                raw_pred_min = row.get("predicted_minutes")
                raw_rate_sigma = row.get("rate_sigma")
                minutes_mu_val: float | None = None
                minutes_sigma_val: float | None = None

                if (
                    raw_pred_min is not None
                    and not (isinstance(raw_pred_min, float) and math.isnan(raw_pred_min))
                    and raw_rate_sigma is not None
                    and not (isinstance(raw_rate_sigma, float) and math.isnan(raw_rate_sigma))
                    and market_str != "fg3m"
                ):
                    minutes_mu_val = float(raw_pred_min)
                    minutes_sigma_val = float(raw_rate_sigma)

                p_over, p_under = monte_carlo_prob(
                    mu=injury_adjusted_mu,
                    sigma=sigma,
                    line=line,
                    market=market_str,
                    n_sims=n_monte_carlo_sims,
                    minutes_mu=minutes_mu_val,
                    minutes_sigma=minutes_sigma_val,
                )
                mc_p_win_val = None  # set to chosen side's p_win after side selection
            except Exception as _mc_exc:
                import warnings
                warnings.warn(f"Monte Carlo failed for {row.get('player_name')}: {_mc_exc}; falling back to analytic")
                use_monte_carlo_row = False
                if market_str == "fg3m":
                    p_over = prob_over_poisson(mu=injury_adjusted_mu, line=line)
                    p_under = prob_under_poisson(mu=injury_adjusted_mu, line=line)
                else:
                    p_over = prob_over(mu=injury_adjusted_mu, sigma=sigma, line=line)
                    p_under = prob_under(mu=injury_adjusted_mu, sigma=sigma, line=line)
            else:
                use_monte_carlo_row = True
        else:
            use_monte_carlo_row = False
            # fg3m is a discrete count (0-10 range): use Poisson CDF for accuracy.
            # All other markets use the Normal CDF approximation.
            if market_str == "fg3m":
                p_over = prob_over_poisson(mu=injury_adjusted_mu, line=line)
            else:
                p_over = prob_over(mu=injury_adjusted_mu, sigma=sigma, line=line)

        # --- Under evaluation ---
        under_price_raw = row.get("under_price")
        under_price_val: int | None = None
        if under_price_raw is not None and not (
            isinstance(under_price_raw, float) and math.isnan(under_price_raw)
        ):
            under_price_val = int(under_price_raw)

        if not use_monte_carlo_row:
            if market_str == "fg3m":
                p_under = prob_under_poisson(mu=injury_adjusted_mu, line=line)
            else:
                p_under = prob_under(mu=injury_adjusted_mu, sigma=sigma, line=line)

        # Use no-vig implied probs when both sides are available; fall back to
        # raw implied probability when only the over side is present.
        if under_price_val is not None:
            implied_prob_over, implied_prob_under = implied_probability_no_vig(
                over_price, under_price_val
            )
        else:
            implied_prob_over = implied_probability(over_price)
            implied_prob_under = None

        over_edge = p_over - implied_prob_over
        under_edge = 0.0
        if under_price_val is not None and implied_prob_under is not None:
            under_edge = p_under - implied_prob_under

        # Pick the better side, or skip if neither meets threshold
        if over_edge >= min_edge and over_edge >= under_edge:
            side = "over"
            p_win = p_over
            edge_pct = over_edge
            price_used = over_price
        elif under_edge >= min_edge:
            side = "under"
            p_win = p_under
            edge_pct = under_edge
            price_used = under_price_val
        else:
            continue

        if use_monte_carlo_row:
            mc_p_win_val = p_win

        # Apply probability calibration if requested
        p_win_raw: float | None = None
        is_calibrated = False
        if calibrator is not None:
            p_win_raw = p_win
            p_win = calibrator.calibrate(p_win, market_str)
            # Recompute edge, ROI, Kelly with calibrated p_win
            if side == "over":
                edge_pct = p_win - implied_prob_over
            else:
                edge_pct = p_win - (implied_prob_under if implied_prob_under is not None else implied_probability(price_used))
            is_calibrated = True

        roi = expected_roi(p_win, price_used)
        kelly = kelly_fraction(p_win, price_used)

        under_price: int | None = under_price_val

        # Confidence engine scoring
        raw_fga = row.get("usage_rate")
        fga_share = (
            float(raw_fga)
            if raw_fga is not None and not (isinstance(raw_fga, float) and math.isnan(raw_fga))
            else 0.0
        )

        raw_vol = row.get("volatility_score")
        vol_score = (
            float(raw_vol)
            if raw_vol is not None and not (isinstance(raw_vol, float) and math.isnan(raw_vol))
            else 50.0
        )

        conf_input = {
            "edge_percentage": edge_pct,
            "mu": projected_value,
            "sigma": sigma,
            "fga_share": fga_share,
            "volatility_score": vol_score,
            "usage_spike": injury_boost > 1.0,  # usage spike during injury periods
        }
        confidence_score = compute_nba_confidence_score(conf_input)
        confidence_tier = assign_nba_tier(confidence_score)

        results.append(
            {
                "season": season,
                "game_date": game_date,
                "player_id": row.get("player_id"),
                "player_name": str(row.get("player_name", "")),
                "team": row.get("team"),
                "event_id": str(row["event_id"]),
                "market": market_str,
                "sportsbook": str(row["sportsbook"]),
                "line": line,
                "over_price": over_price,
                "under_price": under_price,
                "mu": projected_value,
                "sigma": sigma,
                "p_win": round(p_win, 6),
                "edge_percentage": round(edge_pct, 6),
                "expected_roi": round(roi, 6),
                "kelly_fraction": round(kelly, 6),
                "confidence": row.get("confidence"),
                "confidence_score": confidence_score,
                "confidence_tier": confidence_tier,
                "base_mu": base_mu,
                "injury_adjusted_mu": injury_adjusted_mu,
                "injury_boost_multiplier": injury_boost,
                "injury_boost_players": injury_players_json,
                "side": side,
                "mc_p_win": mc_p_win_val,
                "p_win_raw": p_win_raw,
                "calibrated": is_calibrated,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    results.sort(key=lambda r: r["edge_percentage"], reverse=True)
    return results


def materialize_nba_value(
    game_date: str,
    season: int,
    min_edge: float = 0.08,
    use_monte_carlo: bool = False,
    n_monte_carlo_sims: int = 10_000,
    calibrated: bool = False,
) -> int:
    """Persist ranked NBA value bets to *nba_materialized_value_view*.

    Returns the number of rows written.
    """
    rows = rank_nba_value(
        game_date=game_date,
        season=season,
        min_edge=min_edge,
        use_monte_carlo=use_monte_carlo,
        n_monte_carlo_sims=n_monte_carlo_sims,
        calibrated=calibrated,
    )

    if not rows:
        return 0

    sql = """
        INSERT OR REPLACE INTO nba_materialized_value_view (
            season, game_date, player_id, player_name, team,
            event_id, market, sportsbook, line, over_price, under_price,
            mu, sigma, p_win, edge_percentage, expected_roi, kelly_fraction,
            confidence, confidence_score, confidence_tier, generated_at,
            base_mu, injury_adjusted_mu, injury_boost_multiplier, injury_boost_players,
            side, p_win_raw, calibrated
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?
        )
    """

    params = [
        (
            r["season"],
            r["game_date"],
            r["player_id"],
            r["player_name"],
            r["team"],
            r["event_id"],
            r["market"],
            r["sportsbook"],
            r["line"],
            r["over_price"],
            r["under_price"],
            r["mu"],
            r["sigma"],
            r["p_win"],
            r["edge_percentage"],
            r["expected_roi"],
            r["kelly_fraction"],
            r["confidence"],
            r.get("confidence_score"),
            r.get("confidence_tier"),
            r["generated_at"],
            r.get("base_mu"),
            r.get("injury_adjusted_mu"),
            r.get("injury_boost_multiplier"),
            r.get("injury_boost_players"),
            r.get("side", "over"),
            r.get("p_win_raw"),
            1 if r.get("calibrated") else 0,
        )
        for r in rows
    ]

    executemany(sql, params)
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute NBA value bets and materialise results to DB."
    )
    parser.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Game date to process (e.g. 2026-02-17)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="NBA season year (default: 2025)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.08,
        dest="min_edge",
        help="Minimum edge threshold (default: 0.08 = 8%%)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        count = materialize_nba_value(
            game_date=args.date,
            season=args.season,
            min_edge=args.min_edge,
        )
        print(
            f"Materialised {count} NBA value bet(s) for {args.date} "
            f"(min_edge={args.min_edge:.0%})"
        )
    except Exception as exc:
        print(f"Error computing NBA value bets: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
