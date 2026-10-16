"""NBA probability calibration using isotonic regression.

Market-specific calibrators (pts/reb/ast/fg3m) trained from historical
nba_bet_outcomes. Falls back to identity calibration when insufficient data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from utils.db import read_dataframe

if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression


_DEFAULT_MIN_SAMPLES = 50
_MARKETS = ["pts", "reb", "ast", "fg3m"]


def _compute_p_win_from_row(row: pd.Series) -> float | None:
    """Derive raw probability proxy from nba_bet_outcomes row.

    Uses edge_at_placement as proxy if available; otherwise falls back to
    a conservative 0.5 baseline (not useful for calibration, so returns None).
    """
    edge = row.get("edge_at_placement")
    if edge is None or (isinstance(edge, float) and pd.isna(edge)):
        return None
    # edge_at_placement = p_win - implied_prob, so p_win ~ implied_prob + edge
    # Without implied_prob stored, treat edge+0.5 as a rough proxy.
    # Clip to valid probability range.
    raw_p = float(edge) + 0.5
    return max(0.0, min(1.0, raw_p))


class NBACalibrator:
    """Market-specific probability calibration using isotonic regression."""

    def __init__(self) -> None:
        self._calibrators: dict[str, IsotonicRegression] = {}
        self._sample_counts: dict[str, int] = {}

    def fit(self, min_samples: int = _DEFAULT_MIN_SAMPLES) -> dict[str, int]:
        """Train calibrators from nba_bet_outcomes table.

        For each market (pts/reb/ast/fg3m):
        1. Load rows from nba_bet_outcomes where result is not NULL.
        2. Extract raw probability proxy from edge_at_placement.
        3. Binary outcome: result == 'win' -> 1, else -> 0.
        4. Fit IsotonicRegression if n_samples >= min_samples.

        Returns {market: n_samples} for markets with sufficient data.
        """
        from sklearn.isotonic import IsotonicRegression

        sample_counts: dict[str, int] = {}
        calibrators: dict[str, IsotonicRegression] = {}

        outcomes = read_dataframe(
            "SELECT market, result, edge_at_placement "
            "FROM nba_bet_outcomes WHERE result IS NOT NULL"
        )

        if outcomes.empty:
            self._calibrators = {}
            self._sample_counts = {}
            return {}

        for market in _MARKETS:
            market_df = outcomes[outcomes["market"] == market].copy()

            if market_df.empty:
                continue

            # Compute p_win proxy for each row
            p_raw_values = [_compute_p_win_from_row(row) for _, row in market_df.iterrows()]
            valid_mask = [p is not None for p in p_raw_values]
            p_raw_valid = [p for p in p_raw_values if p is not None]

            valid_df = market_df[valid_mask].copy()
            y = (valid_df["result"].str.lower() == "win").astype(int).tolist()

            n = len(p_raw_valid)
            if n < min_samples:
                continue

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit([[p] for p in p_raw_valid], y)

            calibrators[market] = iso
            sample_counts[market] = n

        self._calibrators = calibrators
        self._sample_counts = sample_counts
        return dict(sample_counts)

    def calibrate(self, p_raw: float, market: str) -> float:
        """Apply calibration. Falls back to identity if no calibrator for market."""
        if market not in self._calibrators:
            return p_raw
        calibrated = float(self._calibrators[market].predict([[p_raw]])[0])
        return max(0.0, min(1.0, calibrated))

    def save(self, path: str) -> None:
        """Save calibrators to joblib file."""
        import joblib

        joblib.dump(
            {
                "calibrators": self._calibrators,
                "sample_counts": self._sample_counts,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load calibrators from joblib file."""
        import joblib

        data = joblib.load(path)
        self._calibrators = data["calibrators"]
        self._sample_counts = data["sample_counts"]

