"""Train NBA probability calibration model from historical outcomes.

Reads from nba_bet_outcomes table, fits market-specific isotonic regression
calibrators, and saves the model to models/nba/calibration.joblib.

Usage:
    DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python scripts/train_nba_calibration.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import executemany, read_dataframe
from utils.nba_calibration import NBACalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models" / "nba"
OUTPUT_PATH = MODEL_DIR / "calibration.joblib"
MIN_SAMPLES = 50


def _compute_brier_score(calibrator: NBACalibrator, market: str) -> float | None:
    """Compute Brier score for a calibrated market using leave-one-out estimate.

    Brier score = mean((p_calibrated - y)^2), lower is better.
    Returns None if insufficient data.
    """
    outcomes = read_dataframe(
        "SELECT result, edge_at_placement FROM nba_bet_outcomes "
        "WHERE result IS NOT NULL AND market = ?",
        (market,),
    )
    if outcomes.empty:
        return None

    from utils.nba_calibration import _compute_p_win_from_row

    p_raw_list = [_compute_p_win_from_row(row) for _, row in outcomes.iterrows()]
    valid_pairs = [
        (p, (row["result"].lower() == "win"))
        for p, (_, row) in zip(p_raw_list, outcomes.iterrows())
        if p is not None
    ]

    if len(valid_pairs) < MIN_SAMPLES:
        return None

    brier = sum(
        (calibrator.calibrate(p, market) - float(y)) ** 2
        for p, y in valid_pairs
    ) / len(valid_pairs)
    return round(brier, 6)


def _save_training_metadata(
    trained_at: str,
    market: str,
    n_samples: int,
    brier_score: float | None,
) -> None:
    """Write calibration training run to nba_calibration_history."""
    executemany(
        "INSERT OR REPLACE INTO nba_calibration_history "
        "(trained_at, market, n_samples, brier_score) "
        "VALUES (?, ?, ?, ?)",
        [(trained_at, market, n_samples, brier_score)],
    )


def main() -> None:
    logger.info("Training NBA probability calibration model...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    calibrator = NBACalibrator()
    sample_counts = calibrator.fit(min_samples=MIN_SAMPLES)

    if not sample_counts:
        logger.warning(
            "No markets have sufficient data (>= %d samples). "
            "Model will use identity calibration. "
            "Run outcome recording first to accumulate data.",
            MIN_SAMPLES,
        )
    else:
        logger.info("Calibrated markets: %s", list(sample_counts.keys()))

    trained_at = datetime.now(timezone.utc).isoformat()

    for market, n_samples in sample_counts.items():
        brier = _compute_brier_score(calibrator, market)
        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        logger.info(
            "  %s: n_samples=%d  brier_score=%s",
            market,
            n_samples,
            brier_str,
        )
        _save_training_metadata(trained_at, market, n_samples, brier)

    calibrator.save(str(OUTPUT_PATH))
    logger.info("Calibration model saved to %s", OUTPUT_PATH)

    # Report markets that were skipped
    all_markets = {"pts", "reb", "ast", "fg3m"}
    skipped = all_markets - set(sample_counts.keys())
    if skipped:
        logger.info(
            "Skipped markets (< %d samples): %s — using identity calibration",
            MIN_SAMPLES,
            sorted(skipped),
        )


if __name__ == "__main__":
    main()
