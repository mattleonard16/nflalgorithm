"""Screen odds snapshots that must not reach the value or CLV path.

Pure functions only — no database access — so the rules stay testable in CI
even though the writers and the grading script are gitignored.

Two independent disqualifiers, deliberately kept separate because they have
different lifetimes:

- **Unjoinable rows.** A snapshot whose ``event_id`` is not a canonical game
  key cannot be tied to a kickoff, so it can be neither staleness-filtered nor
  graded against a closing line. Legacy writers minted per-player ids
  (``2025_W22_NE_a_hooper``) and provider ids, which join to zero ``games``
  rows. New writes are blocked at the source; this screens what is already
  stored.

- **Synthetic rows.** ``SimBook`` lines are derived from the same week's
  realized yardage, so an "edge" measured against them is circular: the line
  already knows the outcome. They are useful for exercising the pipeline and
  ruinous for measuring it.

Callers filter rather than delete, so the rows stay available for debugging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.event_keys import is_canonical_event_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# Book name used by the synthetic line generator in data_pipeline.
SYNTHETIC_SPORTSBOOK = "SimBook"


def is_synthetic_book(sportsbook: object) -> bool:
    """Return whether ``sportsbook`` names the synthetic line generator."""
    return isinstance(sportsbook, str) and sportsbook.strip().casefold() == (
        SYNTHETIC_SPORTSBOOK.casefold()
    )


def filter_gradeable_odds(
    odds_df: "pd.DataFrame", *, drop_synthetic: bool = True
) -> "pd.DataFrame":
    """Drop snapshots that cannot honestly be graded.

    Args:
        odds_df: Rows from ``weekly_odds``. An ``event_id`` column is required;
            ``sportsbook`` is required only when ``drop_synthetic`` is set.
        drop_synthetic: Whether to also drop the circular ``SimBook`` rows.
            Callers measuring real edge want this; callers rendering a demo
            dashboard may not.

    Returns:
        A copy holding only rows joinable to ``games`` and, by default, priced
        by a real book. Column order and dtypes are preserved.
    """
    if odds_df.empty:
        return odds_df.copy()

    if "event_id" not in odds_df.columns:
        raise KeyError("odds_df has no event_id column; cannot screen unjoinable rows")

    keep = odds_df["event_id"].map(is_canonical_event_id)

    if drop_synthetic:
        if "sportsbook" not in odds_df.columns:
            raise KeyError("odds_df has no sportsbook column; cannot screen synthetic rows")
        keep &= ~odds_df["sportsbook"].map(is_synthetic_book)

    return odds_df[keep].copy()


def describe_excluded(odds_df: "pd.DataFrame") -> dict[str, int]:
    """Count why rows would be excluded, for logging before a grading run.

    A run that silently grades nothing is indistinguishable from a run with no
    bets; these counts let the caller say which happened.
    """
    if odds_df.empty:
        return {"total": 0, "unjoinable": 0, "synthetic": 0, "gradeable": 0}

    unjoinable = ~odds_df["event_id"].map(is_canonical_event_id)
    synthetic = (
        odds_df["sportsbook"].map(is_synthetic_book)
        if "sportsbook" in odds_df.columns
        else unjoinable & False
    )
    return {
        "total": int(len(odds_df)),
        "unjoinable": int(unjoinable.sum()),
        "synthetic": int(synthetic.sum()),
        "gradeable": int((~unjoinable & ~synthetic).sum()),
    }
