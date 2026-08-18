"""Describe what trained a persisted model bundle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def build_provenance(
    *,
    season_week_tuples: Sequence[tuple[int, int]],
    feature_cols: Sequence[str],
    training_rows: int,
    private_model_path: Path,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "season_week_tuples": [list(pair) for pair in season_week_tuples],
        "feature_cols": list(feature_cols),
        "training_rows": int(training_rows),
        "source_path": str(private_model_path),
        "evaluation": evaluation or {},
    }


def describe_bundle_provenance(bundle: dict[str, Any]) -> str:
    provenance = bundle.get("provenance") or {}
    if not provenance:
        return "no provenance"
    trained_at = provenance.get("trained_at", "unknown")
    rows = provenance.get("training_rows", "?")
    return f"trained_at={trained_at} rows={rows}"
