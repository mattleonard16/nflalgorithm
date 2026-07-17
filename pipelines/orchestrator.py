"""Sport-neutral sequential pipeline orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

VALID_STATUSES = frozenset({"ok", "error", "skipped"})


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """A named, zero-argument stage bound to one sport-specific context."""

    name: str
    handler: Callable[[], Mapping[str, Any]]


def _validate_names(stages: Sequence[PipelineStage]) -> set[str]:
    names = [stage.name for stage in stages]
    name_set = set(names)
    if len(names) != len(name_set):
        raise ValueError("Pipeline stage names must be unique")
    return name_set


def _normalize_result(stage_name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    reported_stage = normalized.setdefault("stage", stage_name)
    if reported_stage != stage_name:
        raise ValueError(f"Pipeline stage '{stage_name}' returned result for '{reported_stage}'")
    status = normalized.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(f"Pipeline stage '{stage_name}' returned invalid status '{status}'")
    return normalized


def run_stages(
    stages: Sequence[PipelineStage],
    *,
    only: Collection[str] | None = None,
    skip: Mapping[str, str] | None = None,
    stop_on_error: bool = True,
    stop_after_skip: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Run bound stages with explicit selection, skip, and failure policies."""
    known_names = _validate_names(stages)
    selected = set(only) if only is not None else None
    skipped = dict(skip or {})

    unknown = ((selected or set()) | set(skipped) | set(stop_after_skip)) - known_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown pipeline stage(s): {names}")

    results: list[dict[str, Any]] = []
    for stage in stages:
        if selected is not None and stage.name not in selected:
            continue
        if stage.name in skipped:
            results.append(
                {
                    "status": "skipped",
                    "stage": stage.name,
                    "reason": skipped[stage.name],
                }
            )
            if stage.name in stop_after_skip:
                break
            continue

        logger.info("Running pipeline stage: %s", stage.name)
        try:
            raw_result = stage.handler()
        except Exception as exc:
            logger.exception("Pipeline stage %s raised unexpectedly", stage.name)
            message = str(exc)
            result = {
                "status": "error",
                "stage": stage.name,
                "error": message,
                "detail": message,
            }
        else:
            result = _normalize_result(stage.name, raw_result)
        results.append(result)
        logger.info("Pipeline stage %s: %s", stage.name, result["status"])

        if result["status"] == "error" and stop_on_error:
            break

    return results
