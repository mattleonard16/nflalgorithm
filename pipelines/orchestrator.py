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
    retry_safe: bool = False


def _validate_names(stages: Sequence[PipelineStage]) -> set[str]:
    names = [stage.name for stage in stages]
    name_set = set(names)
    if len(names) != len(name_set):
        raise ValueError("Pipeline stage names must be unique")
    return name_set


def _normalize_result(stage: PipelineStage, result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    reported_stage = normalized.setdefault("stage", stage.name)
    if reported_stage != stage.name:
        raise ValueError(f"Pipeline stage '{stage.name}' returned result for '{reported_stage}'")
    status = normalized.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(f"Pipeline stage '{stage.name}' returned invalid status '{status}'")
    if stage.retry_safe:
        normalized["retry_safe"] = True
    return normalized


def run_stages(
    stages: Sequence[PipelineStage],
    *,
    only: Collection[str] | None = None,
    skip: Mapping[str, str] | None = None,
    stop_on_error: bool = True,
    stop_after_skip: Collection[str] = (),
    on_stage_start: Callable[[str, int], None] | None = None,
    on_stage_result: Callable[[str, int, Mapping[str, Any]], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
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
    for ordinal, stage in enumerate(stages):
        if cancellation_requested and cancellation_requested():
            logger.info("Pipeline cancellation requested before stage %s", stage.name)
            break
        if selected is not None and stage.name not in selected:
            continue
        if stage.name in skipped:
            result = {
                "status": "skipped",
                "stage": stage.name,
                "reason": skipped[stage.name],
            }
            results.append(result)
            if on_stage_result:
                on_stage_result(stage.name, ordinal, result)
            if stage.name in stop_after_skip:
                break
            continue

        logger.info("Running pipeline stage: %s", stage.name)
        if on_stage_start:
            on_stage_start(stage.name, ordinal)
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
            if stage.retry_safe:
                result["retry_safe"] = True
        else:
            result = _normalize_result(stage, raw_result)
        results.append(result)
        if on_stage_result:
            on_stage_result(stage.name, ordinal, result)
        logger.info("Pipeline stage %s: %s", stage.name, result["status"])

        if result["status"] == "error" and stop_on_error:
            break

    return results
