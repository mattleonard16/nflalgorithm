"""Tests for the shared multi-sport pipeline orchestrator."""

from __future__ import annotations

import pytest

from pipelines.orchestrator import PipelineStage, run_stages


def test_run_stages_preserves_order_and_normalizes_stage_name() -> None:
    calls: list[str] = []
    stages = [
        PipelineStage("ingest", lambda: calls.append("ingest") or {"status": "ok"}),
        PipelineStage("predict", lambda: calls.append("predict") or {"status": "ok"}),
    ]

    results = run_stages(stages)

    assert calls == ["ingest", "predict"]
    assert results == [
        {"status": "ok", "stage": "ingest"},
        {"status": "ok", "stage": "predict"},
    ]


def test_blocking_pipeline_stops_after_error() -> None:
    calls: list[str] = []
    stages = [
        PipelineStage("odds", lambda: {"status": "error", "error": "stale"}),
        PipelineStage("value", lambda: calls.append("value") or {"status": "ok"}),
    ]

    results = run_stages(stages, stop_on_error=True)

    assert [result["stage"] for result in results] == ["odds"]
    assert calls == []


def test_nonblocking_pipeline_can_continue_after_error() -> None:
    stages = [
        PipelineStage("ingest", lambda: {"status": "error", "error": "rate limit"}),
        PipelineStage("drift", lambda: {"status": "ok"}),
    ]

    results = run_stages(stages, stop_on_error=False)

    assert [result["stage"] for result in results] == ["ingest", "drift"]


def test_skip_can_halt_downstream_consumers() -> None:
    stages = [
        PipelineStage("odds", lambda: pytest.fail("skipped stage must not run")),
        PipelineStage("value", lambda: pytest.fail("downstream stage must not run")),
    ]

    results = run_stages(
        stages,
        skip={"odds": "skip_odds"},
        stop_after_skip={"odds"},
    )

    assert results == [{"status": "skipped", "stage": "odds", "reason": "skip_odds"}]


def test_unknown_selected_stage_is_rejected() -> None:
    stages = [PipelineStage("ingest", lambda: {"status": "ok"})]

    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        run_stages(stages, only={"predict"})


def test_unknown_stop_after_skip_stage_is_rejected() -> None:
    stages = [PipelineStage("odds", lambda: {"status": "ok"})]

    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        run_stages(stages, stop_after_skip={"value"})


def test_duplicate_stage_names_are_rejected() -> None:
    stages = [
        PipelineStage("predict", lambda: {"status": "ok"}),
        PipelineStage("predict", lambda: {"status": "ok"}),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        run_stages(stages)


def test_invalid_stage_result_is_rejected() -> None:
    stages = [PipelineStage("ingest", lambda: {"status": "maybe"})]

    with pytest.raises(ValueError, match="invalid status"):
        run_stages(stages)


def test_stage_cannot_report_another_stage_name() -> None:
    stages = [PipelineStage("predict", lambda: {"status": "ok", "stage": "odds"})]

    with pytest.raises(ValueError, match="returned result for 'odds'"):
        run_stages(stages)


def test_handler_exceptions_become_stage_errors() -> None:
    def invalid_input() -> dict[str, str]:
        raise ValueError("bad slate")

    results = run_stages([PipelineStage("ingest", invalid_input)])

    assert results == [
        {
            "status": "error",
            "stage": "ingest",
            "error": "bad slate",
            "detail": "bad slate",
        }
    ]
