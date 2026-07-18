"""Black-box staging probe contract tests."""

from __future__ import annotations

from scripts.pipeline_soak import run_soak
from scripts.validate_deployed_pipeline_auth import validate


class Response:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_authorization_probe_exercises_reader_and_operator_boundaries(monkeypatch) -> None:
    get_statuses = iter([401, 200])
    post_statuses = iter([401, 403, 200])
    monkeypatch.setattr(
        "scripts.validate_deployed_pipeline_auth.requests.get",
        lambda *args, **kwargs: Response(next(get_statuses)),
    )
    monkeypatch.setattr(
        "scripts.validate_deployed_pipeline_auth.requests.post",
        lambda *args, **kwargs: Response(next(post_statuses)),
    )

    checks = validate(
        base_url="https://staging.example",
        reader_token="reader",
        operator_token="operator",
        season=2025,
        week=8,
    )

    assert all(check.passed for check in checks)
    assert [check.actual for check in checks] == [401, 401, 200, 403, 200]


def test_soak_captures_metrics_and_verifies_returned_job_identity(monkeypatch) -> None:
    created = [
        {"run_id": "run-1", "job_id": "job-1"},
        {"run_id": "run-2", "job_id": "job-2"},
    ]

    def post(*args, **kwargs):
        return Response(200, created.pop(0))

    def get(url, *args, **kwargs):
        if url.endswith("pipeline-metrics"):
            return Response(200, {"queue": {"queued": 0}, "stale_running": 0})
        run_id = url.rsplit("/", 1)[-1]
        return Response(
            200,
            {
                "run_id": run_id,
                "job_id": run_id.replace("run", "job"),
                "status": "completed",
                "stages": [{"name": "prepare_week", "attempt": 1, "status": "completed"}],
            },
        )

    monkeypatch.setattr("scripts.pipeline_soak.requests.post", post)
    monkeypatch.setattr("scripts.pipeline_soak.requests.get", get)

    result = run_soak(
        base_url="https://staging.example",
        operator_token="operator",
        season=2025,
        week=8,
        jobs=2,
        timeout_seconds=1,
        poll_seconds=0.01,
        candidate_sha="a" * 40,
    )

    assert result["passed"] is True
    assert result["candidate_sha"] == "a" * 40
    assert result["metrics_before"]["stale_running"] == 0
    assert result["metrics_after"]["stale_running"] == 0
    assert result["identity_mismatches"] == []
    assert result["duplicate_stage_keys"] == []
