"""Point-in-time evaluation coverage for persisted NFL projections."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_nfl_projections import compare_reports, evaluate_projections


def _inputs(
    *, candidate_mu: float = 55.0, candidate_sha: str = "a" * 40
) -> tuple[pd.DataFrame, ...]:
    projections = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "player_id": "p1",
                "team": "BUF",
                "market": "receiving_yards",
                "mu": candidate_mu,
                "model_version": "candidate-v1",
                "featureset_hash": "features-v1",
                "generated_at": "2025-09-03T12:00:00Z",
            },
            {
                "season": 2025,
                "week": 1,
                "player_id": "p2",
                "team": "MIA",
                "market": "rushing_yards",
                "mu": 40.0,
                "model_version": "candidate-v1",
                "featureset_hash": "features-v1",
                "generated_at": "2025-09-03T12:00:00Z",
            },
        ]
    )
    actuals = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "player_id": "p1",
                "receiving_yards": 50.0,
                "rushing_yards": 0.0,
            },
            {
                "season": 2025,
                "week": 1,
                "player_id": "p2",
                "receiving_yards": 0.0,
                "rushing_yards": 50.0,
            },
        ]
    )
    games = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "BUF",
                "away_team": "MIA",
                "kickoff_utc": "2025-09-04T17:00:00Z",
            }
        ]
    )
    runs = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "season": 2025,
                "week": 1,
                "status": "completed",
                "started_at": "2025-09-03T11:00:00Z",
                "finished_at": "2025-09-03T13:00:00Z",
                "report_json": '{"commit_sha":"' + candidate_sha + '"}',
            }
        ]
    )
    return projections, actuals, games, runs


def test_evaluation_scores_only_pregame_production_outputs() -> None:
    report = evaluate_projections(*_inputs(), candidate_sha="a" * 40)

    assert report["passed"] is True
    assert report["candidate_sha"] == "a" * 40
    assert report["scope"] == {"season_weeks": [{"season": 2025, "week": 1}]}
    assert report["metrics"]["projection_count"] == 2
    assert report["metrics"]["mae"] == pytest.approx(7.5)
    assert report["metrics"]["mean_bias"] == pytest.approx(-2.5)
    assert report["metrics"]["by_market"]["receiving_yards"]["mae"] == 5.0
    assert report["metrics"]["by_market"]["rushing_yards"]["mae"] == 10.0


def test_evaluation_rejects_run_from_another_commit() -> None:
    report = evaluate_projections(*_inputs(), candidate_sha="b" * 40)

    assert report["passed"] is False
    assert "completed run producer SHA does not match candidate SHA" in report["blockers"]


def test_evaluation_rejects_projection_outside_completed_run_window() -> None:
    projections, actuals, games, runs = _inputs()
    projections["generated_at"] = "2025-09-03T14:00:00Z"

    report = evaluate_projections(
        projections,
        actuals,
        games,
        runs,
        candidate_sha="a" * 40,
    )

    assert report["passed"] is False
    assert "projections are not bound to the completed producer run" in report["blockers"]


def test_post_kickoff_projection_fails_closed() -> None:
    projections, actuals, games, runs = _inputs(candidate_sha="b" * 40)
    projections.loc[0, "generated_at"] = "2025-09-04T18:00:00Z"

    report = evaluate_projections(projections, actuals, games, runs, candidate_sha="b" * 40)

    assert report["passed"] is False
    assert report["metrics"]["projection_count"] == 1
    assert report["freshness_failures"] == {"projection_after_kickoff": 1}
    assert "projection freshness violations are present" in report["blockers"]


def test_missing_actuals_do_not_become_zero_error() -> None:
    projections, actuals, games, runs = _inputs(candidate_sha="c" * 40)
    actuals = actuals.iloc[0:0]

    report = evaluate_projections(projections, actuals, games, runs, candidate_sha="c" * 40)

    assert report["passed"] is False
    assert report["metrics"]["projection_count"] == 0
    assert "no eligible projections with actual outcomes" in report["blockers"]


def test_partial_actuals_fail_instead_of_improving_score_by_exclusion() -> None:
    projections, actuals, games, runs = _inputs(candidate_sha="d" * 40)
    actuals = actuals[actuals["player_id"] == "p1"]

    report = evaluate_projections(projections, actuals, games, runs, candidate_sha="d" * 40)

    assert report["passed"] is False
    assert report["outcome_failures"] == {"missing_actual": 1}
    assert "projection outcome coverage is incomplete" in report["blockers"]


def test_candidate_comparison_requires_real_overall_improvement() -> None:
    baseline = evaluate_projections(*_inputs(candidate_mu=60.0), candidate_sha="a" * 40)
    candidate = evaluate_projections(
        *_inputs(candidate_mu=52.0, candidate_sha="b" * 40), candidate_sha="b" * 40
    )

    comparison = compare_reports(
        baseline,
        candidate,
        min_improvement_pct=10.0,
        max_market_regression_pct=5.0,
    )

    assert comparison["passed"] is True
    assert comparison["overall"]["mae_improvement_pct"] == pytest.approx(40.0)


def test_candidate_comparison_rejects_market_regression() -> None:
    baseline = evaluate_projections(*_inputs(candidate_mu=55.0), candidate_sha="a" * 40)
    projections, actuals, games, runs = _inputs(candidate_mu=58.0, candidate_sha="b" * 40)
    projections.loc[1, "mu"] = 48.0
    candidate = evaluate_projections(projections, actuals, games, runs, candidate_sha="b" * 40)

    comparison = compare_reports(
        baseline,
        candidate,
        min_improvement_pct=0.0,
        max_market_regression_pct=5.0,
    )

    assert comparison["passed"] is False
    assert any("receiving_yards MAE regressed" in item for item in comparison["blockers"])


def test_candidate_comparison_rejects_different_evaluation_scope() -> None:
    baseline = evaluate_projections(*_inputs(), candidate_sha="a" * 40)
    projections, actuals, games, runs = _inputs(candidate_sha="b" * 40)
    projections["week"] = 2
    actuals["week"] = 2
    games["week"] = 2
    runs["week"] = 2
    candidate = evaluate_projections(projections, actuals, games, runs, candidate_sha="b" * 40)

    comparison = compare_reports(
        baseline,
        candidate,
        min_improvement_pct=0.0,
        max_market_regression_pct=5.0,
    )

    assert comparison["passed"] is False
    assert "evaluation scope differs between baseline and candidate" in comparison["blockers"]


def test_candidate_comparison_requires_positive_improvement_threshold() -> None:
    baseline = evaluate_projections(*_inputs(), candidate_sha="a" * 40)
    candidate = evaluate_projections(*_inputs(candidate_sha="b" * 40), candidate_sha="b" * 40)

    comparison = compare_reports(
        baseline,
        candidate,
        min_improvement_pct=0.0,
        max_market_regression_pct=5.0,
    )

    assert comparison["passed"] is False
    assert "minimum improvement must be greater than zero" in comparison["blockers"]
