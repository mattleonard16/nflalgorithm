"""Fail-closed staging promotion evidence tests."""

from __future__ import annotations

from scripts.pipeline_release_evidence import REQUIRED_EVIDENCE, build_manifest

CANDIDATE_SHA = "a" * 40
MAIN_SHA = "b" * 40
BASELINE_SHA = "c" * 40


def _complete_evidence() -> dict:
    evidence = {
        name: {
            "passed": True,
            "candidate_sha": CANDIDATE_SHA,
            "evidence": f"{name}-proof",
        }
        for name in REQUIRED_EVIDENCE
    }
    evidence["shadow_weekly_run"]["baseline_sha"] = BASELINE_SHA
    evidence["algorithm_evaluation"]["baseline_sha"] = BASELINE_SHA
    return evidence


def test_release_manifest_passes_only_with_every_bound_gate() -> None:
    manifest = build_manifest(
        candidate_sha=CANDIDATE_SHA,
        baseline_sha=BASELINE_SHA,
        origin_main_sha=MAIN_SHA,
        origin_main_is_ancestor=True,
        evidence=_complete_evidence(),
    )

    assert manifest["passed"] is True
    assert manifest["blockers"] == []
    assert set(manifest["gates"]) == set(REQUIRED_EVIDENCE) | {
        "known_commit_sha",
        "known_baseline_sha",
        "integrated_with_origin_main",
    }


def test_release_manifest_rejects_missing_gate() -> None:
    evidence = _complete_evidence()
    del evidence["rollback_rehearsal"]

    manifest = build_manifest(
        candidate_sha=CANDIDATE_SHA,
        baseline_sha=BASELINE_SHA,
        origin_main_sha=MAIN_SHA,
        origin_main_is_ancestor=True,
        evidence=evidence,
    )

    assert manifest["passed"] is False
    assert "rollback_rehearsal evidence is missing" in manifest["blockers"]


def test_release_manifest_rejects_evidence_from_another_sha() -> None:
    evidence = _complete_evidence()
    evidence["staging_soak"]["candidate_sha"] = "c" * 40

    manifest = build_manifest(
        candidate_sha=CANDIDATE_SHA,
        baseline_sha=BASELINE_SHA,
        origin_main_sha=MAIN_SHA,
        origin_main_is_ancestor=True,
        evidence=evidence,
    )

    assert manifest["passed"] is False
    assert "staging_soak evidence belongs to a different candidate SHA" in manifest["blockers"]


def test_release_manifest_rejects_non_ancestor_main_and_failed_proof() -> None:
    evidence = _complete_evidence()
    evidence["monitoring_delivery"]["passed"] = False

    manifest = build_manifest(
        candidate_sha=CANDIDATE_SHA,
        baseline_sha=BASELINE_SHA,
        origin_main_sha=MAIN_SHA,
        origin_main_is_ancestor=False,
        evidence=evidence,
    )

    assert manifest["passed"] is False
    assert "origin/main is not an ancestor of the candidate" in manifest["blockers"]
    assert "monitoring_delivery evidence did not pass" in manifest["blockers"]


def test_release_manifest_rejects_shadow_from_another_baseline() -> None:
    evidence = _complete_evidence()
    evidence["shadow_weekly_run"]["baseline_sha"] = "d" * 40

    manifest = build_manifest(
        candidate_sha=CANDIDATE_SHA,
        baseline_sha=BASELINE_SHA,
        origin_main_sha=MAIN_SHA,
        origin_main_is_ancestor=True,
        evidence=evidence,
    )

    assert manifest["passed"] is False
    assert "shadow_weekly_run evidence belongs to a different baseline SHA" in manifest["blockers"]


def test_release_manifest_requires_algorithm_evaluation_from_same_baseline() -> None:
    evidence = _complete_evidence()
    evidence["algorithm_evaluation"]["baseline_sha"] = "d" * 40

    manifest = build_manifest(
        candidate_sha=CANDIDATE_SHA,
        baseline_sha=BASELINE_SHA,
        origin_main_sha=MAIN_SHA,
        origin_main_is_ancestor=True,
        evidence=evidence,
    )

    assert manifest["passed"] is False
    assert (
        "algorithm_evaluation evidence belongs to a different baseline SHA" in manifest["blockers"]
    )
