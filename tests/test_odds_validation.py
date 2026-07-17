"""Production live-odds freshness and coverage gates."""

from __future__ import annotations

from pipelines.odds_validation import OddsRequirements, validate_odds_snapshot

REQUIREMENTS = OddsRequirements(
    max_age_seconds=300,
    min_event_coverage=1.0,
    min_market_coverage=1.0,
    required_markets=("pass", "rush", "receive"),
)


def complete_observation() -> dict[str, object]:
    return {
        "source_statuses": ["MISS"],
        "response_ages_seconds": [1.5, 2.0],
        "responses_observed": 2,
        "snapshot_at": "2026-09-01T12:00:00+00:00",
        "scheduled_events": 2,
        "covered_events": 2,
        "covered_event_markets": 6,
        "odds_rows": 24,
    }


def test_fresh_complete_live_snapshot_is_accepted() -> None:
    result = validate_odds_snapshot(complete_observation(), requirements=REQUIREMENTS)

    assert result["valid"] is True
    assert result["reason_code"] == "validated"


def test_stale_odds_cache_is_rejected() -> None:
    observed = complete_observation()
    observed["source_statuses"] = ["STALE-ON-ERROR"]

    result = validate_odds_snapshot(observed, requirements=REQUIREMENTS)

    assert result["valid"] is False
    assert result["reason_code"] == "stale_cache"


def test_offline_odds_cache_is_rejected() -> None:
    observed = complete_observation()
    observed["source_statuses"] = ["HIT-OFFLINE"]

    result = validate_odds_snapshot(observed, requirements=REQUIREMENTS)

    assert result["valid"] is False
    assert result["reason_code"] == "offline_cache"


def test_incomplete_response_freshness_is_rejected() -> None:
    observed = complete_observation()
    observed["responses_observed"] = 3

    result = validate_odds_snapshot(observed, requirements=REQUIREMENTS)

    assert result["valid"] is False
    assert result["reason_code"] == "provenance_incomplete"
    assert result["responses_with_age"] == 2


def test_unknown_response_provenance_is_rejected() -> None:
    observed = complete_observation()
    observed["source_statuses"] = ["UNKNOWN"]

    result = validate_odds_snapshot(observed, requirements=REQUIREMENTS)

    assert result["valid"] is False
    assert result["reason_code"] == "untrusted_source"


def test_partial_odds_coverage_is_rejected() -> None:
    observed = complete_observation()
    observed["covered_event_markets"] = 5

    result = validate_odds_snapshot(observed, requirements=REQUIREMENTS)

    assert result["valid"] is False
    assert result["reason_code"] == "market_coverage"
    assert result["market_coverage"] == 5 / 6
