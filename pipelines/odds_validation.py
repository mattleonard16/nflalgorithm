"""Fail-closed validation for point-in-time NFL odds snapshots."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from config import config


@dataclass(frozen=True, slots=True)
class OddsRequirements:
    max_age_seconds: int
    min_event_coverage: float
    min_market_coverage: float
    min_sportsbooks_per_event_market: int
    required_markets: tuple[str, ...]

    @classmethod
    def from_config(cls) -> "OddsRequirements":
        return cls(
            max_age_seconds=int(config.pipeline.odds_max_age_seconds),
            min_event_coverage=float(config.pipeline.odds_min_event_coverage),
            min_market_coverage=float(config.pipeline.odds_min_market_coverage),
            min_sportsbooks_per_event_market=int(
                config.pipeline.odds_min_sportsbooks_per_event_market
            ),
            required_markets=tuple(config.pipeline.odds_required_markets),
        )

    def validate(self) -> None:
        if self.max_age_seconds <= 0:
            raise ValueError("odds max age must be positive")
        if not 0 < self.min_event_coverage <= 1:
            raise ValueError("minimum event coverage must be in (0, 1]")
        if not 0 < self.min_market_coverage <= 1:
            raise ValueError("minimum market coverage must be in (0, 1]")
        if self.min_sportsbooks_per_event_market <= 0:
            raise ValueError("minimum sportsbooks per event-market must be positive")
        if not self.required_markets:
            raise ValueError("at least one required odds market must be configured")


def validate_odds_snapshot(
    observed: Mapping[str, Any],
    *,
    requirements: OddsRequirements | None = None,
) -> dict[str, Any]:
    """Return a persisted-safe decision with explicit reason and coverage metrics."""
    required = requirements or OddsRequirements.from_config()
    required.validate()

    source_statuses = sorted({str(value).upper() for value in observed.get("source_statuses", [])})
    scheduled_events = int(observed.get("scheduled_events", 0))
    covered_events = int(observed.get("covered_events", 0))
    covered_event_markets = int(observed.get("covered_event_markets", 0))
    required_event_markets = scheduled_events * len(required.required_markets)
    odds_rows = int(observed.get("odds_rows", 0))
    response_ages = [float(value) for value in observed.get("response_ages_seconds", [])]
    responses_observed = int(observed.get("responses_observed", len(response_ages)))
    responses_with_age = len(response_ages)
    max_response_age = max(response_ages) if response_ages else None
    event_coverage = covered_events / scheduled_events if scheduled_events else 0.0
    market_coverage = (
        covered_event_markets / required_event_markets if required_event_markets else 0.0
    )
    raw_sportsbook_counts = observed.get("sportsbooks_per_event_market", {})
    if not isinstance(raw_sportsbook_counts, Mapping):
        raise ValueError("sportsbooks_per_event_market must be a mapping")
    sportsbook_counts = {str(pair): int(count) for pair, count in raw_sportsbook_counts.items()}
    if any(count < 0 for count in sportsbook_counts.values()):
        raise ValueError("sportsbook counts cannot be negative")
    sportsbook_pairs_reported = len(sportsbook_counts)
    sportsbook_qualified_event_markets = sum(
        count >= required.min_sportsbooks_per_event_market for count in sportsbook_counts.values()
    )
    sportsbook_coverage = (
        sportsbook_qualified_event_markets / required_event_markets
        if required_event_markets
        else 0.0
    )
    min_sportsbooks_observed = min(sportsbook_counts.values()) if sportsbook_counts else 0

    reason_code = "validated"
    reason = "Odds snapshot meets freshness and coverage requirements"
    if not source_statuses or responses_observed <= 0 or not response_ages:
        reason_code = "provenance_missing"
        reason = "Odds response provenance or age is missing"
    elif responses_with_age != responses_observed:
        reason_code = "provenance_incomplete"
        reason = (
            f"Only {responses_with_age}/{responses_observed} odds responses have freshness metadata"
        )
    elif any(not math.isfinite(age) or age < 0 for age in response_ages):
        reason_code = "invalid_response_age"
        reason = "Odds response age metadata is invalid"
    elif any(status in {"HIT-OFFLINE", "FALLBACK-SNAPSHOT"} for status in source_statuses):
        reason_code = "offline_cache"
        reason = "Offline cache cannot authorize a production betting card"
    elif any("STALE" in status for status in source_statuses):
        reason_code = "stale_cache"
        reason = "Stale-on-error cache cannot authorize a production betting card"
    elif any(status not in {"HIT", "MISS"} for status in source_statuses):
        reason_code = "untrusted_source"
        reason = f"Untrusted odds response provenance: {', '.join(source_statuses)}"
    elif max_response_age is None or max_response_age > required.max_age_seconds:
        reason_code = "stale_snapshot"
        reason = f"Odds response age {max_response_age!r}s exceeds {required.max_age_seconds}s"
    elif scheduled_events <= 0:
        reason_code = "schedule_missing"
        reason = "No scheduled events were available for odds coverage validation"
    elif event_coverage < required.min_event_coverage:
        reason_code = "event_coverage"
        reason = f"Odds cover {covered_events}/{scheduled_events} events " f"({event_coverage:.1%})"
    elif market_coverage < required.min_market_coverage:
        reason_code = "market_coverage"
        reason = (
            f"Odds cover {covered_event_markets}/{required_event_markets} "
            f"required event-market pairs ({market_coverage:.1%})"
        )
    elif sportsbook_pairs_reported != covered_event_markets:
        reason_code = "sportsbook_provenance_incomplete"
        reason = (
            f"Sportsbook breadth is present for {sportsbook_pairs_reported}/"
            f"{covered_event_markets} covered event-market pairs"
        )
    elif sportsbook_coverage < required.min_market_coverage:
        reason_code = "sportsbook_coverage"
        reason = (
            f"Only {sportsbook_qualified_event_markets}/{required_event_markets} "
            "required event-market pairs meet the sportsbook breadth requirement "
            f"of {required.min_sportsbooks_per_event_market} ({sportsbook_coverage:.1%})"
        )
    elif odds_rows <= 0:
        reason_code = "empty_snapshot"
        reason = "Validated odds snapshot contains no complete two-sided rows"

    return {
        "valid": reason_code == "validated",
        "reason_code": reason_code,
        "reason": reason,
        "source_statuses": source_statuses,
        "snapshot_at": observed.get("snapshot_at"),
        "max_response_age_seconds": max_response_age,
        "responses_observed": responses_observed,
        "responses_with_age": responses_with_age,
        "scheduled_events": scheduled_events,
        "covered_events": covered_events,
        "event_coverage": event_coverage,
        "required_markets": list(required.required_markets),
        "required_event_markets": required_event_markets,
        "covered_event_markets": covered_event_markets,
        "market_coverage": market_coverage,
        "sportsbook_pairs_reported": sportsbook_pairs_reported,
        "sportsbook_qualified_event_markets": sportsbook_qualified_event_markets,
        "sportsbook_coverage": sportsbook_coverage,
        "min_sportsbooks_observed": min_sportsbooks_observed,
        "odds_rows": odds_rows,
        "requirements": asdict(required),
    }
