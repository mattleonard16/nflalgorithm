"""Agent orchestration for live NFL betting operations.

Provides specialized agents that independently analyze betting opportunities,
plus a coordinator that merges their recommendations into consensus decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AgentReport:
    """Immutable report produced by each agent after analysis.

    Attributes
    ----------
    agent_name : str
        Identifier of the producing agent.
    recommendation : str
        One of "APPROVE", "REJECT", "NEUTRAL".
    confidence : float
        0.0-1.0 confidence in the recommendation.
    rationale : str
        Human-readable explanation.
    data : dict
        Arbitrary structured payload for downstream consumers.
    player_id : str or None
        Scoped to a specific player when set.
    market : str or None
        Scoped to a specific market when set.
    generated_at : str
        ISO-8601 timestamp of report creation.
    """

    agent_name: str
    recommendation: str
    confidence: float
    rationale: str
    data: Dict[str, Any] = field(default_factory=dict)
    player_id: Optional[str] = None
    market: Optional[str] = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


VALID_RECOMMENDATIONS = frozenset({"APPROVE", "REJECT", "NEUTRAL"})


def validate_report(report: AgentReport) -> List[str]:
    """Return a list of validation errors (empty if valid)."""
    errors: List[str] = []
    if report.recommendation not in VALID_RECOMMENDATIONS:
        errors.append(
            f"Invalid recommendation '{report.recommendation}'; "
            f"must be one of {sorted(VALID_RECOMMENDATIONS)}"
        )
    if not 0.0 <= report.confidence <= 1.0:
        errors.append(
            f"Confidence {report.confidence} out of range [0.0, 1.0]"
        )
    if not report.agent_name:
        errors.append("agent_name must not be empty")
    return errors
