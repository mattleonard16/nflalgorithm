"""Stable stage contract shared by NFL job control and execution."""

NFL_STAGE_NAMES = (
    "prepare_week",
    "odds",
    "value_ranking",
    "risk_assessment",
    "agents",
    "materialize",
)

NFL_STAGE_COUNT = len(NFL_STAGE_NAMES)

# The deployment-supplied value adapter is intentionally excluded until its
# external side effects are verified idempotent under a lost acknowledgement.
NFL_AUTOMATIC_RETRY_SAFE_STAGES = frozenset(
    {"prepare_week", "odds", "risk_assessment", "agents", "materialize"}
)
