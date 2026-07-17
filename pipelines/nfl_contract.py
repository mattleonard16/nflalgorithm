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
