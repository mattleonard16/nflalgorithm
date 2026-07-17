"""Public NFL domain constants shared by data and model adapters."""

from __future__ import annotations

INACTIVE_ROSTER_STATUSES = frozenset({"CUT", "DEV", "INA", "IR", "PUP", "RES", "RET", "SUS"})

__all__ = ["INACTIVE_ROSTER_STATUSES"]
