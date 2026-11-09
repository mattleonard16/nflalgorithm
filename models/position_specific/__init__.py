"""Position-specific model entry points.

A deployment may supply the weekly NFL model as a private ``weekly.py``. When it
is installed it wins. A public clone has no such file, so the entry points fall
back to the tracked ``baseline`` projector rather than failing, the same way
``config/`` falls back to its tracked defaults. Rows the baseline writes carry
its own ``model_version``, so a baseline card never passes for the real one.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

from config import config

from . import baseline
from .base_model import BasePositionModel

logger = logging.getLogger(__name__)


def weekly_implementation() -> Any:
    """Return the deployment's weekly model module, or the public baseline."""
    try:
        return import_module(f"{__name__}.weekly")
    except ModuleNotFoundError as exc:
        # Only weekly.py itself may be absent. A broken dependency inside an
        # installed model must fail, not quietly swap in the baseline.
        if exc.name != f"{__name__}.weekly":
            raise
        if config.features.require_private_models:
            raise RuntimeError(
                "NFL_REQUIRE_PRIVATE_MODELS is set but models/position_specific/weekly.py "
                "is not installed"
            ) from exc
    logger.warning(
        "models/position_specific/weekly.py is not installed; projecting with the public "
        "baseline (%s)",
        baseline.MODEL_VERSION,
    )
    return baseline


def train_weekly_models(*args: Any, **kwargs: Any) -> Any:
    """Train the installed NFL weekly model."""
    return weekly_implementation().train_weekly_models(*args, **kwargs)


def predict_week(*args: Any, **kwargs: Any) -> Any:
    """Generate predictions through the installed NFL weekly model."""
    return weekly_implementation().predict_week(*args, **kwargs)


__all__ = ["BasePositionModel", "predict_week", "train_weekly_models", "weekly_implementation"]
