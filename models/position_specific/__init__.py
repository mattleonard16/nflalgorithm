"""Position-specific model entry points.

The weekly NFL implementation may be supplied as a private deployment module.
Importing the shared package must remain safe when that optional module is not
installed, so the public entry points resolve it only when invoked.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base_model import BasePositionModel
from .rb_model import RBModel


def _weekly_module() -> Any:
    try:
        return import_module(f"{__name__}.weekly")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.weekly":
            raise
        raise RuntimeError(
            "NFL weekly model implementation is not installed. "
            "Provide models.position_specific.weekly in the runtime environment."
        ) from exc


def train_weekly_models(*args: Any, **kwargs: Any) -> Any:
    """Train the optional NFL weekly model implementation."""
    return _weekly_module().train_weekly_models(*args, **kwargs)


def predict_week(*args: Any, **kwargs: Any) -> Any:
    """Generate predictions through the optional NFL weekly model."""
    return _weekly_module().predict_week(*args, **kwargs)


__all__ = ["BasePositionModel", "RBModel", "predict_week", "train_weekly_models"]
