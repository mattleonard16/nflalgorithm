"""Application configuration with an optional local override.

Tracked, environment-driven defaults live in :mod:`config.runtime` so a fresh
checkout is runnable.  A gitignored top-level ``config.py`` may still override
those defaults for local deployments that need private settings.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TypeVar

from .runtime import PROJECT_ROOT as _PROJECT_ROOT
from .runtime import config

_CONFIG_FILE = Path(os.getenv("NFL_CONFIG_PATH", _PROJECT_ROOT / "config.py"))
_ConfigT = TypeVar("_ConfigT")


def _fill_missing_settings(target: _ConfigT, defaults: object) -> _ConfigT:
    """Recursively preserve private overrides while adding tracked defaults."""
    for name, default_value in vars(defaults).items():
        if not hasattr(target, name):
            setattr(target, name, default_value)
            continue
        target_value = getattr(target, name)
        if hasattr(default_value, "__dict__") and hasattr(target_value, "__dict__"):
            _fill_missing_settings(target_value, default_value)
    return target


if _CONFIG_FILE.is_file():
    _spec = importlib.util.spec_from_file_location(
        "_nflalgorithm_runtime_config",
        _CONFIG_FILE,
    )
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Unable to load runtime config from {_CONFIG_FILE}")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    if not hasattr(_module, "config"):
        raise ImportError(f"Runtime config {_CONFIG_FILE} must define 'config'")
    config = _fill_missing_settings(_module.config, config)

__all__ = ["config"]
