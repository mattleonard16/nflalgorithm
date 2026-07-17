"""Tests for the optional NFL weekly-model package boundary."""

from __future__ import annotations

import pytest

import models.position_specific as position_models


def test_position_package_import_does_not_require_weekly_model() -> None:
    assert position_models.BasePositionModel is not None
    assert callable(position_models.predict_week)


def test_weekly_entrypoint_explains_missing_optional_module(monkeypatch) -> None:
    def missing_weekly_module(name: str):
        error = ModuleNotFoundError(name=name)
        raise error

    monkeypatch.setattr(position_models, "import_module", missing_weekly_module)

    with pytest.raises(RuntimeError, match="weekly model implementation is not installed"):
        position_models.predict_week(2026, 1)


def test_unrelated_dependency_import_errors_are_not_hidden(monkeypatch) -> None:
    def broken_weekly_dependency(name: str):
        error = ModuleNotFoundError(name="missing_dependency")
        raise error

    monkeypatch.setattr(position_models, "import_module", broken_weekly_dependency)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        position_models.predict_week(2026, 1)

    assert exc_info.value.name == "missing_dependency"
