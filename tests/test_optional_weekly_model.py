"""Tests for the optional NFL weekly-model package boundary."""

from __future__ import annotations

import pytest

import models.position_specific as position_models


def test_position_package_import_does_not_require_weekly_model() -> None:
    assert position_models.BasePositionModel is not None
    assert callable(position_models.predict_week)


def test_a_clone_without_the_private_model_projects_with_the_public_baseline(
    monkeypatch,
) -> None:
    def missing_weekly_module(name: str):
        error = ModuleNotFoundError(name=name)
        raise error

    monkeypatch.setattr(position_models, "import_module", missing_weekly_module)

    assert position_models.weekly_implementation() is position_models.baseline


def test_unrelated_dependency_import_errors_are_not_hidden(monkeypatch) -> None:
    def broken_weekly_dependency(name: str):
        error = ModuleNotFoundError(name="missing_dependency")
        raise error

    monkeypatch.setattr(position_models, "import_module", broken_weekly_dependency)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        position_models.predict_week(2026, 1)

    assert exc_info.value.name == "missing_dependency"


def test_a_deployment_that_requires_the_private_model_fails_without_it(monkeypatch) -> None:
    def missing_weekly_module(name: str):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(position_models, "import_module", missing_weekly_module)
    monkeypatch.setattr(position_models.config.features, "require_private_models", True)

    with pytest.raises(RuntimeError, match="NFL_REQUIRE_PRIVATE_MODELS"):
        position_models.predict_week(2026, 1)
