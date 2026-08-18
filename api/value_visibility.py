"""Shared SQL scope for value rows exposed through public NFL endpoints."""

from __future__ import annotations

from config import config
from utils.odds_quality import SYNTHETIC_SPORTSBOOK


def value_visibility_scope(*, demo_mode: bool | None = None) -> tuple[str, tuple[object, ...]]:
    """Return the public visibility predicate and its bound parameters."""
    show_fixtures = config.api.demo_mode if demo_mode is None else demo_mode
    if show_fixtures:
        return "1 = 1", ()

    predicate = f"""
        v.published_run_id IS NOT NULL
        AND LOWER(TRIM(v.sportsbook)) <> ?
        AND EXISTS (
            SELECT 1
            FROM pipeline_runs public_run
            WHERE public_run.run_id = v.published_run_id
              AND public_run.status = 'completed'
        )
        AND EXISTS (
            SELECT 1
            FROM pipeline_odds_validations public_validation
            WHERE public_validation.run_id = v.published_run_id
              AND public_validation.attempt = (
                  SELECT MAX(latest_validation.attempt)
                  FROM pipeline_odds_validations latest_validation
                  WHERE latest_validation.run_id = v.published_run_id
              )
              AND public_validation.valid = 1
        )
        AND EXISTS (
            SELECT 1
            FROM games public_game
            WHERE public_game.game_id = v.event_id
              AND public_game.season = v.season
              AND public_game.week = v.week
        )
    """
    return predicate.strip(), (SYNTHETIC_SPORTSBOOK.casefold(),)


__all__ = ["value_visibility_scope"]
