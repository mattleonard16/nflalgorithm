"""Unit tests for QB attempt volume calibration and starter probability gating (U3)."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline import QB_BASELINE_ATTEMPTS, compute_pass_attempts_predicted
from models.position_specific.weekly import _eligible_role_mask


class TestQBBaselineCalibration:
    def test_baseline_attempts_is_31(self):
        assert QB_BASELINE_ATTEMPTS == 31.0

    def test_positive_game_script_suppresses_volume(self):
        hist = pd.Series([30.0, 32.0, 31.0])
        neutral = compute_pass_attempts_predicted(hist, game_script=0.0)
        leading = compute_pass_attempts_predicted(hist, game_script=5.0)
        trailing = compute_pass_attempts_predicted(hist, game_script=-5.0)

        assert leading < neutral
        assert trailing > neutral

    def test_passing_yards_market_min_volume_threshold(self):
        starter = pd.DataFrame([{"expected_passing_attempts": 31.0}])
        backup = pd.DataFrame([{"expected_passing_attempts": 0.62}])

        assert bool(_eligible_role_mask(starter, "passing_yards").iloc[0]) is True
        assert bool(_eligible_role_mask(backup, "passing_yards").iloc[0]) is False
