"""Touchdown columns must survive transform_to_enhanced_stats (T1 #12).

TD props cannot be priced unless passing/rushing/receiving TDs reach
player_stats_enhanced; these tests lock the passthrough and the
default-to-zero path for sources that predate the columns.
"""

from __future__ import annotations

import pandas as pd

from scripts import ingest_real_nfl_data


def _weekly_frame(**extra: list[float]) -> pd.DataFrame:
    frame = {
        "position": ["QB", "WR"],
        "player_id": ["00-0026498", "00-0039067"],
        "player_name": ["M.Stafford", "P.Nacua"],
        "player_display_name": ["Matthew Stafford", "Puka Nacua"],
        "team": ["LAR", "LAR"],
        "opponent_team": ["SEA", "SEA"],
        "season": [2025, 2025],
        "week": [1, 1],
        "rushing_yards": [0.0, 0.0],
        "receiving_yards": [0.0, 80.0],
        "receptions": [0.0, 6.0],
        "targets": [0.0, 8.0],
        "passing_yards": [250.0, 0.0],
        "attempts": [30.0, 0.0],
        "carries": [0.0, 0.0],
    }
    frame.update(extra)
    return pd.DataFrame(frame)


def test_transform_carries_touchdown_columns_through() -> None:
    weekly = _weekly_frame(
        passing_tds=[2.0, 0.0],
        rushing_tds=[0.0, 0.0],
        receiving_tds=[0.0, 1.0],
    )

    result = ingest_real_nfl_data.transform_to_enhanced_stats(weekly, pd.DataFrame())

    by_player = result.set_index("name")
    assert by_player.loc["Matthew Stafford", "passing_tds"] == 2.0
    assert by_player.loc["Puka Nacua", "receiving_tds"] == 1.0
    assert by_player.loc["Puka Nacua", "rushing_tds"] == 0.0


def test_transform_defaults_missing_touchdown_columns_to_zero() -> None:
    result = ingest_real_nfl_data.transform_to_enhanced_stats(
        _weekly_frame(), pd.DataFrame()
    )

    for column in ("passing_tds", "rushing_tds", "receiving_tds"):
        assert column in result.columns
        assert (result[column] == 0.0).all()
