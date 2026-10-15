"""Snap-count merge keys must survive suffix and punctuation drift.

The old key only dropped a suffix when the lowercased string *ended* in
" jr", so "Michael Pittman Jr." (trailing period) kept its suffix and never
matched snap counts' "Michael Pittman", leaving snap_percentage at 0.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.ingest_real_nfl_data import _merge_snap_counts, _normalize_name_for_merge


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Michael Pittman Jr.", "Michael Pittman"),
        ("Kenneth Walker III", "Kenneth Walker"),
        ("Marvin Harrison Jr", "Marvin Harrison"),
        ("Odell Beckham, Jr.", "Odell Beckham"),
        ("A.J. Brown", "AJ Brown"),
        ("Ja'Marr Chase", "Jamarr Chase"),
        ("Amon-Ra St. Brown", "Amon Ra St Brown"),
        ("José Cruz", "Jose Cruz"),
    ],
)
def test_name_variants_share_a_merge_key(left: str, right: str) -> None:
    assert _normalize_name_for_merge(left) == _normalize_name_for_merge(right)
    assert _normalize_name_for_merge(left)


def test_distinct_names_keep_distinct_keys() -> None:
    assert _normalize_name_for_merge("Michael Pittman") != _normalize_name_for_merge(
        "Michael Pitman"
    )


def test_missing_name_is_empty_key() -> None:
    assert _normalize_name_for_merge(None) == ""  # type: ignore[arg-type]
    assert _normalize_name_for_merge(float("nan")) == ""  # type: ignore[arg-type]


def test_suffixed_display_name_gets_its_snap_share() -> None:
    stats = pd.DataFrame(
        {
            "player_name": ["M.Pittman"],
            "player_display_name": ["Michael Pittman Jr."],
            "season": [2025],
            "week": [3],
            "team": ["IND"],
        }
    )
    snaps = pd.DataFrame(
        {
            "player": ["Michael Pittman"],
            "season": [2025],
            "week": [3],
            "team": ["IND"],
            "offense_snaps": [61],
            "offense_pct": [0.91],
        }
    )

    merged = _merge_snap_counts(stats, snaps)

    assert merged["snap_count"].iloc[0] == 61
    assert merged["snap_percentage"].iloc[0] == pytest.approx(91.0)
