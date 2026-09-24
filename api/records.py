"""Query results as JSON-safe rows."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rows as dicts, with every missing value as None.

    pandas reads NULL as NaN when other rows in the column hold a value. NaN is
    not valid JSON and ``int(NaN)`` raises, so one row with a missing field
    failed the whole request.
    """
    records: List[Dict[str, Any]] = (
        df.astype(object).where(df.notna(), None).to_dict(orient="records")
    )
    return records
