"""Shared rendering and read plumbing for the weekly research memo.

Two concerns, both of which every section needs. Rendering: a markdown cell for
a number that may be absent, and the one "no data" shape a section falls back
to. Reading: a query against a table that may not have been migrated yet, and a
chronological max over a TEXT timestamp column -- which SQLite's own ``MAX()``
cannot give, because it compares those strings lexically.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from utils.db import DBConnection, read_dataframe, table_exists

logger = logging.getLogger(__name__)


def _fmt(value: Any, digits: int = 1) -> str:
    """Render a number for a markdown cell, or ``-`` when it is absent."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return "-"
    return f"{number:.{digits}f}"


def _text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and pd.isna(value):
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _no_data(title: str, reason: str, section: str) -> Tuple[str, Dict[str, Any]]:
    """The one shape every section falls back to when its inputs are empty."""
    return (
        f"{title}\n\n_no data: {reason}_",
        {"section": section, "status": "no_data", "note": reason},
    )


def _read_optional(
    sql: str,
    params: Tuple[Any, ...],
    *,
    table: str,
    conn: Optional[DBConnection] = None,
) -> pd.DataFrame:
    """Read from a table that may not exist yet, returning empty if it does not.

    Used for the optional tables only. A migration that has not run is a normal
    state for a fresh clone; the memo says so rather than dying on it.
    """
    if not table_exists(table, conn=conn):
        logger.warning("weekly_research: table %s does not exist; treating as empty", table)
        return pd.DataFrame()
    return read_dataframe(sql, params, conn=conn)


def _max_timestamp(values: pd.Series) -> Tuple[Optional[pd.Timestamp], int]:
    """Newest parseable timestamp in a text column, plus the unparseable count.

    These columns are TEXT in SQLite, so ``MAX()`` in SQL would be a lexical
    comparison. Parsing first is the only way the answer is chronological.
    """
    if values.empty:
        return None, 0
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    unparseable = int(parsed.isna().sum())
    newest = parsed.max()
    if pd.isna(newest):
        return None, unparseable
    return newest, unparseable
