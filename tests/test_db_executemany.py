"""Regression coverage for streaming database batch parameters."""

from __future__ import annotations

from utils.db import execute, executemany, fetchall, get_connection


def test_executemany_accepts_a_one_shot_generator() -> None:
    with get_connection() as conn:
        execute(
            "CREATE TEMPORARY TABLE batch_rows (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            conn=conn,
        )

        executemany(
            "INSERT INTO batch_rows (id, value) VALUES (?, ?)",
            ((identifier, f"value-{identifier}") for identifier in range(3)),
            conn=conn,
        )

        assert fetchall("SELECT id, value FROM batch_rows ORDER BY id", conn=conn) == [
            (0, "value-0"),
            (1, "value-1"),
            (2, "value-2"),
        ]
