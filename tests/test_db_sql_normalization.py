"""Placeholder translation for the MySQL backend (utils.db._normalize_sql_for_backend)."""

from __future__ import annotations

from utils.db import _normalize_sql_for_backend


def _pymysql_interpolate(sql: str, params: tuple) -> str:
    """What PyMySQL's mogrify does when args is not None (escaping aside)."""
    return sql % tuple(repr(p) for p in params)


def test_sqlite_sql_is_untouched():
    sql = "SELECT * FROM t WHERE a = ? AND m LIKE '%vs.%'"

    assert _normalize_sql_for_backend(sql, "sqlite") == sql


def test_mysql_converts_placeholders():
    assert _normalize_sql_for_backend("SELECT 1 FROM t WHERE a = ? AND b = ?", "mysql") == (
        "SELECT 1 FROM t WHERE a = %s AND b = %s"
    )


def test_mysql_escapes_literal_percent_when_driver_formats():
    sql = "SELECT game_id FROM logs WHERE game_date = ? AND matchup LIKE '%vs.%'"

    normalized = _normalize_sql_for_backend(sql, "mysql")

    assert normalized == "SELECT game_id FROM logs WHERE game_date = %s AND matchup LIKE '%%vs.%%'"
    # Interpolation must succeed and leave the literal pattern intact.
    assert _pymysql_interpolate(normalized, ("2026-03-10",)) == (
        "SELECT game_id FROM logs WHERE game_date = '2026-03-10' AND matchup LIKE '%vs.%'"
    )


def test_mysql_escapes_percent_even_with_empty_params():
    # execute/fetchone/fetchall pass `params or ()`, and PyMySQL formats any
    # non-None args, so `()` still triggers interpolation.
    normalized = _normalize_sql_for_backend("SELECT 1 FROM t WHERE m LIKE 'x%'", "mysql")

    assert _pymysql_interpolate(normalized, ()) == "SELECT 1 FROM t WHERE m LIKE 'x%'"


def test_mysql_leaves_percent_alone_when_driver_will_not_format():
    sql = "SELECT 1 FROM t WHERE m LIKE '%vs.%'"

    assert _normalize_sql_for_backend(sql, "mysql", formats_params=False) == sql


def test_read_dataframe_escapes_only_when_params_are_passed(monkeypatch):
    import utils.db as db

    captured = []

    class _Conn:
        pass

    def fake_read_sql_query(sql, conn, params=None):
        captured.append((sql, params))
        return None

    monkeypatch.setattr(db, "_get_backend", lambda: "mysql")
    monkeypatch.setattr(db.pd, "read_sql_query", fake_read_sql_query)

    db.read_dataframe("SELECT 1 WHERE m LIKE '%a%'", conn=_Conn())
    db.read_dataframe("SELECT 1 WHERE m LIKE '%a%' AND x = ?", params=(1,), conn=_Conn())

    assert captured == [
        ("SELECT 1 WHERE m LIKE '%a%'", None),
        ("SELECT 1 WHERE m LIKE '%%a%%' AND x = %s", (1,)),
    ]
