"""Tests for MigrationManager._drop_mysql_index_if_columns_differ.

information_schema.statistics reports bare column names, while the desired
column list may carry MySQL prefix lengths ("name(64)"). The comparator must
strip the prefix before comparing, or a prefixed index is dropped and rebuilt
on every migration run.
"""

from schema_migrations import MigrationManager


class _FakeCursor:
    """Records executed SQL; serves one canned fetchall result."""

    def __init__(self, existing_columns):
        self._existing = [(col,) for col in existing_columns]
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        self.executed.append(" ".join(sql.split()))

    def fetchall(self):
        return self._existing


def _drop_statements(cursor: _FakeCursor) -> list[str]:
    return [sql for sql in cursor.executed if sql.startswith("DROP INDEX")]


def test_prefix_lengths_do_not_trigger_rebuild() -> None:
    cursor = _FakeCursor(["player_id", "name", "position"])
    MigrationManager._drop_mysql_index_if_columns_differ(
        cursor, "player_stats_enhanced", "idx_player_stats_identity",
        "player_id, name(64), position(8)",
    )
    assert _drop_statements(cursor) == []


def test_changed_column_set_still_drops() -> None:
    cursor = _FakeCursor(["season", "week"])
    MigrationManager._drop_mysql_index_if_columns_differ(
        cursor, "materialized_value_view", "idx_materialized_value_view_lookup",
        "season, week, edge_percentage",
    )
    assert len(_drop_statements(cursor)) == 1


def test_missing_index_is_left_for_create_path() -> None:
    cursor = _FakeCursor([])
    MigrationManager._drop_mysql_index_if_columns_differ(
        cursor, "weekly_odds", "idx_weekly_odds_lookup",
        "season, week, player_id, market",
    )
    assert _drop_statements(cursor) == []


def test_case_and_whitespace_are_normalized() -> None:
    cursor = _FakeCursor(["player_id", "name", "position"])
    MigrationManager._drop_mysql_index_if_columns_differ(
        cursor, "player_stats_enhanced", "idx_player_stats_identity",
        " Player_ID ,NAME(64) , Position(8) ",
    )
    assert _drop_statements(cursor) == []
