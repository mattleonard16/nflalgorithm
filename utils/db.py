"""Database abstraction layer for SQLite and Supabase (PostgreSQL).

The goal is to centralize connection handling so callers do not care
whether the backend is a local SQLite file or a remote Postgres DB.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Iterator, Optional, Iterable, Any, Union, Dict
from urllib.parse import urlparse

import pandas as pd
import sqlite3

try:
    import psycopg2
    from psycopg2.extensions import connection as psycopg2_connection
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2_connection = None  # type: ignore

from config import config

# Suppress pandas UserWarning about raw DBAPI connections
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy connectable.*")

# Union type for database connections
DBConnection = Union[sqlite3.Connection, psycopg2_connection] if PSYCOPG2_AVAILABLE else sqlite3.Connection


def _normalize_sql_for_backend(sql: str, backend: str) -> str:
    """Convert SQL parameter placeholders based on backend.
    
    SQLite uses '?' placeholders, PostgreSQL uses '%s'.
    This function converts '?' to '%s' when using Postgres backend.
    
    Note: This assumes '?' only appears as parameter placeholders,
    not inside string literals. Properly parameterized SQL should
    follow this pattern.
    """
    if backend in ("postgresql", "supabase"):
        # Simple replacement: convert ? to %s for PostgreSQL
        # This works because properly parameterized SQL should not
        # have ? inside string literals (values should be in params tuple)
        return sql.replace('?', '%s')
    return sql


def _get_backend() -> str:
    """Get the configured database backend."""
    return getattr(config.database, "backend", "sqlite").lower()


def _create_postgres_connection() -> psycopg2_connection:
    """Create a PostgreSQL connection using Supabase DSN."""
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2-binary is required for PostgreSQL/Supabase support. Install it with: pip install psycopg2-binary")
    
    supabase_dsn = getattr(config.database, "supabase_dsn", "")
    if not supabase_dsn:
        raise RuntimeError("SUPABASE_DB_URL environment variable must be set when using Supabase backend")
    
    # Validate the connection string format
    try:
        parsed = urlparse(supabase_dsn)
        if parsed.scheme not in ("postgresql", "postgres"):
            raise ValueError(f"Invalid DSN scheme: {parsed.scheme}. Expected postgresql:// or postgres://")
    except Exception as e:
        raise RuntimeError(f"Invalid Supabase DSN format: {e}")
    
    try:
        # Connect to Supabase
        # options="-c client_min_messages=ERROR" suppresses notices like "supautils.disable_program"
        conn = psycopg2.connect(supabase_dsn, options="-c client_min_messages=ERROR")
        return conn
    except psycopg2.Error as e:
        raise RuntimeError(f"Failed to connect to Supabase: {e}")


@contextlib.contextmanager
def get_connection() -> Iterator[DBConnection]:
    """Return a context-managed connection.
    
    Supports both SQLite and PostgreSQL/Supabase backends based on
    config.database.backend setting.
    """
    backend = _get_backend()
    
    if backend == "sqlite":
        sqlite_path = config.database.path
        conn = sqlite3.connect(sqlite_path)
        try:
            yield conn
        finally:
            conn.close()
    elif backend in ("postgresql", "supabase"):
        conn = _create_postgres_connection()
        try:
            yield conn
        finally:
            conn.close()
    else:
        raise RuntimeError(f"Unsupported database backend: {backend}. Supported: sqlite, postgresql, supabase")


def read_dataframe(sql: str, params: Optional[tuple] = None, conn: Optional[DBConnection] = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame using the configured backend.

    If ``conn`` is provided, it is used directly and *not* closed. This
    allows callers to reuse a transaction. Otherwise a temporary
    connection from :func:`get_connection` is used.
    
    Note: pandas handles parameter style conversion automatically, so
    both SQLite (?) and PostgreSQL (%s) placeholders work.
    """
    backend = _get_backend()
    normalized_sql = _normalize_sql_for_backend(sql, backend)

    if conn is not None:
        return pd.read_sql_query(normalized_sql, conn, params=params)
    with get_connection() as tmp_conn:
        return pd.read_sql_query(normalized_sql, tmp_conn, params=params)


def execute(sql: str, params: Optional[tuple] = None, conn: Optional[DBConnection] = None) -> None:
    """Execute a single statement and commit immediately.

    If ``conn`` is provided, it is used and the caller is responsible for
    committing/rolling back. Otherwise the function manages a temporary
    connection and commits automatically.
    
    Note: SQL parameter placeholders are automatically converted based on
    the backend (SQLite uses ?, PostgreSQL uses %s).
    """
    backend = _get_backend()
    normalized_sql = _normalize_sql_for_backend(sql, backend)
    
    if conn is not None:
        if isinstance(conn, sqlite3.Connection):
            conn.execute(normalized_sql, params or ())
        else:
            # PostgreSQL connection
            with conn.cursor() as cursor:
                cursor.execute(normalized_sql, params or ())
        return
    
    with get_connection() as tmp_conn:
        if isinstance(tmp_conn, sqlite3.Connection):
            tmp_conn.execute(normalized_sql, params or ())
            tmp_conn.commit()
        else:
            # PostgreSQL connection
            with tmp_conn.cursor() as cursor:
                cursor.execute(normalized_sql, params or ())
            tmp_conn.commit()


def executemany(sql: str, seq_of_params: Iterable[tuple[Any, ...]], conn: Optional[DBConnection] = None) -> None:
    """Execute many statements and commit immediately.

    Follows the same connection semantics as :func:`execute`.
    
    Note: SQL parameter placeholders are automatically converted based on
    the backend (SQLite uses ?, PostgreSQL uses %s).
    """
    backend = _get_backend()
    normalized_sql = _normalize_sql_for_backend(sql, backend)
    params_list = list(seq_of_params)
    
    if conn is not None:
        if isinstance(conn, sqlite3.Connection):
            conn.executemany(normalized_sql, params_list)
        else:
            # PostgreSQL connection
            with conn.cursor() as cursor:
                cursor.executemany(normalized_sql, params_list)
        return
    
    with get_connection() as tmp_conn:
        if isinstance(tmp_conn, sqlite3.Connection):
            tmp_conn.executemany(normalized_sql, params_list)
            tmp_conn.commit()
        else:
            # PostgreSQL connection
            with tmp_conn.cursor() as cursor:
                cursor.executemany(normalized_sql, params_list)
            tmp_conn.commit()


def write_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = 'fail', index: bool = False) -> None:
    """Write a DataFrame to the database using the configured backend.
    
    Args:
        df: DataFrame to write
        table_name: Target table name
        if_exists: How to behave if the table already exists.
                   User 'fail', 'replace', or 'append'.
        index: Write DataFrame index as a column.
    """
    backend = _get_backend()
    
    if backend in ("postgresql", "supabase"):
        # Use SQLAlchemy for robust Postgres support (handles types/schema correctly)
        try:
            from sqlalchemy import create_engine
            # Create engine (pool is managed by SQLAlchemy)
            supabase_dsn = getattr(config.database, "supabase_dsn", "")
            if not supabase_dsn:
                raise RuntimeError("SUPABASE_DB_URL not set")
                
            # options="-c client_min_messages=ERROR" suppresses notices like "supautils.disable_program"
            engine = create_engine(supabase_dsn, connect_args={'options': '-c client_min_messages=ERROR'})
            with engine.begin() as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=index)
        except ImportError:
            raise RuntimeError("SQLAlchemy is required for writing DataFrames to PostgreSQL/Supabase")
        except Exception as e:
            raise RuntimeError(f"Failed to write dataframe to {table_name}: {e}")
    else:
        # SQLite
        # Pandas handles SQLite correctly with raw connection
        with get_connection() as conn:
             df.to_sql(table_name, conn, if_exists=if_exists, index=index)


def get_table_columns(table_name: str, conn: Optional[DBConnection] = None) -> Dict[str, Dict[str, Any]]:
    """Return column metadata for a table across backends.

    The metadata dict for each column includes keys: ``type``, ``notnull``,
    ``default`` and ``pk`` mirroring SQLite's PRAGMA output to minimize
    downstream changes.
    """
    backend = _get_backend()
    cleanup_needed = False

    if conn is None:
        cleanup_needed = True
        connection_cm = get_connection()
        conn = connection_cm.__enter__()

    try:
        columns: Dict[str, Dict[str, Any]] = {}
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            for row in cursor.fetchall():
                columns[row[1]] = {
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default": row[4],
                    "pk": bool(row[5]),
                }
        else:
            with conn.cursor() as cursor:  # type: ignore[arg-type]
                cursor.execute(
                    """
                    WITH pk_columns AS (
                        SELECT kcu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                          ON tc.constraint_name = kcu.constraint_name
                         AND tc.table_schema = kcu.table_schema
                        WHERE tc.table_schema = current_schema()
                          AND tc.table_name = %s
                          AND tc.constraint_type = 'PRIMARY KEY'
                    )
                    SELECT
                        c.column_name,
                        c.data_type,
                        (c.is_nullable = 'NO') AS notnull,
                        c.column_default,
                        (c.column_name IN (SELECT column_name FROM pk_columns)) AS pk
                    FROM information_schema.columns c
                    WHERE c.table_schema = current_schema()
                      AND c.table_name = %s
                    ORDER BY c.ordinal_position
                    """,
                    (table_name, table_name),
                )
                for row in cursor.fetchall():
                    col_name = row[0]
                    columns[col_name] = {
                        "type": row[1],
                        "notnull": bool(row[2]),
                        "default": row[3],
                        "pk": bool(row[4]),
                    }
        return columns
    finally:
        if cleanup_needed and conn is not None:
            assert 'connection_cm' in locals()
            connection_cm.__exit__(None, None, None)


def column_exists(table_name: str, column_name: str, conn: Optional[DBConnection] = None) -> bool:
    """Return True if ``column_name`` exists on ``table_name``."""
    columns = get_table_columns(table_name, conn=conn)
    return column_name in columns


def table_exists(table_name: str, conn: Optional[DBConnection] = None) -> bool:
    """Return True if the given table exists in the current backend."""
    cleanup_needed = False
    if conn is None:
        cleanup_needed = True
        connection_cm = get_connection()
        conn = connection_cm.__enter__()

    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cursor.fetchone() is not None
        else:
            with conn.cursor() as cursor:  # type: ignore[arg-type]
                cursor.execute(
                    "SELECT to_regclass(%s)",
                    (f"public.{table_name}",),
                )
                result = cursor.fetchone()
                return bool(result and result[0])
    finally:
        if cleanup_needed:
            connection_cm.__exit__(None, None, None)
