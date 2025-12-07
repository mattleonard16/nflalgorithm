"""Database abstraction layer for SQLite and MySQL.

The goal is to centralize connection handling so callers do not care
whether the backend is a local SQLite file or a remote MySQL DB.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Iterator, Optional, Iterable, Any, Union, Dict
from urllib.parse import urlparse

import pandas as pd
import sqlite3

# MySQL support
try:
    import pymysql
    from pymysql.connections import Connection as pymysql_connection
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False
    pymysql_connection = None # type: ignore

from config import config

# Suppress pandas UserWarning about raw DBAPI connections
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy connectable.*")

# Union type for database connections
DBConnection = Union[sqlite3.Connection, pymysql_connection]

def _normalize_sql_for_backend(sql: str, backend: str) -> str:
    """Convert SQL parameter placeholders based on backend.
    
    SQLite uses '?' placeholders
    MySQL uses '%s'
    """
    if backend == "mysql":
        # MySQL (PyMySQL) use %s
        return sql.replace('?', '%s')
    return sql


def _get_backend() -> str:
    """Get the configured database backend."""
    # Force re-evaluate from config which now reads env dynamically or refresh config
    # For now, just trust config.database.backend
    backend = getattr(config.database, "backend", "sqlite").lower()
    
    # Double check env var because config might be cached/stale
    import os
    env_backend = os.getenv("DB_BACKEND", "").lower()
    if env_backend and env_backend != backend:
        return env_backend
        
    return backend


def _create_mysql_connection() -> pymysql_connection:
    """Create a MySQL connection using connection URL."""
    if not PYMYSQL_AVAILABLE:
        raise RuntimeError("pymysql is required for MySQL support.")

    # Use env var directly to ensure freshness and correctness for now
    import os
    dsn = os.getenv("DB_URL", "")
    
    # Fallback to config if env var missing
    if not dsn:
         dsn = getattr(config.database, "db_url", "")

    if not dsn:
        raise RuntimeError("DB_URL environment variable must be set for MySQL backend")

    # Parse mysql://user:pass@host:port/db
    try:
        parsed = urlparse(dsn)
        # Allow mysql, mysql+pymysql schemes
        if parsed.scheme not in ("mysql", "mysql+pymysql"):
            raise ValueError(f"DSN scheme must be 'mysql' or 'mysql+pymysql', got '{parsed.scheme}'")
            
        conn = pymysql.connect(
            host=parsed.hostname,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.lstrip('/'),
            port=parsed.port or 3306,
            cursorclass=pymysql.cursors.Cursor
        )
        return conn
    except Exception as e:
        raise RuntimeError(f"Failed to connect to MySQL: {e}")

@contextlib.contextmanager
def get_connection() -> Iterator[DBConnection]:
    """Return a context-managed connection."""
    backend = _get_backend()
    
    if backend == "sqlite":
        sqlite_path = config.database.path
        conn = sqlite3.connect(sqlite_path)
        try:
            yield conn
        finally:
            conn.close()
    elif backend == "mysql":
        conn = _create_mysql_connection()
        try:
            yield conn
        finally:
            conn.close()
    else:
        raise RuntimeError(f"Unsupported database backend: {backend}. Supported: sqlite, mysql")


def read_dataframe(sql: str, params: Optional[tuple] = None, conn: Optional[DBConnection] = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    backend = _get_backend()
    normalized_sql = _normalize_sql_for_backend(sql, backend)

    if conn is not None:
        return pd.read_sql_query(normalized_sql, conn, params=params)
    with get_connection() as tmp_conn:
        return pd.read_sql_query(normalized_sql, tmp_conn, params=params)


def execute(sql: str, params: Optional[tuple] = None, conn: Optional[DBConnection] = None) -> None:
    """Execute a single statement and commit immediately."""
    backend = _get_backend()
    normalized_sql = _normalize_sql_for_backend(sql, backend)
    
    if conn is not None:
        if isinstance(conn, sqlite3.Connection):
            conn.execute(normalized_sql, params or ())
        else:
            # MySQL
            with conn.cursor() as cursor:
                cursor.execute(normalized_sql, params or ())
        return
    
    with get_connection() as tmp_conn:
        if isinstance(tmp_conn, sqlite3.Connection):
            tmp_conn.execute(normalized_sql, params or ())
            tmp_conn.commit()
        else:
             # MySQL
            with tmp_conn.cursor() as cursor:
                cursor.execute(normalized_sql, params or ())
            tmp_conn.commit()


def executemany(sql: str, seq_of_params: Iterable[tuple[Any, ...]], conn: Optional[DBConnection] = None) -> None:
    """Execute many statements and commit immediately."""
    backend = _get_backend()
    normalized_sql = _normalize_sql_for_backend(sql, backend)
    params_list = list(seq_of_params)
    
    if conn is not None:
        if isinstance(conn, sqlite3.Connection):
            conn.executemany(normalized_sql, params_list)
        else:
            with conn.cursor() as cursor:
                cursor.executemany(normalized_sql, params_list)
        return
    
    with get_connection() as tmp_conn:
        if isinstance(tmp_conn, sqlite3.Connection):
            tmp_conn.executemany(normalized_sql, params_list)
            tmp_conn.commit()
        else:
            with tmp_conn.cursor() as cursor:
                cursor.executemany(normalized_sql, params_list)
            tmp_conn.commit()


def write_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = 'fail', index: bool = False) -> None:
    """Write a DataFrame to the database."""
    backend = _get_backend()
    
    if backend == "mysql":
        try:
            from sqlalchemy import create_engine
            # Check env var first
            import os
            dsn = os.getenv("DB_URL", "")
            if not dsn:
                 dsn = getattr(config.database, "db_url", "")

            if not dsn:
                raise RuntimeError("Database URL not set")
                
            # MySQL needs pymysql driver in connection string usually: mysql+pymysql://
            if dsn.startswith("mysql://"):
                dsn = dsn.replace("mysql://", "mysql+pymysql://")
                
            engine = create_engine(dsn)
            with engine.begin() as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=index)
        except ImportError:
            raise RuntimeError("SQLAlchemy is required for writing DataFrames")
        except Exception as e:
            raise RuntimeError(f"Failed to write dataframe to {table_name}: {e}")
    else:
        # SQLite
        with get_connection() as conn:
             df.to_sql(table_name, conn, if_exists=if_exists, index=index)


def get_table_columns(table_name: str, conn: Optional[DBConnection] = None) -> Dict[str, Dict[str, Any]]:
    """Return column metadata for a table."""
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
        elif backend == "mysql":
             with conn.cursor() as cursor:
                cursor.execute(f"DESCRIBE {table_name}")
                for row in cursor.fetchall():
                    # Row: Field, Type, Null, Key, Default, Extra
                    # PyMySQL returns tuples
                    col_name = row[0]
                    columns[col_name] = {
                        "type": row[1],
                        "notnull": row[2] == "NO",
                        "default": row[4],
                        "pk": row[3] == "PRI"
                    }
        return columns
    finally:
        if cleanup_needed and conn is not None:
            assert 'connection_cm' in locals()
            connection_cm.__exit__(None, None, None)


def column_exists(table_name: str, column_name: str, conn: Optional[DBConnection] = None) -> bool:
    columns = get_table_columns(table_name, conn=conn)
    return column_name in columns


def table_exists(table_name: str, conn: Optional[DBConnection] = None) -> bool:
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
        elif _get_backend() == "mysql":
             with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES LIKE %s", (table_name,))
                return cursor.fetchone() is not None
        return False
    finally:
        if cleanup_needed:
            connection_cm.__exit__(None, None, None)


def fetchone(sql: str, params: Optional[tuple] = None, conn: Optional[DBConnection] = None) -> Optional[tuple]:
    """Execute a query and return a single row."""
    backend = _get_backend()
    normalized_sql = _normalize_sql_for_backend(sql, backend)

    if conn is not None:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.execute(normalized_sql, params or ())
            return cursor.fetchone()
        else:
            with conn.cursor() as cursor:
                cursor.execute(normalized_sql, params or ())
                return cursor.fetchone()

    with get_connection() as tmp_conn:
        if isinstance(tmp_conn, sqlite3.Connection):
            cursor = tmp_conn.execute(normalized_sql, params or ())
            return cursor.fetchone()
        else:
            with tmp_conn.cursor() as cursor:
                cursor.execute(normalized_sql, params or ())
                return cursor.fetchone()


def fetchall(sql: str, params: Optional[tuple] = None, conn: Optional[DBConnection] = None) -> list[tuple]:
    """Execute a query and return all rows."""
    backend = _get_backend()
    normalized_sql = _normalize_sql_for_backend(sql, backend)

    if conn is not None:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.execute(normalized_sql, params or ())
            return cursor.fetchall()
        else:
            with conn.cursor() as cursor:
                cursor.execute(normalized_sql, params or ())
                return cursor.fetchall()

    with get_connection() as tmp_conn:
        if isinstance(tmp_conn, sqlite3.Connection):
            cursor = tmp_conn.execute(normalized_sql, params or ())
            return cursor.fetchall()
        else:
            with tmp_conn.cursor() as cursor:
                cursor.execute(normalized_sql, params or ())
                return cursor.fetchall()
