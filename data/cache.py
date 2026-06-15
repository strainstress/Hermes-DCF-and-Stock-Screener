"""DuckDB singleton cache connection.

Provides a single, shared DuckDB connection backed by the project's
data/cache/ directory. All modules use this same connection for consistent
caching of fundamentals, prices, and financials.
"""

from pathlib import Path
import duckdb
import threading

# Resolve cache directory from config.yaml's data.cache_dir, default to data/cache/
_CACHE_ROOT = Path(__file__).resolve().parent / "cache"
CACHE_DIR = _CACHE_ROOT

_conn: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()


def get_cache() -> duckdb.DuckDBPyConnection:
    """Return the singleton DuckDB connection.

    Creates the cache directory and connection on first call.
    Subsequent calls return the same connection. Thread-safe.

    Returns:
        A DuckDB connection backed by data/cache/cache.db.
    """
    global _conn
    if _conn is not None:
        return _conn

    with _lock:
        # Double-check after acquiring lock
        if _conn is not None:
            return _conn

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        db_path = str(CACHE_DIR / "cache.db")
        _conn = duckdb.connect(db_path)
        return _conn


def reset_cache() -> None:
    """Close and reset the singleton connection. For testing only."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
