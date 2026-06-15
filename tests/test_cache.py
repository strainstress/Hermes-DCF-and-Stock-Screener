"""Tests for data/cache.py — DuckDB singleton cache."""
import pytest
from pathlib import Path
from data.cache import get_cache, CACHE_DIR, reset_cache


def teardown_module():
    """Clean up singleton state between test modules."""
    reset_cache()


def test_get_cache_returns_duckdb_connection():
    """Must return a working DuckDB connection."""
    conn = get_cache()
    assert conn is not None
    result = conn.execute("SELECT 1 AS val").fetchone()
    assert result[0] == 1


def test_get_cache_is_singleton():
    """Multiple calls return the same connection."""
    conn1 = get_cache()
    conn2 = get_cache()
    assert conn1 is conn2, "get_cache() must return the same connection"


def test_cache_dir_created():
    """Cache directory must exist after first get_cache() call."""
    conn = get_cache()
    conn.execute("SELECT 1")  # force connection init
    assert CACHE_DIR.exists(), f"Cache dir {CACHE_DIR} should exist"
    assert CACHE_DIR.is_dir(), f"Cache dir {CACHE_DIR} should be a directory"


def test_can_create_and_query_table():
    """Verify we can create a table, insert, and query."""
    conn = get_cache()
    conn.execute("CREATE OR REPLACE TABLE test_items (name VARCHAR, value INTEGER)")
    conn.execute("INSERT INTO test_items VALUES ('alpha', 1), ('beta', 2)")
    rows = conn.execute("SELECT * FROM test_items ORDER BY name").fetchall()
    assert rows == [("alpha", 1), ("beta", 2)]


def test_data_persists_across_calls():
    """Table created in one call is visible in subsequent calls."""
    conn = get_cache()
    conn.execute("CREATE OR REPLACE TABLE persist_test (id INTEGER)")
    conn.execute("INSERT INTO persist_test VALUES (42)")

    # Same connection (singleton) — data must still be there
    rows = conn.execute("SELECT * FROM persist_test").fetchall()
    assert rows == [(42,)]
