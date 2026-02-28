"""
SQLite database layer for voice cloning.

Uses WAL mode for concurrent reads and an asyncio lock to serialize writes.
Each call opens a fresh synchronous sqlite3 connection for simplicity.
"""

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = os.environ.get("VOICES_DB", "/mnt/voices/voices.db")

_write_lock = asyncio.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS voices (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_by  TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN ('youtube', 'upload', 'recording', 'curated')),
    source_url  TEXT,
    language    TEXT NOT NULL,
    ref_audio   TEXT NOT NULL,
    transcript  TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    is_public   BOOLEAN DEFAULT 1,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with WAL mode and row factory."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database: create tables if they don't exist."""
    conn = _get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def execute_read(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Execute a read query and return results as a list of dicts."""
    conn = _get_connection()
    try:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


async def execute_write(query: str, params: tuple[Any, ...] = ()) -> int:
    """Execute a write query under the async lock. Returns rowcount."""
    async with _write_lock:
        conn = _get_connection()
        try:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()


async def execute_write_returning(
    query: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    """Execute a write query that returns rows (INSERT ... RETURNING, etc.)."""
    async with _write_lock:
        conn = _get_connection()
        try:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            conn.commit()
            return [dict(row) for row in rows]
        finally:
            conn.close()
