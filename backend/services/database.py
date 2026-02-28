"""
SQLite database layer for voice cloning.

Uses WAL mode for concurrent reads and an asyncio lock to serialize writes.
All async functions use aiosqlite so they never block the event loop.
init_db() stays synchronous (called once at startup before the loop serves requests).
"""

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

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


def init_db() -> None:
    """Initialize the database: create tables if they don't exist.

    Uses plain sqlite3 because this runs once at startup before the event
    loop begins serving requests.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


async def execute_read(
    query: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    """Execute a read query and return results as a list of dicts."""
    async with aiosqlite.connect(DB_PATH, timeout=10) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = aiosqlite.Row
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def execute_write(query: str, params: tuple[Any, ...] = ()) -> int:
    """Execute a write query under the async lock. Returns rowcount."""
    async with _write_lock:
        async with aiosqlite.connect(DB_PATH, timeout=10) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            cursor = await conn.execute(query, params)
            await conn.commit()
            return cursor.rowcount


async def execute_write_returning(
    query: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    """Execute a write query that returns rows (INSERT ... RETURNING, etc.)."""
    async with _write_lock:
        async with aiosqlite.connect(DB_PATH, timeout=10) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            await conn.commit()
            return [dict(row) for row in rows]
