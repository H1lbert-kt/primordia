"""Persistent world store on SQLite (stage 19: the daemon).

The database holds everything a self-running world needs and nothing
the tick reads: a ``meta`` key/value table (seed, format version,
config, last checkpoint tick, schema version), full-state checkpoints
(npz blobs, same format as :mod:`primordia.io`) and a light ``timeline``
of population gauges sampled outside the tick. WAL journaling keeps the
window free to read the latest checkpoint while the daemon writes.

Only stdlib ``sqlite3`` — no new dependency. The simulation core stays
pygame-free; this module never imports the renderer.
"""

from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import asdict
from typing import Any

from .io import FORMAT_VERSION, load_world, save_world
from .world import World

SCHEMA_VERSION = 1

# Timeline columns, in table order: gauges only — cheap reads off World,
# no RNG, never consulted by the tick. Full per-tick stats stay on the
# `--stats` npz of normal runs (the StatsRecorder grows unbounded, which
# a forever-running daemon must not do).
TIMELINE_COLUMNS: tuple[str, ...] = (
    "tick",
    "alive",
    "births",
    "deaths_famine",
    "deaths_age",
    "deaths_predation",
    "mean_energy",
    "n_fire",
    "n_house",
    "wood_total",
)


def open_db(path: str) -> sqlite3.Connection:
    """Open (creating if needed) a world database with WAL journaling."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection, world: World) -> None:
    """Create tables if missing and write meta keys (never clobbering).

    Idempotent: reopening an existing database keeps its meta; a schema
    bump from a future stage is rejected instead of silently migrated.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            tick  INTEGER PRIMARY KEY,
            state BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS timeline (
            tick            INTEGER PRIMARY KEY,
            alive           INTEGER NOT NULL,
            births          INTEGER NOT NULL,
            deaths_famine   INTEGER NOT NULL,
            deaths_age      INTEGER NOT NULL,
            deaths_predation INTEGER NOT NULL,
            mean_energy     REAL NOT NULL,
            n_fire          INTEGER NOT NULL,
            n_house         INTEGER NOT NULL,
            wood_total      REAL NOT NULL
        );
        """
    )
    existing = conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    if existing is not None and int(existing[0]) != SCHEMA_VERSION:
        raise ValueError(
            f"database schema {existing[0]} is not this build's "
            f"{SCHEMA_VERSION}"
        )
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "format_version": str(FORMAT_VERSION),
        "seed": str(world.seed),
        "config_json": json.dumps(asdict(world.config), sort_keys=True),
        "tick": str(world.tick),
    }
    conn.executemany(
        "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
        list(meta.items()),
    )
    conn.commit()


def save_checkpoint(conn: sqlite3.Connection, world: World) -> None:
    """Serialize the full world into the checkpoints table."""
    buf = io.BytesIO()
    save_world(world, buf)
    conn.execute(
        "INSERT OR REPLACE INTO checkpoints(tick, state) VALUES (?, ?)",
        (int(world.tick), buf.getvalue()),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('tick', ?)",
        (str(int(world.tick)),),
    )
    conn.commit()


def load_checkpoint(conn: sqlite3.Connection) -> World:
    """Rebuild the most recent checkpoint (RNG streams restored)."""
    fmt = conn.execute(
        "SELECT value FROM meta WHERE key = 'format_version'"
    ).fetchone()
    if fmt is not None and int(fmt[0]) != FORMAT_VERSION:
        raise ValueError(
            f"database format {fmt[0]} is not this build's {FORMAT_VERSION}"
        )
    row = conn.execute(
        "SELECT state FROM checkpoints ORDER BY tick DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("database has no checkpoints")
    return load_world(io.BytesIO(row[0]))


def latest_tick(conn: sqlite3.Connection) -> int | None:
    """Tick of the newest checkpoint, or None if the table is empty."""
    row = conn.execute("SELECT MAX(tick) FROM checkpoints").fetchone()
    return None if row is None or row[0] is None else int(row[0])


def prune_checkpoints(conn: sqlite3.Connection, keep: int) -> int:
    """Keep only the ``keep`` newest checkpoints; return rows deleted."""
    if keep < 1:
        raise ValueError("keep must be >= 1")
    cur = conn.execute(
        "DELETE FROM checkpoints WHERE tick NOT IN "
        "(SELECT tick FROM checkpoints ORDER BY tick DESC LIMIT ?)",
        (keep,),
    )
    conn.commit()
    return int(cur.rowcount)


def timeline_row(world: World) -> tuple[Any, ...]:
    """One gauge row for the timeline table (order = TIMELINE_COLUMNS)."""
    mean_energy = (
        float(world.energy[world.alive].mean()) if world.alive_count else 0.0
    )
    return (
        int(world.tick),
        int(world.alive_count),
        int(world.births),
        int(world.deaths_famine),
        int(world.deaths_age),
        int(world.deaths_predation),
        mean_energy,
        int(world.struct_alive.sum()),
        int((world.struct_alive & (world.struct_type == 1)).sum()),
        float(world.wood.sum()),
    )


def insert_timeline(
    conn: sqlite3.Connection, rows: list[tuple[Any, ...]]
) -> None:
    """Batch-insert timeline rows (one transaction); empty rows = no-op."""
    if not rows:
        return
    placeholders = ", ".join("?" * len(TIMELINE_COLUMNS))
    conn.executemany(
        f"INSERT OR REPLACE INTO timeline VALUES ({placeholders})", rows
    )
    conn.commit()


def timeline_rows(
    conn: sqlite3.Connection, limit: int | None = None
) -> list[tuple[Any, ...]]:
    """Timeline rows ordered by tick (newest last); test/inspection aid."""
    sql = f"SELECT * FROM timeline ORDER BY tick"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql))
