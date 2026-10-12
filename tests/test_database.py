"""Stage-19 tests: SQLite world store (checkpoints, timeline, prune)."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from primordia import database as db
from primordia.config import Config
from primordia.io import FORMAT_VERSION, save_world
from primordia.step import advance
from primordia.world import World


def run_world(ticks: int = 40, seed: int = 5) -> World:
    cfg = Config(max_creatures=64, initial_creatures=32)
    w = World(cfg, seed=seed)
    for _ in range(ticks):
        advance(w)
    return w


def assert_worlds_equal(a: World, b: World) -> None:
    assert a.tick == b.tick
    assert a.alive_count == b.alive_count
    assert a.births == b.births
    for name in ("pos", "vel", "angle", "energy", "age", "genome", "alive"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_checkpoint_roundtrip_bit_exact(tmp_path: Path) -> None:
    conn = db.open_db(str(tmp_path / "w.db"))
    w = run_world()
    db.init_db(conn, w)
    db.save_checkpoint(conn, w)
    v = db.load_checkpoint(conn)
    assert_worlds_equal(w, v)
    assert v.rng.bit_generator.state == w.rng.bit_generator.state
    conn.close()


def test_meta_written_and_idempotent(tmp_path: Path) -> None:
    path = str(tmp_path / "w.db")
    w = run_world(ticks=10, seed=9)
    conn = db.open_db(path)
    db.init_db(conn, w)
    meta = dict(conn.execute("SELECT key, value FROM meta"))
    assert int(meta["schema_version"]) == db.SCHEMA_VERSION
    assert int(meta["format_version"]) == FORMAT_VERSION
    assert int(meta["seed"]) == 9
    assert int(meta["tick"]) == w.tick
    assert db.latest_tick(conn) is None  # no checkpoint yet
    db.save_checkpoint(conn, w)
    conn.close()

    # reopen: meta must not be clobbered, checkpoint survives
    conn2 = db.open_db(path)
    db.init_db(conn2, run_world(ticks=0, seed=99))  # different world
    meta2 = dict(conn2.execute("SELECT key, value FROM meta"))
    assert int(meta2["seed"]) == 9  # original seed kept
    assert db.latest_tick(conn2) == w.tick
    assert_worlds_equal(w, db.load_checkpoint(conn2))
    conn2.close()


def test_prune_keeps_newest(tmp_path: Path) -> None:
    conn = db.open_db(str(tmp_path / "w.db"))
    w = run_world(ticks=10)
    db.init_db(conn, w)
    db.save_checkpoint(conn, w)  # tick 10
    for _ in range(5):
        advance(w)
        db.save_checkpoint(conn, w)
    ticks = [row[0] for row in conn.execute("SELECT tick FROM checkpoints")]
    assert len(ticks) == 6

    deleted = db.prune_checkpoints(conn, keep=2)
    assert deleted == 4
    kept = [row[0] for row in conn.execute("SELECT tick FROM checkpoints ORDER BY tick")]
    assert kept == sorted(ticks)[-2:]
    assert db.latest_tick(conn) == max(ticks)
    with pytest.raises(ValueError, match="keep"):
        db.prune_checkpoints(conn, keep=0)
    conn.close()


def test_load_checkpoint_rejects_foreign_format(tmp_path: Path) -> None:
    conn = db.open_db(str(tmp_path / "w.db"))
    w = run_world(ticks=5)
    db.init_db(conn, w)
    # forge a blob whose format_version is an older release
    buf = io.BytesIO()
    save_world(w, buf)
    forged = io.BytesIO()
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as f:
        payload = {name: f[name] for name in f.files}
    payload["format_version"] = np.int32(FORMAT_VERSION - 1)
    np.savez(forged, **payload)
    conn.execute(
        "INSERT OR REPLACE INTO checkpoints(tick, state) VALUES (?, ?)",
        (w.tick, forged.getvalue()),
    )
    conn.commit()
    with pytest.raises(ValueError, match="format"):
        db.load_checkpoint(conn)
    conn.close()


def test_empty_db_load_raises(tmp_path: Path) -> None:
    conn = db.open_db(str(tmp_path / "empty.db"))
    w = run_world(ticks=0)
    db.init_db(conn, w)
    with pytest.raises(ValueError, match="no checkpoints"):
        db.load_checkpoint(conn)
    conn.close()


def test_timeline_row_and_batch(tmp_path: Path) -> None:
    conn = db.open_db(str(tmp_path / "w.db"))
    w = run_world(ticks=30)
    db.init_db(conn, w)
    rows = [db.timeline_row(w)]
    db.insert_timeline(conn, rows)  # empty batch after this
    db.insert_timeline(conn, [])
    stored = db.timeline_rows(conn)
    assert len(stored) == 1
    tick, alive, births, *rest, mean_e, n_fire, n_house, wood = stored[0]
    assert tick == w.tick and alive == w.alive_count and births == w.births
    assert n_fire == int(w.struct_alive.sum())
    assert n_house == int(
        (w.struct_alive & (w.struct_type == 1)).sum()
    )
    assert mean_e == pytest.approx(
        float(w.energy[w.alive].mean()) if w.alive_count else 0.0
    )
    assert wood == pytest.approx(float(w.wood.sum()))
    assert len(rest) == 3  # three death counters
    conn.close()


def test_timeline_is_monotone_across_checkpoints(tmp_path: Path) -> None:
    conn = db.open_db(str(tmp_path / "w.db"))
    w = run_world(ticks=5)
    db.init_db(conn, w)
    rows = []
    for _ in range(20):
        advance(w)
        rows.append(db.timeline_row(w))
    db.insert_timeline(conn, rows)
    stored = db.timeline_rows(conn)
    ticks = [r[0] for r in stored]
    assert ticks == sorted(ticks)
    births = [r[2] for r in stored]
    assert births == sorted(births)  # cumulative counter never decreases
    conn.close()


def test_wal_second_connection_reads_clean(tmp_path: Path) -> None:
    path = str(tmp_path / "wal.db")
    w = run_world(ticks=20)
    writer = db.open_db(path)
    db.init_db(writer, w)
    for _ in range(3):
        advance(w)
        db.save_checkpoint(writer, w)
        db.insert_timeline(writer, [db.timeline_row(w)])

    reader = db.open_db(path)  # second connection, WAL snapshot
    mode = reader.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    assert db.latest_tick(reader) == w.tick
    assert len(db.timeline_rows(reader)) == 3
    integrity = reader.execute("PRAGMA integrity_check").fetchone()[0]
    assert str(integrity).lower() == "ok"
    reader.close()
    writer.close()


def test_schema_version_mismatch_rejected(tmp_path: Path) -> None:
    path = str(tmp_path / "schema.db")
    conn = db.open_db(path)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '999')")
    conn.commit()
    with pytest.raises(ValueError, match="schema"):
        db.init_db(conn, run_world(ticks=0))
    conn.close()
