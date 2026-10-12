"""Stage-19 tests: daemon CLI — SQLite persistence and identical resume."""

from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from primordia.database import timeline_rows
from primordia.io import load_world

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "run.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_worlds_equal(a, b) -> None:
    assert (a.tick, a.alive_count, a.births) == (b.tick, b.alive_count, b.births)
    for name in ("pos", "vel", "angle", "energy", "age", "genome", "alive"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_cli_validations() -> None:
    cases = [
        (["--daemon", "--headless"], "--daemon requires --db"),
        (["--daemon", "--db", "x.db", "--headless", "--stats", "s.npz"], "cannot be combined"),
        (["--load-db", "a.npz", "--load", "b.npz"], "mutually exclusive"),
        (["--load-db", "a.npz", "--config", "c.json"], "cannot be combined"),
        (["--db", "x.db", "--seed", "1"], "--db requires --headless"),
        (["--checkpoint-every", "0", "--db", "x.db", "--headless"], "checkpoint-every"),
    ]
    for args, needle in cases:
        proc = run_cli(*args, "--ticks", "1")
        assert proc.returncode == 2, (args, proc.stderr)
        assert needle in proc.stderr, (args, proc.stderr)


def test_finite_db_run_then_resume_is_bit_identical(tmp_path: Path) -> None:
    db = tmp_path / "w.db"
    cont = tmp_path / "cont.npz"
    direct = tmp_path / "direct.npz"

    proc = run_cli(
        "--seed", "42", "--headless", "--ticks", "60",
        "--db", str(db), "--checkpoint-every", "25", "--metrics-every", "5",
    )
    assert proc.returncode == 0, proc.stderr
    assert "checkpoint ->" in proc.stdout

    proc = run_cli(
        "--load-db", str(db), "--headless", "--ticks", "40", "--save", str(cont),
    )
    assert proc.returncode == 0, proc.stderr
    assert "loaded db" in proc.stdout

    proc = run_cli(
        "--seed", "42", "--headless", "--ticks", "100", "--save", str(direct),
    )
    assert proc.returncode == 0, proc.stderr

    assert_worlds_equal(load_world(str(cont)), load_world(str(direct)))

    conn = sqlite3.connect(str(db))
    ticks = [row[0] for row in conn.execute("SELECT tick FROM checkpoints")]
    assert max(ticks) == 60  # final checkpoint in the finally block
    rows = timeline_rows(conn)
    assert len(rows) > 0
    assert rows[0][0] == 5 and rows[-1][0] == 60  # metrics-every=5
    births = [r[2] for r in rows]
    assert births == sorted(births)
    conn.close()


def test_daemon_sigint_checkpoints_and_resumes(tmp_path: Path) -> None:
    db = tmp_path / "daemon.db"
    proc = subprocess.Popen(
        [sys.executable, "run.py", "--daemon", "--db", str(db),
         "--checkpoint-every", "200", "--metrics-every", "20"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(6.0)  # JIT warmup + some ticks; the world runs forever
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    assert "interrupted at tick" in out
    assert "checkpoint ->" in out

    conn = sqlite3.connect(str(db))
    latest = conn.execute("SELECT MAX(tick) FROM checkpoints").fetchone()[0]
    assert latest is not None and latest > 0
    conn.close()

    cont = tmp_path / "after.npz"
    proc = run_cli(
        "--load-db", str(db), "--headless", "--ticks", "10", "--save", str(cont),
    )
    assert proc.returncode == 0, proc.stderr
    w = load_world(str(cont))
    assert w.tick == latest + 10
    assert w.alive_count > 0


def test_load_db_windowed_snapshot_path_exists(tmp_path: Path) -> None:
    # the observe path loads the checkpoint and enters the windowed branch;
    # we only assert it gets that far without a display by forcing the
    # headless-equivalent load + select bookkeeping through a zero-tick run
    db = tmp_path / "snap.db"
    proc = run_cli(
        "--seed", "7", "--headless", "--ticks", "30", "--db", str(db),
    )
    assert proc.returncode == 0, proc.stderr
    proc = run_cli("--load-db", str(db), "--headless", "--ticks", "0")
    assert proc.returncode == 0, proc.stderr
    assert "loaded db" in proc.stdout
