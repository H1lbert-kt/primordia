"""Stage-7 tests: world save/load roundtrip, identical continuation, lineage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numba import njit

from primordia import step
from primordia.config import Config
from primordia.genome import get_numba_rng_state, set_numba_rng_state
from primordia.io import FORMAT_VERSION, load_world, save_world
from primordia.step import advance
from primordia.world import World

REPO_ROOT = Path(__file__).resolve().parents[1]

STATE_ARRAYS = (
    "pos", "vel", "angle", "energy", "age", "genome", "species_id",
    "parent_id", "creature_id", "alive", "free_list", "food", "food_field",
    "hidden_prev",
)
COUNTERS = (
    "tick", "births", "deaths_famine", "deaths_age", "deaths_predation",
    "free_count", "next_id", "genealogy_used", "genealogy_overflow", "seed",
)


def assert_worlds_equal(a: World, b: World) -> None:
    for name in STATE_ARRAYS:
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for name in COUNTERS:
        assert getattr(a, name) == getattr(b, name), name
    la = a.genealogy[: a.genealogy_used]
    lb = b.genealogy[: b.genealogy_used]
    assert np.array_equal(la, lb), "genealogy"


def test_roundtrip_immediate(tmp_path: Path) -> None:
    cfg = Config(max_creatures=64, initial_creatures=32)
    w = World(cfg, seed=11)
    for _ in range(100):
        advance(w)
    saved_stream = get_numba_rng_state()

    path = tmp_path / "w.npz"
    save_world(w, str(path))
    v = load_world(str(path))

    assert_worlds_equal(w, v)
    assert v.config == w.config
    assert get_numba_rng_state() == saved_stream  # Numba stream restored
    assert v.rng.bit_generator.state == w.rng.bit_generator.state
    # cell grids are valid right after load (rebuild ran), not garbage from
    # the freshly constructed world
    w.rebuild_counts()
    for name in ("cell_counts", "cell_diet", "cell_offsets"):
        assert np.array_equal(getattr(w, name), getattr(v, name)), name
    # cell_slots is scratch (not saved): only the CSR prefix up to the live
    # count is meaningful, and the rebuild wrote exactly that prefix
    used = int(w.cell_offsets[-1])
    assert used == int(v.cell_offsets[-1])
    assert np.array_equal(w.cell_slots[:used], v.cell_slots[:used]), "cell_slots"


def test_continuation_identical(tmp_path: Path) -> None:
    """load(save(w)) advanced N ticks == w advanced N ticks, bit for bit."""
    cfg = Config(
        max_creatures=64,
        initial_creatures=32,
        reproduce_threshold=60.0,
        max_age=140,  # the spawn generation dies inside the continuation window,
        # generous economy: this test pins the IO/RNG stream, not the balance
        metabolic_cost=0.0,
        move_cost=0.0,
        turn_cost=0.0,
        food_growth_rate=0.2,
    )
    a = World(cfg, seed=42)
    for _ in range(120):
        advance(a)
    path = tmp_path / "mid.npz"
    save_world(a, str(path))
    saved_stream = get_numba_rng_state()
    births_at_save = a.births

    # b picks up the stream; the Numba RNG is process-global, so the two
    # continuations must run sequentially with an explicit restore between
    # them (the same sequential-worlds rule as everywhere else).
    b = load_world(str(path))
    for _ in range(60):
        advance(b)

    set_numba_rng_state(saved_stream)
    for _ in range(60):
        advance(a)

    assert a.tick == b.tick == 180
    assert_worlds_equal(a, b)
    assert b.births > births_at_save  # reproduction ran during the continuation


def test_unsupported_format_version(tmp_path: Path) -> None:
    path = tmp_path / "future.npz"
    np.savez(path, format_version=np.int32(FORMAT_VERSION + 999))
    with pytest.raises(ValueError, match="format"):
        load_world(str(path))


def test_v1_save_is_rejected_clearly(tmp_path: Path) -> None:
    """Pre-stage-9 saves lack food_field; the version gate must say so."""
    path = tmp_path / "old.npz"
    np.savez(path, format_version=np.int32(1))
    with pytest.raises(ValueError, match="format"):
        load_world(str(path))


def test_numba_rng_state_helpers_roundtrip() -> None:
    """Guard: numba._helperlib must keep exposing the stream (see genome.py)."""

    @njit
    def draw() -> float:
        return np.random.random()

    s0 = get_numba_rng_state()
    first = draw()
    s1 = get_numba_rng_state()
    assert s1 != s0
    set_numba_rng_state(s0)
    assert draw() == first  # restored stream replays the same draw
    assert get_numba_rng_state() == s1


def test_genealogy_ids_and_log() -> None:
    cfg = Config(max_creatures=64, initial_creatures=32, reproduce_threshold=40.0)
    w = World(cfg, seed=7)
    for _ in range(60):
        advance(w)
    assert w.births > 0

    ids = w.creature_id[w.alive]
    assert ids.size == w.alive_count
    assert np.unique(ids).size == ids.size  # unique among the living
    assert ids.min() >= 0 and ids.max() < w.next_id

    assert w.genealogy_used + w.genealogy_overflow == w.births
    log = w.genealogy[: w.genealogy_used]
    assert log.shape[1] == 3
    assert np.all(np.diff(log[:, 1]) > 0)  # child ids strictly increase
    assert np.all(log[:, 2] >= 0)  # a logged parent is never a founder marker
    assert np.all(log[:, 2] < log[:, 1])  # parent born before the child
    # every living creature spawned by birth points at an id that existed
    born = w.alive & (w.parent_id >= 0)
    assert np.all(w.parent_id[born] < w.next_id)


def test_lineage_survives_slot_recycling() -> None:
    cfg = Config(max_creatures=8, initial_creatures=4, reproduce_threshold=100.0)
    w = World(cfg, seed=3)
    parent = int(np.flatnonzero(w.alive)[0])
    parent_creature = int(w.creature_id[parent])
    w.energy[parent] = 200.0
    step.phase_reproduce(w)

    children = np.flatnonzero(w.alive & (w.parent_id == parent_creature))
    assert children.size == 1
    child = int(children[0])
    child_id = int(w.creature_id[child])

    # Kill the parent and recycle its slot for a newborn.
    w.energy[parent] = 0.0
    step.phase_deaths(w)
    assert not w.alive[parent]
    slots = w.spawn(1)
    assert int(slots[0]) == parent  # free-list LIFO reused the parent's slot
    assert w.creature_id[slots[0]] != parent_creature

    # The child still points at its true parent, dead slot or not.
    assert w.parent_id[child] == parent_creature
    assert not np.any(w.alive & (w.creature_id == parent_creature))
    log = w.genealogy[: w.genealogy_used]
    rows = log[(log[:, 1] == child_id) & (log[:, 2] == parent_creature)]
    assert rows.shape[0] == 1  # the birth survived the recycling


def test_genealogy_overflow_counter() -> None:
    cfg = Config(
        max_creatures=20,
        initial_creatures=4,
        reproduce_threshold=100.0,
        genealogy_capacity=5,
    )
    w = World(cfg, seed=1)
    for _ in range(10):
        w.energy[w.alive] = 200.0
        step.phase_reproduce(w)
        if w.free_count == 0:
            break

    assert w.genealogy_used == 5
    assert w.genealogy_overflow > 0
    assert w.genealogy_used + w.genealogy_overflow == w.births


def test_cli_save_then_load(tmp_path: Path) -> None:
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"

    proc = subprocess.run(
        [sys.executable, "run.py", "--seed", "42", "--headless", "--ticks", "60",
         "--save", str(first)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr

    proc = subprocess.run(
        [sys.executable, "run.py", "--load", str(first), "--headless",
         "--ticks", "10", "--save", str(second)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert "loaded" in proc.stdout and "world ->" in proc.stdout

    w = load_world(str(second))
    assert w.tick == 70
    assert w.alive_count > 0
    assert load_world(str(first)).tick == 60
