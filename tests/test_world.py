"""Stage-1 world tests: invariants, determinism, energy conservation, wrap, free-list."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.step import advance
from primordia.world import World


def make_config(**overrides: object) -> Config:
    return replace(Config(), **overrides)


def test_dtypes_and_shapes() -> None:
    cfg = make_config()
    w = World(cfg, seed=0)
    n = cfg.max_creatures
    assert w.pos.shape == (n, 2) and w.pos.dtype == np.float32
    assert w.vel.shape == (n, 2) and w.vel.dtype == np.float32
    assert w.angle.shape == (n,) and w.angle.dtype == np.float32
    assert w.energy.shape == (n,) and w.energy.dtype == np.float32
    assert w.age.shape == (n,) and w.age.dtype == np.int32
    assert w.genome.shape == (n, cfg.genome_size) and w.genome.dtype == np.float32
    assert w.species_id.dtype == np.int32
    assert w.parent_id.dtype == np.int32
    assert w.creature_id.dtype == np.int32
    assert w.genealogy.shape == (cfg.genealogy_capacity, 3)
    assert w.genealogy.dtype == np.int32
    assert w.hidden_prev.shape == (n, cfg.hidden_size)
    assert w.hidden_prev.dtype == np.float32
    assert w.alive.dtype == np.bool_
    assert w.food.dtype == np.float32
    assert w.alive_count == cfg.initial_creatures
    assert w.alive_count + w.free_count == n


def test_invariants_after_1000_ticks() -> None:
    cfg = make_config(max_creatures=500, initial_creatures=500)
    w = World(cfg, seed=1)
    for _ in range(1000):
        advance(w)

    alive = w.alive
    assert np.all(w.energy[alive] > 0.0)
    assert np.all(w.age[alive] < cfg.max_age)
    limit = np.array([cfg.width, cfg.height], dtype=np.float32)
    assert np.all(w.pos >= 0.0) and np.all(w.pos < limit)
    assert w.alive_count + w.free_count == cfg.max_creatures
    assert np.all(w.food >= 0.0) and np.all(w.food <= cfg.food_capacity)
    assert w.tick == 1000


def test_determinism_same_seed() -> None:
    cfg = make_config(max_creatures=100, initial_creatures=100)
    # Worlds run sequentially: the Numba RNG (used by reproduction) is a
    # process-global stream, so interleaved worlds would share draws.
    w1 = World(cfg, seed=42)
    for _ in range(500):
        advance(w1)
    w2 = World(cfg, seed=42)
    for _ in range(500):
        advance(w2)
    arrays = ("pos", "vel", "angle", "energy", "age", "genome", "species_id", "parent_id", "creature_id", "alive", "food", "hidden_prev")
    for name in arrays:
        assert np.array_equal(getattr(w1, name), getattr(w2, name)), name
    assert (w1.births, w1.deaths_famine, w1.deaths_age) == (
        w2.births,
        w2.deaths_famine,
        w2.deaths_age,
    )

    w3 = World(cfg, seed=43)
    for _ in range(500):
        advance(w3)
    assert not np.array_equal(w1.pos, w3.pos)


def test_energy_conservation_without_growth_or_deaths() -> None:
    cfg = make_config(
        max_creatures=100,
        initial_creatures=100,
        initial_energy=1000.0,
        metabolic_cost=0.1,
        food_growth_rate=0.0,  # food then only transfers to creatures
        move_cost=0.0,  # isolate the food <-> energy accounting from balance
    )
    w = World(cfg, seed=7)
    food0 = float(w.food.sum(dtype=np.float64))
    energy0 = float(w.energy.sum(dtype=np.float64))

    ticks = 100
    for _ in range(ticks):
        advance(w)

    assert w.alive_count == cfg.initial_creatures
    removed = cfg.metabolic_cost * cfg.initial_creatures * ticks
    total = float(w.food.sum(dtype=np.float64)) + float(w.energy.sum(dtype=np.float64))
    assert np.isclose(food0 + energy0 - removed, total, rtol=1e-4)


def test_wrap_borders() -> None:
    # Unit-test the kernel directly: `advance` overwrites vel from brain outputs.
    cfg = make_config(width=100.0, height=100.0, max_creatures=20, initial_creatures=20)
    w = World(cfg, seed=3)
    w.vel[:, 0] = 7.0
    w.vel[:, 1] = -5.0
    slot = np.flatnonzero(w.alive)[0]
    w.pos[slot] = (99.0, 1.0)

    step._integrate(w.pos, w.vel, w.alive, np.float32(100.0), np.float32(100.0))
    assert w.pos[slot, 0] == pytest.approx(6.0)
    assert w.pos[slot, 1] == pytest.approx(96.0)

    limit = np.array([100.0, 100.0], dtype=np.float32)
    for _ in range(50):
        step._integrate(w.pos, w.vel, w.alive, np.float32(100.0), np.float32(100.0))
        assert np.all(w.pos >= 0.0) and np.all(w.pos < limit)


def test_free_list_recycles_dead_slots() -> None:
    cfg = make_config(max_creatures=50, initial_creatures=50)
    w = World(cfg, seed=9)
    # Kill by old age: eating happens before deaths in the tick, so low energy
    # alone would be "cured" by foraging in the same tick.
    w.age[w.alive] = cfg.max_age
    advance(w)
    assert w.alive_count == 0
    assert w.free_count == 50

    slots = w.spawn(10)
    assert w.alive_count == 10
    assert w.free_count == 40
    assert slots.size == 10 and np.unique(slots).size == 10
    assert np.all(slots < cfg.max_creatures)
    assert np.all(w.alive[slots])
    assert np.all(w.energy[slots] == cfg.initial_energy)
    assert np.all(w.parent_id[slots] == -1)


def test_food_field_patches() -> None:
    """Stage-9: K is heterogeneous within bounds; amplitude 0 = uniform."""
    uniform = World(make_config(food_patch_amplitude=0.0), seed=3)
    assert np.all(uniform.food_field == uniform.config.food_capacity)

    cfg = make_config()
    w = World(cfg, seed=3)
    field = w.food_field
    assert field.shape == w.food.shape and field.dtype == np.float32
    assert field.std() > 1.0  # real patches, not a flat field
    lo = cfg.food_capacity * (1.0 - cfg.food_patch_amplitude)
    assert field.min() >= lo - 1e-3
    assert field.max() <= cfg.food_capacity + 1e-3


def test_food_field_deterministic() -> None:
    a = World(make_config(), seed=9)
    b = World(make_config(), seed=9)
    assert np.array_equal(a.food_field, b.food_field)
    c = World(make_config(), seed=10)
    assert not np.array_equal(a.food_field, c.food_field)


def test_patches_persist_under_growth() -> None:
    """Food still follows the terrain after hundreds of ticks (no flattening)."""
    w = World(make_config(), seed=5)
    for _ in range(300):
        advance(w)
    corr = np.corrcoef(w.food.ravel(), w.food_field.ravel())[0, 1]
    assert corr > 0.7
