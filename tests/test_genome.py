"""Stage-3 tests: energy split, inheritance, mutation, clamps, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.step import advance
from primordia.world import World

ARRAYS = (
    "pos",
    "vel",
    "angle",
    "energy",
    "age",
    "genome",
    "species_id",
    "parent_id",
    "creature_id",
    "alive",
    "food",
)


def first_alive(w: World) -> int:
    return int(np.flatnonzero(w.alive)[0])


def birth_of_one_parent(cfg: Config, seed: int = 2) -> tuple[World, int, int]:
    """One eligible parent, one phase_reproduce call; returns (world, parent, child)."""
    w = World(cfg, seed=seed)
    parent = first_alive(w)
    w.energy[parent] = 200.0
    step.phase_reproduce(w)
    children = np.flatnonzero(w.alive & (w.parent_id == w.creature_id[parent]))
    assert children.size == 1
    return w, parent, int(children[0])


def test_child_starts_with_empty_memory() -> None:
    """The Elman state is world state, but never inherited across births."""
    cfg = Config(
        max_creatures=10,
        initial_creatures=5,
        reproduce_threshold=100.0,
    )
    w, parent, child = birth_of_one_parent(cfg)
    w.hidden_prev[parent] = 0.7  # simulate a lived-in parent...
    step.phase_reproduce(w)  # (a second birth, parent still eligible)
    kids = np.flatnonzero(w.alive & (w.parent_id == w.creature_id[parent]))
    assert kids.size >= 1
    assert np.all(w.hidden_prev[kids] == 0.0)
    assert w.hidden_prev[child].max() <= 0.0  # the first child too


def test_reproduce_splits_energy() -> None:
    cfg = Config(max_creatures=10, initial_creatures=5, reproduce_threshold=100.0)
    w = World(cfg, seed=0)
    parent = first_alive(w)
    w.energy[parent] = 200.0
    step.phase_reproduce(w)

    children = np.flatnonzero(w.alive & (w.parent_id == w.creature_id[parent]))
    assert children.size == 1
    child = int(children[0])
    total = float(w.energy[parent]) + float(w.energy[child])
    assert total == pytest.approx(200.0, rel=1e-6)  # no energy created
    assert w.energy[parent] == pytest.approx(100.0, rel=1e-6)
    assert w.energy[child] == pytest.approx(100.0, rel=1e-6)
    assert w.alive[parent] and w.alive[child]
    assert w.free_count == 4
    assert w.births == 1


def test_child_inherits_state_and_genome() -> None:
    cfg = Config(
        max_creatures=10,
        initial_creatures=5,
        reproduce_threshold=100.0,
        mutation_rate=0.0,  # pure clone
    )
    w, parent, child = birth_of_one_parent(cfg)

    assert np.array_equal(w.genome[child], w.genome[parent])
    assert w.age[child] == 0
    assert w.parent_id[child] == w.creature_id[parent]
    np.testing.assert_array_equal(w.pos[child], w.pos[parent])
    np.testing.assert_array_equal(w.vel[child], 0.0)
    assert w.angle[child] == w.angle[parent]
    assert w.species_id[child] == w.species_id[parent]


def test_mutation_zero_std_is_clone_and_full_std_changes_genes() -> None:
    base = dict(max_creatures=10, initial_creatures=5, reproduce_threshold=100.0)

    w, parent, child = birth_of_one_parent(
        Config(mutation_rate=1.0, mutation_std=0.0, trait_mutation_std=0.0, **base)
    )
    assert np.array_equal(w.genome[child], w.genome[parent])  # draws happen, add 0

    cfg = Config(mutation_rate=1.0, mutation_std=0.05, trait_mutation_std=0.1, **base)
    w, parent, child = birth_of_one_parent(cfg)
    p = cfg.brain_params
    assert np.all(w.genome[child, :p] != w.genome[parent, :p])  # every brain gene
    assert w.genome[child, p] != w.genome[parent, p]  # max_speed changed
    assert 0.0 <= w.genome[child, p + 3] <= 1.0  # diet stays a valid fraction


def test_trait_clamps_stay_valid() -> None:
    cfg = Config(
        max_creatures=50,
        initial_creatures=10,
        initial_energy=1000.0,  # energy stays far above threshold: 30 generations
        reproduce_threshold=40.0,
        mutation_rate=1.0,
        mutation_std=0.5,
        trait_mutation_std=5.0,  # violent: exp(+-5) swings force the clamps
    )
    w = World(cfg, seed=3)
    for _ in range(30):
        step.phase_reproduce(w)

    alive = w.alive
    assert np.count_nonzero(alive) == cfg.max_creatures  # filled the capacity
    t = cfg.brain_params
    max_dim = float(max(cfg.width, cfg.height))
    speed = w.genome[alive, t]
    size = w.genome[alive, t + 1]
    vision = w.genome[alive, t + 2]
    diet = w.genome[alive, t + 3]
    assert np.all(speed > 0.0) and np.all(speed <= max_dim)
    assert np.all(size > 0.0) and np.all(size <= cfg.max_size)
    assert np.all(vision > 0.0) and np.all(vision <= max_dim)
    assert np.all(diet >= 0.0) and np.all(diet <= 1.0)
    assert np.any(diet > 0.0)  # additive mutation lets diet leave 0


def test_energy_conserved_with_births() -> None:
    cfg = Config(
        max_creatures=40,
        initial_creatures=10,
        initial_energy=100.0,
        reproduce_threshold=40.0,
        metabolic_cost=0.0,
        move_cost=0.0,
        turn_cost=0.0,  # isolate the food <-> energy accounting from steering
        food_growth_rate=0.0,  # food only transfers to creatures
    )
    w = World(cfg, seed=4)
    total0 = float(w.food.sum(dtype=np.float64)) + float(w.energy.sum(dtype=np.float64))
    for _ in range(100):
        advance(w)
    assert w.births > 0
    total = float(w.food.sum(dtype=np.float64)) + float(w.energy.sum(dtype=np.float64))
    assert np.isclose(total0, total, rtol=1e-4)


def test_determinism_with_reproduction() -> None:
    cfg = Config(max_creatures=150, initial_creatures=50, reproduce_threshold=40.0)

    def run(seed: int) -> World:
        w = World(cfg, seed=seed)
        for _ in range(300):
            advance(w)
        return w

    w1 = run(7)  # sequential: the Numba RNG stream is process-global
    w2 = run(7)
    for name in ARRAYS:
        assert np.array_equal(getattr(w1, name), getattr(w2, name)), name
    assert (w1.births, w1.deaths_famine, w1.deaths_age) == (
        w2.births,
        w2.deaths_famine,
        w2.deaths_age,
    )
    assert w1.births > 0  # the test is vacuous otherwise


def test_capacity_blocks_net_growth() -> None:
    cfg = Config(max_creatures=20, initial_creatures=20, reproduce_threshold=1.0)
    w = World(cfg, seed=5)
    for _ in range(50):
        advance(w)
    assert w.alive_count + w.free_count == cfg.max_creatures
    # Slots only free up through death, so births can never exceed deaths.
    assert w.births <= w.deaths_famine + w.deaths_age


def test_move_cost_quadratic() -> None:
    cfg = Config(
        max_creatures=4,
        initial_creatures=4,
        initial_energy=100.0,
        metabolic_cost=0.1,
        move_cost=0.05,
    )
    w = World(cfg, seed=6)
    w.vel[:, 0] = 3.0
    w.vel[:, 1] = 4.0  # |vel|^2 = 25
    e0 = w.energy.copy()
    step.phase_age_and_metabolize(w)
    np.testing.assert_allclose(w.energy, e0 - 0.1 - 0.05 * 25.0, rtol=1e-6)


def test_death_causes_counted() -> None:
    cfg = Config(max_creatures=6, initial_creatures=6, max_age=10)
    w = World(cfg, seed=7)
    slots = np.flatnonzero(w.alive)
    w.energy[slots[:3]] = 0.0  # starve
    w.age[slots[3:5]] = cfg.max_age  # old age
    step.phase_deaths(w)
    assert w.deaths_famine == 3
    assert w.deaths_age == 2
    assert w.alive_count == 1
    assert w.free_count == 5
