"""Stage-3/11 tests: sexual birth, inheritance, mutation, clamps, determinism."""

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
    "meat",
)


def first_alive(w: World) -> int:
    return int(np.flatnonzero(w.alive)[0])


def birth_of_couple(
    cfg: Config, seed: int = 2, partner_energy: float = 90.0
) -> tuple[World, int, int, int]:
    """One eligible initiator + one nearby partner; returns (world, i, j, child).

    The partner sits within ``mate_range`` of the initiator, far from everyone
    else, and below the threshold (but above its half of the child) so only
    the initiator breeds: exactly one birth per call. The child's slot is
    always below the initiator's (free-list LIFO), so the ascending scan
    never re-breeds the newborn in the same pass.
    """
    w = World(cfg, seed=seed)
    slots = np.flatnonzero(w.alive)
    i = int(slots[0])
    j = int(slots[1])
    w.energy[i] = 200.0
    w.energy[j] = partner_energy
    w.pos[j] = w.pos[i] + np.float32([1.0, 0.0])  # distance 1 < default mate_range
    # park everyone else far away so they never interfere
    for k in slots[2:]:
        w.pos[k] = w.pos[i] + np.float32([100.0, 100.0])
    step.phase_rebuild_counts(w)  # the CSR must list the moved slots
    step.phase_reproduce(w)
    children = np.flatnonzero(w.alive & (w.parent_id == w.creature_id[i]))
    assert children.size == 1
    return w, i, j, int(children[0])


def test_child_starts_with_empty_memory() -> None:
    """The Elman state is world state, but never inherited across births."""
    cfg = Config(
        max_creatures=10,
        initial_creatures=5,
        reproduce_threshold=100.0,
    )
    w, i, _j, child = birth_of_couple(cfg)
    w.hidden_prev[i] = 0.7  # simulate a lived-in initiator...
    step.phase_reproduce(w)  # (a second birth, initiator still eligible)
    kids = np.flatnonzero(w.alive & (w.parent_id == w.creature_id[i]))
    assert kids.size >= 1
    assert np.all(w.hidden_prev[kids] == 0.0)
    assert w.hidden_prev[child].max() <= 0.0  # the first child too


def test_reproduce_splits_energy_between_two_parents() -> None:
    cfg = Config(max_creatures=10, initial_creatures=5, reproduce_threshold=100.0)
    w, i, j, child = birth_of_couple(cfg)

    total = float(w.energy[i]) + float(w.energy[j]) + float(w.energy[child])
    assert total == pytest.approx(290.0, rel=1e-6)  # no energy created (200+90)
    # child_e = 0.5 * 200 = 100; each parent paid 50
    assert w.energy[i] == pytest.approx(150.0, rel=1e-6)
    assert w.energy[j] == pytest.approx(40.0, rel=1e-6)
    assert w.energy[child] == pytest.approx(100.0, rel=1e-6)
    assert w.alive[i] and w.alive[j] and w.alive[child]
    assert w.free_count == 4  # 5 free slots, one consumed by the birth
    assert w.births == 1


def test_no_partner_no_birth() -> None:
    """An eligible initiator alone in the world simply does not reproduce."""
    cfg = Config(max_creatures=10, initial_creatures=5, reproduce_threshold=100.0)
    w = World(cfg, seed=0)
    i = first_alive(w)
    w.energy[i] = 200.0
    # everyone else parked beyond mate_range
    slots = np.flatnonzero(w.alive)
    for k in slots[1:]:
        w.pos[k] = w.pos[i] + np.float32([500.0, 500.0])
    step.phase_rebuild_counts(w)
    step.phase_reproduce(w)
    assert w.births == 0
    assert w.energy[i] == 200.0  # no debit without a partner
    assert w.free_count == cfg.max_creatures - cfg.initial_creatures


def test_child_inherits_state_and_genome() -> None:
    cfg = Config(
        max_creatures=10,
        initial_creatures=5,
        reproduce_threshold=100.0,
        mutation_rate=0.0,  # crossover only, no noise
    )
    w, i, j, child = birth_of_couple(cfg)

    # every gene is one parent's copy (uniform crossover, no mutation)
    from_parent = (w.genome[child] == w.genome[i]) | (w.genome[child] == w.genome[j])
    assert np.all(from_parent)
    assert w.age[child] == 0
    assert w.parent_id[child] == w.creature_id[i]  # lineage tracks the initiator
    np.testing.assert_array_equal(w.pos[child], w.pos[i])
    np.testing.assert_array_equal(w.vel[child], 0.0)
    assert w.angle[child] == w.angle[i]
    assert w.species_id[child] == w.species_id[i]


def test_crossover_mixes_both_parents() -> None:
    """With distinguishable parents the child's genes come from both."""
    cfg = Config(
        max_creatures=10,
        initial_creatures=5,
        reproduce_threshold=100.0,
        mutation_rate=0.0,
    )
    w = World(cfg, seed=2)
    slots = np.flatnonzero(w.alive)
    i, j = int(slots[0]), int(slots[1])
    w.genome[i] = 1.0
    w.genome[j] = -1.0
    w.energy[i] = 200.0
    w.energy[j] = 90.0
    w.pos[j] = w.pos[i] + np.float32([1.0, 0.0])
    for k in slots[2:]:
        w.pos[k] = w.pos[i] + np.float32([100.0, 100.0])
    step.phase_rebuild_counts(w)
    step.phase_reproduce(w)

    assert w.births == 1
    child = int(np.flatnonzero(w.alive & (w.parent_id == w.creature_id[i]))[0])
    genes = w.genome[child]
    assert np.any(genes == 1.0) and np.any(genes == -1.0)


def test_mutation_zero_std_is_crossover_only_and_full_std_changes_genes() -> None:
    base = dict(max_creatures=10, initial_creatures=5, reproduce_threshold=100.0)

    def couple(**extra) -> tuple[World, int, int, int]:
        cfg = Config(**(base | extra))
        w = World(cfg, seed=2)
        slots = np.flatnonzero(w.alive)
        i, j = int(slots[0]), int(slots[1])
        w.energy[i] = 200.0
        w.energy[j] = 90.0
        w.pos[j] = w.pos[i] + np.float32([1.0, 0.0])
        for k in slots[2:]:
            w.pos[k] = w.pos[i] + np.float32([100.0, 100.0])
        return w, i, j, 0

    # identical parents + zero mutation std: child genes equal one parent exactly
    w, i, j, _ = couple(mutation_rate=1.0, mutation_std=0.0, trait_mutation_std=0.0)
    w.genome[j] = w.genome[i]
    step.phase_rebuild_counts(w)
    step.phase_reproduce(w)
    child = int(np.flatnonzero(w.alive & (w.parent_id == w.creature_id[i]))[0])
    assert np.array_equal(w.genome[child], w.genome[i])

    # full mutation on every gene: nothing survives unchanged from either parent
    w, i, j, _ = couple(mutation_rate=1.0, mutation_std=0.05, trait_mutation_std=0.1)
    step.phase_rebuild_counts(w)
    step.phase_reproduce(w)
    child = int(np.flatnonzero(w.alive & (w.parent_id == w.creature_id[i]))[0])
    p = Config(**base).brain_params
    not_i = w.genome[child, :p] != w.genome[i, :p]
    not_j = w.genome[child, :p] != w.genome[j, :p]
    assert np.all(not_i & not_j)
    assert 0.0 <= w.genome[child, p + 3] <= 1.0  # diet stays a valid fraction


def test_trait_clamps_stay_valid() -> None:
    cfg = Config(
        max_creatures=50,
        initial_creatures=10,
        initial_energy=1000.0,  # energy stays far above threshold: many generations
        reproduce_threshold=40.0,
        mutation_rate=1.0,
        mutation_std=0.5,
        trait_mutation_std=5.0,  # violent: exp(+-5) swings force the clamps
        mate_range=1e6,  # everyone can reach everyone: fill the capacity
    )
    w = World(cfg, seed=3)
    for _ in range(30):
        step.phase_rebuild_counts(w)
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
        mate_range=1e6,
    )
    w = World(cfg, seed=4)
    total0 = float(w.food.sum(dtype=np.float64)) + float(w.energy.sum(dtype=np.float64))
    for _ in range(100):
        advance(w)
    assert w.births > 0
    total = float(w.food.sum(dtype=np.float64)) + float(w.energy.sum(dtype=np.float64))
    assert np.isclose(total0, total, rtol=1e-4)


def test_determinism_with_reproduction() -> None:
    cfg = Config(
        max_creatures=150,
        initial_creatures=50,
        reproduce_threshold=40.0,
        mate_range=1e6,
    )

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
    cfg = Config(
        max_creatures=20,
        initial_creatures=20,
        reproduce_threshold=1.0,
        mate_range=1e6,
    )
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
        senescence_rate=0.0,  # isolate the quadratic move cost from wear
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
