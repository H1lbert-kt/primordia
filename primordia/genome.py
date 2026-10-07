"""Genome variation: mutation and asexual reproduction.

There is no fitness anywhere: a living creature whose energy reaches
``Config.reproduce_threshold`` splits in two. The child is a mutated copy of
the parent; the parent keeps the remainder of the energy, so no energy is
created (``child = e * child_fraction; parent = e - child``).

Mutation policy (per gene, probability ``Config.mutation_rate``):
    brain genes [0, brain_params)   additive   + N(0, mutation_std)
    max_speed, size, vision_range   multiplicative  * exp(N(0, trait_std))
    diet [brain_params+3]           additive   + N(0, trait_std), clamped
Diet is additive because it starts at 0 (herbivore) and a multiplicative
mutation could never make it leave zero. After mutation, traits are clamped
to physically valid ranges (``max_dim`` caps speed and vision: on a torus,
sensing or moving farther than the world itself is redundant, and the cap
bounds ray sampling cost).

Determinism: the serial kernel draws from the Numba RNG, which is a global
stream seeded once at ``World`` construction (``seed_numba_rng``). Because the
stream is process-global, tests that compare two worlds with active
reproduction must advance them sequentially, never interleaved. Draws happen
in a fixed order (ascending slots; brain genes before traits) and only inside
this serial kernel — never inside ``prange``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Numerical floor keeping positive traits strictly positive; not a balance
# parameter (mutation magnitudes never approach it with valid Config values).
TRAIT_EPS = 1e-4


@njit(cache=True)
def seed_numba_rng(seed: int) -> None:
    """Seed the Numba RNG stream (separate from ``World.rng``)."""
    np.random.seed(seed & 0xFFFFFFFF)


@njit(cache=True)
def _reproduce(
    pos: np.ndarray,
    vel: np.ndarray,
    age: np.ndarray,
    angle: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    species_id: np.ndarray,
    parent_id: np.ndarray,
    free_list: np.ndarray,
    free_count: np.int32,
    threshold: np.float32,
    child_fraction: np.float32,
    mutation_rate: np.float32,
    mutation_std: np.float32,
    trait_std: np.float32,
    traits_off: np.int32,
    max_dim: np.float32,
) -> tuple[int, int]:
    """One serial pass: split every eligible parent that has a free slot.

    Returns ``(births, free_count)``. Slots are popped LIFO, matching the
    free-list convention of ``World.spawn``. The scan stops early when the
    population reaches capacity.
    """
    births = 0
    n_genes = genome.shape[1]
    for i in range(alive.size):
        if not alive[i] or energy[i] < threshold:
            continue
        if free_count <= 0:
            break
        free_count = free_count - 1
        child = free_list[free_count]

        # Energy split: child first, parent keeps the remainder (exact).
        child_e = energy[i] * child_fraction
        energy[child] = child_e
        energy[i] = energy[i] - child_e

        # Clone, then mutate only the child.
        for g in range(n_genes):
            genome[child, g] = genome[i, g]

        for g in range(traits_off):
            if np.random.random() < mutation_rate:
                genome[child, g] = genome[child, g] + np.random.normal(
                    0.0, mutation_std
                )

        t = traits_off
        # max_speed
        if np.random.random() < mutation_rate:
            v = genome[child, t] * np.exp(np.random.normal(0.0, trait_std))
            if not (v > TRAIT_EPS):
                v = TRAIT_EPS
            if v > max_dim:
                v = max_dim
            genome[child, t] = v
        # size
        if np.random.random() < mutation_rate:
            v = genome[child, t + 1] * np.exp(np.random.normal(0.0, trait_std))
            if not (v > TRAIT_EPS):
                v = TRAIT_EPS
            genome[child, t + 1] = v
        # vision_range
        if np.random.random() < mutation_rate:
            v = genome[child, t + 2] * np.exp(np.random.normal(0.0, trait_std))
            if not (v > TRAIT_EPS):
                v = TRAIT_EPS
            if v > max_dim:
                v = max_dim
            genome[child, t + 2] = v
        # diet (additive: must be able to leave 0)
        if np.random.random() < mutation_rate:
            d = genome[child, t + 3] + np.random.normal(0.0, trait_std)
            if d < 0.0:
                d = 0.0
            if d > 1.0:
                d = 1.0
            genome[child, t + 3] = d

        # Child state: born in place, parent tracked for the lineage.
        pos[child, 0] = pos[i, 0]
        pos[child, 1] = pos[i, 1]
        vel[child, 0] = 0.0
        vel[child, 1] = 0.0
        angle[child] = angle[i]
        age[child] = 0
        species_id[child] = species_id[i]
        parent_id[child] = i
        alive[child] = True
        births = births + 1

    return births, free_count
