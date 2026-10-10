"""Genome variation: mutation and sexual reproduction.

There is no fitness anywhere: a living creature whose energy reaches
``Config.reproduce_threshold`` looks for a partner alive within
``Config.mate_range`` (nearest first, exact toroidal distance) and, if one
exists, they split a child. Each parent pays half of the child's energy
(``child_e = energy[i] * child_fraction``; each pays ``child_e / 2``, so no
energy is created and a creature that cannot find a partner simply does not
reproduce — mobility and aggregation get selected without being
programmed). The child's genome is a per-gene uniform crossover of the two
parents followed by mutation.

Mutation policy (per gene, probability ``Config.mutation_rate``):
    brain genes [0, brain_params)   additive   + N(0, mutation_std)
    max_speed, size, vision_range   multiplicative  * exp(N(0, trait_std))
    diet [brain_params+3]           additive   + N(0, trait_std), clamped
Diet is additive because it starts at 0 (herbivore) and a multiplicative
mutation could never make it leave zero. After mutation, traits are clamped
to physically valid ranges (``max_dim`` caps speed and vision: on a torus,
sensing or moving farther than the world itself is redundant, and the cap
bounds ray sampling cost; ``max_size`` caps contact reach so neighbour scans
stay bounded).

Determinism: the serial kernel draws from the Numba RNG, which is a global
stream seeded once at ``World`` construction (``seed_numba_rng``). Because the
stream is process-global, tests that compare two worlds with active
reproduction must advance them sequentially, never interleaved. Draws happen
in a fixed order (ascending initiator slots; per gene: one crossover coin
then the mutation coin; genes ascending) and only inside this serial kernel —
never inside ``prange``. The stream can be snapshotted and restored for
save/load (``get_numba_rng_state``/``set_numba_rng_state``).
"""

from __future__ import annotations

import numpy as np
from numba import _helperlib, njit

# Numerical floor keeping positive traits strictly positive; not a balance
# parameter (mutation magnitudes never approach it with valid Config values).
TRAIT_EPS = 1e-4


@njit(cache=True)
def seed_numba_rng(seed: int) -> None:
    """Seed the Numba RNG stream (separate from ``World.rng``)."""
    np.random.seed(seed & 0xFFFFFFFF)


def get_numba_rng_state() -> tuple[int, list[int]]:
    """Snapshot the process-global Numba RNG stream (for save/load).

    Uses ``numba._helperlib`` — a private API, but the only one exposing the
    stream; it is the same call the numba test suite uses. Verified against
    numba 0.68 (the pinned version in AGENTS.md); tests/test_io.py fails
    loudly if a future numba removes it.
    """
    index, key = _helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())
    return int(index), [int(v) for v in key]


def set_numba_rng_state(state: tuple[int, list[int]]) -> None:
    """Restore a stream captured by :func:`get_numba_rng_state`, bit-exact."""
    index, key = state
    _helperlib.rnd_set_state(
        _helperlib.rnd_get_np_state_ptr(), (int(index), [int(v) for v in key])
    )


@njit(cache=True)
def _reproduce(
    pos: np.ndarray,
    vel: np.ndarray,
    age: np.ndarray,
    angle: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    hidden_prev: np.ndarray,
    species_id: np.ndarray,
    parent_id: np.ndarray,
    creature_id: np.ndarray,
    genealogy: np.ndarray,
    free_list: np.ndarray,
    free_count: np.int32,
    cell_offsets: np.ndarray,
    cell_slots: np.ndarray,
    threshold: np.float32,
    child_fraction: np.float32,
    mutation_rate: np.float32,
    mutation_std: np.float32,
    trait_std: np.float32,
    traits_off: np.int32,
    max_dim: np.float32,
    max_size: np.float32,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    width: np.float32,
    height: np.float32,
    mate_range: np.float32,
    tick: np.int32,
    next_id: np.int32,
    genealogy_used: np.int32,
    genealogy_overflow: np.int32,
) -> tuple[int, int, int, int, int]:
    """One serial pass: every eligible initiator mates with its nearest partner.

    Returns ``(births, free_count, genealogy_used, genealogy_overflow,
    next_id)``. Slots are popped LIFO, matching the free-list convention of
    ``World.spawn``. The scan stops early when the population reaches
    capacity.

    Mate search: scan the CSR cells whose centers fall within ``mate_range``
    of the initiator (the grid was rebuilt at the start of the tick, so a
    mover may be listed in its old cell — the exact toroidal distance below
    is the final word on eligibility; the staleness is the same the bite
    phase already accepts). Candidate = alive, not the initiator, energy
    above its half of the child; nearest wins, ties go to the lower slot.
    No partner -> no birth (no energy is debited, the free list is untouched).

    Each child gets a monotonic ``creature_id`` and a ``parent_id`` holding
    the initiator's creature id (never its slot: slots are recycled and
    would corrupt the lineage; the partner's id is not logged — documented
    limitation); the birth is appended to the ``genealogy`` log until it is
    full (then only ``overflow`` grows). RNG draws are the crossover coin
    and the mutation draws, in ascending initiator/gene order.
    """
    births = 0
    n_genes = genome.shape[1]
    log_cap = genealogy.shape[0]
    reach_x = int(mate_range * inv_cell_w) + 1
    reach_y = int(mate_range * inv_cell_h) + 1
    if reach_x > gw // 2:
        reach_x = gw // 2
    if reach_y > gh // 2:
        reach_y = gh // 2
    hw = width * np.float32(0.5)
    hh = height * np.float32(0.5)
    range2 = mate_range * mate_range

    for i in range(alive.size):
        if not alive[i] or energy[i] < threshold:
            continue
        if free_count <= 0:
            break

        # Energy split: child_e total, each parent pays half (exact).
        child_e = energy[i] * child_fraction
        half = child_e * np.float32(0.5)
        # partner must stay strictly positive after paying
        if energy[i] <= half:
            continue

        # nearest eligible partner within mate_range (exact toroidal distance)
        px = pos[i, 0]
        py = pos[i, 1]
        ix0 = int(px * inv_cell_w)
        iy0 = int(py * inv_cell_h)
        if ix0 < 0:
            ix0 = 0
        elif ix0 >= gw:
            ix0 = gw - 1
        if iy0 < 0:
            iy0 = 0
        elif iy0 >= gh:
            iy0 = gh - 1
        best_j = -1
        best_d2 = np.float32(np.inf)
        for dy in range(-reach_y, reach_y + 1):
            iy = iy0 + dy
            while iy < 0:
                iy = iy + gh
            while iy >= gh:
                iy = iy - gh
            for dx in range(-reach_x, reach_x + 1):
                ix = ix0 + dx
                while ix < 0:
                    ix = ix + gw
                while ix >= gw:
                    ix = ix - gw
                k = iy * gw + ix
                for p in range(cell_offsets[k], cell_offsets[k + 1]):
                    j = cell_slots[p]
                    if j == i or not alive[j] or energy[j] <= half:
                        continue
                    ddx = px - pos[j, 0]
                    if ddx > hw:
                        ddx = ddx - width
                    elif ddx < -hw:
                        ddx = ddx + width
                    ddy = py - pos[j, 1]
                    if ddy > hh:
                        ddy = ddy - height
                    elif ddy < -hh:
                        ddy = ddy + height
                    d2 = ddx * ddx + ddy * ddy
                    if d2 <= range2 and (d2 < best_d2 or (d2 == best_d2 and j < best_j)):
                        best_d2 = d2
                        best_j = j
        if best_j < 0:
            continue

        j = best_j
        free_count = free_count - 1
        child = free_list[free_count]
        energy[i] = energy[i] - half
        energy[j] = energy[j] - half
        energy[child] = child_e

        # uniform crossover per gene, then mutate only the child
        for g in range(n_genes):
            if np.random.random() < 0.5:
                genome[child, g] = genome[i, g]
            else:
                genome[child, g] = genome[j, g]
        for h in range(hidden_prev.shape[1]):
            hidden_prev[child, h] = 0.0  # the child starts with no memory

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
            if v > max_size:
                v = max_size
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

        # Child state: born in place, initiator tracked for the lineage.
        pos[child, 0] = pos[i, 0]
        pos[child, 1] = pos[i, 1]
        vel[child, 0] = 0.0
        vel[child, 1] = 0.0
        angle[child] = angle[i]
        age[child] = 0
        species_id[child] = species_id[i]
        child_id = next_id
        next_id = next_id + 1
        creature_id[child] = child_id
        parent_id[child] = creature_id[i]
        if genealogy_used < log_cap:
            genealogy[genealogy_used, 0] = tick
            genealogy[genealogy_used, 1] = child_id
            genealogy[genealogy_used, 2] = creature_id[i]
            genealogy_used = genealogy_used + 1
        else:
            genealogy_overflow = genealogy_overflow + 1
        alive[child] = True
        births = births + 1

    return births, free_count, genealogy_used, genealogy_overflow, next_id
