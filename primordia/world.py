"""Fixed-capacity structure-of-arrays world state.

A creature is an index into parallel arrays, never an object. Dead slots are
recycled through a free-list, so arrays are allocated once at construction and
never resized. Array dtypes are part of the module contract:

    pos (N,2) float32, vel (N,2) float32, angle (N,) float32,
    energy (N,) float32, age (N,) int32, genome (N,G) float32,
    species_id (N,) int32, parent_id (N,) int32, alive (N,) bool,
    free_list (N,) int32, food (H,W) float32, cell_counts (H,W) int32,
    ray_unit (n_rays,2) float32,
    sensor_buf (N, input_dim) float32, hidden_buf (N, hidden) float32,
    actions (N, 3) float32   # scratch buffers, zeroed once, reused every tick

Genome layout (see brain.py): brain parameters then body traits; body traits
are filled from Config at spawn and vary only through mutation (stage 3).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .config import Config


@njit(cache=True)
def _count_alive(
    pos: np.ndarray,
    alive: np.ndarray,
    counts: np.ndarray,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
) -> None:
    """One serial pass: counts[cy, cx] += 1 for every living creature."""
    for i in range(alive.size):
        if alive[i]:
            ix = int(pos[i, 0] * inv_cell_w)
            iy = int(pos[i, 1] * inv_cell_h)
            if ix < 0:
                ix = 0
            elif ix >= gw:
                ix = gw - 1
            if iy < 0:
                iy = 0
            elif iy >= gh:
                iy = gh - 1
            counts[iy, ix] = counts[iy, ix] + 1


class World:
    """Pre-allocated simulation state: arrays, grids, RNG and tick counter."""

    def __init__(self, config: Config, seed: int = 0) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        n = config.max_creatures

        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.angle = np.zeros(n, dtype=np.float32)
        self.energy = np.zeros(n, dtype=np.float32)
        self.age = np.zeros(n, dtype=np.int32)
        self.genome = np.zeros((n, config.genome_size), dtype=np.float32)
        self.species_id = np.zeros(n, dtype=np.int32)
        self.parent_id = np.full(n, -1, dtype=np.int32)
        self.alive = np.zeros(n, dtype=bool)

        # Free-list stack: pop from the end (free_count-1), push at free_count.
        self.free_list = np.arange(n, dtype=np.int32)
        self.free_count = n

        # Scratch buffers reused every tick (never reallocated inside a tick).
        self.sensor_buf = np.zeros((n, config.sensor_input_dim), dtype=np.float32)
        self.hidden_buf = np.zeros((n, config.hidden_size), dtype=np.float32)
        self.actions = np.zeros((n, config.N_OUTPUTS), dtype=np.float32)

        gh, gw = config.grid_shape
        self.cell_w = np.float32(config.width / gw)
        self.cell_h = np.float32(config.height / gh)
        self.food = np.full((gh, gw), config.initial_food, dtype=np.float32)
        self.cell_counts = np.zeros((gh, gw), dtype=np.int32)

        # Ray fan as unit vectors of the fovea-relative offsets; each creature
        # rotates this table by its angle (one sin/cos per creature, not per ray).
        if config.n_rays == 1:
            offsets = np.zeros(1, dtype=np.float64)
        else:
            offsets = np.linspace(
                -config.field_of_view / 2.0, config.field_of_view / 2.0, config.n_rays
            )
        self.ray_unit = np.stack((np.cos(offsets), np.sin(offsets)), axis=1).astype(
            np.float32
        )

        self.tick = 0
        self.spawn(config.initial_creatures)

    @property
    def alive_count(self) -> int:
        return int(np.count_nonzero(self.alive))

    def rebuild_counts(self) -> None:
        """Rebuild the creature-per-cell grid that the ray sensors read."""
        self.cell_counts.fill(0)
        gh, gw = self.cell_counts.shape
        _count_alive(
            self.pos,
            self.alive,
            self.cell_counts,
            np.float32(1.0 / self.cell_w),
            np.float32(1.0 / self.cell_h),
            np.int32(gw),
            np.int32(gh),
        )

    def spawn(self, count: int) -> np.ndarray:
        """Pop ``count`` slots from the free-list and initialize them.

        Vectorized on purpose: stage-3 reproduction will call this once per
        tick with all newborns of that tick. Returns the slot indices.
        """
        if count > self.free_count:
            raise ValueError(f"not enough free slots: {count} > {self.free_count}")
        start = self.free_count - count
        # Copy: the slice is a view into free_list and later deaths reuse it.
        slots = self.free_list[start : self.free_count].copy()
        self.free_count = start

        cfg = self.config
        self.pos[slots, 0] = self.rng.uniform(0.0, cfg.width, count)
        self.pos[slots, 1] = self.rng.uniform(0.0, cfg.height, count)
        self.vel[slots] = 0.0
        self.angle[slots] = self.rng.uniform(0.0, 2.0 * np.pi, count)
        self.energy[slots] = cfg.initial_energy
        self.age[slots] = 0

        params = cfg.brain_params
        self.genome[slots, :params] = self.rng.normal(
            0.0, cfg.weight_init_std, (count, params)
        )
        traits = params  # body traits sit right after the brain parameters
        self.genome[slots, traits] = cfg.max_speed
        self.genome[slots, traits + 1] = 1.0  # size (contact, stage 5)
        self.genome[slots, traits + 2] = cfg.vision_range
        self.genome[slots, traits + 3] = 0.0  # diet: 0 = herbivore (stage 5)

        self.species_id[slots] = 0
        self.parent_id[slots] = -1
        self.alive[slots] = True
        return slots
