"""Fixed-capacity structure-of-arrays world state.

A creature is an index into parallel arrays, never an object. Dead slots are
recycled through a free-list, so arrays are allocated once at construction and
never resized. Array dtypes are part of the module contract:

    pos (N,2) float32, vel (N,2) float32, angle (N,) float32,
    energy (N,) float32, age (N,) int32, genome (N,G) float32,
    species_id (N,) int32, parent_id (N,) int32, alive (N,) bool,
    free_list (N,) int32, food (H,W) float32
"""

from __future__ import annotations

import numpy as np

from .config import Config


class World:
    """Pre-allocated simulation state: arrays, food grid, RNG and tick counter."""

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

        gh, gw = config.grid_shape
        self.cell_w = np.float32(config.width / gw)
        self.cell_h = np.float32(config.height / gh)
        self.food = np.full((gh, gw), config.initial_food, dtype=np.float32)

        self.tick = 0
        self.spawn(config.initial_creatures)

    @property
    def alive_count(self) -> int:
        return int(np.count_nonzero(self.alive))

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
        self.genome[slots] = 0.0
        self.species_id[slots] = 0
        self.parent_id[slots] = -1
        self.alive[slots] = True
        return slots
