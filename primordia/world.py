"""Fixed-capacity structure-of-arrays world state.

A creature is an index into parallel arrays, never an object. Dead slots are
recycled through a free-list, so arrays are allocated once at construction and
never resized. Array dtypes are part of the module contract:

    pos (N,2) float32, vel (N,2) float32, angle (N,) float32,
    energy (N,) float32, age (N,) int32, genome (N,G) float32,
    species_id (N,) int32, parent_id (N,) int32, creature_id (N,) int32,
    alive (N,) bool, free_list (N,) int32, food (H,W) float32,
    food_field (H,W) float32 (local capacity K, stage 9 patches),
    terrain (H,W) float32 (fertility in [0,1], exact 0.0 = water, stage 10),
    cell_counts (H,W) int32,
    cell_diet (H,W) float32, cell_offsets (H*W+1,) int32,
    cell_slots (N,) int32, cell_cursor (H*W,) int32 scratch,
    genealogy (L,3) int32,   # birth log rows (tick, child_id, parent_id)
    ray_unit (n_rays,2) float32,
    sensor_buf (N, input_dim) float32, hidden_buf (N, hidden) float32,
    hidden_prev (N, hidden) float32,   # Elman state: last tick's hidden (saved)
    actions (N, 3) float32   # scratch buffers, zeroed once, reused every tick

``creature_id`` is a monotonic birth id (``next_id`` counts up); dead slots
keep their stale id, so every lookup must also filter ``alive``.
``parent_id`` stores the parent's **creature id** (not its slot: slots are
recycled and would corrupt the lineage), ``-1`` for spawn. ``genealogy`` is
the append-only birth log (capacity ``Config.genealogy_capacity``); when it
fills, further births only bump ``genealogy_overflow``.

``cell_offsets``/``cell_slots`` are a CSR listing of living slots per cell
(counting sort rebuilt every tick; stage-5 ``bite`` scans neighbours through
it). ``cell_cursor`` is its pre-allocated write-head scratch.

Genome layout (see brain.py): brain parameters then body traits; body traits
are filled from Config at spawn and vary only through mutation (stage 3).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .config import Config
from .genome import seed_numba_rng


@njit(cache=True)
def _rebuild_grid(
    pos: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    traits_off: np.int32,
    counts: np.ndarray,
    diet_sum: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    cursor: np.ndarray,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
) -> None:
    """Serial counting sort: per-cell counts, diet sums and CSR slot lists.

    Slots are visited in ascending order in both passes, so the float32 diet
    sums and the CSR layout are deterministic bit for bit. ``counts`` and
    ``diet_sum`` must be zeroed by the caller; ``offsets`` is fully rewritten.
    """
    # Pass A: counts and diet sums per cell.
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
            diet_sum[iy, ix] = diet_sum[iy, ix] + genome[i, traits_off + 3]

    # Prefix sums over cells in row-major order. The reshape keeps a flat
    # view (counts is C-contiguous), avoiding a divide+modulo per cell.
    flat = counts.reshape(-1)
    offsets[0] = 0
    for k in range(gh * gw):
        offsets[k + 1] = offsets[k] + flat[k]

    # Pass B: scatter living slots into their cell ranges.
    for k in range(gh * gw):
        cursor[k] = offsets[k]
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
            k = iy * gw + ix
            p = cursor[k]
            slots[p] = i
            cursor[k] = p + 1


class World:
    """Pre-allocated simulation state: arrays, grids, RNG and tick counter."""

    def __init__(self, config: Config, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        seed_numba_rng(seed)  # second stream, used only by genome._reproduce
        n = config.max_creatures

        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.angle = np.zeros(n, dtype=np.float32)
        self.energy = np.zeros(n, dtype=np.float32)
        self.age = np.zeros(n, dtype=np.int32)
        self.genome = np.zeros((n, config.genome_size), dtype=np.float32)
        self.species_id = np.zeros(n, dtype=np.int32)
        self.parent_id = np.full(n, -1, dtype=np.int32)
        self.creature_id = np.full(n, -1, dtype=np.int32)
        self.alive = np.zeros(n, dtype=bool)

        # Free-list stack: pop from the end (free_count-1), push at free_count.
        self.free_list = np.arange(n, dtype=np.int32)
        self.free_count = n

        # Elman recurrent state: persists across ticks (save/load world state)
        self.hidden_prev = np.zeros((n, config.hidden_size), dtype=np.float32)
        # Scratch buffers reused every tick (never reallocated inside a tick).
        self.sensor_buf = np.zeros((n, config.sensor_input_dim), dtype=np.float32)
        self.hidden_buf = np.zeros((n, config.hidden_size), dtype=np.float32)
        self.actions = np.zeros((n, config.N_OUTPUTS), dtype=np.float32)

        gh, gw = config.grid_shape
        self.cell_w = np.float32(config.width / gw)
        self.cell_h = np.float32(config.height / gh)
        self.food = np.full((gh, gw), config.initial_food, dtype=np.float32)
        self.cell_counts = np.zeros((gh, gw), dtype=np.int32)
        self.cell_diet = np.zeros((gh, gw), dtype=np.float32)
        n_cells = gh * gw
        self.cell_offsets = np.zeros(n_cells + 1, dtype=np.int32)
        self.cell_slots = np.zeros(n, dtype=np.int32)
        self.cell_cursor = np.zeros(n_cells, dtype=np.int32)

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
        # Cumulative counters for balance reporting (stats.py, stage 6, will
        # consume these; Python ints, never allocated inside a tick).
        self.births = 0
        self.deaths_famine = 0
        self.deaths_age = 0
        self.deaths_predation = 0

        # Genealogy (stage 7): birth-id allocator plus the pre-allocated
        # birth log; _reproduce writes rows through the cursor in-place.
        self.next_id = 0
        self.genealogy = np.zeros((config.genealogy_capacity, 3), dtype=np.int32)
        self.genealogy_used = 0
        self.genealogy_overflow = 0
        self.spawn(config.initial_creatures)
        # Local carrying capacity per cell (stage 9 patches). Drawn after
        # spawn so initial positions match the stage-8 stream for a seed.
        self.food_field = self._make_food_field()
        # Terrain (stage 10) uses the same noise machinery from the same
        # NumPy stream, drawn after food_field so spawn and patch history
        # stay unchanged; amplitude 0 draws nothing.
        self.terrain = self._make_terrain()

    def _value_noise(self, scale: float) -> np.ndarray:
        """Bilinear value noise in [0, 1] on a lattice spaced ``scale`` world
        units apart (wraps toroidally, one ``rng.uniform`` draw).

        Shared by ``food_field`` (patches) and ``terrain`` (stage 10); the
        draw order and shapes are what make a seed's fields reproducible.
        """
        gh, gw = self.food.shape
        step = max(1, int(round(scale / float(self.config.cell_size))))
        lh = max(1, (gh + step - 1) // step)
        lw = max(1, (gw + step - 1) // step)
        lattice = self.rng.uniform(0.0, 1.0, (lh, lw)).astype(np.float32)
        ys = np.arange(gh, dtype=np.float32) / step
        xs = np.arange(gw, dtype=np.float32) / step
        y0 = np.floor(ys).astype(np.int64)
        x0 = np.floor(xs).astype(np.int64)
        fy = (ys - y0).astype(np.float32)[:, None]
        fx = (xs - x0).astype(np.float32)[None, :]
        y0 %= lh
        x0 %= lw
        y1 = (y0 + 1) % lh
        x1 = (x0 + 1) % lw
        v00 = lattice[np.ix_(y0, x0)]
        v01 = lattice[np.ix_(y0, x1)]
        v10 = lattice[np.ix_(y1, x0)]
        v11 = lattice[np.ix_(y1, x1)]
        return (
            v00 * (1.0 - fy) * (1.0 - fx)
            + v01 * (1.0 - fy) * fx
            + v10 * fy * (1.0 - fx)
            + v11 * fy * fx
        )

    def _make_food_field(self) -> np.ndarray:
        """Value-noise field K: per-cell capacity in [1-amp, 1] * capacity.

        Growth refills toward K instead of the global capacity, so rich and
        poor patches persist. ``amplitude = 0`` returns the uniform field
        (bit-identical to the stage-8 behavior). Uses ``World.rng`` (the
        NumPy stream), never the Numba stream; the result is saved and
        restored by ``io`` like any other state array.
        """
        cfg = self.config
        gh, gw = self.food.shape
        amp = cfg.food_patch_amplitude
        if amp <= 0.0:
            return np.full((gh, gw), cfg.food_capacity, dtype=np.float32)
        noise = self._value_noise(cfg.food_patch_scale)
        return (cfg.food_capacity * (1.0 - amp + amp * noise)).astype(np.float32)

    def _make_terrain(self) -> np.ndarray:
        """Terrain fertility with amplitude as the land share (stage 10).

        Raw value noise ``n``; cells with ``n < 1 - amplitude`` become water
        (stored as **exact 0.0** so kernels use an ``== 0`` compare), land is
        rescaled to (0, 1]. ``amplitude = 0`` returns flat fertility 1.0
        without drawing RNG — the pre-stage-10 world (no water, uniform
        growth target).
        """
        cfg = self.config
        gh, gw = self.food.shape
        amp = cfg.terrain_amplitude
        if amp <= 0.0:
            return np.full((gh, gw), 1.0, dtype=np.float32)
        noise = self._value_noise(cfg.terrain_scale)
        threshold = 1.0 - amp
        terrain = (noise - threshold) / amp
        terrain[noise < threshold] = 0.0  # water: exact zero
        np.clip(terrain, 0.0, 1.0, out=terrain)
        return terrain.astype(np.float32)

    @property
    def alive_count(self) -> int:
        return int(np.count_nonzero(self.alive))

    def rebuild_counts(self) -> None:
        """Rebuild the per-cell grids: creature counts, diet sums, CSR lists."""
        self.cell_counts.fill(0)
        self.cell_diet.fill(0.0)
        gh, gw = self.cell_counts.shape
        _rebuild_grid(
            self.pos,
            self.alive,
            self.genome,
            np.int32(self.config.brain_params),
            self.cell_counts,
            self.cell_diet,
            self.cell_offsets,
            self.cell_slots,
            self.cell_cursor,
            np.float32(1.0 / self.cell_w),
            np.float32(1.0 / self.cell_h),
            np.int32(gw),
            np.int32(gh),
        )

    def spawn(self, count: int) -> np.ndarray:
        """Pop ``count`` slots from the free-list and initialize them.

        Vectorized: creates all ``count`` newborns in one pass. Used for the
        initial population (and manual resets); tick-born newborns are
        initialized inside ``genome._reproduce`` instead. Returns the slots.
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
        self.hidden_prev[slots] = 0.0  # newborns start with an empty memory

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
        self.creature_id[slots] = np.arange(
            self.next_id, self.next_id + count, dtype=np.int32
        )
        self.next_id += count
        self.alive[slots] = True
        return slots
