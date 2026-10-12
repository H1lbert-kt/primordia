"""Fixed-capacity structure-of-arrays world state.

A creature is an index into parallel arrays, never an object. Dead slots are
recycled through a free-list, so arrays are allocated once at construction and
never resized. Array dtypes are part of the module contract:

    pos (N,2) float32, vel (N,2) float32, angle (N,) float32,
    energy (N,) float32, age (N,) int32, genome (N,G) float32,
    species_id (N,) int32, parent_id (N,) int32, creature_id (N,) int32,
    alive (N,) bool, free_list (N,) int32, inventory (N,) float32
        (carried wood, stage 17),
    food (H,W) float32,
    food_field (H,W) float32 (local capacity K, stage 9 patches),
    terrain (H,W) float32 (fertility in [0,1], exact 0.0 = water, stage 10),
    meat (H,W) float32 (carrion energy per cell, stage 11),
    wood (H,W) float32 (wood per cell, stage 17; wood_field (H,W) is its
    patch capacity K, same machinery as food_field),
    signal (H,W) float32 (scalar channel, stage 13; diffusion scratch
    signal_buf (H,W) float32 is the stage-14 Jacobi staging buffer),
    struct_pos (S,2) float32, struct_type (S,) uint8 (0=fire, 1=house),
    struct_fuel (S,) float32, struct_store (S,) float32 (unused for
        fires; house completeness u in [0, 1] since stage 22 — the
        shelter stamped on the ground is cover_reduction * u),
    struct_owner (S,) int32
        (builder's creature id), struct_alive (S,) bool,
    struct_free_list (S,) int32   # stage-17 structures, fixed capacity S
    struct_grid/struct_fire (H,W) int32, fire_heat/cover (H,W) float32
        (per-tick structure scratch, rebuilt by rebuild_struct_fields,
        never saved),
    cell_counts (H,W) int32,
    cell_diet (H,W) float32, cell_offsets (H*W+1,) int32,
    cell_tag_re/im (H,W) float32,  # per-cell (cos,sin) of tag_hue, stage 15
    hue_re/im (N,) float32,        # per-slot (cos,sin) of tag_hue, stage 15
    cell_slots (N,) int32, cell_cursor (H*W,) int32 scratch,
    genealogy (L,4) int32,   # birth log rows (tick, child_id, parent_id, partner_id)
    ray_unit (n_rays,2) float32,
    sensor_buf (N, input_dim) float32, hidden_buf (N, hidden) float32,
    hidden_prev (N, hidden) float32,   # Elman state: last tick's hidden (saved)
    dw (N, plastic_params) float32,    # stage-21 Oja delta: W_eff = genome + dw,
                                       # zeroed at birth, decays to 0, NOT inherited
                                       # (Baldwin — the genome stays immutable in life)
    e_last_plastic (N,) float32,       # energy at the previous tick, credit signal
                                       # for the stage-21 delta-E ablation (saved)
    actions (N, 8) float32   # scratch buffers, zeroed once, reused every tick

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
    tag_re: np.ndarray,
    tag_im: np.ndarray,
    hue_re: np.ndarray,
    hue_im: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    cursor: np.ndarray,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
) -> None:
    """Serial counting sort: per-cell counts, diet sums, tag vectors, CSR.

    Slots are visited in ascending order in both passes, so the float32
    diet and tag sums and the CSR layout are deterministic bit for bit.
    ``counts``, ``diet_sum`` and the tag grids must be zeroed by the
    caller; ``offsets`` is fully rewritten. ``hue_re``/``hue_im`` are the
    precomputed (cos, sin) of each slot's tag_hue (stage 15).
    """
    # Pass A: counts, diet sums and hue unit vectors per cell.
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
            tag_re[iy, ix] = tag_re[iy, ix] + hue_re[i]
            tag_im[iy, ix] = tag_im[iy, ix] + hue_im[i]

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


@njit(cache=True)
def _stamp_structures(
    struct_pos: np.ndarray,
    struct_type: np.ndarray,
    struct_store: np.ndarray,
    struct_alive: np.ndarray,
    struct_grid: np.ndarray,
    struct_fire: np.ndarray,
    fire_heat: np.ndarray,
    cover: np.ndarray,
    cell_w: np.float32,
    cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    width: np.float32,
    height: np.float32,
    fire_radius: np.float32,
    house_radius: np.float32,
) -> None:
    """Stamp alive structures into the per-tick grids (serial, stage 17).

    The caller zero-fills the four grids first. Ascending slot order makes
    overlapping fires deterministic: the *lowest* slot owns a heated cell
    (``struct_fire``), so the fire that pays for cooking on it is always
    the same one. A structure occupies its exact cell (``struct_grid``)
    plus, within its radius, heat/cover stamped at cell-center distance
    with a toroidal delta (``fire_radius = 0`` keeps only the exact cell).
    Houses stamp their completeness ``struct_store`` (0..1, stage 22)
    with a max-merge, so cover is the shelter share, never more than 1.
    """
    hw = width * np.float32(0.5)
    hh = height * np.float32(0.5)
    for s in range(struct_pos.shape[0]):
        if not struct_alive[s]:
            continue
        x = struct_pos[s, 0]
        y = struct_pos[s, 1]
        cx = int(x / cell_w)
        cy = int(y / cell_h)
        if cx < 0:
            cx = 0
        elif cx >= gw:
            cx = gw - 1
        if cy < 0:
            cy = 0
        elif cy >= gh:
            cy = gh - 1
        struct_grid[cy, cx] = s
        if struct_type[s] == 0:
            r = fire_radius
            r2 = r * r
            ix0 = int((x - r) / cell_w)
            ix1 = int((x + r) / cell_w)
            iy0 = int((y - r) / cell_h)
            iy1 = int((y + r) / cell_h)
            for iy in range(iy0, iy1 + 1):
                dy = (np.float32(iy) + np.float32(0.5)) * cell_h - y
                if dy > hh:
                    dy = dy - height
                elif dy < -hh:
                    dy = dy + height
                for ix in range(ix0, ix1 + 1):
                    dx = (np.float32(ix) + np.float32(0.5)) * cell_w - x
                    if dx > hw:
                        dx = dx - width
                    elif dx < -hw:
                        dx = dx + width
                    if dx * dx + dy * dy <= r2:
                        jy = iy % gh
                        jx = ix % gw
                        fire_heat[jy, jx] = 1.0
                        if struct_fire[jy, jx] < 0:
                            struct_fire[jy, jx] = s
        else:
            r = house_radius
            r2 = r * r
            ix0 = int((x - r) / cell_w)
            ix1 = int((x + r) / cell_w)
            iy0 = int((y - r) / cell_h)
            iy1 = int((y + r) / cell_h)
            for iy in range(iy0, iy1 + 1):
                dy = (np.float32(iy) + np.float32(0.5)) * cell_h - y
                if dy > hh:
                    dy = dy - height
                elif dy < -hh:
                    dy = dy + height
                for ix in range(ix0, ix1 + 1):
                    dx = (np.float32(ix) + np.float32(0.5)) * cell_w - x
                    if dx > hw:
                        dx = dx - width
                    elif dx < -hw:
                        dx = dx + width
                    if dx * dx + dy * dy <= r2:
                        jy = iy % gh
                        jx = ix % gw
                        # stage 22: shelter scales with completeness u
                        # (struct_store); max-merge so overlapping houses
                        # never erase each other's cover
                        v = struct_store[s]
                        if v > cover[jy, jx]:
                            cover[jy, jx] = v


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
        # Carried wood (stage 17): clamped to Config.max_inventory by the
        # gather kernel; spent on construction and fire refuelling.
        self.inventory = np.zeros(n, dtype=np.float32)

        # Elman recurrent state: persists across ticks (save/load world state)
        self.hidden_prev = np.zeros((n, config.hidden_size), dtype=np.float32)
        # Scratch buffers reused every tick (never reallocated inside a tick).
        self.sensor_buf = np.zeros((n, config.sensor_input_dim), dtype=np.float32)
        self.hidden_buf = np.zeros((n, config.hidden_size), dtype=np.float32)
        self.actions = np.zeros((n, config.N_OUTPUTS), dtype=np.float32)
        # Stage-21 plasticity: the Oja delta lives in its own scratch so the
        # genome is never written during a life (Baldwin, not Lamarck).
        self.dw = np.zeros((n, config.plastic_params), dtype=np.float32)
        # Energy at the previous tick for the delta-E credit ablation; set
        # to the birth energy, updated every tick by phase_plasticity.
        self.e_last_plastic = np.zeros(n, dtype=np.float32)

        gh, gw = config.grid_shape
        self.cell_w = np.float32(config.width / gw)
        self.cell_h = np.float32(config.height / gh)
        self.food = np.full((gh, gw), config.initial_food, dtype=np.float32)
        self.meat = np.zeros((gh, gw), dtype=np.float32)
        # Wood field (stage 17): second ground resource; starts empty by
        # default and is refilled by phase_grow_wood toward wood_field.
        self.wood = np.full((gh, gw), config.initial_wood, dtype=np.float32)
        # Signal field (stage 13): scalar broadcast channel, zero at spawn.
        # Emission deposits here; decay_signal bleeds it; the tick never
        # reads it back into energy (it is information, not food).
        self.signal = np.zeros((gh, gw), dtype=np.float32)
        # Scratch for the stage-14 Jacobi diffusion pass: written by the
        # kernel, copied back over signal right after (copy, not swap —
        # references held to self.signal stay valid). Never reallocated.
        self.signal_buf = np.zeros((gh, gw), dtype=np.float32)
        self.cell_counts = np.zeros((gh, gw), dtype=np.int32)
        self.cell_diet = np.zeros((gh, gw), dtype=np.float32)
        # Per-cell hue unit vectors (stage 15): (cos, sin) of tag_hue summed
        # over living slots, rebuilt with the counts; scratch, never saved.
        self.cell_tag_re = np.zeros((gh, gw), dtype=np.float32)
        self.cell_tag_im = np.zeros((gh, gw), dtype=np.float32)
        # Per-slot (cos, sin) of tag_hue, refreshed by rebuild_counts so
        # perceive reads them instead of drawing trig per creature.
        self.hue_re = np.zeros(n, dtype=np.float32)
        self.hue_im = np.zeros(n, dtype=np.float32)
        n_cells = gh * gw
        self.cell_offsets = np.zeros(n_cells + 1, dtype=np.int32)
        self.cell_slots = np.zeros(n, dtype=np.int32)
        self.cell_cursor = np.zeros(n_cells, dtype=np.int32)

        # Structures (stage 17): fixed-capacity SoA with its own free-list,
        # same recycling rule as creatures. One structure per world cell.
        s = config.max_structures
        self.struct_pos = np.zeros((s, 2), dtype=np.float32)
        self.struct_type = np.zeros(s, dtype=np.uint8)
        self.struct_fuel = np.zeros(s, dtype=np.float32)
        # Deposit slot: written at build time (starts 0), but no rule reads
        # it until the territory stage — kept in the save so state is whole.
        self.struct_store = np.zeros(s, dtype=np.float32)
        self.struct_owner = np.full(s, -1, dtype=np.int32)
        self.struct_alive = np.zeros(s, dtype=bool)
        self.struct_free_list = np.arange(s, dtype=np.int32)
        self.struct_free_count = s
        # Per-tick structure scratch (rebuilt every advance, never saved):
        # struct_grid maps exact-cell occupancy (build slots), struct_fire
        # maps each heated cell to the fire that pays for cooking on it,
        # fire_heat/cover are the [0,1] cook/shelter fields (stage 17).
        self.struct_grid = np.full((gh, gw), -1, dtype=np.int32)
        self.struct_fire = np.full((gh, gw), -1, dtype=np.int32)
        self.fire_heat = np.zeros((gh, gw), dtype=np.float32)
        self.cover = np.zeros((gh, gw), dtype=np.float32)

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
        # Total energy ever extracted from carrion (stage 11 gate metric).
        self.meat_eaten = 0.0
        # Total signal ever deposited (stage 13 gate metric); float64 so
        # long runs do not lose counts, same pattern as meat_eaten. Saved
        # as a scalar in io.py.
        self.signal_emitted = 0.0
        # Total energy ever gifted to neighbours (stage 16 gate metric);
        # float64, saved as a scalar in io.py, same pattern.
        self.energy_donated = 0.0
        # Total wood ever harvested into inventories (stage 17 gate metric);
        # float64, saved as a scalar in io.py, same pattern.
        self.wood_gathered = 0.0
        # Total extra energy ever granted by cooking, i.e. the fuel burned
        # to pay for it (stage 17 conservation metric): float64, saved.
        self.energy_cooked = 0.0
        # Cumulative bites by the *attacker's* diet class (stage 12.5):
        # [herbivore (< HERB_MAX), omnivore, predator (> PRED_MIN)] — the
        # measurement buckets of ecology.py, int64 counts of successful
        # takes. Pure diagnostic: the tick never reads it, and it is
        # deliberately not saved (io.py): a resumed run restarts the
        # stats baseline at 0, same as the recorder does.
        self.bite_by_diet = np.zeros(3, dtype=np.int64)
        # Per-tick donation diagnostics (stage 16): sum of the circular
        # hue alignment over gift events plus the event count, reset by
        # the donate phase each tick so stats can read the last tick's
        # values. Diagnostic only, not saved (io.py), same contract as
        # bite_by_diet.
        self.donate_align_sum = 0.0
        self.donate_pairs = 0
        # Per-tick gift map for the renderer (stage 16): donor slot ->
        # recipient slot, -1 otherwise, rewritten every tick by the
        # donate phase (always cleared, even when the rate is 0). The
        # simulation never reads it; only the overlay draws transfer
        # lines from it, so recording stays bit-identical to headless.
        # Diagnostic scratch, not saved (io.py), like bite_by_diet.
        self.donate_to = np.full(config.max_creatures, -1, dtype=np.int32)

        # Genealogy (stage 7): birth-id allocator plus the pre-allocated
        # birth log; _reproduce writes rows through the cursor in-place.
        self.next_id = 0
        self.genealogy = np.zeros((config.genealogy_capacity, 4), dtype=np.int32)
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
        # Wood patch capacity (stage 17), drawn last: every earlier field
        # keeps its historic stream position for a seed; amplitude 0 draws
        # nothing (uniform K), same rule as food patches and terrain.
        self.wood_field = self._make_wood_field()

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

    def _make_wood_field(self) -> np.ndarray:
        """Wood patch capacity K in [1-amp, 1] * wood_capacity (stage 17).

        Same value-noise machinery as ``_make_food_field`` (drawn after
        terrain so earlier stream positions never move); ``amplitude = 0``
        returns the uniform field without drawing RNG.
        """
        cfg = self.config
        gh, gw = self.food.shape
        amp = cfg.wood_patch_amplitude
        if amp <= 0.0:
            return np.full((gh, gw), cfg.wood_capacity, dtype=np.float32)
        noise = self._value_noise(cfg.wood_patch_scale)
        return (cfg.wood_capacity * (1.0 - amp + amp * noise)).astype(np.float32)

    @property
    def alive_count(self) -> int:
        return int(np.count_nonzero(self.alive))

    def rebuild_counts(self) -> None:
        """Rebuild the per-cell grids: counts, diet/tag sums, CSR lists."""
        self.cell_counts.fill(0)
        self.cell_diet.fill(0.0)
        self.cell_tag_re.fill(0.0)
        self.cell_tag_im.fill(0.0)
        # vectorized trig once per tick (cheaper than per-cell cos/sin)
        hue = self.genome[:, self.config.brain_params + 4]
        np.cos(np.float32(2.0 * np.pi) * hue, out=self.hue_re)
        np.sin(np.float32(2.0 * np.pi) * hue, out=self.hue_im)
        gh, gw = self.cell_counts.shape
        _rebuild_grid(
            self.pos,
            self.alive,
            self.genome,
            np.int32(self.config.brain_params),
            self.cell_counts,
            self.cell_diet,
            self.cell_tag_re,
            self.cell_tag_im,
            self.hue_re,
            self.hue_im,
            self.cell_offsets,
            self.cell_slots,
            self.cell_cursor,
            np.float32(1.0 / self.cell_w),
            np.float32(1.0 / self.cell_h),
            np.int32(gw),
            np.int32(gh),
        )

    def rebuild_struct_fields(self) -> None:
        """Zero and re-stamp the stage-17 structure grids (phase 15 + load).

        ``struct_grid`` (exact-cell occupancy), ``struct_fire`` (which fire
        heats/pays for each cell), ``fire_heat`` and ``cover`` are scratch:
        rebuilt from the structure arrays, never saved. Numpy fills the
        grids (C-speed), the kernel only stamps.
        """
        self.struct_grid.fill(-1)
        self.struct_fire.fill(-1)
        self.fire_heat.fill(0.0)
        self.cover.fill(0.0)
        _stamp_structures(
            self.struct_pos,
            self.struct_type,
            self.struct_store,
            self.struct_alive,
            self.struct_grid,
            self.struct_fire,
            self.fire_heat,
            self.cover,
            self.cell_w,
            self.cell_h,
            np.int32(self.food.shape[1]),
            np.int32(self.food.shape[0]),
            np.float32(self.config.width),
            np.float32(self.config.height),
            np.float32(self.config.fire_radius),
            np.float32(self.config.house_radius),
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
        self.dw[slots] = 0.0  # no learned delta yet (stage 21, Baldwin)
        self.e_last_plastic[slots] = cfg.initial_energy

        params = cfg.brain_params
        self.genome[slots, :params] = self.rng.normal(
            0.0, cfg.weight_init_std, (count, params)
        )
        traits = params  # body traits sit right after the brain parameters
        self.genome[slots, traits] = cfg.max_speed
        self.genome[slots, traits + 1] = 1.0  # size (contact, stage 5)
        self.genome[slots, traits + 2] = cfg.vision_range
        self.genome[slots, traits + 3] = 0.0  # diet: 0 = herbivore (stage 5)
        self.genome[slots, traits + 4] = self.rng.uniform(
            0.0, 1.0, count
        )  # tag_hue: neutral lineage colour (stage 15)
        self.genome[slots, traits + 5] = 0.0  # plasticity eta (stage 21): born
        # unlearned; only mutation can turn learning on, and it always pays
        # plasticity_cost — selection decides whether the gene is worth it
        if cfg.build_house_bias != 0.0:
            # stage 22: a spawn-time prior against attempting a house; the
            # logit sits at genome column b2[7] (build_house output). No
            # RNG is drawn, so worlds with bias 0 keep their old stream.
            b2_off = (
                cfg.hidden_size * cfg.sensor_input_dim
                + cfg.hidden_size
                + cfg.N_OUTPUTS * cfg.hidden_size
            )
            self.genome[slots, b2_off + 7] += np.float32(cfg.build_house_bias)

        self.species_id[slots] = 0
        self.parent_id[slots] = -1
        self.creature_id[slots] = np.arange(
            self.next_id, self.next_id + count, dtype=np.int32
        )
        self.next_id += count
        self.alive[slots] = True
        return slots
