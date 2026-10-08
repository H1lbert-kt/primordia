"""Per-tick metrics recorder: history for plots and balance reports (stage 6).

``StatsRecorder`` samples the world **read-only** from the CLI loop (never
inside ``advance``, so the tick and its benchmark stay untouched) and keeps
one row per recorded tick in pre-allocated int32/float32 columns:

    tick, alive, mean_energy, mean_age                      (state)
    births, deaths_famine, deaths_age, deaths_predation      (per-tick deltas
        of the World's cumulative counters)
    speed/size/vision/diet mean and std over living slots    (diversity)
    genome_dist   (sampled genetic diversity, see below)

``genome_dist`` is the RMS pairwise distance between sampled genomes,
``sqrt(mean_pairs(||a-b||^2) / G)``, over a genome normalized by the spawn
scales (brain genes by ``weight_init_std``; traits by ``max_speed``, ``1.0``,
``vision_range``, ``1.0``). A deterministic ``linspace`` sample of
``sample_size`` living slots is taken every ``sample_every`` ticks — no RNG,
so recordings are bit-reproducible — and the value is forward-filled on the
in-between ticks. It reads ``0`` when fewer than two creatures are alive at
a sample tick.

Saving writes a plain ``.npz`` (column arrays plus seed, recorder settings
and the Config as JSON) so ``plot.py`` can redraw without re-simulating.
No matplotlib here: the core must never import it (tests/test_render_guard).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .world import World

INT_COLUMNS = (
    "tick",
    "alive",
    "births",
    "deaths_famine",
    "deaths_age",
    "deaths_predation",
)
FLOAT_COLUMNS = (
    "mean_energy",
    "mean_age",
    "speed_mean",
    "speed_std",
    "size_mean",
    "size_std",
    "vision_mean",
    "vision_std",
    "diet_mean",
    "diet_std",
    "genome_dist",
)
COLUMNS = INT_COLUMNS + FLOAT_COLUMNS


@dataclass(frozen=True)
class StatsHistory:
    """Loaded stats file: trimmed column arrays plus recording metadata."""

    data: dict[str, np.ndarray]
    seed: int | None
    config: dict
    sample_every: int
    sample_size: int

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[name]

    def __len__(self) -> int:
        return len(self.data["tick"])

    @property
    def seed_label(self) -> str:
        return "?" if self.seed is None else str(self.seed)


class StatsRecorder:
    """Accumulates one metrics row per ``record(world)`` call.

    Buffers are pre-allocated and doubled on demand; every column shares the
    same capacity and length, so growth is a simple block copy. Recording
    never touches the world and draws no random numbers.
    """

    def __init__(self, capacity: int = 4096, sample_every: int = 10,
                 sample_size: int = 64) -> None:
        if capacity < 1 or sample_every < 1 or sample_size < 2:
            raise ValueError("capacity >= 1, sample_every >= 1, sample_size >= 2")
        self.capacity = capacity
        self.sample_every = sample_every
        self.sample_size = sample_size
        self._length = 0
        self._columns: dict[str, np.ndarray] = {
            name: np.zeros(capacity, dtype=np.int32 if name in INT_COLUMNS
                           else np.float32)
            for name in COLUMNS
        }
        self._prev_cum = np.zeros(4, dtype=np.int64)  # famine/age/predation/births
        self._last_dist = np.float32(0.0)
        self._config: dict | None = None
        self._inv_scale: np.ndarray | None = None

    @property
    def length(self) -> int:
        """Number of recorded ticks."""
        return self._length

    def to_dict(self) -> dict[str, np.ndarray]:
        """Trimmed views of every column (shared buffer; copies if edited)."""
        return {name: arr[: self._length] for name, arr in self._columns.items()}

    def _grow(self) -> None:
        new_cap = self.capacity * 2
        for name, arr in self._columns.items():
            grown = np.zeros(new_cap, dtype=arr.dtype)
            grown[: self.length] = arr[: self.length]
            self._columns[name] = grown
        self.capacity = new_cap

    def _genome_scales(self, world: World) -> np.ndarray:
        """Per-gene spawn scales so distances mix dimensionless genes."""
        cfg = world.config
        params = cfg.brain_params
        scale = np.full(cfg.genome_size, cfg.weight_init_std, dtype=np.float32)
        if cfg.weight_init_std <= 0.0:
            scale[:params] = 1.0
        scale[params] = cfg.max_speed
        scale[params + 1] = 1.0
        scale[params + 2] = cfg.vision_range
        scale[params + 3] = 1.0
        return scale

    def _genome_dist(self, world: World) -> np.float32:
        """RMS pairwise genome distance over a deterministic living sample.

        The Gram matrix is contracted with ``np.einsum`` (not ``@``): the
        BLAS path spawns an 8-thread barrier per micro-gemm whose spin-waiting
        threads then starve the following prange kernels of the tick
        (measured: a full 600-tick loop slowed ~7x with ``@``).
        """
        alive = np.flatnonzero(world.alive)
        if alive.size < 2:
            return np.float32(0.0)
        idx = np.linspace(0, alive.size - 1, min(alive.size, self.sample_size))
        sample = alive[idx.astype(np.int32)]
        x = world.genome[sample] * self._inv_scale
        sq = np.einsum("ij,ij->i", x, x)
        d2 = sq[:, None] + sq[None, :] - 2.0 * np.einsum("ik,jk->ij", x, x)
        iu = np.triu_indices(sample.size, k=1)
        mean_d2 = float(np.mean(d2[iu]))
        return np.float32(np.sqrt(max(mean_d2, 0.0) / x.shape[1]))

    def record(self, world: World) -> None:
        """Append one row of current metrics; reads the world, never writes."""
        if self._length >= self.capacity:
            self._grow()
        if self._config is None:
            self._config = asdict(world.config)
            self._inv_scale = self._genome_scales(world)

        cols = self._columns
        i = self._length
        cols["tick"][i] = world.tick
        alive = np.flatnonzero(world.alive)
        cols["alive"][i] = alive.size
        if alive.size:
            cols["mean_energy"][i] = np.float32(world.energy[alive].mean())
            cols["mean_age"][i] = np.float32(world.age[alive].mean())
            traits = world.genome[alive, world.config.brain_params:
                                  world.config.brain_params + 4]
            means = traits.mean(axis=0)
            stds = traits.std(axis=0)
        else:
            cols["mean_energy"][i] = 0.0
            cols["mean_age"][i] = 0.0
            means = stds = np.zeros(4, dtype=np.float32)
        for k, trait in enumerate(("speed", "size", "vision", "diet")):
            cols[f"{trait}_mean"][i] = np.float32(means[k])
            cols[f"{trait}_std"][i] = np.float32(stds[k])

        cum = (world.births, world.deaths_famine, world.deaths_age,
               world.deaths_predation)
        for k, name in enumerate(("births", "deaths_famine", "deaths_age",
                                  "deaths_predation")):
            cols[name][i] = np.int32(cum[k] - self._prev_cum[k])
            self._prev_cum[k] = cum[k]

        if i == 0 or world.tick % self.sample_every == 0:
            self._last_dist = self._genome_dist(world)
        cols["genome_dist"][i] = self._last_dist
        self._length = i + 1

    def save(self, path: str, seed: int | None = None) -> None:
        """Write columns and metadata (seed, settings, Config JSON) to .npz."""
        payload: dict[str, np.ndarray] = {
            name: arr[: self.length].copy() for name, arr in self._columns.items()
        }
        payload["columns"] = np.array(list(COLUMNS))
        payload["seed"] = np.int64(-1 if seed is None else seed)
        payload["sample_every"] = np.int32(self.sample_every)
        payload["sample_size"] = np.int32(self.sample_size)
        payload["config_json"] = np.array(
            json.dumps(self._config or {}, sort_keys=True)
        )
        np.savez(path, **payload)

    @staticmethod
    def load(path: str) -> StatsHistory:
        """Read a file written by :meth:`save` into a :class:`StatsHistory`."""
        with np.load(path, allow_pickle=False) as f:
            names = [str(n) for n in f["columns"]]
            data = {name: f[name] for name in names}
            seed = int(f["seed"])
            sample_every = int(f["sample_every"])
            sample_size = int(f["sample_size"])
            config = json.loads(str(f["config_json"]))
        return StatsHistory(
            data=data,
            seed=None if seed < 0 else seed,
            config=config,
            sample_every=sample_every,
            sample_size=sample_size,
        )
