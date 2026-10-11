"""Per-tick metrics recorder: history for plots and balance reports (stage 6).

``StatsRecorder`` samples the world **read-only** from the CLI loop (never
inside ``advance``, so the tick and its benchmark stay untouched) and keeps
one row per recorded tick in pre-allocated int32/float32 columns:

    tick, alive, mean_energy, mean_age                      (state)
    births, deaths_famine, deaths_age, deaths_predation      (per-tick deltas
        of the World's cumulative counters)
    meat_eaten       (per-tick delta of World.meat_eaten, stage 11 carrion)
    signal_emitted   (per-tick delta of World.signal_emitted, stage 13:
                      how much signal was deposited this tick — 0 means
                      no living creature opened the emit gate)
    signal_mass      (current sum of the signal field, gauge every tick:
                      mass at equilibrium is deposit_rate / signal_decay)
    emit_mean        (mean emit-gate fraction over living creatures, from
                      the last think's logits — reads 0 when nothing lives)
    bite_herb, bite_omni, bite_pred
                     (per-tick deltas of World.bite_by_diet, stage 12.5:
                     successful takes grouped by the *attacker's* diet class)
    n_herbivore, n_omnivore, n_predator                     (diet classes,
        stage 12: gauge over living slots, from ecology.classify_diets)
    speed/size/vision/diet mean and std over living slots    (diversity)
    diet_max, diet_p50, diet_p90, diet_p99
                     (distribution of the diet trait over living slots,
                     stage 12.5: diet_max is the ">= 1 predator alive"
                     gauge, recorded every tick; the percentiles are
                     sampled with genome_dist and forward-filled; all
                     read 0 when no creatures are alive)
    genome_dist   (sampled genetic diversity, see below)
    n_species, barrier, matings_valid                       (stage 12;
        sampled every ``cluster_every`` ticks and forward-filled — see below)

``genome_dist`` is the RMS pairwise distance between sampled genomes,
``sqrt(mean_pairs(||a-b||^2) / G)``, over a genome normalized by the spawn
scales (brain genes by ``weight_init_std``; traits by ``max_speed``, ``1.0``,
``vision_range``, ``1.0``). A deterministic ``linspace`` sample of
``sample_size`` living slots is taken every ``sample_every`` ticks — no RNG,
so recordings are bit-reproducible — and the value is forward-filled on the
in-between ticks. It reads ``0`` when fewer than two creatures are alive at
a sample tick.

Stage-12 columns: ``n_species`` is the number of greedy-leader clusters
(``ecology.cluster_living`` with ``Config.species_threshold``) over *all*
living slots; ``barrier`` is the fraction of logged births since the last
cluster sample whose two parents (validated against the currently living
creature ids) carried *different* cluster labels; ``matings_valid`` is the
denominator of that fraction (births where both parents were still alive at
the sample — 0 means no data, not "no barrier"). All three are
forward-filled between cluster samples.

Saving writes a plain ``.npz`` (column arrays plus seed, recorder settings
and the Config as JSON) so ``plot.py`` can redraw without re-simulating.
No matplotlib here: the core must never import it (tests/test_render_guard).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .ecology import classify_diets, cluster_living, genome_scales
from .world import World

INT_COLUMNS = (
    "tick",
    "alive",
    "births",
    "deaths_famine",
    "deaths_age",
    "deaths_predation",
    "bite_herb",
    "bite_omni",
    "bite_pred",
    "n_herbivore",
    "n_omnivore",
    "n_predator",
    "matings_valid",
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
    "diet_max",
    "diet_p50",
    "diet_p90",
    "diet_p99",
    "meat_eaten",
    "signal_emitted",
    "signal_mass",
    "emit_mean",
    "genome_dist",
    "n_species",
    "barrier",
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
                 sample_size: int = 64, cluster_every: int = 100) -> None:
        if capacity < 1 or sample_every < 1 or sample_size < 2:
            raise ValueError("capacity >= 1, sample_every >= 1, sample_size >= 2")
        if cluster_every < 1:
            raise ValueError("cluster_every must be >= 1")
        self.capacity = capacity
        self.sample_every = sample_every
        self.sample_size = sample_size
        self.cluster_every = cluster_every
        self._length = 0
        self._columns: dict[str, np.ndarray] = {
            name: np.zeros(capacity, dtype=np.int32 if name in INT_COLUMNS
                           else np.float32)
            for name in COLUMNS
        }
        self._prev_cum = np.zeros(4, dtype=np.int64)  # famine/age/predation/births
        self._prev_meat = np.float64(0.0)  # cumulative meat_eaten delta baseline
        self._prev_bite = np.zeros(3, dtype=np.int64)  # bite_by_diet baseline
        self._prev_signal = np.float64(0.0)  # signal_emitted delta baseline
        self._last_dist = np.float32(0.0)
        self._last_diet_p = (np.float32(0.0), np.float32(0.0), np.float32(0.0))
        self._last_n_species = np.float32(0.0)
        self._last_barrier = np.float32(0.0)
        self._last_matings_valid = 0
        self._prev_log_used = 0
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
        return genome_scales(world.config)

    def _diet_percentiles(self, world: World) -> tuple[np.float32, np.float32, np.float32]:
        """p50/p90/p99 of the diet trait over living slots (stage 12.5).

        Computed on ``sample_every`` ticks (same cadence as genome_dist) and
        forward-filled in ``record``; reads (0, 0, 0) when nothing is alive.
        The three values bracket the "how far has diet climbed" question that
        diet_max alone (a single extreme order statistic) cannot answer.
        """
        alive = np.flatnonzero(world.alive)
        if alive.size == 0:
            return np.float32(0.0), np.float32(0.0), np.float32(0.0)
        diet = world.genome[alive, world.config.brain_params + 3]
        p50, p90, p99 = np.percentile(diet, (50.0, 90.0, 99.0))
        return np.float32(p50), np.float32(p90), np.float32(p99)

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
            diet_labels = classify_diets(traits[:, 3])
            cols["n_herbivore"][i] = np.int32(np.count_nonzero(diet_labels == 0))
            cols["n_omnivore"][i] = np.int32(np.count_nonzero(diet_labels == 1))
            cols["n_predator"][i] = np.int32(np.count_nonzero(diet_labels == 2))
            cols["diet_max"][i] = np.float32(traits[:, 3].max())
            cols["emit_mean"][i] = np.float32(
                0.5 * (1.0 + np.tanh(world.actions[alive, 3])).mean()
            )
        else:
            cols["mean_energy"][i] = 0.0
            cols["mean_age"][i] = 0.0
            cols["n_herbivore"][i] = 0
            cols["n_omnivore"][i] = 0
            cols["n_predator"][i] = 0
            cols["diet_max"][i] = 0.0
            cols["emit_mean"][i] = 0.0
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
        cols["meat_eaten"][i] = np.float32(world.meat_eaten - self._prev_meat)
        self._prev_meat = np.float64(world.meat_eaten)
        cols["signal_emitted"][i] = np.float32(
            world.signal_emitted - self._prev_signal
        )
        self._prev_signal = np.float64(world.signal_emitted)
        # full-field sum: 40k float32 adds (~40 µs), cheap next to record's
        # existing trait means; the mass shows whether the field is alive
        cols["signal_mass"][i] = np.float32(world.signal.sum(dtype=np.float64))
        for k, name in enumerate(("bite_herb", "bite_omni", "bite_pred")):
            cols[name][i] = np.int32(world.bite_by_diet[k] - self._prev_bite[k])
            self._prev_bite[k] = world.bite_by_diet[k]

        if i == 0 or world.tick % self.sample_every == 0:
            self._last_dist = self._genome_dist(world)
            self._last_diet_p = self._diet_percentiles(world)
        cols["genome_dist"][i] = self._last_dist
        cols["diet_p50"][i] = self._last_diet_p[0]
        cols["diet_p90"][i] = self._last_diet_p[1]
        cols["diet_p99"][i] = self._last_diet_p[2]

        if i == 0 or world.tick % self.cluster_every == 0:
            self._sample_species(world)
        cols["n_species"][i] = self._last_n_species
        cols["barrier"][i] = self._last_barrier
        cols["matings_valid"][i] = self._last_matings_valid
        self._length = i + 1

    def _sample_species(self, world: World) -> None:
        """Cluster the living genomes and measure the reproductive barrier.

        Reads the genealogy rows appended since the last cluster sample;
        a birth counts only when *both* parents (by creature id) are still
        alive at this sample — ``matings_valid`` exposes how many did, so a
        low denominator is visible instead of hidden behind a noisy ratio.
        """
        slots, labels = cluster_living(world, world.config.species_threshold)
        self._last_n_species = (
            np.float32(np.unique(labels).size) if slots.size else np.float32(0.0)
        )
        log_end = world.genealogy_used
        window = world.genealogy[self._prev_log_used:log_end]
        self._prev_log_used = log_end
        if window.shape[0] == 0 or slots.size < 2:
            self._last_barrier = np.float32(0.0)
            self._last_matings_valid = 0
            return
        id_to_slot = {int(world.creature_id[s]): s for s in slots}
        label_of = {int(world.creature_id[s]): int(lab)
                    for s, lab in zip(slots, labels)}
        cross = 0
        valid = 0
        for _tick, _child, parent, partner in window:
            p = int(parent)
            q = int(partner)
            if p in id_to_slot and q in id_to_slot:
                valid += 1
                if label_of[p] != label_of[q]:
                    cross += 1
        self._last_matings_valid = valid
        self._last_barrier = (
            np.float32(cross / valid) if valid else np.float32(0.0)
        )

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
