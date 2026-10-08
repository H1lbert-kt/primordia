"""Save and load the entire world as a versioned ``.npz`` (stage 7).

Format v1 stores exactly what is needed to continue a run bit-for bit:

    format_version  int32 scalar; readers reject other versions
    config_json     Config as JSON (rebuilds every balance parameter)
    rng_json        {"numpy": bit_generator.state, "numba": [index, key]}
                    — both RNG streams, so continuation is identical
    seed            int64, the construction seed (labels only)
    tick, births, deaths_famine, deaths_age, deaths_predation,
    free_count, next_id, genealogy_used, genealogy_overflow
                    int64 scalars
    pos vel angle energy age genome species_id parent_id creature_id
    alive free_list food
                    the SoA state arrays
    genealogy       (used, 3) int32 birth log, trimmed to genealogy_used

Deliberately **not** stored: ``sensor_buf``/``hidden_buf``/``actions``
(rewritten by phases 3–5 of every tick), ``cell_*`` grids (rebuilt as phase
0 of every ``advance``, and again by ``load_world`` so grids are valid even
before the first tick) and ``ray_unit``/``cell_w``/``cell_h`` (derived from
Config).

The Numba RNG stream is process-global: ``save_world`` must be called while
the world being saved is the active stream — the same "advance worlds
sequentially, never interleaved" rule documented in genome.py. Loading
restores both streams, so ``load(save(w))`` continued for N ticks matches
advancing ``w`` for N more ticks (tests/test_io.py proves it).
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .config import Config
from .genome import get_numba_rng_state, set_numba_rng_state
from .world import World

FORMAT_VERSION = 1

_INT_SCALARS = (
    "seed",
    "tick",
    "births",
    "deaths_famine",
    "deaths_age",
    "deaths_predation",
    "free_count",
    "next_id",
    "genealogy_used",
    "genealogy_overflow",
)
_ARRAYS = (
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
    "free_list",
    "food",
)


def save_world(world: World, path: str) -> None:
    """Write the full simulation state to ``path`` (npz, format v1)."""
    payload: dict[str, np.ndarray] = {
        "format_version": np.int32(FORMAT_VERSION),
        "config_json": np.array(json.dumps(asdict(world.config), sort_keys=True)),
        "rng_json": np.array(
            json.dumps(
                {
                    "numpy": world.rng.bit_generator.state,
                    "numba": get_numba_rng_state(),
                }
            )
        ),
    }
    for name in _INT_SCALARS:
        payload[name] = np.int64(getattr(world, name))
    for name in _ARRAYS:
        payload[name] = np.asarray(getattr(world, name))
    payload["genealogy"] = world.genealogy[: world.genealogy_used].copy()
    np.savez(path, **payload)


def load_world(path: str) -> World:
    """Rebuild a world saved by :func:`save_world`, RNG streams restored.

    Raises ``ValueError`` on an unknown format version so old files fail
    with a clear message instead of a shape mismatch.
    """
    with np.load(path, allow_pickle=False) as f:
        version = int(f["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported save format {version} (this build reads {FORMAT_VERSION})"
            )
        cfg = Config(**json.loads(str(f["config_json"])))
        rng = json.loads(str(f["rng_json"]))
        scalars = {name: int(f[name]) for name in _INT_SCALARS}
        arrays = {name: np.array(f[name]) for name in _ARRAYS}
        log = np.array(f["genealogy"])

    # Fresh construction spawns the initial population and reseeds both RNG
    # streams; every array, scalar and stream below is then overwritten, so
    # the construction seed only has to be valid.
    world = World(cfg, seed=scalars["seed"])
    for name, arr in arrays.items():
        getattr(world, name)[...] = arr
    for name, value in scalars.items():
        setattr(world, name, value)
    world.genealogy[: log.shape[0]] = log
    world.rng.bit_generator.state = rng["numpy"]
    set_numba_rng_state(tuple(rng["numba"]))
    world.rebuild_counts()
    return world
