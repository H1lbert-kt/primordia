"""Save and load the entire world as a versioned ``.npz`` (stage 7).

Format v11 stores exactly what is needed to continue a run bit-for bit
(v2 added ``food_field`` and ``hidden_prev``; v3 adds ``terrain``, the
stage-10 geography; v4 adds ``meat`` and ``meat_eaten``, the stage-11
carrion cycle; v5 widens the genealogy log to 4 columns so both parents
are recorded, the stage-12 reproductive-barrier metric; v6 adds ``signal``
and ``signal_emitted``, the stage-13 communication field; v7 is a layout
bump — the stage-15 tag adds a body trait; v8 adds the stage-16
``energy_donated`` counter; v9 adds the stage-17 wood cycle: the ``wood``
field, ``inventory``, the structure SoA (positions, types, fuel, store,
owner, alive mask and free-list) plus the ``wood_gathered`` /
``energy_cooked`` counters, and the genome grows to 490 columns with the
8-output layout; v10 is a layout bump for stage-21 plasticity — the genome
grows to 491 columns with the evolvable-eta trait, ``dw`` (the Oja delta,
never inherited but required to continue learning bit-for bit) and
``e_last_plastic`` (the delta-E credit signal) join the payload; v11 is
the stage-24 layout bump — the brain is 37 inputs at hidden 8, so the
genome shrinks to 446 columns, ``hidden_prev`` to 8 wide and ``dw`` to
360; readers reject older versions with a clear message):

    format_version  int32 scalar; readers reject other versions
    config_json     Config as JSON (rebuilds every balance parameter)
    rng_json        {"numpy": bit_generator.state, "numba": [index, key]}
                    — both RNG streams, so continuation is identical
    seed            int64, the construction seed (labels only)
    tick, births, deaths_famine, deaths_age, deaths_predation,
    free_count, next_id, genealogy_used, genealogy_overflow,
    struct_free_count
                    int64 scalars
    meat_eaten      float64 scalar (cumulative carrion energy extracted)
    signal_emitted  float64 scalar (cumulative signal deposited)
    energy_donated  float64 scalar (cumulative energy gifted to neighbours)
    wood_gathered   float64 scalar (cumulative wood moved into inventories)
    energy_cooked   float64 scalar (cumulative cooking bonus = fuel burned)
    pos vel angle energy age genome species_id parent_id creature_id
    alive free_list inventory food food_field terrain meat signal wood
    struct_pos struct_type struct_fuel struct_store struct_owner
    struct_alive struct_free_list hidden_prev dw e_last_plastic
                    the SoA state arrays
    genealogy       (used, 4) int32 birth log, trimmed to genealogy_used

Light and season are deliberately **not** stored: they are pure functions
of ``tick`` (see ``cycles.py``), which is saved — so a restored world
resumes mid-cycle with the exact same phase.

Deliberately **not** stored: ``sensor_buf``/``hidden_buf``/``actions``
(rewritten by phases 8, 9 and 11 of every tick — ``hidden_prev`` *is* stored: it is
persistent Elman state), ``cell_*`` grids (rebuilt as phase
0 of every ``advance``, and again by ``load_world`` so grids are valid even
before the first tick), the stage-17 structure scratch
(``struct_grid``/``struct_fire``/``fire_heat``/``cover`` — rebuilt by
``World.rebuild_struct_fields``, also called on load),
``ray_unit``/``cell_w``/``cell_h`` (derived from Config) and the stage-24
affect scratches (``stress``/``attacker_hue``/``attacker_age`` — lifetime
bite scars like ``dw``: a resumed run restarts with clean scratches,
which only matters while a bite is fresh, a few hundred ticks at the
default ``grudge_decay``).

The Numba RNG stream is process-global: ``save_world`` must be called while
the world being saved is the active stream — the same "advance worlds
sequentially, never interleaved" rule documented in genome.py. Loading
restores both streams, so ``load(save(w))`` continued for N ticks matches
advancing ``w`` for N more ticks (tests/test_io.py proves it).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import BinaryIO

import numpy as np

from .config import Config
from .genome import get_numba_rng_state, set_numba_rng_state
from .world import World

# v11 (stage 24): the brain layout changed (37 inputs, hidden 8), so the
# saved genome (N, 446), hidden_prev (N, 8) and dw (N, 360) have new shapes
# and old files cannot be reshaped safely. The stage-24 affect scratches
# (stress / attacker_hue / attacker_age) are lifetime state and stay out of
# the save, like bite_by_diet and donate_to.
FORMAT_VERSION = 11

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
    "struct_free_count",
)
_FLOAT_SCALARS = (
    "meat_eaten",
    "signal_emitted",
    "energy_donated",
    "wood_gathered",
    "energy_cooked",
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
    "inventory",
    "food",
    "food_field",
    "terrain",
    "meat",
    "signal",
    "wood",
    "struct_pos",
    "struct_type",
    "struct_fuel",
    "struct_store",
    "struct_owner",
    "struct_alive",
    "struct_free_list",
    "hidden_prev",
    "dw",
    "e_last_plastic",
)


def save_world(world: World, path: str | BinaryIO) -> None:
    """Write the full simulation state as npz (format v11).

    ``path`` is a filesystem path or any binary file-like object (the
    stage-19 daemon serializes checkpoints into a SQLite BLOB via
    ``BytesIO``); both go through the same payload.
    """
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
    for name in _FLOAT_SCALARS:
        payload[name] = np.float64(getattr(world, name))
    for name in _ARRAYS:
        payload[name] = np.asarray(getattr(world, name))
    payload["genealogy"] = world.genealogy[: world.genealogy_used].copy()
    np.savez(path, **payload)


def load_world(path: str | BinaryIO) -> World:
    """Rebuild a world saved by :func:`save_world`, RNG streams restored.

    ``path`` is a filesystem path or a binary file-like object holding
    the npz bytes (stage-19 database checkpoints).

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
        scalars_f = {name: float(f[name]) for name in _FLOAT_SCALARS}
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
    for name, value in scalars_f.items():
        setattr(world, name, value)
    world.genealogy[: log.shape[0]] = log
    world.rng.bit_generator.state = rng["numpy"]
    set_numba_rng_state(tuple(rng["numba"]))
    world.rebuild_counts()
    world.rebuild_struct_fields()  # stage-17 scratch: heat/cover/grid valid
    return world
