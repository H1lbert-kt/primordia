"""The tick: the only function that advances the world.

Per-creature logic lives in compiled kernels (``@njit``); there is no Python
loop over creatures anywhere. Kernel order inside :func:`advance` is part of
the deterministic contract:

    1. grow food      (parallel over cells; models photosynthesis)
    2. age + metabolize (parallel over slots, alive only)
    3. eat            (serial: several creatures share a cell; parallel would race)
    4. integrate motion + wrap borders (parallel, alive only)
    5. deaths         (serial: kill, zero energy, push slot to free-list)

No random numbers are drawn inside any kernel: draws happen only in
``World.__init__``/``World.spawn`` (NumPy Generator), never inside ``prange``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .config import Config
from .world import World


@njit(cache=True, parallel=True)
def _grow_food(food: np.ndarray, rate: np.float32, capacity: np.float32) -> None:
    for i in prange(food.shape[0]):
        for j in range(food.shape[1]):
            f = food[i, j]
            food[i, j] = f + rate * (capacity - f)


@njit(cache=True, parallel=True)
def _age_and_metabolize(
    energy: np.ndarray, age: np.ndarray, alive: np.ndarray, cost: np.float32
) -> None:
    for i in prange(alive.size):
        if alive[i]:
            age[i] = age[i] + 1
            energy[i] = energy[i] - cost


@njit(cache=True)
def _eat(
    pos: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    food: np.ndarray,
    cell_w: np.float32,
    cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    eat_rate: np.float32,
) -> None:
    for i in range(alive.size):
        if not alive[i]:
            continue
        cx = int(pos[i, 0] / cell_w)
        cy = int(pos[i, 1] / cell_h)
        if cx < 0:
            cx = 0
        elif cx >= gw:
            cx = gw - 1
        if cy < 0:
            cy = 0
        elif cy >= gh:
            cy = gh - 1
        take = food[cy, cx]
        if take > eat_rate:
            take = eat_rate
        if take > 0.0:
            food[cy, cx] = food[cy, cx] - take
            energy[i] = energy[i] + take


@njit(cache=True, parallel=True)
def _integrate(
    pos: np.ndarray,
    vel: np.ndarray,
    alive: np.ndarray,
    width: np.float32,
    height: np.float32,
) -> None:
    for i in prange(alive.size):
        if not alive[i]:
            continue
        x = pos[i, 0] + vel[i, 0]
        y = pos[i, 1] + vel[i, 1]
        while x < 0.0:
            x = x + width
        while x >= width:
            x = x - width
        while y < 0.0:
            y = y + height
        while y >= height:
            y = y - height
        pos[i, 0] = x
        pos[i, 1] = y


@njit(cache=True)
def _kill(
    energy: np.ndarray,
    age: np.ndarray,
    alive: np.ndarray,
    free_list: np.ndarray,
    free_count: np.int32,
    max_age: np.int32,
) -> np.int32:
    """Kill and recycle in one serial pass (ascending slot order = deterministic)."""
    for i in range(alive.size):
        if alive[i] and (energy[i] <= 0.0 or age[i] >= max_age):
            alive[i] = False
            energy[i] = 0.0
            free_list[free_count] = i
            free_count = free_count + 1
    return free_count


def advance(world: World) -> None:
    """Advance the simulation by one tick, modifying ``world`` in place."""
    cfg = world.config
    _grow_food(world.food, np.float32(cfg.food_growth_rate), np.float32(cfg.food_capacity))
    _age_and_metabolize(
        world.energy, world.age, world.alive, np.float32(cfg.metabolic_cost)
    )
    gh, gw = world.food.shape
    _eat(
        world.pos,
        world.energy,
        world.alive,
        world.food,
        world.cell_w,
        world.cell_h,
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.eat_rate),
    )
    _integrate(world.pos, world.vel, world.alive, np.float32(cfg.width), np.float32(cfg.height))
    world.free_count = int(
        _kill(
            world.energy,
            world.age,
            world.alive,
            world.free_list,
            np.int32(world.free_count),
            np.int32(cfg.max_age),
        )
    )
    world.tick += 1
