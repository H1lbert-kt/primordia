"""The tick: the only function that advances the world.

Per-creature logic lives in compiled kernels (``@njit``); there is no Python
loop over creatures anywhere. Phase order inside :func:`advance` is part of
the deterministic contract (``PHASES`` exposes the same sequence for the
profiler in ``bench``):

    0. rebuild creature-count grid (serial; the spatial grid sensors read)
    1. grow food        (parallel over cells; models photosynthesis)
    2. age + metabolize (parallel over slots, alive only; energy drains by
       metabolic_cost plus move_cost * |vel|^2)
    3. perceive         (parallel: rays + smell + internal -> sensor_buf)
    4. think            (parallel: MLP forward -> hidden_buf, raw actions)
    5. apply actions    (parallel: accel/turn -> vel/angle, eat gate)
    6. eat              (serial: shared cells; extraction * gate)
    7. integrate motion + wrap borders (parallel, alive only)
    8. deaths           (serial: kill, zero energy, push slot to free-list)
    9. reproduce        (serial: energy >= threshold -> split + mutate child;
       uses the slots freed in phase 8; newborns skip perceive/think until
       the next tick)

Perception uses positions from the start of the tick, movement is applied
afterwards. No random numbers are drawn inside any parallel kernel. The
serial reproduce kernel draws from the Numba RNG, seeded once at World
construction (see genome.py); it is never touched by prange.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .brain import apply_actions, think
from .genome import _reproduce
from .sensors import perceive
from .world import World


@njit(cache=True, parallel=True)
def _grow_food(food: np.ndarray, rate: np.float32, capacity: np.float32) -> None:
    for i in prange(food.shape[0]):
        for j in range(food.shape[1]):
            f = food[i, j]
            food[i, j] = f + rate * (capacity - f)


@njit(cache=True, parallel=True)
def _age_and_metabolize(
    energy: np.ndarray,
    age: np.ndarray,
    alive: np.ndarray,
    vel: np.ndarray,
    cost: np.float32,
    move_cost: np.float32,
) -> None:
    for i in prange(alive.size):
        if alive[i]:
            age[i] = age[i] + 1
            speed2 = vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1]
            energy[i] = energy[i] - cost - move_cost * speed2


@njit(cache=True)
def _eat(
    pos: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    food: np.ndarray,
    actions: np.ndarray,
    cell_w: np.float32,
    cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    eat_rate: np.float32,
) -> None:
    """Serial: shared food cells would make a parallel scatter a race."""
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
        take = take * actions[i, 2]  # eat gate from the brain
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
) -> tuple[int, int, int]:
    """Kill and recycle in one serial pass (ascending slot order = deterministic).

    Returns ``(free_count, deaths_famine, deaths_age)``; a creature dying of
    both counts as famine (checked first).
    """
    n_famine = 0
    n_age = 0
    for i in range(alive.size):
        if alive[i] and (energy[i] <= 0.0 or age[i] >= max_age):
            if energy[i] <= 0.0:
                n_famine = n_famine + 1
            else:
                n_age = n_age + 1
            alive[i] = False
            energy[i] = 0.0
            free_list[free_count] = i
            free_count = free_count + 1
    return free_count, n_famine, n_age


def phase_rebuild_counts(world: World) -> None:
    world.rebuild_counts()


def phase_grow_food(world: World) -> None:
    cfg = world.config
    _grow_food(world.food, np.float32(cfg.food_growth_rate), np.float32(cfg.food_capacity))


def phase_age_and_metabolize(world: World) -> None:
    cfg = world.config
    _age_and_metabolize(
        world.energy,
        world.age,
        world.alive,
        world.vel,
        np.float32(cfg.metabolic_cost),
        np.float32(cfg.move_cost),
    )


def phase_perceive(world: World) -> None:
    cfg = world.config
    gh, gw = world.food.shape
    cell_w = float(world.cell_w)
    cell_h = float(world.cell_h)
    step_len = cell_w if cell_w < cell_h else cell_h
    energy_scale = cfg.initial_energy * 2.0
    if energy_scale <= 0.0:
        energy_scale = 1.0
    perceive(
        world.pos,
        world.angle,
        world.vel,
        world.energy,
        world.alive,
        world.genome,
        world.food,
        world.cell_counts,
        world.ray_unit,
        world.sensor_buf,
        np.int32(cfg.n_rays),
        np.int32(cfg.smell_radius_cells),
        np.int32(cfg.brain_params),
        np.float32(1.0 / cell_w),
        np.float32(1.0 / cell_h),
        np.float32(step_len),
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.width),
        np.float32(cfg.height),
        np.float32(cfg.food_capacity),
        np.float32(energy_scale),
    )


def phase_think(world: World) -> None:
    cfg = world.config
    think(
        world.sensor_buf,
        world.hidden_buf,
        world.actions,
        world.alive,
        world.genome,
        np.int32(cfg.sensor_input_dim),
        np.int32(cfg.hidden_size),
        np.int32(cfg.N_OUTPUTS),
    )


def phase_apply_actions(world: World) -> None:
    cfg = world.config
    apply_actions(
        world.actions,
        world.angle,
        world.vel,
        world.alive,
        world.genome,
        np.float32(cfg.max_turn),
        np.int32(cfg.brain_params),
    )


def phase_eat(world: World) -> None:
    cfg = world.config
    gh, gw = world.food.shape
    _eat(
        world.pos,
        world.energy,
        world.alive,
        world.food,
        world.actions,
        world.cell_w,
        world.cell_h,
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.eat_rate),
    )


def phase_integrate(world: World) -> None:
    cfg = world.config
    _integrate(
        world.pos, world.vel, world.alive, np.float32(cfg.width), np.float32(cfg.height)
    )


def phase_deaths(world: World) -> None:
    cfg = world.config
    free_count, n_famine, n_age = _kill(
        world.energy,
        world.age,
        world.alive,
        world.free_list,
        np.int32(world.free_count),
        np.int32(cfg.max_age),
    )
    world.free_count = int(free_count)
    world.deaths_famine += int(n_famine)
    world.deaths_age += int(n_age)


def phase_reproduce(world: World) -> None:
    cfg = world.config
    births, free_count = _reproduce(
        world.pos,
        world.vel,
        world.age,
        world.angle,
        world.energy,
        world.alive,
        world.genome,
        world.species_id,
        world.parent_id,
        world.free_list,
        np.int32(world.free_count),
        np.float32(cfg.reproduce_threshold),
        np.float32(cfg.child_fraction),
        np.float32(cfg.mutation_rate),
        np.float32(cfg.mutation_std),
        np.float32(cfg.trait_mutation_std),
        np.int32(cfg.brain_params),
        np.float32(max(cfg.width, cfg.height)),
    )
    world.free_count = int(free_count)
    world.births += int(births)


# Tick phases in execution order; consumed by `python -m primordia.bench --profile`.
PHASES = (
    ("rebuild_counts", phase_rebuild_counts),
    ("grow_food", phase_grow_food),
    ("age_metabolize", phase_age_and_metabolize),
    ("perceive", phase_perceive),
    ("think", phase_think),
    ("apply_actions", phase_apply_actions),
    ("eat", phase_eat),
    ("integrate", phase_integrate),
    ("deaths", phase_deaths),
    ("reproduce", phase_reproduce),
)


def advance(world: World) -> None:
    """Advance the simulation by one tick, modifying ``world`` in place."""
    for _name, phase in PHASES:
        phase(world)
    world.tick += 1
