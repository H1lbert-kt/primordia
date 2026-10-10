"""The tick: the only function that advances the world.

Per-creature logic lives in compiled kernels (``@njit``); there is no Python
loop over creatures anywhere. Phase order inside :func:`advance` is part of
the deterministic contract (``PHASES`` exposes the same sequence for the
profiler in ``bench``):

    0. rebuild creature grid  (serial; counts, diet sums and CSR cell lists
       that sensors, bite and mate search read)
    1. grow food        (parallel over cells; models photosynthesis)
    2. decay meat       (parallel over cells; carrion loses meat_decay per
       tick, stage 11 — nutrients are borrowed, not stored)
    3. age + metabolize (parallel over slots, alive only; energy drains by
       metabolic_cost * wear plus move_cost * |vel|^2, wear =
       exp(senescence_rate * age), stage 11)
    4. perceive         (parallel: rays + smells + internal -> sensor_buf;
       ray reach also divided by wear)
    5. think            (parallel: MLP forward -> hidden_buf, raw actions)
    6. apply actions    (parallel: accel/turn -> vel/angle, eat gate; turning
        drains turn_cost * |radians| — steering is never free; top speed
        divided by wear)
    7. eat              (serial: shared cells; food first, then carrion;
        extraction * gate)
    8. bite             (serial: contact predation; gate * diet budget, energy
       transfer at bite_efficiency, the dissipating share lands as carrion,
       prey at 0 energy is killed here and counted as predation)
    9. integrate motion + wrap borders (parallel, alive only)
    10. deaths          (serial: kill, deposit leftover body energy as carrion,
       zero energy, push slot to free-list)
    11. reproduce       (serial: energy >= threshold -> nearest partner within
       mate_range -> child; uses the slots freed in phases 8 and 10;
       newborns skip perceive/think until the next tick)

Perception uses positions from the start of the tick, movement is applied
afterwards (bite uses those same positions). The CSR cell lists are one
movement old inside reproduce — the exact toroidal distance check is the
final eligibility word (genome.py documents the staleness). No random
numbers are drawn inside any parallel kernel. The serial reproduce kernel
draws from the Numba RNG, seeded once at World construction (see
genome.py); it is never touched by prange or by bite.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .brain import apply_actions, think
from .cycles import growth_multiplier, light_level
from .genome import _reproduce
from .sensors import perceive
from .world import World


@njit(cache=True, parallel=True)
def _grow_food(
    food: np.ndarray, field: np.ndarray, terrain: np.ndarray, rate: np.float32
) -> None:
    """Refill each cell toward K = ``field * terrain`` (stages 9 + 10).

    Terrain is flat 1.0 when disabled, which reduces the target to
    ``field`` exactly — the stage-9 behavior. Water cells (terrain == 0.0)
    target 0: they never grow food and drain what they inherited.
    """
    for i in prange(food.shape[0]):
        for j in range(food.shape[1]):
            f = food[i, j]
            target = field[i, j] * terrain[i, j]
            food[i, j] = f + rate * (target - f)


@njit(cache=True, parallel=True)
def _age_and_metabolize(
    energy: np.ndarray,
    age: np.ndarray,
    alive: np.ndarray,
    vel: np.ndarray,
    pos: np.ndarray,
    cost: np.float32,
    move_cost: np.float32,
    terrain: np.ndarray,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    water_move_cost: np.float32,
    senescence_rate: np.float32,
) -> None:
    """Age, then drain metabolism (x wear) plus movement (x water, stage 10).

    Wear = ``exp(senescence_rate * age)`` (stage 11): the metabolic bill
    grows exponentially with age; rate 0 restores the flat stage-10 cost.
    """
    gh = terrain.shape[0]
    gw = terrain.shape[1]
    for i in prange(alive.size):
        if alive[i]:
            age[i] = age[i] + 1
            metabolic = cost
            if senescence_rate > 0.0:
                metabolic = cost * np.exp(senescence_rate * np.float32(age[i]))
            speed2 = vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1]
            move = move_cost * speed2
            # swimming costs extra: water cells are exact 0.0 in terrain
            cx = int(pos[i, 0] * inv_cell_w)
            cy = int(pos[i, 1] * inv_cell_h)
            if cx < 0:
                cx = 0
            elif cx >= gw:
                cx = gw - 1
            if cy < 0:
                cy = 0
            elif cy >= gh:
                cy = gh - 1
            if terrain[cy, cx] == 0.0:
                move = move * water_move_cost
            energy[i] = energy[i] - metabolic - move


@njit(cache=True, parallel=True)
def _decay_meat(meat: np.ndarray, rate: np.float32) -> None:
    """Carrion decays by ``meat_decay`` per tick (stage 11 nutrients loop)."""
    for i in prange(meat.shape[0]):
        for j in range(meat.shape[1]):
            meat[i, j] = meat[i, j] * (np.float32(1.0) - rate)


@njit(cache=True)
def _eat(
    pos: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    food: np.ndarray,
    meat: np.ndarray,
    actions: np.ndarray,
    cell_w: np.float32,
    cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    eat_rate: np.float32,
) -> float:
    """Serial: shared food cells would make a parallel scatter a race.

    Food is drained first up to the gate-scaled budget; whatever the budget
    still allows drains carrion (stage 11 — a scavenger eats what is there).
    Returns the total meat energy extracted (the caller adds it to
    ``World.meat_eaten``).
    """
    meat_eaten = np.float32(0.0)
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
        budget = eat_rate * actions[i, 2]  # eat gate from the brain
        if budget <= 0.0:
            continue
        take = food[cy, cx]
        if take > budget:
            take = budget
        if take > 0.0:
            food[cy, cx] = food[cy, cx] - take
            energy[i] = energy[i] + take
            budget = budget - take
        take_m = meat[cy, cx]
        if take_m > budget:
            take_m = budget
        if take_m > 0.0:
            meat[cy, cx] = meat[cy, cx] - take_m
            energy[i] = energy[i] + take_m
            meat_eaten = meat_eaten + take_m
    return meat_eaten


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
    pos: np.ndarray,
    meat: np.ndarray,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    carrion_yield: np.float32,
) -> tuple[int, int, int]:
    """Kill and recycle in one serial pass (ascending slot order = deterministic).

    Returns ``(free_count, deaths_famine, deaths_age)``; a creature dying of
    both counts as famine (checked first). Death deposits carrion on the
    cell (stage 11): ``carrion_yield * max(energy, 0)`` — an old body leaves
    its leftover energy, a starved one leaves nothing. The deposit happens
    before energy is zeroed.
    """
    n_famine = 0
    n_age = 0
    for i in range(alive.size):
        if alive[i] and (energy[i] <= 0.0 or age[i] >= max_age):
            if energy[i] <= 0.0:
                n_famine = n_famine + 1
            else:
                n_age = n_age + 1
                # only a body with leftover energy feeds the ground
                deposit = carrion_yield * energy[i]
                cx = int(pos[i, 0] * inv_cell_w)
                cy = int(pos[i, 1] * inv_cell_h)
                if cx < 0:
                    cx = 0
                elif cx >= gw:
                    cx = gw - 1
                if cy < 0:
                    cy = 0
                elif cy >= gh:
                    cy = gh - 1
                meat[cy, cx] = meat[cy, cx] + deposit
            alive[i] = False
            energy[i] = 0.0
            free_list[free_count] = i
            free_count = free_count + 1
    return free_count, n_famine, n_age


def phase_rebuild_counts(world: World) -> None:
    world.rebuild_counts()


def phase_grow_food(world: World) -> None:
    cfg = world.config
    # season x light (stage 10): pure function of the tick, no state; the
    # applied rate is clamped to [0, 1] so a strong boom cannot overshoot
    # K and drive food negative (f + r*(K-f) with r in [0,1] stays in [f,K])
    scale = float(growth_multiplier(cfg, world.tick))
    rate = min(cfg.food_growth_rate * scale, 1.0)
    _grow_food(world.food, world.food_field, world.terrain, np.float32(rate))


def phase_decay_meat(world: World) -> None:
    cfg = world.config
    if cfg.meat_decay > 0.0:
        _decay_meat(world.meat, np.float32(cfg.meat_decay))


def phase_age_and_metabolize(world: World) -> None:
    cfg = world.config
    _age_and_metabolize(
        world.energy,
        world.age,
        world.alive,
        world.vel,
        world.pos,
        np.float32(cfg.metabolic_cost),
        np.float32(cfg.move_cost),
        world.terrain,
        np.float32(1.0 / world.cell_w),
        np.float32(1.0 / world.cell_h),
        np.float32(cfg.water_move_cost),
        np.float32(cfg.senescence_rate),
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
    # ambient light (stage 10): shortens rays at night and feeds the
    # "light" input; same pure formula the renderer reads
    light = light_level(cfg, world.tick)
    perceive(
        world.pos,
        world.angle,
        world.vel,
        world.energy,
        world.age,
        world.alive,
        world.genome,
        world.food,
        world.meat,
        world.cell_counts,
        world.cell_diet,
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
        light,
        np.float32(cfg.senescence_rate),
    )


def phase_think(world: World) -> None:
    cfg = world.config
    think(
        world.sensor_buf,
        world.hidden_buf,
        world.hidden_prev,
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
        world.energy,
        world.age,
        world.alive,
        world.genome,
        np.float32(cfg.max_turn),
        np.float32(cfg.turn_cost),
        np.int32(cfg.brain_params),
        np.float32(cfg.senescence_rate),
    )


def phase_eat(world: World) -> None:
    cfg = world.config
    gh, gw = world.food.shape
    meat_eaten = _eat(
        world.pos,
        world.energy,
        world.alive,
        world.food,
        world.meat,
        world.actions,
        world.cell_w,
        world.cell_h,
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.eat_rate),
    )
    world.meat_eaten += float(meat_eaten)


@njit(cache=True)
def _bite(
    pos: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    actions: np.ndarray,
    offsets: np.ndarray,
    slots: np.ndarray,
    free_list: np.ndarray,
    free_count: np.int32,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    width: np.float32,
    height: np.float32,
    reach_x: np.int32,
    reach_y: np.int32,
    bite_rate: np.float32,
    bite_efficiency: np.float32,
    contact_range: np.float32,
    traits_off: np.int32,
    meat: np.ndarray,
    carrion_yield: np.float32,
) -> tuple[int, int]:
    """Serial contact predation: gate * diet budget drains neighbours.

    For every living attacker whose ``bite_rate * gate * diet`` budget is
    positive, scan the CSR cells within ``reach`` of the attacker cell and
    drain each victim inside contact range ``contact_range * (size_a + size_v)``
    (shortest toroidal delta). The victim loses ``take`` and the attacker
    keeps ``bite_efficiency * take``; a carrion share
    ``carrion_yield * (1 - bite_efficiency) * take`` lands on the victim's
    cell (stage 11 — the rest of the "heat" still dissipates), so energy is
    never created. A victim left at zero energy dies here: its slot is
    pushed onto the free-list and counted as a predation death.

    Slot order is fixed and no random numbers are drawn, so the outcome is
    deterministic. Returns ``(free_count, deaths_predation)``.
    """
    n_pred = 0
    hw = width * np.float32(0.5)
    hh = height * np.float32(0.5)
    for i in range(alive.size):
        if not alive[i]:
            continue
        budget = bite_rate * actions[i, 2] * genome[i, traits_off + 3]
        if budget <= 0.0:
            continue
        px = pos[i, 0]
        py = pos[i, 1]
        ix0 = int(px * inv_cell_w)
        iy0 = int(py * inv_cell_h)
        if ix0 < 0:
            ix0 = 0
        elif ix0 >= gw:
            ix0 = gw - 1
        if iy0 < 0:
            iy0 = 0
        elif iy0 >= gh:
            iy0 = gh - 1
        size_i = genome[i, traits_off + 1]

        for dy in range(-reach_y, reach_y + 1):
            iy = iy0 + dy
            while iy < 0:
                iy = iy + gh
            while iy >= gh:
                iy = iy - gh
            for dx in range(-reach_x, reach_x + 1):
                ix = ix0 + dx
                while ix < 0:
                    ix = ix + gw
                while ix >= gw:
                    ix = ix - gw
                k = iy * gw + ix
                for p in range(offsets[k], offsets[k + 1]):
                    j = slots[p]
                    if j == i or not alive[j] or energy[j] <= 0.0:
                        continue
                    reach = contact_range * (size_i + genome[j, traits_off + 1])
                    ddx = px - pos[j, 0]
                    if ddx > hw:
                        ddx = ddx - width
                    elif ddx < -hw:
                        ddx = ddx + width
                    ddy = py - pos[j, 1]
                    if ddy > hh:
                        ddy = ddy - height
                    elif ddy < -hh:
                        ddy = ddy + height
                    if ddx * ddx + ddy * ddy > reach * reach:
                        continue
                    take = energy[j] if energy[j] < budget else budget
                    if take <= 0.0:
                        continue
                    energy[j] = energy[j] - take
                    energy[i] = energy[i] + bite_efficiency * take
                    # the dissipating share lands as carrion on the victim cell
                    meat[iy, ix] = (
                        meat[iy, ix] + carrion_yield * (np.float32(1.0) - bite_efficiency) * take
                    )
                    budget = budget - take
                    if energy[j] <= 0.0:
                        # killed by this bite: attribute the death now, before
                        # the famine pass could see the slot
                        alive[j] = False
                        energy[j] = 0.0
                        free_list[free_count] = j
                        free_count = free_count + 1
                        n_pred = n_pred + 1
                    if budget <= 0.0:
                        break
                if budget <= 0.0:
                    break
            if budget <= 0.0:
                break
    return free_count, n_pred


def phase_bite(world: World) -> None:
    cfg = world.config
    gh, gw = world.cell_counts.shape
    # Cells to scan per axis so the largest possible contact
    # (contact_range * 2 * max_size) can never be missed across cell edges.
    r_max = cfg.contact_range * 2.0 * cfg.max_size
    reach_x = int(r_max // float(world.cell_w)) + 1
    reach_y = int(r_max // float(world.cell_h)) + 1
    reach_x = min(reach_x, gw // 2)
    reach_y = min(reach_y, gh // 2)
    free_count, n_pred = _bite(
        world.pos,
        world.energy,
        world.alive,
        world.genome,
        world.actions,
        world.cell_offsets,
        world.cell_slots,
        world.free_list,
        np.int32(world.free_count),
        np.float32(1.0 / world.cell_w),
        np.float32(1.0 / world.cell_h),
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.width),
        np.float32(cfg.height),
        np.int32(reach_x),
        np.int32(reach_y),
        np.float32(cfg.bite_rate),
        np.float32(cfg.bite_efficiency),
        np.float32(cfg.contact_range),
        np.int32(cfg.brain_params),
        world.meat,
        np.float32(cfg.carrion_yield),
    )
    world.free_count = int(free_count)
    world.deaths_predation += int(n_pred)


def phase_integrate(world: World) -> None:
    cfg = world.config
    _integrate(
        world.pos, world.vel, world.alive, np.float32(cfg.width), np.float32(cfg.height)
    )


def phase_deaths(world: World) -> None:
    cfg = world.config
    gh, gw = world.food.shape
    free_count, n_famine, n_age = _kill(
        world.energy,
        world.age,
        world.alive,
        world.free_list,
        np.int32(world.free_count),
        np.int32(cfg.max_age),
        world.pos,
        world.meat,
        np.float32(1.0 / world.cell_w),
        np.float32(1.0 / world.cell_h),
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.carrion_yield),
    )
    world.free_count = int(free_count)
    world.deaths_famine += int(n_famine)
    world.deaths_age += int(n_age)


def phase_reproduce(world: World) -> None:
    cfg = world.config
    gh, gw = world.food.shape
    births, free_count, log_used, log_overflow, next_id = _reproduce(
        world.pos,
        world.vel,
        world.age,
        world.angle,
        world.energy,
        world.alive,
        world.genome,
        world.hidden_prev,
        world.species_id,
        world.parent_id,
        world.creature_id,
        world.genealogy,
        world.free_list,
        np.int32(world.free_count),
        world.cell_offsets,
        world.cell_slots,
        np.float32(cfg.reproduce_threshold),
        np.float32(cfg.child_fraction),
        np.float32(cfg.mutation_rate),
        np.float32(cfg.mutation_std),
        np.float32(cfg.trait_mutation_std),
        np.int32(cfg.brain_params),
        np.float32(max(cfg.width, cfg.height)),
        np.float32(cfg.max_size),
        np.float32(1.0 / world.cell_w),
        np.float32(1.0 / world.cell_h),
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.width),
        np.float32(cfg.height),
        np.float32(cfg.mate_range),
        np.int32(world.tick),
        np.int32(world.next_id),
        np.int32(world.genealogy_used),
        np.int32(world.genealogy_overflow),
    )
    world.free_count = int(free_count)
    world.births += int(births)
    world.genealogy_used = int(log_used)
    world.genealogy_overflow = int(log_overflow)
    world.next_id = int(next_id)


# Tick phases in execution order; consumed by `python -m primordia.bench --profile`.
PHASES = (
    ("rebuild_counts", phase_rebuild_counts),
    ("grow_food", phase_grow_food),
    ("decay_meat", phase_decay_meat),
    ("age_metabolize", phase_age_and_metabolize),
    ("perceive", phase_perceive),
    ("think", phase_think),
    ("apply_actions", phase_apply_actions),
    ("eat", phase_eat),
    ("bite", phase_bite),
    ("integrate", phase_integrate),
    ("deaths", phase_deaths),
    ("reproduce", phase_reproduce),
)


def advance(world: World) -> None:
    """Advance the simulation by one tick, modifying ``world`` in place."""
    for _name, phase in PHASES:
        phase(world)
    world.tick += 1
