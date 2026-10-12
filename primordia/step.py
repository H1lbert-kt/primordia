"""The tick: the only function that advances the world.

Per-creature logic lives in compiled kernels (``@njit``); there is no Python
loop over creatures anywhere. Phase order inside :func:`advance` is part of
the deterministic contract (``PHASES`` exposes the same sequence for the
profiler in ``bench``):

    0. rebuild creature grid  (serial; counts, diet sums and CSR cell lists
       that sensors, bite and mate search read)
    1. grow food        (parallel over cells; models photosynthesis)
    2. grow wood        (parallel over cells; stage 17 — same refill law as
       food toward wood_field * terrain, constant rate, no seasonality)
    3. decay meat       (parallel over cells; carrion loses meat_decay per
       tick, stage 11 — nutrients are borrowed, not stored)
    4. decay signal     (parallel over cells; signal loses signal_decay per
       tick, stage 13 — a broadcast is information, it fades)
    5. diffuse signal   (parallel over cells; Jacobi 5-point pass over the
       4 toroidal neighbors, stage 14 — pheromone spreads before this
       tick's sources emit, so a deposit is picked up from the *next*
       tick onward)
    6. decay structures (serial; every fire burns fire_fuel_decay fuel and
       dies at 0, stage 17; every incomplete house rots at
       house_rot_rate * (1 - u) and dies at 0, stage 22 — a finished
       house (u = 1) persists)
    7. age + metabolize (parallel over slots, alive only; energy drains by
       metabolic_cost * wear plus move_cost * |vel|^2, wear =
       exp(senescence_rate * age), stage 11)
    8. perceive         (parallel: rays + smells + internal -> sensor_buf;
       ray reach also divided by wear)
    9. think            (parallel: MLP forward -> hidden_buf, raw actions;
       effective synapses genome + dw when plasticity is active, stage 21)
    10. plasticity      (parallel: local Oja on dw — W1 from the sensors,
        W_rec from the pre-commit hidden — then decay toward 0, clamp,
        energy cost * |dw| moved, credit ablation, and the hidden_prev
        commit that used to live inside think; zero RNG, genome untouched)
    11. apply actions   (parallel: accel/turn -> vel/angle, eat/emit/donate/
        gather/build_fire/build_house gates; turning drains turn_cost *
        |radians| — steering is never free; top speed divided by wear)
    12. emit signal     (serial: own-cell deposit of emit_rate * gate,
        drained from energy at signal_emit_cost * gate, stage 13 — the
        deposit is only visible to perceive from the *next* tick, since
        perceive already ran; the contract is deliberate, never reordered)
    13. donate          (serial: gift donate_rate * gate to the nearest
        living neighbour within donate_range, stage 16 — energy moves in
        the same tick, no latency; who may receive is physics, whether to
        give is the gate)
    14. gather          (serial: wood moves cell -> inventory, budgeted by
        gather_rate * gate and clamped both ends, stage 17 — storing is
        free; since stage 22 carrying can cost extra via carry_weight)
    15. build           (serial: gates > 0.5 act; fire is still a lump
        build/refuel, houses are raised in wood deliveries with
        energy charged per delivered fraction (stage 22), completeness
        u lives in struct_store and the shelter scales with u; if
        house_fire_radius > 0 a live fire must burn within that radius)
    16. rebuild structure fields (zero + stamp: exact-cell occupancy,
        fire heat radius, house cover stamped at completeness u —
        a fire built this tick already cooks this tick: an object is
        not a signal, stage 17)
    17. eat             (serial: shared cells; food first, then carrion;
        extraction * gate; food eaten under fire earns cook_bonus x energy
        paid 1:1 from that fire's fuel, stage 17 — never created energy)
    18. bite            (serial: contact predation; gate * diet budget,
        energy transfer at bite_efficiency reduced by house cover, the
        dissipating share lands as carrion, prey at 0 energy is killed
        here and counted as predation)
    19. integrate motion + wrap borders (parallel, alive only)
    20. deaths          (serial: kill, deposit leftover body energy as carrion,
        zero energy, push slot to free-list)
    21. reproduce       (serial: energy >= threshold -> nearest partner within
        mate_range -> child; uses the slots freed in phases 17 and 20;
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

from .brain import apply_actions, plasticity, think
from .cycles import growth_multiplier, light_level
from .ecology import HERB_MAX, PRED_MIN
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
    inventory: np.ndarray,
    cost: np.float32,
    move_cost: np.float32,
    terrain: np.ndarray,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    water_move_cost: np.float32,
    senescence_rate: np.float32,
    carry_weight: np.float32,
    max_inventory: np.float32,
) -> None:
    """Age, then drain metabolism (x wear) plus movement (x water, stage 10).

    Wear = ``exp(senescence_rate * age)`` (stage 11): the metabolic bill
    grows exponentially with age; rate 0 restores the flat stage-10 cost.
    Hauling tax (stage 22): movement is multiplied by
    ``1 + carry_weight * inventory / max_inventory`` — wood is free to
    store, dear to carry, so supply lines price long builds. The branch
    is skipped entirely at ``carry_weight = 0`` (bit-identical world).
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
            if carry_weight > 0.0 and inventory[i] > 0.0:
                move = move * (
                    np.float32(1.0)
                    + carry_weight * (inventory[i] / max_inventory)
                )
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


@njit(cache=True, parallel=True)
def _decay_signal(signal: np.ndarray, rate: np.float32) -> None:
    """Signal fades by ``signal_decay`` per tick (stage 13; the Jacobi
    spread of stage 14 runs right after this phase)."""
    for i in prange(signal.shape[0]):
        for j in range(signal.shape[1]):
            signal[i, j] = signal[i, j] * (np.float32(1.0) - rate)


@njit(cache=True, parallel=True)
def _diffuse_signal(
    signal: np.ndarray, scratch: np.ndarray, rate: np.float32
) -> None:
    """Jacobi pass: the field spreads to its 4 toroidal neighbors (stage 14).

    ``scratch[i,j] = signal[i,j] + rate * (N + S + E + W - 4*signal[i,j])``.
    Over a torus every edge pair transfers ``rate * value`` between cells,
    so the pass conserves total mass exactly (up to float rounding); the
    filter is stable for ``rate <= 0.25`` (the highest Laplacian mode is
    critically damped there — Config validates the bound). Rows are
    independent; reading only ``signal`` and writing only ``scratch``
    keeps the prange deterministic. The caller copies ``scratch`` back
    over ``signal`` (copy, not swap — held references stay valid).
    """
    gh = signal.shape[0]
    gw = signal.shape[1]
    for i in prange(gh):
        im = i - 1
        if im < 0:
            im = gh - 1
        ip = i + 1
        if ip >= gh:
            ip = 0
        for j in range(gw):
            jm = j - 1
            if jm < 0:
                jm = gw - 1
            jp = j + 1
            if jp >= gw:
                jp = 0
            lap = (
                signal[im, j]
                + signal[ip, j]
                + signal[i, jm]
                + signal[i, jp]
                - np.float32(4.0) * signal[i, j]
            )
            scratch[i, j] = signal[i, j] + rate * lap


@njit(cache=True)
def _decay_structures(
    struct_type: np.ndarray,
    struct_fuel: np.ndarray,
    struct_store: np.ndarray,
    struct_alive: np.ndarray,
    struct_free_list: np.ndarray,
    struct_free_count: np.int32,
    fuel_decay: np.float32,
    house_rot_rate: np.float32,
) -> tuple[int, int]:
    """Serial: fires burn fuel, incomplete houses rot (stage 17 + 22).

    Every fire loses ``fire_fuel_decay`` fuel per tick and dies at 0.
    Every house below completeness 1.0 loses ``house_rot_rate * (1 - u)``
    per tick (stage 22: an abandoned stub cannot linger) and dies at
    u <= 0; a complete house (u = 1) never decays. Dead slots return to
    the structure free-list. Returns ``(free_count, n_died)``. Cells
    still point at the dead slot until phase 15's rebuild, so readers
    must check ``struct_alive`` (the build kernel does).
    """
    n_died = 0
    for s in range(struct_alive.size):
        if not struct_alive[s]:
            continue
        if struct_type[s] == 0:
            fuel = struct_fuel[s] - fuel_decay
            if fuel <= 0.0:
                struct_alive[s] = False
                struct_fuel[s] = 0.0
                struct_free_list[struct_free_count] = s
                struct_free_count = struct_free_count + 1
                n_died = n_died + 1
            else:
                struct_fuel[s] = fuel
        else:
            u = struct_store[s]
            if u < 1.0:
                u = u - house_rot_rate * (np.float32(1.0) - u)
                if u <= 0.0:
                    struct_alive[s] = False
                    struct_store[s] = 0.0
                    struct_free_list[struct_free_count] = s
                    struct_free_count = struct_free_count + 1
                    n_died = n_died + 1
                else:
                    struct_store[s] = u
    return struct_free_count, n_died


@njit(cache=True)
def _eat(
    pos: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    food: np.ndarray,
    meat: np.ndarray,
    actions: np.ndarray,
    struct_fire: np.ndarray,
    struct_fuel: np.ndarray,
    cell_w: np.float32,
    cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    eat_rate: np.float32,
    cook_bonus: np.float32,
) -> tuple[float, float]:
    """Serial: shared food cells would make a parallel scatter a race.

    Food is drained first up to the gate-scaled budget; whatever the budget
    still allows drains carrion (stage 11 — a scavenger eats what is there).
    Cooking (stage 17): food eaten on a cell heated by a fire yields
    ``take * (cook_bonus - 1)`` extra energy, drained **from that fire's
    fuel** at 1:1 — fuel short of the claim pays the proportional bonus and
    then stops, so the eater's gain and the fuel's loss are always equal
    and no energy is created. Carrion is never cooked (v1). Returns
    ``(meat_eaten, energy_cooked)``; the caller adds them to
    ``World.meat_eaten`` and ``World.energy_cooked``.
    """
    meat_eaten = np.float32(0.0)
    cooked = np.float32(0.0)
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
            extra = np.float32(0.0)
            fire_slot = struct_fire[cy, cx]
            if fire_slot >= 0 and cook_bonus > 1.0:
                want = take * (cook_bonus - np.float32(1.0))
                have = struct_fuel[fire_slot]
                if want >= have:
                    extra = have
                    struct_fuel[fire_slot] = 0.0
                else:
                    extra = want
                    struct_fuel[fire_slot] = have - want
            energy[i] = energy[i] + take + extra
            cooked = cooked + extra
            budget = budget - take
        take_m = meat[cy, cx]
        if take_m > budget:
            take_m = budget
        if take_m > 0.0:
            meat[cy, cx] = meat[cy, cx] - take_m
            energy[i] = energy[i] + take_m
            meat_eaten = meat_eaten + take_m
    return meat_eaten, cooked


@njit(cache=True)
def _emit_signal(
    pos: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    signal: np.ndarray,
    actions: np.ndarray,
    cell_w: np.float32,
    cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    emit_rate: np.float32,
    emit_cost: np.float32,
) -> float:
    """Serial: own-cell deposits only, but keep the ordering deterministic.

    The brain's emit gate (output 3, decoded in apply_actions) deposits
    ``emit_rate * gate`` signal on the creature's own cell and drains
    ``signal_emit_cost * gate`` energy (stage 13 — communicating is a
    physical cost, never a rewarded behavior). Returns the total deposited
    (the caller adds it to ``World.signal_emitted``).
    """
    emitted = np.float32(0.0)
    for i in range(alive.size):
        if not alive[i]:
            continue
        gate = actions[i, 3]
        if gate <= 0.0:
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
        deposit = emit_rate * gate
        signal[cy, cx] = signal[cy, cx] + deposit
        energy[i] = energy[i] - emit_cost * gate
        emitted = emitted + deposit
    return emitted


@njit(cache=True)
def _donate(
    pos: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    actions: np.ndarray,
    hue_re: np.ndarray,
    hue_im: np.ndarray,
    cell_offsets: np.ndarray,
    cell_slots: np.ndarray,
    donate_to: np.ndarray,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    width: np.float32,
    height: np.float32,
    reach_x: np.int32,
    reach_y: np.int32,
    donate_rate: np.float32,
    donate_range: np.float32,
) -> tuple[float, float, int]:
    """Serial: every donor gifts its nearest living neighbour in range.

    The donate gate (output 4, decoded in apply_actions) scales the gift
    ``donate_rate * gate`` (stage 16 — the tick opens a physical transfer,
    never rewards giving). The recipient is the nearest living creature
    within ``donate_range`` by exact toroidal distance over the CSR cells:
    self excluded, ties to the lower slot — the mate search's rule, so the
    choice of *who can receive* is pure physics and kin (born in place,
    clustered by stage 15) are what "nearest" happens to mean. The donor
    pays at most what it has, so a gift can empty it; the deaths phase
    comes later and then finds nothing to deposit — the energy is already
    with the recipient, so the transfer conserves the world total up to
    float rounding. Returns ``(total, align_sum, pairs)`` where alignment
    is the dot of the two unit hue vectors (the stage-15 similarity
    metric): the experiment reads it against the global chance level to
    see whether gifts follow colour or not.
    """
    total = np.float32(0.0)
    align_sum = np.float64(0.0)
    pairs = 0
    if donate_rate <= 0.0:
        return total, align_sum, pairs
    hw = width * np.float32(0.5)
    hh = height * np.float32(0.5)
    range2 = donate_range * donate_range

    for i in range(alive.size):
        if not alive[i]:
            continue
        gate = actions[i, 4]
        avail = energy[i]
        if gate <= 0.0 or avail <= 0.0:
            continue
        # nearest living neighbour within donate_range (mirror of the mate
        # search: CSR cells in reach, exact toroidal distance, no self)
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
        best_j = -1
        best_d2 = np.float32(np.inf)
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
                for p in range(cell_offsets[k], cell_offsets[k + 1]):
                    j = cell_slots[p]
                    if j == i or not alive[j]:
                        continue
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
                    d2 = ddx * ddx + ddy * ddy
                    if d2 <= range2 and (
                        d2 < best_d2 or (d2 == best_d2 and j < best_j)
                    ):
                        best_d2 = d2
                        best_j = j
        if best_j < 0:
            continue
        j = best_j
        amount = donate_rate * gate
        if amount > avail:
            amount = avail
        energy[i] = avail - amount
        energy[j] = energy[j] + amount
        total = total + amount
        align_sum = align_sum + np.float64(
            hue_re[i] * hue_re[j] + hue_im[i] * hue_im[j]
        )
        pairs = pairs + 1
        donate_to[i] = j  # renderer-only gift map, cleared each tick
    return total, align_sum, pairs


@njit(cache=True)
def _gather(
    pos: np.ndarray,
    inventory: np.ndarray,
    alive: np.ndarray,
    actions: np.ndarray,
    wood: np.ndarray,
    cell_w: np.float32,
    cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    gather_rate: np.float32,
    max_inventory: np.float32,
) -> float:
    """Serial: move wood from the cell into the carrier (stage 17).

    Budget ``gather_rate * gather_gate`` clamped at both ends: the cell
    donates at most what it holds, the inventory accepts at most the room
    left under ``max_inventory`` — an empty cell or a full pack is a cheap
    no-op. Shared cells stay deterministic through slot order. Returns the
    total moved (the caller adds it to ``World.wood_gathered``).
    """
    moved = np.float32(0.0)
    for i in range(alive.size):
        if not alive[i]:
            continue
        budget = gather_rate * actions[i, 5]
        room = max_inventory - inventory[i]
        if budget <= 0.0 or room <= 0.0:
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
        take = wood[cy, cx]
        if take > budget:
            take = budget
        if take > room:
            take = room
        if take > 0.0:
            wood[cy, cx] = wood[cy, cx] - take
            inventory[i] = inventory[i] + take
            moved = moved + take
    return moved


@njit(cache=True)
def _build(
    pos: np.ndarray,
    energy: np.ndarray,
    inventory: np.ndarray,
    alive: np.ndarray,
    actions: np.ndarray,
    creature_id: np.ndarray,
    struct_pos: np.ndarray,
    struct_type: np.ndarray,
    struct_fuel: np.ndarray,
    struct_store: np.ndarray,
    struct_owner: np.ndarray,
    struct_alive: np.ndarray,
    struct_free_list: np.ndarray,
    struct_free_count: np.int32,
    struct_grid: np.ndarray,
    cell_w: np.float32,
    cell_h: np.float32,
    gw: np.int32,
    gh: np.int32,
    fire_wood_cost: np.float32,
    fire_energy_cost: np.float32,
    house_wood_cost: np.float32,
    house_energy_cost: np.float32,
    fire_initial_fuel: np.float32,
    houses_enabled: bool,
    house_fire_radius: np.float32,
    width: np.float32,
    height: np.float32,
) -> int:
    """Serial: construction and refuelling (stage 17; stage 22 rewrite).

    A gate decoded above 0.5 (raw logit positive) is a *decision*; physics
    then demands payment. Rules per cell:

    - ``build_fire`` on an existing live fire refuels it: carried wood
      converts 1:1 into fuel up to ``fire_initial_fuel``, free of energy
      cost (stocking the pile is effort enough).
    - ``build_fire`` on an empty cell (or a dead occupant's cell — the
      grid is one rebuild stale) raises a fire at full lump price.
    - ``build_house`` (only when ``houses_enabled``) raises a house in
      **increments** (stage 22): each attempt delivers
      ``dW = min(carried wood, wood still needed)`` and pays
      ``house_energy_cost * dW / house_wood_cost`` energy — all-or-nothing
      *per delivery*, never per house. Completeness ``u`` lives in
      ``struct_store`` (0..1); the shelter stamped on the ground is
      ``cover_reduction * u``, so a half-built house half-protects. If
      ``house_fire_radius > 0`` every delivery additionally requires a
      live fire within that radius of the building cell — the fire is the
      step before the house. A house slot is only created when a delivery
      actually completes (no empty shells from failed attempts).

    The fire gate is evaluated first and writes ``struct_grid`` on
    creation, so same-tick duplicates see the fresh structure. Slot order
    is ascending and no random numbers are drawn. Returns the new
    ``struct_free_count``.
    """
    hw = width * np.float32(0.5)
    hh = height * np.float32(0.5)
    for i in range(alive.size):
        if not alive[i]:
            continue
        gate_fire = actions[i, 6]
        gate_house = actions[i, 7]
        if gate_fire <= 0.5 and gate_house <= 0.5:
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
        if gate_fire > 0.5:
            slot = struct_grid[cy, cx]
            if slot >= 0 and struct_alive[slot] and struct_type[slot] == 0:
                want = fire_initial_fuel - struct_fuel[slot]
                if want > 0.0 and inventory[i] > 0.0:
                    take = inventory[i] if inventory[i] < want else want
                    inventory[i] = inventory[i] - take
                    struct_fuel[slot] = struct_fuel[slot] + take
            elif (slot < 0 or not struct_alive[slot]) and struct_free_count > 0:
                if (
                    inventory[i] >= fire_wood_cost
                    and energy[i] >= fire_energy_cost
                ):
                    s = struct_free_list[struct_free_count - 1]
                    struct_free_count = struct_free_count - 1
                    struct_pos[s, 0] = pos[i, 0]
                    struct_pos[s, 1] = pos[i, 1]
                    struct_type[s] = 0
                    struct_fuel[s] = fire_initial_fuel
                    struct_store[s] = 0.0
                    struct_owner[s] = creature_id[i]
                    struct_alive[s] = True
                    struct_grid[cy, cx] = s  # same tick sees the new fire
                    inventory[i] = inventory[i] - fire_wood_cost
                    energy[i] = energy[i] - fire_energy_cost
        if gate_house > 0.5 and houses_enabled:
            # re-read: the fire branch may have just built on this cell
            slot = struct_grid[cy, cx]
            s = -1
            if slot >= 0 and struct_alive[slot] and struct_type[slot] == 1:
                s = slot  # continue the existing (incomplete) house
            elif (slot < 0 or not struct_alive[slot]) and struct_free_count > 0:
                s = -2  # candidate: only materialise once a delivery pays
            # s == -1 means a live fire owns the cell: no delivery at all
            if s >= 0 or s == -2:
                u = struct_store[s] if s >= 0 else np.float32(0.0)
                wood_need = house_wood_cost * (np.float32(1.0) - u)
                d_w = inventory[i] if inventory[i] < wood_need else wood_need
                ok = d_w > 0.0 and energy[i] >= house_energy_cost * (
                    d_w / house_wood_cost
                )
                if ok and house_fire_radius > 0.0:
                    # a live fire must burn within radius of this cell
                    ok = False
                    for t in range(struct_pos.shape[0]):
                        if not struct_alive[t] or struct_type[t] != 0:
                            continue
                        dx = struct_pos[t, 0] - pos[i, 0]
                        dy = struct_pos[t, 1] - pos[i, 1]
                        if dx > hw:
                            dx = dx - width
                        elif dx < -hw:
                            dx = dx + width
                        if dy > hh:
                            dy = dy - height
                        elif dy < -hh:
                            dy = dy + height
                        if dx * dx + dy * dy <= house_fire_radius * house_fire_radius:
                            ok = True
                            break
                if ok:
                    if s == -2:  # first delivery: raise the structure
                        s = struct_free_list[struct_free_count - 1]
                        struct_free_count = struct_free_count - 1
                        struct_pos[s, 0] = pos[i, 0]
                        struct_pos[s, 1] = pos[i, 1]
                        struct_type[s] = 1
                        struct_fuel[s] = 0.0
                        struct_store[s] = 0.0
                        struct_owner[s] = creature_id[i]
                        struct_alive[s] = True
                        struct_grid[cy, cx] = s
                    du = d_w / house_wood_cost
                    u = u + du
                    if u > 1.0:
                        u = np.float32(1.0)
                    struct_store[s] = u
                    inventory[i] = inventory[i] - d_w
                    energy[i] = energy[i] - house_energy_cost * du
    return struct_free_count


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


def phase_grow_wood(world: World) -> None:
    """Stage 17 wood: the same refill law as food, without the seasons.

    ``wood`` moves toward ``wood_field * terrain`` at a constant
    ``wood_growth_rate`` (clamped to [0, 1]) — light and season modulate
    food only, so a wood line in a stats plot is comparable across ticks.
    Water cells target 0 exactly, same as food. Rate 0 skips the kernel.
    """
    cfg = world.config
    if cfg.wood_growth_rate <= 0.0:
        return
    rate = min(cfg.wood_growth_rate, 1.0)
    _grow_food(world.wood, world.wood_field, world.terrain, np.float32(rate))


def phase_decay_meat(world: World) -> None:
    cfg = world.config
    if cfg.meat_decay > 0.0:
        _decay_meat(world.meat, np.float32(cfg.meat_decay))


def phase_decay_signal(world: World) -> None:
    cfg = world.config
    if cfg.signal_decay > 0.0:
        _decay_signal(world.signal, np.float32(cfg.signal_decay))


def phase_diffuse_signal(world: World) -> None:
    """Stage 14 pheromone: one Jacobi spread step over the signal field.

    Runs right after decay so decay and diffusion evolve the existing
    field together; the tick's own deposits (emit) come later and are
    only spread by the *next* tick's pass. ``signal_diffuse = 0`` is an
    exact no-op.
    """
    cfg = world.config
    if cfg.signal_diffuse > 0.0:
        _diffuse_signal(world.signal, world.signal_buf, np.float32(cfg.signal_diffuse))
        np.copyto(world.signal, world.signal_buf)


def phase_decay_structures(world: World) -> None:
    cfg = world.config
    if cfg.fire_fuel_decay <= 0.0 and cfg.house_rot_rate <= 0.0:
        return
    struct_free_count, _n_died = _decay_structures(
        world.struct_type,
        world.struct_fuel,
        world.struct_store,
        world.struct_alive,
        world.struct_free_list,
        np.int32(world.struct_free_count),
        np.float32(cfg.fire_fuel_decay),
        np.float32(cfg.house_rot_rate),
    )
    world.struct_free_count = int(struct_free_count)


def phase_age_and_metabolize(world: World) -> None:
    cfg = world.config
    _age_and_metabolize(
        world.energy,
        world.age,
        world.alive,
        world.vel,
        world.pos,
        world.inventory,
        np.float32(cfg.metabolic_cost),
        np.float32(cfg.move_cost),
        world.terrain,
        np.float32(1.0 / world.cell_w),
        np.float32(1.0 / world.cell_h),
        np.float32(cfg.water_move_cost),
        np.float32(cfg.senescence_rate),
        np.float32(cfg.carry_weight),
        np.float32(cfg.max_inventory),
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
        world.signal,
        world.wood,
        world.cell_counts,
        world.cell_diet,
        world.cell_tag_re,
        world.cell_tag_im,
        world.hue_re,
        world.hue_im,
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
        np.float32(cfg.signal_capacity),
        np.float32(cfg.wood_capacity),
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
        world.dw,
        cfg.plasticity_active,  # stage 21: read effective synapses genome + dw
        np.int32(cfg.sensor_input_dim),
        np.int32(cfg.hidden_size),
        np.int32(cfg.N_OUTPUTS),
    )


def phase_plasticity(world: World) -> None:
    """Stage 21: local Oja on ``dw``, energy cost, hidden-state commit.

    Runs right after think so ``hidden_buf`` holds this tick's response and
    ``hidden_prev`` still holds the previous one (the recurrent presynaptic
    side). The interval gate decides whether the rule fires at all; the
    commit of ``hidden_prev`` happens unconditionally — that is the job
    think gave up in this stage. See :func:`primordia.brain.plasticity`.
    """
    cfg = world.config
    mode_gene = cfg.plasticity_mode == "gene"
    eta_fixed = np.float32(
        cfg.plasticity_rate if cfg.plasticity_mode == "fixed" else 0.0
    )  # gene reads the per-creature trait; off must never move dw (config
    # validation already forbids rate>0 outside "fixed" — belt and braces)
    plasticity(
        world.sensor_buf,
        world.hidden_buf,
        world.hidden_prev,
        world.energy,
        world.e_last_plastic,
        world.alive,
        world.genome,
        world.dw,
        eta_fixed,
        mode_gene,
        np.int32(cfg.plasticity_gene_offset),
        cfg.plasticity_credit,
        np.float32(cfg.plasticity_credit_strength),
        (world.tick % cfg.plasticity_interval) == 0,
        np.float32(cfg.plasticity_decay),
        np.float32(cfg.plasticity_clip),
        np.float32(cfg.plasticity_cost),
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


def phase_emit_signal(world: World) -> None:
    cfg = world.config
    gh, gw = world.signal.shape
    emitted = _emit_signal(
        world.pos,
        world.energy,
        world.alive,
        world.signal,
        world.actions,
        world.cell_w,
        world.cell_h,
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.emit_rate),
        np.float32(cfg.signal_emit_cost),
    )
    world.signal_emitted += float(emitted)


def phase_donate(world: World) -> None:
    cfg = world.config
    world.donate_to.fill(-1)  # fresh map: render sees only this tick's gifts
    if cfg.donate_rate <= 0.0:
        # default off: no kernel call at all; the scratch is reset so
        # stats never reads last tick's values (same as a quiet tick)
        world.donate_align_sum = 0.0
        world.donate_pairs = 0
        return
    gh, gw = world.cell_counts.shape
    # Cells to scan per axis so the largest allowed gift range can never
    # be missed across cell edges (same reach formula as the bite phase).
    reach_x = int(cfg.donate_range // float(world.cell_w)) + 1
    reach_y = int(cfg.donate_range // float(world.cell_h)) + 1
    reach_x = min(reach_x, gw // 2)
    reach_y = min(reach_y, gh // 2)
    total, align_sum, pairs = _donate(
        world.pos,
        world.energy,
        world.alive,
        world.actions,
        world.hue_re,
        world.hue_im,
        world.cell_offsets,
        world.cell_slots,
        world.donate_to,
        np.float32(1.0 / world.cell_w),
        np.float32(1.0 / world.cell_h),
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.width),
        np.float32(cfg.height),
        np.int32(reach_x),
        np.int32(reach_y),
        np.float32(cfg.donate_rate),
        np.float32(cfg.donate_range),
    )
    world.energy_donated += float(total)
    world.donate_align_sum = float(align_sum)
    world.donate_pairs = int(pairs)


def phase_gather(world: World) -> None:
    cfg = world.config
    if cfg.gather_rate <= 0.0:
        return
    gh, gw = world.wood.shape
    moved = _gather(
        world.pos,
        world.inventory,
        world.alive,
        world.actions,
        world.wood,
        world.cell_w,
        world.cell_h,
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.gather_rate),
        np.float32(cfg.max_inventory),
    )
    world.wood_gathered += float(moved)


def phase_build(world: World) -> None:
    cfg = world.config
    gh, gw = world.struct_grid.shape
    world.struct_free_count = int(
        _build(
            world.pos,
            world.energy,
            world.inventory,
            world.alive,
            world.actions,
            world.creature_id,
            world.struct_pos,
            world.struct_type,
            world.struct_fuel,
            world.struct_store,
            world.struct_owner,
            world.struct_alive,
            world.struct_free_list,
            np.int32(world.struct_free_count),
            world.struct_grid,
            world.cell_w,
            world.cell_h,
            np.int32(gw),
            np.int32(gh),
            np.float32(cfg.fire_wood_cost),
            np.float32(cfg.fire_energy_cost),
            np.float32(cfg.house_wood_cost),
            np.float32(cfg.house_energy_cost),
            np.float32(cfg.fire_initial_fuel),
            cfg.houses_enabled,
            np.float32(cfg.house_fire_radius),
            np.float32(cfg.width),
            np.float32(cfg.height),
        )
    )


def phase_rebuild_struct_fields(world: World) -> None:
    """Restamp occupancy, fire heat and house cover after this tick's
    build/decay — so a fire raised this tick already cooks this tick
    (stage 17: an object is not a signal; compare emit_signal's latency)."""
    world.rebuild_struct_fields()


def phase_eat(world: World) -> None:
    cfg = world.config
    gh, gw = world.food.shape
    meat_eaten, energy_cooked = _eat(
        world.pos,
        world.energy,
        world.alive,
        world.food,
        world.meat,
        world.actions,
        world.struct_fire,
        world.struct_fuel,
        world.cell_w,
        world.cell_h,
        np.int32(gw),
        np.int32(gh),
        np.float32(cfg.eat_rate),
        np.float32(cfg.cook_bonus),
    )
    world.meat_eaten += float(meat_eaten)
    world.energy_cooked += float(energy_cooked)


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
    bite_by_diet: np.ndarray,
    herb_max: np.float32,
    pred_min: np.float32,
    cover: np.ndarray,
    cover_reduction: np.float32,
) -> tuple[int, int]:
    """Serial contact predation: gate * diet budget drains neighbours.

    For every living attacker whose ``bite_rate * gate * diet`` budget is
    positive, scan the CSR cells within ``reach`` of the attacker cell and
    drain each victim inside contact range ``contact_range * (size_a + size_v)``
    (shortest toroidal delta). The victim loses ``land`` and the attacker
    keeps ``bite_efficiency * land``; a carrion share
    ``carrion_yield * (1 - bite_efficiency) * land`` lands on the victim's
    cell (stage 11 — the rest of the "heat" still dissipates), so energy is
    never created.

    House cover (stage 17): a victim standing under cover only suffers the
    uncovered share ``land = take * (1 - cover_reduction * cover)`` — the
    shield absorbs the rest — while the attacker's budget still spends the
    full attempt ``take``, so shelter wastes a blow instead of delaying it
    for free. With no cover the factors are exactly 1.0 and the transfer is
    bit-identical to stage 16. A victim left at zero energy dies here: its
    slot is pushed onto the free-list and counted as a predation death.

    Each successful take is also attributed to the attacker's diet class
    (stage 12.5 measurement: ``bite_by_diet[herb/omn/pred] += 1``, buckets
    from ``ecology.HERB_MAX``/``PRED_MIN``) — a counter the tick never
    reads, so trajectories are unchanged.

    Slot order is fixed and no random numbers are drawn, so the outcome is
    deterministic. Returns ``(free_count, deaths_predation)``.
    """
    n_pred = 0
    hw = width * np.float32(0.5)
    hh = height * np.float32(0.5)
    for i in range(alive.size):
        if not alive[i]:
            continue
        diet_i = genome[i, traits_off + 3]
        budget = bite_rate * actions[i, 2] * diet_i
        if budget <= 0.0:
            continue
        if diet_i < herb_max:
            bucket = 0
        elif diet_i > pred_min:
            bucket = 2
        else:
            bucket = 1
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
                shield = np.float32(1.0) - cover_reduction * cover[iy, ix]
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
                    land = take * shield  # full attempt is spent below
                    bite_by_diet[bucket] = bite_by_diet[bucket] + 1
                    energy[j] = energy[j] - land
                    energy[i] = energy[i] + bite_efficiency * land
                    # the dissipating share lands as carrion on the victim cell
                    meat[iy, ix] = (
                        meat[iy, ix] + carrion_yield * (np.float32(1.0) - bite_efficiency) * land
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
        world.bite_by_diet,
        np.float32(HERB_MAX),
        np.float32(PRED_MIN),
        world.cover,
        np.float32(cfg.cover_reduction),
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
        world.inventory,
        world.dw,
        world.e_last_plastic,
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
        np.float32(cfg.plasticity_eta_std),
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
        np.float32(cfg.birth_spread),
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
    ("grow_wood", phase_grow_wood),
    ("decay_meat", phase_decay_meat),
    ("decay_signal", phase_decay_signal),
    ("diffuse_signal", phase_diffuse_signal),
    ("decay_structures", phase_decay_structures),
    ("age_metabolize", phase_age_and_metabolize),
    ("perceive", phase_perceive),
    ("think", phase_think),
    ("plasticity", phase_plasticity),
    ("apply_actions", phase_apply_actions),
    ("emit_signal", phase_emit_signal),
    ("donate", phase_donate),
    ("gather", phase_gather),
    ("build", phase_build),
    ("rebuild_struct_fields", phase_rebuild_struct_fields),
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
