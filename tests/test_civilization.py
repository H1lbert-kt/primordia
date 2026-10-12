"""Stage 17 tests: civilization base — wood, fire, houses.

Physics and contract only: construction is paid in full, cooking never
creates energy (the bonus comes out of the fire's fuel 1:1), shelter
absorbs bites, one structure per cell, structures are deterministic and
the layout pins hold. Whether building *pays off* is the stage-17
experiment's question (principle 1: no fitness, nothing here asserts
that builders outlive non-builders).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.io import load_world, save_world
from primordia.stats import StatsRecorder
from primordia.world import World


def make_config(**overrides: object) -> Config:
    return replace(Config(), **overrides)


def placed(**overrides: object) -> tuple[World, int, int]:
    """Four creatures; slot i parked at (50, 50), slot j at (10, 10).

    Both are controlled (energy, gates) per test; the other two keep
    zero gates, so they are inert no-ops in every serial phase.
    """
    cfg = make_config(
        max_creatures=8,
        initial_creatures=4,
        width=100.0,
        height=100.0,
        **overrides,
    )
    w = World(cfg, seed=3)
    slots = np.flatnonzero(w.alive)
    i, j = int(slots[0]), int(slots[1])
    w.pos[i] = (50.0, 50.0)
    w.pos[j] = (10.0, 10.0)
    w.actions.fill(0.0)
    return w, i, j


def cell_of(w: World, i: int) -> tuple[int, int]:
    return int(w.pos[i, 1] / w.cell_h), int(w.pos[i, 0] / w.cell_w)


# --- layout and config -----------------------------------------------------


def test_layout_pins() -> None:
    cfg = Config()
    assert Config.N_OUTPUTS == 8  # accel, turn, eat, emit, donate, gather,
    # build_fire, build_house
    assert cfg.sensor_input_dim == 35  # stage 17 added `smell wood` at 3n+13
    assert cfg.hidden_size == 9
    assert cfg.brain_params == 485  # 9*36 + 8*10 + 9*9 (W1, W2, W_rec)
    assert cfg.genome_size == 490
    assert 100 <= cfg.brain_params <= 500
    names = [name for name, _ in step.PHASES]
    assert len(names) == 21
    assert names.index("grow_wood") == names.index("grow_food") + 1
    assert names.index("decay_structures") == names.index("diffuse_signal") + 1
    # gather -> build -> rebuild -> eat: a fire built this tick cooks now
    assert names.index("gather") < names.index("build")
    assert names.index("build") < names.index("rebuild_struct_fields")
    assert names.index("rebuild_struct_fields") < names.index("eat")


def test_civilization_config_validation() -> None:
    bad_params: list[dict[str, object]] = [
        {"wood_capacity": 0.0},
        {"wood_growth_rate": 1.5},
        {"wood_growth_rate": -0.1},
        {"initial_wood": -0.1},
        {"wood_patch_amplitude": 1.5},
        {"wood_patch_scale": 0.0},
        {"gather_rate": -1.0},
        {"max_inventory": -1.0},
        {"fire_wood_cost": -1.0},
        {"fire_energy_cost": -1.0},
        {"house_wood_cost": -1.0},
        {"house_energy_cost": -1.0},
        # affordability: inventory is the only wood depot, so a price above
        # the pack size is unbuildable (the stage-17 plan shipped 6/4)
        {"house_wood_cost": 6.0},
        {"fire_wood_cost": 5.0},
        {"fire_initial_fuel": -1.0},
        {"fire_fuel_decay": -1.0},
        {"fire_radius": -1.0},
        {"house_radius": -1.0},
        {"cook_bonus": 0.5},  # 1.0 = cooking off; below would be a discount
        {"cover_reduction": 1.5},
        {"max_structures": -1},
    ]
    for bad in bad_params:
        with pytest.raises(ValueError):
            replace(Config(), **bad)


# --- wood field and the gather gate ---------------------------------------


def test_wood_grows_toward_field_and_rate_zero_is_noop() -> None:
    # flat terrain (stage 10 off): the growth target is wood_field exactly
    cfg = make_config(wood_patch_amplitude=0.0, terrain_amplitude=0.0)
    w = World(cfg, seed=5)
    for _ in range(100):
        step.phase_grow_wood(w)
    wood_mean = float(w.wood.mean(dtype=np.float64))
    # 2.0 * (1 - 0.99**100) ~= 1.27 with the default rate and capacity
    assert 1.0 < wood_mean <= cfg.wood_capacity
    assert np.all(w.wood <= w.wood_field + 1e-6)

    cfg0 = make_config(wood_growth_rate=0.0)
    w0 = World(cfg0, seed=5)
    for _ in range(50):
        step.phase_grow_wood(w0)
    assert np.all(w0.wood == cfg0.initial_wood)  # fast path: no RNG, no drift


def test_gather_moves_wood_at_the_budget() -> None:
    w, i, _j = placed()
    cfg = w.config
    iy, ix = cell_of(w, i)
    w.wood[iy, ix] = cfg.wood_capacity
    e0 = float(w.energy[i])
    w.actions[i, 5] = 1.0  # gather gate fully open
    step.phase_gather(w)

    moved = cfg.gather_rate  # budget = rate * gate, cell has plenty
    assert float(w.inventory[i]) == pytest.approx(moved, rel=1e-6)
    assert float(w.wood[iy, ix]) == pytest.approx(cfg.wood_capacity - moved, rel=1e-6)
    assert float(w.wood_gathered) == pytest.approx(moved, rel=1e-6)
    assert float(w.energy[i]) == e0  # hauling is not billed (gate cost only)


def test_gather_stops_at_the_pack_limit() -> None:
    w, i, _j = placed()
    cfg = w.config
    iy, ix = cell_of(w, i)
    w.wood[iy, ix] = cfg.wood_capacity
    w.inventory[i] = cfg.max_inventory - 0.1
    w.actions[i, 5] = 1.0
    step.phase_gather(w)

    assert float(w.inventory[i]) == pytest.approx(cfg.max_inventory, rel=1e-6)
    assert float(w.wood[iy, ix]) == pytest.approx(cfg.wood_capacity - 0.1, rel=1e-6)


def test_gather_rate_zero_is_noop() -> None:
    w, i, _j = placed(gather_rate=0.0)
    cfg = w.config
    iy, ix = cell_of(w, i)
    w.wood[iy, ix] = cfg.wood_capacity
    w.actions[i, 5] = 1.0
    step.phase_gather(w)  # returns before the kernel

    assert float(w.wood[iy, ix]) == cfg.wood_capacity
    assert float(w.inventory[i]) == 0.0
    assert float(w.wood_gathered) == 0.0


# --- construction ----------------------------------------------------------


def test_build_fire_pays_the_full_price() -> None:
    w, i, _j = placed()
    cfg = w.config
    iy, ix = cell_of(w, i)
    w.inventory[i] = cfg.fire_wood_cost + 1.0
    w.energy[i] = 50.0
    w.actions[i, 6] = 1.0
    step.phase_build(w)

    assert int(w.struct_free_count) == cfg.max_structures - 1
    s = int(w.struct_grid[iy, ix])
    assert s >= 0 and w.struct_alive[s]
    assert int(w.struct_type[s]) == 0
    assert float(w.struct_fuel[s]) == pytest.approx(cfg.fire_initial_fuel)
    assert int(w.struct_owner[s]) == int(w.creature_id[i])
    assert float(w.inventory[i]) == pytest.approx(1.0, rel=1e-6)
    assert float(w.energy[i]) == pytest.approx(50.0 - cfg.fire_energy_cost, rel=1e-6)


@pytest.mark.parametrize(
    "inv,energy",
    [
        (2.9, 50.0),  # short of fire_wood_cost: no debt allowed
        (3.0, 9.9),  # short of fire_energy_cost
    ],
)
def test_build_fire_requires_full_payment(inv: float, energy: float) -> None:
    w, i, _j = placed()
    w.inventory[i] = inv
    w.energy[i] = energy
    w.actions[i, 6] = 1.0
    step.phase_build(w)

    assert not np.any(w.struct_alive)
    assert int(w.struct_free_count) == w.config.max_structures
    assert float(w.inventory[i]) == pytest.approx(inv)
    assert float(w.energy[i]) == pytest.approx(energy)


def test_one_structure_per_cell_fire_blocks_house() -> None:
    w, i, j = placed()
    cfg = w.config
    w.inventory[i] = 100.0
    w.energy[i] = 100.0
    w.actions[i, 6] = 1.0  # both gates open on purpose
    w.actions[i, 7] = 1.0
    step.phase_build(w)

    alive = np.flatnonzero(w.struct_alive)
    assert alive.size == 1
    s = int(alive[0])
    assert int(w.struct_type[s]) == 0  # fire wins the cell, house skipped
    assert float(w.inventory[i]) == pytest.approx(
        100.0 - cfg.fire_wood_cost, rel=1e-6
    )
    assert float(w.energy[i]) == pytest.approx(
        100.0 - cfg.fire_energy_cost, rel=1e-6
    )

    # the second creature builds a house alone on its own (empty) cell
    w.inventory[j] = cfg.house_wood_cost + 1.0
    w.energy[j] = 60.0
    w.actions[j, 7] = 1.0
    step.phase_build(w)
    alive = np.flatnonzero(w.struct_alive)
    assert alive.size == 2
    types = sorted(int(w.struct_type[s]) for s in alive)
    assert types == [0, 1]
    assert int(w.struct_grid[*cell_of(w, j)]) >= 0


def test_refuel_tops_up_fuel_without_energy_cost() -> None:
    w, i, _j = placed()
    cfg = w.config
    w.inventory[i] = cfg.fire_wood_cost + 5.0
    w.energy[i] = 50.0
    w.actions[i, 6] = 1.0
    step.phase_build(w)
    s = int(w.struct_grid[*cell_of(w, i)])
    w.struct_fuel[s] = 1.0  # a fire that has been burning for a while
    e_after_build = float(w.energy[i])
    inv_after_build = float(w.inventory[i])  # 8 - fire_wood_cost = 5

    step.phase_build(w)  # same gate: the live fire is refuelled, not rebuilt

    assert int(w.struct_free_count) == cfg.max_structures - 1  # no second slot
    assert float(w.struct_fuel[s]) == pytest.approx(cfg.fire_initial_fuel, rel=1e-6)
    # stock the pile: only wood moved, energy untouched
    assert float(w.energy[i]) == e_after_build
    assert float(w.inventory[i]) == pytest.approx(inv_after_build - 2.0, rel=1e-6)


def test_fire_burns_out_and_returns_its_slot() -> None:
    w, i, j = placed()
    cfg = w.config
    w.inventory[i] = cfg.fire_wood_cost
    w.energy[i] = 50.0
    w.actions[i, 6] = 1.0
    step.phase_build(w)
    s_fire = int(w.struct_grid[*cell_of(w, i)])

    w.inventory[j] = cfg.house_wood_cost
    w.energy[j] = 60.0
    w.actions[j, 7] = 1.0
    step.phase_build(w)
    s_house = int(w.struct_grid[*cell_of(w, j)])
    assert w.struct_alive[s_fire] and w.struct_alive[s_house]

    w.struct_fuel[s_fire] = 0.001  # one decay step away from going out
    step.phase_decay_structures(w)

    assert not w.struct_alive[s_fire]
    assert w.struct_alive[s_house]  # houses never decay: no rule owns it yet
    assert int(w.struct_free_count) == cfg.max_structures - 1  # fire slot back
    iy, ix = cell_of(w, i)
    step.phase_rebuild_struct_fields(w)
    assert int(w.struct_grid[iy, ix]) == -1  # the dead cell is free again
    assert float(w.fire_heat[iy, ix]) == 0.0


# --- stamped fields --------------------------------------------------------


def test_heat_and_cover_stamp_within_radius() -> None:
    w, i, j = placed()
    cfg = w.config
    s_fire, s_house = 7, 3  # fixed slots: no build, just the stamping rules
    w.struct_pos[s_fire] = (50.0, 50.0)
    w.struct_type[s_fire] = 0
    w.struct_alive[s_fire] = True
    w.struct_pos[s_house] = (10.0, 10.0)
    w.struct_type[s_house] = 1
    w.struct_alive[s_house] = True
    w.rebuild_struct_fields()

    fy, fx = cell_of(w, i)
    hy, hx = cell_of(w, j)
    assert float(w.fire_heat[fy, fx]) == 1.0  # exact cell: distance 0
    assert int(w.struct_fire[fy, fx]) == s_fire  # who pays for cooking here
    assert float(w.cover[hy, hx]) == 1.0
    assert float(w.fire_heat[hy, hx]) == 0.0  # house is not a heat source
    assert float(w.cover[fy, fx]) == 0.0  # fire gives no shelter
    assert float(w.fire_heat[0, 0]) == 0.0  # far outside fire_radius
    assert float(w.cover[0, 0]) == 0.0


def test_overlapping_fires_pinned_to_the_lowest_slot() -> None:
    """Ascending scan: one heated cell always bills the same fire."""
    w, _i, _j = placed()
    w.struct_pos[7] = (48.0, 50.0)  # cell 9
    w.struct_type[7] = 0
    w.struct_alive[7] = True
    w.struct_pos[3] = (52.0, 50.0)  # cell 10, within both radii
    w.struct_type[3] = 0
    w.struct_alive[3] = True
    w.rebuild_struct_fields()

    overlap_y, overlap_x = 10, 10  # cell of (52.5, 50): 4.5 and 0.5 away
    assert float(w.fire_heat[overlap_y, overlap_x]) == 1.0
    assert int(w.struct_fire[overlap_y, overlap_x]) == 3  # min(3, 7)


# --- cooking ---------------------------------------------------------------


def cook_world(fuel: float = 3.0) -> tuple[World, int]:
    """A creature standing on a live fire with food in its mouth's budget."""
    w, i, _j = placed()
    cfg = w.config
    w.inventory[i] = cfg.fire_wood_cost
    w.energy[i] = 50.0
    w.actions[i, 6] = 1.0
    step.phase_build(w)
    s = int(w.struct_grid[*cell_of(w, i)])
    w.struct_fuel[s] = fuel
    step.phase_rebuild_struct_fields(w)
    iy, ix = cell_of(w, i)
    w.food[iy, ix] = 5.0
    w.actions.fill(0.0)
    w.actions[i, 2] = 1.0  # eat gate fully open, budget = eat_rate
    return w, i


def test_cook_bonus_paid_from_fuel() -> None:
    w, i = cook_world()
    cfg = w.config
    iy, ix = cell_of(w, i)
    s = int(w.struct_fire[iy, ix])
    e0 = float(w.energy[i])

    step.phase_eat(w)

    take = cfg.eat_rate  # food is plentiful; budget caps the take
    extra = take * (cfg.cook_bonus - 1.0)
    assert float(w.food[iy, ix]) == pytest.approx(5.0 - take, rel=1e-6)
    assert float(w.energy[i]) == pytest.approx(e0 + take + extra, rel=1e-6)
    assert float(w.struct_fuel[s]) == pytest.approx(3.0 - extra, rel=1e-6)
    assert float(w.energy_cooked) == pytest.approx(extra, rel=1e-6)


def test_cook_never_creates_energy_even_with_starving_fuel() -> None:
    # full fuel: the bonus flows food + fire -> eater
    w, i = cook_world(fuel=3.0)
    iy, ix = cell_of(w, i)
    s = int(w.struct_fire[iy, ix])
    before = float(w.food.sum()) + float(w.energy.sum()) + float(w.struct_fuel.sum())
    step.phase_eat(w)
    after = float(w.food.sum()) + float(w.energy.sum()) + float(w.struct_fuel.sum())
    assert after == pytest.approx(before, rel=1e-6)
    assert float(w.struct_fuel[s]) > 0.0

    # starving fire: pays what it has, then stops (proportional payout)
    w2, i2 = cook_world(fuel=0.1)
    iy2, ix2 = cell_of(w2, i2)
    s2 = int(w2.struct_fire[iy2, ix2])
    e0 = float(w2.energy[i2])
    before2 = (
        float(w2.food.sum()) + float(w2.energy.sum()) + float(w2.struct_fuel.sum())
    )
    step.phase_eat(w2)
    after2 = (
        float(w2.food.sum()) + float(w2.energy.sum()) + float(w2.struct_fuel.sum())
    )
    assert after2 == pytest.approx(before2, rel=1e-6)
    assert float(w2.struct_fuel[s2]) == 0.0  # drained to the last drop
    take = w2.config.eat_rate
    assert float(w2.energy[i2]) == pytest.approx(e0 + take + 0.1, rel=1e-6)


def test_cook_requires_a_fire_and_bonus() -> None:
    cfg = make_config(cook_bonus=1.0)  # explicitly disabled
    w = World(cfg, seed=3)
    i = int(np.flatnonzero(w.alive)[0])
    iy, ix = cell_of(w, i)
    w.food[iy, ix] = 5.0
    w.actions.fill(0.0)
    w.actions[i, 2] = 1.0
    e0 = float(w.energy[i])
    step.phase_eat(w)
    assert float(w.energy[i]) == pytest.approx(e0 + cfg.eat_rate, rel=1e-6)
    assert float(w.energy_cooked) == 0.0


# --- shelter ---------------------------------------------------------------


def test_house_cover_shields_the_victim() -> None:
    cfg = make_config(max_creatures=8, initial_creatures=2, cover_reduction=0.5)

    def run(with_house: bool) -> tuple[World, int, int]:
        w = World(cfg, seed=0)
        a, v = (int(s) for s in np.flatnonzero(w.alive))
        t = cfg.brain_params
        w.pos[a] = (500.0, 500.0)
        w.energy[a] = 10.0
        w.genome[a, t + 1] = 1.0  # size
        w.genome[a, t + 3] = 1.0  # diet: a full predator
        w.actions[a, 2] = 1.0  # bite gate
        w.pos[v] = (501.0, 500.0)  # distance 1 < contact range
        w.energy[v] = 50.0
        w.genome[v, t + 1] = 1.0
        if with_house:
            w.inventory[v] = cfg.house_wood_cost
            w.actions[v, 7] = 1.0
            step.phase_build(w)
            w.actions.fill(0.0)
            w.actions[a, 2] = 1.0  # rebuild the attacker's gate after fill
            w.energy[v] = 50.0  # undo the house payment for a clean measure
            step.phase_rebuild_struct_fields(w)
            iy, ix = int(w.pos[v, 1] / w.cell_h), int(w.pos[v, 0] / w.cell_w)
            assert float(w.cover[iy, ix]) == 1.0
        step.phase_rebuild_counts(w)
        step.phase_bite(w)
        return w, a, v

    w_open, a_open, v_open = run(False)
    w_cover, a_cover, v_cover = run(True)

    # uncovered: the stage-16 transfer, bit for bit (budget 4, land 4)
    assert float(w_open.energy[v_open]) == pytest.approx(46.0, rel=1e-4)
    assert float(w_open.energy[a_open]) == pytest.approx(10.0 + 0.7 * 4.0, rel=1e-4)
    # full cover: the shield absorbs cover_reduction of the blow, but the
    # attacker's budget still spends the full attempt
    assert float(w_cover.energy[v_cover]) == pytest.approx(50.0 - 4.0 * 0.5, rel=1e-4)
    assert float(w_cover.energy[a_cover]) == pytest.approx(10.0 + 0.7 * 2.0, rel=1e-4)


# --- smell slot ------------------------------------------------------------


def test_smell_wood_reads_field_normalized_by_capacity() -> None:
    w, i, _j = placed()
    cfg = w.config
    idx = 3 * cfg.n_rays + 13  # layout pin: slot after `tag`

    w.wood.fill(cfg.wood_capacity)
    step.phase_perceive(w)
    assert float(w.sensor_buf[i, idx]) == pytest.approx(1.0, rel=1e-5)

    w.wood.fill(0.0)
    step.phase_perceive(w)
    assert float(w.sensor_buf[i, idx]) == pytest.approx(0.0, abs=1e-6)


# --- stats -----------------------------------------------------------------


def test_stats_civilization_columns() -> None:
    w, i, _j = placed()
    cfg = w.config

    w.wood.fill(cfg.wood_capacity)
    w.actions[i, 5] = 1.0
    step.phase_gather(w)  # wood_gathered += gather_rate

    w.inventory[i] = cfg.fire_wood_cost
    w.energy[i] = 50.0
    w.actions[i, 6] = 1.0
    step.phase_build(w)
    step.phase_rebuild_struct_fields(w)

    iy, ix = cell_of(w, i)
    w.food[iy, ix] = 2.0
    w.actions.fill(0.0)
    w.actions[i, 2] = 1.0
    step.phase_eat(w)  # cooks: energy_cooked += take * (bonus - 1)

    rec = StatsRecorder(capacity=8, sample_every=1, sample_size=16)
    rec.record(w)
    data = rec.to_dict()
    assert float(data["wood_total"][0]) > 0.0
    assert float(data["wood_gathered"][0]) == pytest.approx(cfg.gather_rate, rel=1e-5)
    assert int(data["n_fire"][0]) == 1
    assert int(data["n_house"][0]) == 0
    take = cfg.eat_rate
    assert float(data["energy_cooked"][0]) == pytest.approx(
        take * (cfg.cook_bonus - 1.0), rel=1e-5
    )


# --- determinism and io ----------------------------------------------------


def test_civilization_is_deterministic_across_sequential_worlds() -> None:
    """Same seed, scripted gather/fire gates: structures land bit-identically.

    The Numba RNG stream is process-global, so the two worlds run in
    sequence (the same rule every multi-world test follows).
    """
    cfg = make_config(
        max_creatures=16,
        initial_creatures=8,
        initial_energy=50.0,
        initial_wood=2.0,  # wood from tick 0: the gates build early
        wood_growth_rate=0.5,
    )
    b2 = (
        cfg.hidden_size * cfg.sensor_input_dim
        + cfg.hidden_size
        + Config.N_OUTPUTS * cfg.hidden_size
    )

    def make() -> World:
        w = World(cfg, seed=42)
        w.genome[:, b2 + 5] = 20.0  # gather gate ~ 1
        w.genome[:, b2 + 6] = 20.0  # build_fire gate ~ 1
        return w

    a = make()
    for _ in range(40):
        step.advance(a)
    b = make()
    for _ in range(40):
        step.advance(b)

    assert int(a.struct_alive.sum()) > 0, "the scripted gates must build"
    for name in (
        "struct_alive",
        "struct_pos",
        "struct_type",
        "struct_fuel",
        "struct_owner",
        "struct_grid",
        "struct_fire",
        "inventory",
        "wood",
        "energy",
    ):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.struct_free_count == b.struct_free_count
    assert a.wood_gathered == b.wood_gathered
    assert a.energy_cooked == b.energy_cooked


def test_save_load_restores_structures(tmp_path: Path) -> None:
    w, i, j = placed()
    cfg = w.config

    w.inventory[i] = cfg.fire_wood_cost
    w.energy[i] = 50.0
    w.actions[i, 6] = 1.0
    step.phase_build(w)

    w.inventory[j] = cfg.house_wood_cost
    w.energy[j] = 60.0
    w.actions[j, 7] = 1.0
    step.phase_build(w)
    step.phase_rebuild_struct_fields(w)

    alive = np.flatnonzero(w.struct_alive)
    assert sorted(int(w.struct_type[s]) for s in alive) == [0, 1]

    path = tmp_path / "civ.npz"
    save_world(w, str(path))
    v = load_world(str(path))

    for name in (
        "struct_alive",
        "struct_pos",
        "struct_type",
        "struct_fuel",
        "struct_store",
        "struct_owner",
        "struct_free_list",
        "struct_grid",  # rebuilt on load: must match the source layout
        "struct_fire",
        "fire_heat",
        "cover",
        "inventory",
        "wood",
        "wood_field",
    ):
        assert np.array_equal(getattr(v, name), getattr(w, name)), name
    assert v.struct_free_count == w.struct_free_count
    assert v.wood_gathered == w.wood_gathered
    assert v.energy_cooked == w.energy_cooked
