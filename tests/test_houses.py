"""Stage 22 tests: houses raised in wood deliveries (incremental build).

Physics and contract only: payment is all-or-nothing *per delivery*, cover
scales with completeness u, unfinished houses rot away, a fire gate can
guard the work, carrying wood is dearer than holding it, and the two
spawn knobs (build bias, birth spread) move the world only when asked.
No test asserts that building pays off — that is the experiment's job.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.genome import _reproduce
from primordia.world import World


def make_config(**overrides: object) -> Config:
    overrides.setdefault("houses_enabled", True)
    # cheap totals so a test pack (max_inventory 4) can finish a house
    overrides.setdefault("house_wood_cost", 4.0)
    overrides.setdefault("house_energy_cost", 20.0)
    return replace(Config(), **overrides)


def placed(**overrides: object) -> tuple[World, int]:
    cfg = make_config(
        max_creatures=8,
        initial_creatures=2,
        width=100.0,
        height=100.0,
        **overrides,
    )
    w = World(cfg, seed=3)
    slots = np.flatnonzero(w.alive)
    i = int(slots[0])
    w.pos[i] = (50.0, 50.0)
    w.actions.fill(0.0)
    return w, i


def cell_of(w: World, i: int) -> tuple[int, int]:
    return int(w.pos[i, 1] / w.cell_h), int(w.pos[i, 0] / w.cell_w)


def alive_houses(w: World) -> np.ndarray:
    return np.flatnonzero(w.struct_alive & (w.struct_type == 1))


# --- config -----------------------------------------------------------------


def test_config_validation() -> None:
    for bad in (
        {"house_wood_cost": 0.0},
        {"house_wood_cost": -1.0},
        {"house_energy_cost": -1.0},
        {"house_rot_rate": -0.1},
        {"house_rot_rate": 1.0},
        {"house_fire_radius": -1.0},
        {"carry_weight": -1.0},
        {"birth_spread": -1.0},
        {"build_house_bias": 5.0},
    ):
        with pytest.raises(ValueError):
            make_config(**bad)


def test_defaults_are_quiet() -> None:
    cfg = Config()
    assert cfg.houses_enabled is False
    assert cfg.house_rot_rate == pytest.approx(0.01)
    assert cfg.house_fire_radius == 0.0
    assert cfg.carry_weight == 0.0
    assert cfg.build_house_bias == 0.0
    assert cfg.birth_spread == 0.0


# --- incremental build ------------------------------------------------------


def test_disabled_is_a_noop() -> None:
    w, i = placed(houses_enabled=False)
    w.inventory[i] = 4.0
    w.energy[i] = 100.0
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    assert alive_houses(w).size == 0
    assert float(w.inventory[i]) == 4.0  # untouched
    assert float(w.energy[i]) == 100.0


def test_one_delivery_pays_per_fraction() -> None:
    w, i = placed()
    cfg = w.config
    w.inventory[i] = 1.0  # a quarter of the house
    w.energy[i] = 100.0
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    houses = alive_houses(w)
    assert houses.size == 1
    u = float(w.struct_store[int(houses[0])])
    assert u == pytest.approx(0.25, rel=1e-6)
    assert float(w.inventory[i]) == pytest.approx(0.0, abs=1e-6)
    # energy charged for the delivered fraction only
    assert float(w.energy[i]) == pytest.approx(100.0 - 20.0 * 0.25, rel=1e-6)


def test_failed_delivery_creates_no_shell() -> None:
    w, i = placed()
    w.inventory[i] = 2.0
    w.energy[i] = 1.0  # not enough for the 10 energy a half costs
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    assert alive_houses(w).size == 0
    assert float(w.inventory[i]) == 2.0  # nothing taken
    assert float(w.energy[i]) == 1.0


def test_deliveries_continue_the_same_house() -> None:
    w, i = placed()
    w.inventory[i] = 1.0
    w.energy[i] = 100.0
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    s0 = int(alive_houses(w)[0])
    w.inventory[i] = 1.0
    w.energy[i] = 100.0
    step.phase_build(w)
    houses = alive_houses(w)
    assert houses.size == 1  # one house, two deliveries
    assert int(houses[0]) == s0
    assert float(w.struct_store[s0]) == pytest.approx(0.5, rel=1e-6)
    w.inventory[i] = 2.0  # finish it
    w.energy[i] = 100.0
    step.phase_build(w)
    assert float(w.struct_store[s0]) == pytest.approx(1.0, rel=1e-6)
    # a finished house keeps taking no more deliveries
    w.inventory[i] = 1.0
    w.energy[i] = 100.0
    step.phase_build(w)
    assert float(w.struct_store[s0]) == pytest.approx(1.0, abs=1e-6)
    assert float(w.inventory[i]) == pytest.approx(1.0, abs=1e-6)


def test_fire_owns_the_cell_against_houses() -> None:
    w, i = placed()
    w.inventory[i] = 4.0
    w.energy[i] = 100.0
    w.actions[i, 6] = 1.0  # both gates open: fire is evaluated first
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    assert alive_houses(w).size == 0
    fires = np.flatnonzero(w.struct_alive & (w.struct_type == 0))
    assert fires.size == 1
    # only the fire's lump price left the pack
    assert float(w.inventory[i]) == pytest.approx(4.0 - 3.0, rel=1e-6)


def test_fire_gate_blocks_without_a_fire_in_range() -> None:
    w, i = placed(house_fire_radius=3.0)
    w.inventory[i] = 4.0
    w.energy[i] = 100.0
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    assert alive_houses(w).size == 0
    assert float(w.inventory[i]) == 4.0


def test_fire_gate_admits_a_delivery_with_a_fire_nearby() -> None:
    # cell_size is 5.0, so a radius of 3 could never reach a *free* cell:
    # the gate needs room for "next to the fire but not on it"
    w, i = placed(house_fire_radius=6.0)
    w.inventory[i] = 4.0
    w.energy[i] = 100.0
    # raise the guard fire (the fire branch runs first and owns the
    # builder's cell — a house can never share it), then step one cell
    # away so the delivery lands within radius of the fire
    w.actions[i, 6] = 1.0
    step.phase_build(w)
    fires = np.flatnonzero(w.struct_alive & (w.struct_type == 0))
    assert fires.size == 1
    w.pos[i] = (56.0, 50.0)  # next cell over, 6 world units from the fire
    w.inventory[i] = 4.0  # the fire's lump price consumed part of the pack
    w.actions.fill(0.0)
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    houses = alive_houses(w)
    assert houses.size == 1
    assert float(w.struct_store[int(houses[0])]) == pytest.approx(1.0, rel=1e-6)


# --- cover, rot, stamp ------------------------------------------------------


def test_cover_stamps_completeness() -> None:
    w, i = placed(cover_reduction=0.5)
    w.inventory[i] = 2.0
    w.energy[i] = 100.0
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    step.phase_rebuild_struct_fields(w)
    cy, cx = cell_of(w, i)
    # the grid carries the raw completeness; the bite kernel applies
    # the shield as (1 - cover_reduction * cover), so a half-built
    # house under reduction 0.5 shields a quarter of the bite
    assert float(w.cover[cy, cx]) == pytest.approx(0.5, rel=1e-6)


def test_incomplete_house_rots_away() -> None:
    w, i = placed(house_rot_rate=0.1, fire_fuel_decay=0.0)
    w.inventory[i] = 1.0
    w.energy[i] = 100.0
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    s = int(alive_houses(w)[0])
    u0 = float(w.struct_store[s])
    step.phase_decay_structures(w)
    u1 = float(w.struct_store[s])
    assert u1 < u0 and u1 > 0.0
    # u_{t+1} = u - rot*(1-u): geometric approach to zero, then death
    for _ in range(200):
        if not w.struct_alive[s]:
            break
        step.phase_decay_structures(w)
    assert not w.struct_alive[s]
    assert float(w.struct_store[s]) == 0.0
    assert int(w.struct_free_count) == int(w.config.max_structures)


def test_finished_house_never_rots() -> None:
    w, i = placed(house_rot_rate=0.5, fire_fuel_decay=0.0)
    w.inventory[i] = 4.0
    w.energy[i] = 100.0
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    s = int(alive_houses(w)[0])
    assert float(w.struct_store[s]) == pytest.approx(1.0, rel=1e-6)
    for _ in range(50):
        step.phase_decay_structures(w)
    assert w.struct_alive[s]
    assert float(w.struct_store[s]) == pytest.approx(1.0, abs=1e-6)


# --- stats ------------------------------------------------------------------


def test_house_progress_column() -> None:
    from primordia.stats import COLUMNS, StatsRecorder

    assert "house_progress" in COLUMNS
    cfg = make_config(max_creatures=8, initial_creatures=2, width=100.0, height=100.0)
    w = World(cfg, seed=1)
    rec = StatsRecorder(capacity=10, sample_every=1)
    rec.record(w)
    assert float(rec.to_dict()["house_progress"][0]) == 0.0  # no houses yet


# --- spawn knobs ------------------------------------------------------------


def test_build_bias_shifts_only_the_house_logit() -> None:
    cfg = make_config(build_house_bias=-1.5, max_creatures=8, initial_creatures=4)
    w = World(cfg, seed=5)
    b2_off = (
        cfg.hidden_size * cfg.sensor_input_dim
        + cfg.hidden_size
        + cfg.N_OUTPUTS * cfg.hidden_size
    )
    slots = np.flatnonzero(w.alive)
    plain = make_config(max_creatures=8, initial_creatures=4)
    w0 = World(plain, seed=5)
    # every other column is bit-identical to the unbiased world
    mask = np.ones(cfg.genome_size, dtype=bool)
    mask[b2_off + 7] = False
    assert np.array_equal(w.genome[:, mask], w0.genome[:, mask])
    # every living spawn carries the shifted house logit; the shift is
    # a constant, so the mutation stream that drew w0's column shifted
    # identically (dead slots never get the bias)
    assert np.allclose(
        w.genome[slots, b2_off + 7],
        w0.genome[slots, b2_off + 7] - 1.5,
        atol=1e-6,
    )


def test_bias_zero_draws_no_new_stream() -> None:
    cfg = make_config(max_creatures=8, initial_creatures=4)
    w = World(cfg, seed=9)
    w2 = World(cfg, seed=9)
    assert np.array_equal(w.genome, w2.genome)


def test_birth_spread_jitters_the_child() -> None:
    cfg = make_config(birth_spread=2.0, max_creatures=8, initial_creatures=2,
                      width=100.0, height=100.0)
    w = World(cfg, seed=4)
    slots = np.flatnonzero(w.alive)
    i, j = int(slots[0]), int(slots[1])
    # reproduce_threshold defaults to 100: both parents must clear it
    w.energy[i] = 200.0
    w.energy[j] = 200.0
    w.pos[i] = (30.0, 30.0)
    w.pos[j] = (30.0, 30.0)
    step.phase_rebuild_counts(w)
    moved = False
    for _ in range(10):
        w.energy[i] = 200.0
        w.energy[j] = 200.0
        before = int(w.next_id)
        step.phase_reproduce(w)
        if int(w.next_id) > before:
            slot = np.flatnonzero(w.creature_id == before)
            assert slot.size == 1
            p = w.pos[slot[0]]
            if abs(float(p[0]) - 30.0) > 1e-6 or abs(float(p[1]) - 30.0) > 1e-6:
                moved = True
                break
    assert moved


def test_birth_spread_zero_keeps_the_child_on_the_parent() -> None:
    cfg = make_config(birth_spread=0.0, max_creatures=8, initial_creatures=2,
                      width=100.0, height=100.0)
    w = World(cfg, seed=4)
    slots = np.flatnonzero(w.alive)
    i, j = int(slots[0]), int(slots[1])
    w.energy[i] = 200.0
    w.energy[j] = 200.0
    w.pos[i] = (30.0, 30.0)
    w.pos[j] = (30.0, 30.0)
    step.phase_rebuild_counts(w)
    step.phase_reproduce(w)
    child = int(w.next_id) - 1
    slot = np.flatnonzero(w.creature_id == child)
    assert slot.size == 1
    assert float(w.pos[slot[0], 0]) == 30.0
    assert float(w.pos[slot[0], 1]) == 30.0


# --- carrying ---------------------------------------------------------------
#
# The drain is metabolic_cost * exp(senescence_rate * age) + move_cost *
# |vel|^2 * (1 + carry_weight * fill), plus the water multiplier. These
# tests pin only the carry term, with defaults for the rest.
#
# age increments before the wear is computed, so age = 1 on the first tick.


def _drain(cfg: Config, fill: float) -> float:
    metabolic = cfg.metabolic_cost * np.exp(cfg.senescence_rate * 1.0)
    move = cfg.move_cost * 1.0 * (1.0 + cfg.carry_weight * fill)
    return metabolic + move


def test_carry_weight_scales_movement_cost() -> None:
    w, i = placed(carry_weight=1.0)
    cfg = w.config
    w.inventory[i] = cfg.max_inventory  # fully laden
    w.energy[i] = 100.0
    w.vel[i] = (1.0, 0.0)
    step.phase_age_and_metabolize(w)
    assert float(w.energy[i]) == pytest.approx(100.0 - _drain(cfg, 1.0), rel=1e-4)


def test_carry_weight_zero_is_the_old_drain() -> None:
    w, i = placed(carry_weight=0.0)
    cfg = w.config
    w.inventory[i] = cfg.max_inventory
    w.energy[i] = 100.0
    w.vel[i] = (1.0, 0.0)
    step.phase_age_and_metabolize(w)
    assert float(w.energy[i]) == pytest.approx(100.0 - _drain(cfg, 1.0), rel=1e-4)
    # with the knob at 0 the inventory is invisible to the drain
    assert float(w.energy[i]) == pytest.approx(
        100.0 - (cfg.metabolic_cost * np.exp(cfg.senescence_rate) + cfg.move_cost),
        rel=1e-4,
    )


def test_carry_scales_with_fill_not_only_full() -> None:
    w, i = placed(carry_weight=1.0)
    cfg = w.config
    w.inventory[i] = cfg.max_inventory * 0.25
    w.energy[i] = 100.0
    w.vel[i] = (1.0, 0.0)
    step.phase_age_and_metabolize(w)
    assert float(w.energy[i]) == pytest.approx(100.0 - _drain(cfg, 0.25), rel=1e-4)


# --- determinism / conservation ---------------------------------------------


def test_build_is_deterministic() -> None:
    def run() -> tuple[float, float]:
        w, i = placed()
        w.inventory[i] = 1.5
        w.energy[i] = 50.0
        w.actions[i, 7] = 1.0
        step.phase_build(w)
        houses = alive_houses(w)
        u = float(w.struct_store[int(houses[0])]) if houses.size else -1.0
        return u, float(w.energy[i])

    assert run() == run()


def test_energy_never_negative_from_a_delivery() -> None:
    w, i = placed()
    w.inventory[i] = 4.0
    w.energy[i] = 9.99  # short of the 20 a full house costs
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    # partial delivery the energy can afford: dE = 20 * dW/4 <= 9.99
    houses = alive_houses(w)
    if houses.size:  # a partial house is legal; negative energy is not
        assert float(w.energy[i]) >= 0.0
    else:
        assert float(w.energy[i]) == pytest.approx(9.99, rel=1e-6)


def test_save_load_restores_completeness() -> None:
    from primordia.io import load_world, save_world
    import tempfile, pathlib

    w, i = placed()
    w.inventory[i] = 1.0
    w.energy[i] = 100.0
    w.actions[i, 7] = 1.0
    step.phase_build(w)
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "w.npz"
        save_world(w, str(p))
        w2 = load_world(str(p))
    houses = np.flatnonzero(w2.struct_alive & (w2.struct_type == 1))
    assert houses.size == 1
    assert float(w2.struct_store[int(houses[0])]) == pytest.approx(0.25, rel=1e-6)
