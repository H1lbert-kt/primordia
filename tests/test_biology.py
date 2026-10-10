"""Stage-11 tests: senescence (wear), carrion deposits/decay/eat, carrion smell."""

from __future__ import annotations

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.step import advance
from primordia.world import World


def first_alive(w: World) -> int:
    return int(np.flatnonzero(w.alive)[0])


def _solo(**overrides) -> tuple[World, Config]:
    cfg = Config(max_creatures=4, initial_creatures=1, **overrides)
    w = World(cfg, seed=0)
    return w, cfg


# --- senescence ------------------------------------------------------------


def test_senescence_scales_metabolic_bill() -> None:
    """Wear = exp(r * age): the metabolic drain grows with age, not movement."""
    w, cfg = _solo(senescence_rate=0.001, move_cost=0.0, metabolic_cost=0.1)
    slot = first_alive(w)
    e0 = float(w.energy[slot])
    step.phase_age_and_metabolize(w)  # age 0 -> 1
    wear1 = np.exp(0.001 * 1)
    assert float(w.energy[slot]) == pytest.approx(e0 - 0.1 * wear1, rel=1e-6)
    e1 = float(w.energy[slot])
    step.phase_age_and_metabolize(w)  # age 1 -> 2
    wear2 = np.exp(0.001 * 2)
    assert float(w.energy[slot]) == pytest.approx(e1 - 0.1 * wear2, rel=1e-6)


def test_senescence_zero_is_flat_cost() -> None:
    w, cfg = _solo(senescence_rate=0.0, move_cost=0.0, metabolic_cost=0.1)
    slot = first_alive(w)
    e0 = float(w.energy[slot])
    step.phase_age_and_metabolize(w)
    assert float(w.energy[slot]) == pytest.approx(e0 - 0.1, rel=1e-6)


def test_senescence_shrinks_effective_speed() -> None:
    """apply_actions divides top speed by wear: old bodies crawl."""
    from primordia.brain import apply_actions

    w, cfg = _solo(senescence_rate=0.01)
    slot = first_alive(w)
    w.genome[slot, cfg.brain_params] = cfg.max_speed  # trait
    w.age[slot] = 100  # wear = e^1 ~ 2.72
    w.actions[slot] = (9.0, 0.0, 0.0)  # accel logit -> tanh ~= 1
    apply_actions(
        w.actions,
        w.angle,
        w.vel,
        w.energy,
        w.age,
        w.alive,
        w.genome,
        np.float32(cfg.max_turn),
        np.float32(cfg.turn_cost),
        np.int32(cfg.brain_params),
        np.float32(cfg.senescence_rate),
    )
    expected = cfg.max_speed * np.tanh(9.0) / np.exp(0.01 * 100)
    speed = float(np.hypot(w.vel[slot, 0], w.vel[slot, 1]))
    assert speed == pytest.approx(expected, rel=1e-3)
    assert speed < cfg.max_speed * np.tanh(9.0)


def test_senescence_shrinks_vision() -> None:
    """Ray reach is divided by wear: an old creature sees less far."""
    base = dict(
        max_creatures=4,
        initial_creatures=1,
        width=100.0,
        height=100.0,
        cell_size=5.0,
        n_rays=5,
        vision_range=50.0,
        senescence_rate=0.01,
    )

    def food_seen(age: int) -> float:
        cfg = Config(**base)
        w = World(cfg, seed=6)
        slot = first_alive(w)
        w.pos[slot] = (50.0, 50.0)
        w.angle[slot] = 0.0
        w.age[slot] = age
        w.food[:] = 0.0
        # food only at x >= 70: reachable at age 0 (50 units), not at age 100
        # (50 / e^1 ~ 18 units)
        w.food[9:11, 14:20] = cfg.food_capacity
        step.phase_rebuild_counts(w)
        step.phase_perceive(w)
        return float(w.sensor_buf[slot, 0:5].sum())

    assert food_seen(0) > 0.0
    assert food_seen(100) == pytest.approx(0.0)


# --- carrion: deposits -----------------------------------------------------


def test_old_age_deposits_carrion_famine_deposits_nothing() -> None:
    w, cfg = _solo(max_age=10, carrion_yield=0.3)
    slot = first_alive(w)
    w.energy[slot] = 40.0
    w.age[slot] = cfg.max_age
    step.phase_deaths(w)
    assert w.deaths_age == 1
    ix = int(w.pos[slot, 0] / w.cell_w)
    iy = int(w.pos[slot, 1] / w.cell_h)
    assert w.meat[iy, ix] == pytest.approx(0.3 * 40.0, rel=1e-6)

    w, cfg = _solo(max_age=10, carrion_yield=0.3)
    slot = first_alive(w)
    w.energy[slot] = 0.0
    step.phase_deaths(w)
    assert w.deaths_famine == 1
    assert float(w.meat.sum()) == 0.0  # a starved body leaves nothing


def test_predation_kill_leaves_carrion() -> None:
    """The bite's dissipating share lands on the victim's cell."""
    cfg = Config(max_creatures=8, initial_creatures=2, carrion_yield=0.3)
    w = World(cfg, seed=0)
    a, v = np.flatnonzero(w.alive)

    def place(slot: int, x: float, energy: float, diet: float, gate: float) -> None:
        t = cfg.brain_params
        w.pos[slot, 0] = x
        w.pos[slot, 1] = 500.0
        w.energy[slot] = energy
        w.genome[slot, t + 3] = diet
        w.actions[slot, 2] = gate

    place(a, 500.0, 10.0, 1.0, 1.0)
    place(v, 501.0, 2.0, 0.0, 0.0)  # budget 4 kills it in one tick
    step.phase_rebuild_counts(w)
    step.phase_bite(w)
    assert w.deaths_predation == 1
    ix = int(w.pos[v, 0] / w.cell_w)
    iy = int(w.pos[v, 1] / w.cell_h)
    # take = 2 (all the victim had); carrion = 0.3 * 0.3 * 2
    assert w.meat[iy, ix] == pytest.approx(0.3 * 0.3 * 2.0, rel=1e-4)


# --- carrion: decay and eating ---------------------------------------------


def test_meat_decays_per_tick() -> None:
    w, cfg = _solo(meat_decay=0.5)
    w.meat[10, 10] = 8.0
    step.phase_decay_meat(w)
    assert w.meat[10, 10] == pytest.approx(4.0)
    step.phase_decay_meat(w)
    assert w.meat[10, 10] == pytest.approx(2.0)


def test_meat_decay_zero_preserves() -> None:
    w, cfg = _solo(meat_decay=0.0)
    w.meat[10, 10] = 8.0
    step.phase_decay_meat(w)
    assert w.meat[10, 10] == pytest.approx(8.0)


def test_eat_drains_food_first_then_meat() -> None:
    """The gate budget covers food first; carrion fills the rest."""
    w, cfg = _solo(eat_rate=1.0, food_growth_rate=0.0)
    slot = first_alive(w)
    ix = int(w.pos[slot, 0] / w.cell_w)
    iy = int(w.pos[slot, 1] / w.cell_h)
    w.food[iy, ix] = 0.4
    w.meat[iy, ix] = 2.0
    w.actions[slot, 2] = 1.0  # full gate
    e0 = float(w.energy[slot])
    step.phase_eat(w)
    # budget 1: takes 0.4 food + 0.6 meat
    assert w.food[iy, ix] == pytest.approx(0.0)
    assert w.meat[iy, ix] == pytest.approx(1.4)
    assert float(w.energy[slot]) == pytest.approx(e0 + 1.0, rel=1e-6)
    assert w.meat_eaten == pytest.approx(0.6, rel=1e-6)


def test_meat_only_still_feeds() -> None:
    """A scavenger with empty ground eats carrion alone."""
    w, cfg = _solo(eat_rate=1.0, food_growth_rate=0.0)
    slot = first_alive(w)
    ix = int(w.pos[slot, 0] / w.cell_w)
    iy = int(w.pos[slot, 1] / w.cell_h)
    w.food[iy, ix] = 0.0
    w.meat[iy, ix] = 2.0
    w.actions[slot, 2] = 1.0
    e0 = float(w.energy[slot])
    step.phase_eat(w)
    assert w.meat[iy, ix] == pytest.approx(1.0)
    assert float(w.energy[slot]) == pytest.approx(e0 + 1.0, rel=1e-6)
    assert w.meat_eaten == pytest.approx(1.0, rel=1e-6)


def test_carrion_smell_sensor() -> None:
    """The 4th smell channel reads the meat grid."""
    cfg = Config(max_creatures=8, initial_creatures=2)
    w = World(cfg, seed=0)
    me, other = np.flatnonzero(w.alive)
    carrion_idx = 3 * cfg.n_rays + 3
    w.pos[me] = (500.0, 500.0)
    w.pos[other] = (800.0, 800.0)
    w.meat[30, 30] = 5.0  # far from both creatures
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    assert w.sensor_buf[me, carrion_idx] == 0.0

    ix = int(w.pos[me, 0] / w.cell_w)
    iy = int(w.pos[me, 1] / w.cell_h)
    w.meat[iy, ix] = cfg.food_capacity  # full cell under my feet
    step.phase_perceive(w)
    # one full cell in a 3x3 window: food_capacity / capacity / 9
    assert w.sensor_buf[me, carrion_idx] == pytest.approx(1.0 / 9.0, rel=1e-5)


def test_carrion_cycle_never_creates_energy() -> None:
    """Death -> meat: the yield share is recovered, the rest dissipates."""
    cfg = Config(
        max_creatures=8,
        initial_creatures=2,
        max_age=5,
        carrion_yield=0.3,
        metabolic_cost=0.0,
        move_cost=0.0,
        turn_cost=0.0,
        food_growth_rate=0.0,
        senescence_rate=0.0,
    )
    w = World(cfg, seed=1)
    slots = np.flatnonzero(w.alive)
    w.energy[slots[0]] = 50.0
    w.energy[slots[1]] = 0.0  # will starve; the other dies of age
    w.age[:] = cfg.max_age
    total0 = float(w.energy.sum(dtype=np.float64)) + float(
        w.meat.sum(dtype=np.float64)
    ) + float(w.food.sum(dtype=np.float64))
    step.phase_deaths(w)
    assert w.deaths_age == 1 and w.deaths_famine == 1
    # old body deposited its yield share; starved body deposited nothing
    assert float(w.meat.sum()) == pytest.approx(0.3 * 50.0, rel=1e-5)
    total1 = float(w.energy.sum(dtype=np.float64)) + float(
        w.meat.sum(dtype=np.float64)
    ) + float(w.food.sum(dtype=np.float64))
    # (1 - yield) * 50 dissipated as heat; nothing was created
    assert total1 == pytest.approx(total0 - 0.7 * 50.0, rel=1e-5)
    assert total1 <= total0


def test_determinism_with_carrion_and_sex() -> None:
    cfg = Config(
        max_creatures=40,
        initial_creatures=20,
        reproduce_threshold=40.0,
        mate_range=1e6,
        carrion_yield=0.5,
        meat_decay=0.1,
        max_age=80,
        senescence_rate=0.005,
    )

    def run() -> World:
        w = World(cfg, seed=11)
        for _ in range(200):
            advance(w)
        return w

    w1 = run()
    w2 = run()
    assert np.array_equal(w1.meat, w2.meat)
    assert np.array_equal(w1.energy, w2.energy)
    assert np.array_equal(w1.genome, w2.genome)
    assert w1.meat_eaten == w2.meat_eaten
    assert (w1.births, w1.deaths_famine, w1.deaths_age) == (
        w2.births,
        w2.deaths_famine,
        w2.deaths_age,
    )
    assert w1.births > 0  # sex ran
