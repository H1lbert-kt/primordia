"""Stage 10 environment physics: terrain, water cost, season, light.

Every law is measured, not assumed: water blocks growth and doubles the
move bill, seasons scale food gain by S(t), night shortens the rays, and
the light input tracks L(t). Cycles are pure functions of (cfg, tick), so
these tests pin the formulas the whole simulation runs on.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.cycles import light_level
from primordia.step import advance
from primordia.world import World


def make_config(**overrides: object) -> Config:
    return replace(Config(), **overrides)


def first_alive(w: World) -> int:
    return int(np.flatnonzero(w.alive)[0])


def test_water_blocks_food_growth() -> None:
    """Water cells drain toward 0; land keeps refilling toward K (measured)."""
    cfg = make_config(
        terrain_amplitude=0.85,
        food_patch_amplitude=0.0,
        day_amp=0.0,
        season_amp=0.0,
        food_growth_rate=0.1,
    )
    w = World(cfg, seed=3)
    water = w.terrain == 0.0
    land = ~water
    assert water.any() and land.any()
    w.food[:] = cfg.food_capacity  # both start full
    for _ in range(200):
        step.phase_grow_food(w)
    assert float(w.food[water].max()) == pytest.approx(0.0, abs=1e-4)
    # land converges to its growth target K = food_field * terrain (fertility
    # varies per cell, so the target is per cell — not the raw capacity)
    target = w.food_field * w.terrain
    assert np.allclose(w.food[land], target[land], rtol=1e-3, atol=1e-4)
    assert float(w.food[land].mean()) > 0.2 * cfg.food_capacity


def test_water_costs_more_move_energy() -> None:
    """Same velocity: a water cell drains exactly (water_move_cost-1)x more."""
    cfg = make_config(
        terrain_amplitude=0.85,
        day_amp=0.0,
        season_amp=0.0,
        max_creatures=4,
        initial_creatures=2,
        water_move_cost=3.0,
        senescence_rate=0.0,  # flat metabolic bill so the delta is exact
    )
    w = World(cfg, seed=3)
    water_cells = np.argwhere(w.terrain == 0.0)
    land_cells = np.argwhere(w.terrain > 0.0)
    assert water_cells.size and land_cells.size
    slots = np.flatnonzero(w.alive)
    cw, ch = float(w.cell_w), float(w.cell_h)
    wy, wx = water_cells[0]
    ly, lx = land_cells[0]
    w.pos[slots[0]] = ((wx + 0.5) * cw, (wy + 0.5) * ch)
    w.pos[slots[1]] = ((lx + 0.5) * cw, (ly + 0.5) * ch)
    w.vel[:] = (0.6, 0.8)  # |v|^2 = 1 exactly
    e0 = w.energy.copy()
    step.phase_age_and_metabolize(w)
    d_water = float(e0[slots[0]] - w.energy[slots[0]])
    d_land = float(e0[slots[1]] - w.energy[slots[1]])
    move = cfg.move_cost  # speed^2 = 1
    assert d_land == pytest.approx(cfg.metabolic_cost + move, rel=1e-4)
    expected_gap = move * (cfg.water_move_cost - 1.0)
    assert d_water - d_land == pytest.approx(expected_gap, rel=1e-3)


def test_season_modulates_growth_measured() -> None:
    """Food gain follows S(t): peak = 2x mid, trough = 0 (season_amp=1)."""
    cfg = make_config(
        season_amp=1.0,
        season_period=4,
        day_amp=0.0,
        food_patch_amplitude=0.0,
        terrain_amplitude=0.0,
        food_growth_rate=0.1,
    )
    w = World(cfg, seed=1)

    def total_gain(tick: int) -> float:
        w.tick = tick
        w.food[:] = 0.0
        step.phase_grow_food(w)
        return float(w.food.sum())

    gain_mid = total_gain(0)  # S = 1
    gain_peak = total_gain(1)  # S = 2
    gain_trough = total_gain(3)  # S = 0
    assert gain_mid > 0.0
    assert gain_peak == pytest.approx(2.0 * gain_mid, rel=1e-4)
    assert gain_trough == pytest.approx(0.0, abs=1e-5)


def test_growth_rate_never_overshoots() -> None:
    """S * L can exceed 1, but the applied rate is clamped: food stays in [0, K]."""
    cfg = make_config(
        season_amp=1.0,
        season_period=4,
        day_amp=0.0,
        food_patch_amplitude=0.0,
        terrain_amplitude=0.0,
        food_growth_rate=1.0,  # without the rate clamp, S=2 would give 2*K
    )
    w = World(cfg, seed=1)
    w.tick = 1  # season peak: S = 2
    w.food[:] = 0.0
    step.phase_grow_food(w)
    assert float(w.food.max()) == pytest.approx(cfg.food_capacity)
    assert float(w.food.min()) >= 0.0


def test_night_shortens_vision() -> None:
    """Food inside daytime reach but beyond nighttime reach: seen only by day."""
    cfg = make_config(
        max_creatures=4,
        initial_creatures=1,
        width=100.0,
        height=100.0,
        cell_size=5.0,
        n_rays=5,
        vision_range=50.0,
        food_patch_amplitude=0.0,
        terrain_amplitude=0.0,
        day_amp=0.5,  # night L = 0.5 -> effective reach 25
    )
    w = World(cfg, seed=6)
    slot = first_alive(w)
    w.pos[slot] = (50.0, 50.0)
    w.angle[slot] = 0.0  # east; ray 2 points along the heading
    w.food[:] = 0.0
    # food only at x >= 80: beyond night reach (steps end at x = 72.5),
    # inside day reach (steps end at x = 97.5)
    w.food[9:11, 16:20] = cfg.food_capacity

    def ray_sum(tick: int) -> float:
        w.tick = tick
        step.phase_rebuild_counts(w)
        step.phase_perceive(w)
        return float(w.sensor_buf[slot, 0:5].sum())

    assert float(light_level(cfg, 0)) == 1.0
    assert float(light_level(cfg, 200)) == 0.5
    day = ray_sum(0)
    night = ray_sum(200)
    assert day > 0.0
    assert night == pytest.approx(0.0)


def test_light_sensor_tracks_cycle() -> None:
    """The light input carries exactly L(tick) for every living creature."""
    cfg = make_config(day_amp=0.5)
    w = World(cfg, seed=2)
    slots = np.flatnonzero(w.alive)
    light_idx = cfg.sensor_input_dim - 1  # light is the last input

    w.tick = 200
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    expected = float(light_level(cfg, 200))
    assert expected == 0.5
    assert np.all(w.sensor_buf[slots, light_idx] == np.float32(expected))

    w.tick = 0
    step.phase_perceive(w)
    assert np.all(w.sensor_buf[slots, light_idx] == 1.0)


def test_determinism_with_environment_active() -> None:
    """Terrain + hard cycles: two runs with the same seed stay bit-identical."""
    cfg = make_config(
        terrain_amplitude=0.85,
        day_amp=1.0,
        season_amp=1.0,
        day_period=50,
        season_period=130,
        max_creatures=100,
        initial_creatures=100,
    )
    # worlds are advanced sequentially (never interleaved): the second
    # construction reseeds the process-global Numba RNG to the same start
    a = World(cfg, seed=11)
    for _ in range(300):
        advance(a)
    b = World(cfg, seed=11)
    for _ in range(300):
        advance(b)
    for name in (
        "pos", "vel", "angle", "energy", "age", "genome", "hidden_prev",
        "sensor_buf", "food", "terrain",
    ):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.tick == b.tick == 300
    assert a.births == b.births
    assert a.deaths_famine == b.deaths_famine
    assert a.deaths_age == b.deaths_age
