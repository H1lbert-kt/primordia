"""Brain tests: exact genome layout, hand-computed forward pass, motion, eat gate."""

from __future__ import annotations

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.world import World


def first_alive(w: World) -> int:
    return int(np.flatnonzero(w.alive)[0])


def offsets(cfg: Config) -> tuple[int, int, int, int, int]:
    """(w1, b1, w2, b2, wrec) offsets for a config, mirroring brain.py."""
    n_in = cfg.sensor_input_dim
    n_hid = cfg.hidden_size
    n_out = Config.N_OUTPUTS
    w1 = 0
    b1 = w1 + n_hid * n_in
    w2 = b1 + n_hid
    b2 = w2 + n_out * n_hid
    wrec = b2 + n_out
    return w1, b1, w2, b2, wrec


def test_genome_layout() -> None:
    cfg = Config()
    assert cfg.sensor_input_dim == 28  # 7*3 rays + 3 smell + 2 antennas + 2 internal
    assert cfg.brain_params == 423  # 10*29 + 3*11 + 10*10 (W1, W2, W_rec)
    assert 100 <= cfg.brain_params <= 500
    assert cfg.genome_size == cfg.brain_params + Config.N_BODY_TRAITS == 427

    _w1, _b1, _w2, b2, wrec = offsets(cfg)
    assert b2 + Config.N_OUTPUTS == wrec
    assert wrec + cfg.hidden_size**2 == cfg.brain_params

    w = World(cfg, seed=0)
    assert w.genome.shape == (cfg.max_creatures, cfg.genome_size)
    assert w.sensor_buf.shape == (cfg.max_creatures, cfg.sensor_input_dim)
    assert w.hidden_buf.shape == (cfg.max_creatures, cfg.hidden_size)
    assert w.hidden_prev.shape == (cfg.max_creatures, cfg.hidden_size)
    assert w.actions.shape == (cfg.max_creatures, Config.N_OUTPUTS)
    # body traits are filled from Config at spawn
    t = cfg.brain_params
    assert np.all(w.genome[w.alive, t] == np.float32(cfg.max_speed))
    assert np.all(w.genome[w.alive, t + 2] == np.float32(cfg.vision_range))


def test_forward_hand_computed() -> None:
    cfg = Config(n_rays=1, hidden_size=2)  # input_dim 10, params 35
    w = World(cfg, seed=1)
    slot = first_alive(w)
    _w1, _b1, w2, _b2, _wrec = offsets(cfg)
    n_in = cfg.sensor_input_dim
    n_hid = cfg.hidden_size

    w.genome[slot] = 0.0
    w.genome[slot, 0 * n_in + 0] = 1.0  # hidden 0 reads input 0
    w.genome[slot, 1 * n_in + 1] = 1.0  # hidden 1 reads input 1
    w.genome[slot, w2 + 0 * n_hid + 0] = 1.0  # out0 <- hidden0
    w.genome[slot, w2 + 1 * n_hid + 1] = 1.0  # out1 <- hidden1
    w.genome[slot, w2 + 2 * n_hid + 0] = 1.0  # out2 <- hidden0 - hidden1
    w.genome[slot, w2 + 2 * n_hid + 1] = -1.0

    x0, x1 = 0.5, -0.5
    w.sensor_buf[slot] = 0.0
    w.sensor_buf[slot, 0] = x0
    w.sensor_buf[slot, 1] = x1

    step.phase_think(w)

    h0 = np.tanh(x0)
    h1 = np.tanh(x1)
    # think stores raw logits; tanh is applied later in apply_actions
    assert w.actions[slot, 0] == pytest.approx(h0, abs=1e-5)
    assert w.actions[slot, 1] == pytest.approx(h1, abs=1e-5)
    assert w.actions[slot, 2] == pytest.approx(h0 - h1, abs=1e-5)


def test_recurrent_state_drives_hidden() -> None:
    """W_rec feeds the previous hidden state into the next one (Elman)."""
    cfg = Config(n_rays=1, hidden_size=2)
    w = World(cfg, seed=4)
    slot = first_alive(w)
    _w1, b1, w2, _b2, wrec = offsets(cfg)

    w.genome[slot] = 0.0  # no sensor weights: hidden = tanh(W_rec @ prev)
    w.genome[slot, wrec + 0] = 1.0  # hidden0 <- prev hidden0
    w.genome[slot, wrec + 1 * cfg.hidden_size + 1] = 1.0  # hidden1 <- prev hidden1
    w.genome[slot, w2 + 0 * cfg.hidden_size + 0] = 1.0  # out0 <- hidden0
    w.sensor_buf[slot] = 0.0
    w.hidden_prev[slot] = (0.9, -0.4)

    step.phase_think(w)

    assert w.actions[slot, 0] == pytest.approx(np.tanh(0.9), abs=1e-5)
    # the new state is committed for the next tick
    assert w.hidden_prev[slot, 0] == pytest.approx(np.tanh(0.9), abs=1e-5)
    assert w.hidden_prev[slot, 1] == pytest.approx(np.tanh(-0.4), abs=1e-5)


def test_outputs_drive_motion() -> None:
    cfg = Config(max_creatures=4, initial_creatures=1)
    w = World(cfg, seed=2)
    slot = first_alive(w)
    _w1, _b1, _w2, b2, _wrec = offsets(cfg)

    w.genome[slot] = 0.0
    w.genome[slot, cfg.brain_params] = cfg.max_speed  # restore trait zeroed above
    w.genome[slot, b2 + 0] = 9.0  # accel logit -> tanh ~= 1; turn/gate stay 0
    angle0 = 0.7
    w.angle[slot] = angle0

    step.phase_think(w)
    step.phase_apply_actions(w)

    speed = cfg.max_speed * np.tanh(9.0)
    assert w.vel[slot, 0] == pytest.approx(np.cos(angle0) * speed, abs=1e-3)
    assert w.vel[slot, 1] == pytest.approx(np.sin(angle0) * speed, abs=1e-3)
    assert w.angle[slot] == pytest.approx(angle0, abs=1e-6)  # tanh(0) = 0
    assert 0.0 < w.actions[slot, 2] < 1.0


def test_turn_costs_energy() -> None:
    """Stage 9.5: energy drain is turn_cost * |tanh(turn logit)| * max_turn."""
    cfg = Config(max_creatures=4, initial_creatures=1)
    w = World(cfg, seed=2)
    slot = first_alive(w)
    _w1, _b1, _w2, b2, _wrec = offsets(cfg)

    w.genome[slot] = 0.0
    w.genome[slot, cfg.brain_params] = cfg.max_speed
    w.genome[slot, b2 + 1] = 9.0  # turn logit -> tanh ~= 1; accel/gate stay 0
    e0 = float(w.energy[slot])

    step.phase_think(w)
    step.phase_apply_actions(w)

    drain = cfg.turn_cost * float(np.tanh(9.0)) * cfg.max_turn
    assert float(w.energy[slot]) == pytest.approx(e0 - drain, abs=1e-3)
    assert abs(float(w.angle[slot])) > 0.25  # it did turn


def test_turn_cost_zero_is_free() -> None:
    cfg = Config(max_creatures=4, initial_creatures=1, turn_cost=0.0)
    w = World(cfg, seed=2)
    slot = first_alive(w)
    _w1, _b1, _w2, b2, _wrec = offsets(cfg)
    w.genome[slot] = 0.0
    w.genome[slot, cfg.brain_params] = cfg.max_speed
    w.genome[slot, b2 + 1] = 9.0
    e0 = float(w.energy[slot])
    step.phase_think(w)
    step.phase_apply_actions(w)
    assert float(w.energy[slot]) == pytest.approx(e0, abs=1e-4)


def test_eat_gate() -> None:
    cfg = Config(max_creatures=4, initial_creatures=1, food_growth_rate=0.0)
    _w1, _b1, _w2, b2, _wrec = offsets(cfg)

    def run(gate_logit: float) -> tuple[float, float]:
        w = World(cfg, seed=3)
        slot = first_alive(w)
        w.genome[slot] = 0.0
        w.genome[slot, cfg.brain_params] = cfg.max_speed
        w.genome[slot, b2 + 2] = gate_logit  # accel/turn logits 0 -> stand still
        step.advance(w)
        ix = int(w.pos[slot, 0] / w.cell_w)
        iy = int(w.pos[slot, 1] / w.cell_h)
        return float(w.food[iy, ix]), float(w.energy[slot])

    cost = cfg.metabolic_cost
    food_low, energy_low = run(-9.0)  # gate ~ 1.5e-7: essentially no foraging
    food_high, energy_high = run(9.0)  # gate ~ 1: full bite

    assert food_low == pytest.approx(cfg.initial_food, abs=0.01)
    assert energy_low == pytest.approx(cfg.initial_energy - cost, abs=0.01)
    assert food_high == pytest.approx(cfg.initial_food - cfg.eat_rate, abs=0.01)
    assert energy_high == pytest.approx(
        cfg.initial_energy - cost + cfg.eat_rate, abs=0.01
    )
