"""Stage 13 core tests: the signal channel is wired, costs energy, decays.

No rule about *when* to communicate exists — these tests only prove the
physical channel (field + emit gate + decay) behaves as documented.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.world import World


def make_config(**overrides: object) -> Config:
    return replace(Config(), **overrides)


def test_signal_shape_dtype_and_zero_start() -> None:
    cfg = make_config()
    w = World(cfg, seed=0)
    gh, gw = cfg.grid_shape
    assert w.signal.shape == (gh, gw) and w.signal.dtype == np.float32
    assert float(w.signal.sum()) == 0.0
    assert w.signal_emitted == 0.0


def test_emit_gate_one_deposits_and_costs() -> None:
    """gate=1 for K ticks: energy drops K*cost, cell gains K*rate."""
    cfg = make_config(
        max_creatures=4,
        initial_creatures=1,
        emit_rate=2.0,
        signal_emit_cost=0.5,
        signal_decay=0.0,  # isolate the deposit from decay
        move_cost=0.0,
        turn_cost=0.0,
        senescence_rate=0.0,
    )
    w = World(cfg, seed=1)
    slot = int(np.flatnonzero(w.alive)[0])
    w.actions[slot, 3] = 1.0  # force the emit gate open
    ix = int(w.pos[slot, 0] / w.cell_w)
    iy = int(w.pos[slot, 1] / w.cell_h)
    e0 = float(w.energy[slot])

    k = 5
    for _ in range(k):
        step.phase_emit_signal(w)
        # the phase only reads alive+actions; re-assert the gate each pass
        w.actions[slot, 3] = 1.0

    assert float(w.signal[iy, ix]) == pytest.approx(k * cfg.emit_rate, abs=1e-4)
    assert float(w.energy[slot]) == pytest.approx(e0 - k * cfg.signal_emit_cost,
                                                 abs=1e-3)
    assert w.signal_emitted == pytest.approx(k * cfg.emit_rate, abs=1e-4)


def test_emit_gate_zero_is_noop() -> None:
    cfg = make_config(
        max_creatures=4,
        initial_creatures=1,
        signal_emit_cost=0.5,
        signal_decay=0.0,
    )
    w = World(cfg, seed=1)
    slot = int(np.flatnonzero(w.alive)[0])
    w.actions[slot, 3] = 0.0
    e0 = float(w.energy[slot])
    step.phase_emit_signal(w)
    assert float(w.signal.sum()) == 0.0
    assert float(w.energy[slot]) == e0
    assert w.signal_emitted == 0.0


def test_signal_decays_geometrically() -> None:
    cfg = make_config(
        max_creatures=4,
        initial_creatures=1,
        signal_decay=0.1,
        signal_emit_cost=0.0,
    )
    w = World(cfg, seed=2)
    w.signal[:, :] = 1.0
    ticks = 20
    for _ in range(ticks):
        step.phase_decay_signal(w)
    expected = (1.0 - cfg.signal_decay) ** ticks
    assert float(w.signal.mean()) == pytest.approx(expected, rel=1e-5)


def test_emit_cost_default_is_zero_preserves_economy() -> None:
    """The default Config must not tax the emit gate: the sigmoid never
    closes, so a nonzero default cost would drain every creature each tick.
    """
    cfg = Config()
    assert cfg.signal_emit_cost == 0.0
    w = World(cfg, seed=3)
    slot = int(np.flatnonzero(w.alive)[0])
    e0 = float(w.energy[slot])
    step.phase_emit_signal(w)  # gate defaults to 0 in the scratch buffer
    assert float(w.energy[slot]) == e0
    assert float(w.signal.sum()) == 0.0


def test_signal_carry_no_energy() -> None:
    """Eating a cell with signal gains nothing: signal is information."""
    cfg = make_config(
        max_creatures=4,
        initial_creatures=1,
        food_growth_rate=0.0,
        eat_rate=1.0,
    )
    w = World(cfg, seed=4)
    slot = int(np.flatnonzero(w.alive)[0])
    ix = int(w.pos[slot, 0] / w.cell_w)
    iy = int(w.pos[slot, 1] / w.cell_h)
    w.signal[iy, ix] = 50.0  # a huge signal on the eater's own cell
    food0 = float(w.food[iy, ix])
    step.phase_eat(w)  # no gate forced; food/meat accounting only
    assert float(w.food[iy, ix]) == food0  # signal never enters _eat
    assert float(w.signal[iy, ix]) == 50.0  # eating never drains signal


def test_determinism_with_signal_active() -> None:
    """Two runs, same seed, cost nonzero: arrays stay bit-identical."""
    cfg = make_config(
        max_creatures=32,
        initial_creatures=16,
        signal_emit_cost=0.3,
        signal_decay=0.1,
    )
    a = World(cfg, seed=11)
    b = World(cfg, seed=11)
    for _ in range(50):
        step.advance(a)
        step.advance(b)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.energy, b.energy)
    assert np.array_equal(a.genome, b.genome)
    assert np.array_equal(a.signal, b.signal)
    assert a.signal_emitted == b.signal_emitted
    assert a.tick == b.tick


def test_signal_config_validation() -> None:
    for bad in ({"signal_decay": 1.0}, {"signal_decay": -0.1},
                {"emit_rate": -1.0}, {"signal_emit_cost": -0.1},
                {"signal_capacity": 0.0}):
        with pytest.raises(ValueError):
            replace(Config(), **bad)


def test_brain_layout_signal_numbers() -> None:
    """Formula guards: the layout pins from stage 13 grew through 15/17."""
    cfg = Config()
    assert Config.N_OUTPUTS == 8
    assert cfg.sensor_input_dim == 35
    assert cfg.brain_params == 485
    assert cfg.genome_size == 490
    w = World(cfg, seed=0)
    assert w.actions.shape == (cfg.max_creatures, 8)
    assert w.sensor_buf.shape == (cfg.max_creatures, 35)
