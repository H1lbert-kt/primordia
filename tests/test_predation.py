"""Stage-5 predation: contact transfer, diet/gate gates, killing, CSR, smell."""

from __future__ import annotations

import numpy as np

from primordia import step
from primordia.config import Config
from primordia.world import World


def _place(w: World, slot: int, x: float, y: float, energy: float,
           diet: float = 0.0, gate: float | None = None, size: float = 1.0) -> None:
    """Put a living creature at a known spot with controlled traits."""
    t = w.config.brain_params
    w.pos[slot, 0] = x
    w.pos[slot, 1] = y
    w.energy[slot] = energy
    w.genome[slot, t + 1] = size
    w.genome[slot, t + 3] = diet
    if gate is not None:
        w.actions[slot, 2] = gate


def _bite(w: World) -> None:
    step.phase_rebuild_counts(w)
    step.phase_bite(w)


def test_transfer_and_conservation() -> None:
    cfg = Config(max_creatures=8, initial_creatures=2)
    w = World(cfg, seed=0)
    a, v = np.flatnonzero(w.alive)  # spawn pops the last slots (LIFO)
    _place(w, a, 500.0, 500.0, 10.0, diet=1.0, gate=1.0)
    _place(w, v, 501.0, 500.0, 50.0)  # distance 1 < contact 3.0
    _bite(w)

    # budget = 4 * gate 1 * diet 1 = 4; victim loses 4, attacker keeps 0.7*4
    assert np.isclose(w.energy[v], 46.0, rtol=1e-4)
    assert np.isclose(w.energy[a], 10.0 + 0.7 * 4.0, rtol=1e-4)
    # global loss equals the dissipated share only (no energy created)
    assert np.isclose(w.energy[a] + w.energy[v], 60.0 - 0.3 * 4.0, rtol=1e-4)
    assert w.deaths_predation == 0


def test_diet_zero_cannot_drain() -> None:
    cfg = Config(max_creatures=8, initial_creatures=2)
    w = World(cfg, seed=0)
    a, v = np.flatnonzero(w.alive)
    _place(w, a, 500.0, 500.0, 10.0, diet=0.0, gate=1.0)
    _place(w, v, 501.0, 500.0, 50.0, diet=0.5)  # meaty, but mouth shut (gate 0)
    _bite(w)
    assert w.energy[a] == 10.0 and w.energy[v] == 50.0


def test_gate_zero_cannot_drain() -> None:
    cfg = Config(max_creatures=8, initial_creatures=2)
    w = World(cfg, seed=0)
    a, v = np.flatnonzero(w.alive)
    _place(w, a, 500.0, 500.0, 10.0, diet=1.0, gate=0.0)
    _place(w, v, 501.0, 500.0, 50.0)
    _bite(w)
    assert w.energy[a] == 10.0 and w.energy[v] == 50.0


def test_contact_range_and_edge_wrap() -> None:
    cfg = Config(max_creatures=8, initial_creatures=4)
    w = World(cfg, seed=0)
    a, far, wl, wr = np.flatnonzero(w.alive)
    _place(w, a, 500.0, 500.0, 10.0, diet=1.0, gate=1.0)
    _place(w, far, 510.0, 500.0, 50.0)  # distance 10 > contact 3.0
    # wrapped pair: toroidal distance sqrt(2^2 + 2^2) ~= 2.83 < 3.0
    _place(w, wl, 1.0, 1.0, 40.0, diet=1.0, gate=1.0)
    _place(w, wr, 999.0, 999.0, 30.0)
    _bite(w)
    assert w.energy[far] == 50.0  # out of range: untouched
    assert w.energy[wr] < 30.0  # across the seam: drained
    assert w.energy[wl] > 40.0


def test_kill_attributes_predation() -> None:
    cfg = Config(max_creatures=8, initial_creatures=3)
    w = World(cfg, seed=0)
    a, b, v = np.flatnonzero(w.alive)
    _place(w, a, 500.0, 500.0, 10.0, diet=1.0, gate=1.0)
    # B is beyond A's contact (4 > 3) but within V's (exactly 3), so only A
    # gets to drain this tick: the victim dies once, no double count.
    _place(w, b, 504.0, 500.0, 10.0, diet=1.0, gate=1.0)
    _place(w, v, 501.0, 500.0, 1.0)  # budget 4 finishes it in one tick
    free_before = w.free_count
    _bite(w)

    assert not w.alive[v]
    assert w.energy[v] == 0.0
    assert w.deaths_predation == 1  # the second attacker must not re-kill
    assert w.free_count == free_before + 1
    assert w.free_list[free_before] == v
    assert w.energy[a] > 10.0  # first attacker got the energy
    assert w.energy[b] == 10.0  # nothing left for the second one


def test_determinism_with_predation() -> None:
    cfg = Config(
        max_creatures=16,
        initial_creatures=2,
        reproduce_threshold=40.0,  # force births into the Numba RNG stream
    )

    def run() -> World:
        w = World(cfg, seed=7)
        a, v = np.flatnonzero(w.alive)
        _place(w, a, 500.0, 500.0, 30.0, diet=0.5, gate=1.0)
        _place(w, v, 501.0, 500.0, 30.0, diet=0.5, gate=1.0)
        # near-zero speed keeps the pair in contact long enough to kill
        w.genome[[a, v], cfg.brain_params] = 0.01
        for _ in range(300):  # sequential worlds: process-global Numba RNG
            step.advance(w)
        return w

    w1 = run()
    w2 = run()
    assert np.array_equal(w1.pos, w2.pos)
    assert np.array_equal(w1.energy, w2.energy)
    assert np.array_equal(w1.genome, w2.genome)
    assert np.array_equal(w1.alive, w2.alive)
    assert (w1.births, w1.deaths_famine, w1.deaths_age, w1.deaths_predation) == (
        w2.births,
        w2.deaths_famine,
        w2.deaths_age,
        w2.deaths_predation,
    )
    assert w1.deaths_predation > 0  # predation actually happened


def test_csr_grid_consistent() -> None:
    cfg = Config(max_creatures=100, initial_creatures=80)
    w = World(cfg, seed=1)
    w.rebuild_counts()

    alive = np.flatnonzero(w.alive)
    offs = w.cell_offsets
    assert offs[0] == 0 and offs[-1] == alive.size
    assert np.all(np.diff(offs) >= 0)

    listed = w.cell_slots[: alive.size]
    assert np.array_equal(np.sort(listed), alive)  # every slot exactly once

    # independent recount: per-cell histogram and diet sums must match
    gh, gw = w.cell_counts.shape
    ix = np.clip((w.pos[alive, 0] / w.cell_w).astype(int), 0, gw - 1)
    iy = np.clip((w.pos[alive, 1] / w.cell_h).astype(int), 0, gh - 1)
    counts = np.zeros((gh, gw), dtype=np.int32)
    diet_sums = np.zeros((gh, gw), dtype=np.float32)
    diet = w.genome[alive, cfg.brain_params + 3]
    np.add.at(counts, (iy, ix), 1)
    np.add.at(diet_sums, (iy, ix), diet)
    assert np.array_equal(w.cell_counts, counts)
    assert np.allclose(w.cell_diet, diet_sums, rtol=1e-6)


def test_meat_smell_sensor() -> None:
    cfg = Config(max_creatures=8, initial_creatures=2)
    w = World(cfg, seed=0)
    me, prey = np.flatnonzero(w.alive)
    meat_idx = 3 * cfg.n_rays + 2  # third smell channel (layout in sensors.py)
    _place(w, me, 500.0, 500.0, 30.0, diet=0.0)
    _place(w, prey, 507.0, 500.0, 30.0, diet=1.0)  # adjacent smell cell
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    assert w.sensor_buf[me, meat_idx] > 0.0  # I smell the prey's diet
    assert w.sensor_buf[prey, meat_idx] >= 0.0

    # a lone creature never smells its own meat (self excluded)
    _place(w, prey, 800.0, 800.0, 30.0, diet=1.0)
    _place(w, me, 200.0, 200.0, 30.0, diet=1.0)
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    assert w.sensor_buf[me, meat_idx] == 0.0
