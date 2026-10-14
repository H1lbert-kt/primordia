"""Stage-24 affect states: bite scars (stress), rancor (grudge), off-switch."""

from __future__ import annotations

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.sensors import input_groups
from primordia.world import NO_MEMORY, World


def _place(w: World, slot: int, x: float, y: float, energy: float,
           diet: float = 0.0, gate: float | None = None,
           hue: float = 0.0) -> None:
    t = w.config.brain_params
    w.pos[slot, 0] = x
    w.pos[slot, 1] = y
    w.energy[slot] = energy
    w.genome[slot, t + 1] = 1.0
    w.genome[slot, t + 3] = diet
    w.genome[slot, t + 4] = hue
    if gate is not None:
        w.actions[slot, 2] = gate


def _bite_pair(hue_victim: float = 0.5, hue_attacker: float = 0.25,
               **cfg_kw) -> tuple[World, int, int]:
    cfg = Config(max_creatures=8, initial_creatures=2, **cfg_kw)
    w = World(cfg, seed=0)
    a, v = np.flatnonzero(w.alive)
    _place(w, a, 500.0, 500.0, 10.0, diet=1.0, gate=1.0, hue=hue_attacker)
    _place(w, v, 501.0, 500.0, 50.0, hue=hue_victim)
    step.phase_rebuild_counts(w)
    step.phase_bite(w)
    return w, int(a), int(v)


def test_layout_pins() -> None:
    cfg = Config()
    assert cfg.sensor_input_dim == 38
    assert cfg.hidden_size == 8
    assert cfg.brain_params == 448
    assert cfg.genome_size == 454
    assert cfg.plastic_params == 8 * 38 + 8 * 8
    assert cfg.affect_enabled is False
    assert cfg.stress_decay == pytest.approx(0.02)
    assert cfg.stress_scale == pytest.approx(1.0)
    assert cfg.grudge_decay == pytest.approx(0.01)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="stress_decay"):
        Config(stress_decay=-0.1)
    with pytest.raises(ValueError, match="stress_scale"):
        Config(stress_scale=0.0)
    with pytest.raises(ValueError, match="grudge_decay"):
        Config(grudge_decay=-0.1)


def test_input_groups_cover_all_inputs() -> None:
    cfg = Config()
    groups = input_groups(cfg)
    parts = []
    for _, idx in groups:
        if isinstance(idx, slice):
            parts.append(np.arange(*idx.indices(cfg.sensor_input_dim)))
        else:
            parts.append(np.asarray(idx))
    covered = np.concatenate(parts)
    assert np.array_equal(np.sort(covered), np.arange(cfg.sensor_input_dim))
    by_name = dict(groups)
    assert by_name["stress"].start == 3 * cfg.n_rays + 14
    assert by_name["grudge"].start == 3 * cfg.n_rays + 15
    assert by_name["store"].start == 3 * cfg.n_rays + 16


def test_bite_fills_victim_scratches() -> None:
    w, a, v = _bite_pair(hue_victim=0.5, hue_attacker=0.25)
    assert w.stress[v] > 0.0  # damage taken becomes stress
    assert int(w.attacker_age[v]) == 0  # fresh scar
    assert w.attacker_hue[v] == pytest.approx(0.25)
    assert w.attacker_age[a] == NO_MEMORY  # attacker carries no scar


def test_stress_decays_and_age_advances() -> None:
    w, _a, v = _bite_pair()
    w.stress[v] = 1.0
    stress_before = float(w.stress[v])
    step.phase_perceive(w)
    age = int(w.attacker_age[v])
    assert age == 1
    assert w.stress[v] == pytest.approx(stress_before * np.exp(-0.02))
    assert w.stress[v] < stress_before


def test_stress_decay_zero_freezes() -> None:
    w, _a, v = _bite_pair(stress_decay=0.0)
    w.stress[v] = 2.5
    step.phase_perceive(w)
    assert w.stress[v] == 2.5  # no decay, but age still advances
    assert int(w.attacker_age[v]) == 1


def test_stress_input_clamps() -> None:
    w, _a, _v = _bite_pair(affect_enabled=True)
    slot = int(np.flatnonzero(w.alive)[0])
    w.stress[slot] = 10.0  # far above stress_scale
    step.phase_perceive(w)
    assert w.sensor_buf[slot, 35] == 1.0


def test_grudge_same_hue_opposite_and_no_memory() -> None:
    # same hue: cos(0) = 1, decayed by exp(-grudge_decay * age)
    w, _a, v = _bite_pair(hue_victim=0.5, hue_attacker=0.5, affect_enabled=True)
    step.phase_perceive(w)  # age becomes 1, then the sensor reads
    expected = float(np.exp(-0.01))
    assert w.sensor_buf[v, 36] == pytest.approx(expected)

    # opposite hue (0.5 apart): cos(2pi*0.5) = -1
    w2, _a2, v2 = _bite_pair(hue_victim=0.0, hue_attacker=0.5, affect_enabled=True)
    step.phase_perceive(w2)
    assert w2.sensor_buf[v2, 36] == pytest.approx(-np.exp(-0.01))

    # no memory: grudge reads 0 even with grudge_decay=0
    w3, _a3, v3 = _bite_pair(affect_enabled=True, grudge_decay=0.0)
    w3.attacker_age[v3] = NO_MEMORY
    step.phase_perceive(w3)
    assert w3.sensor_buf[v3, 36] == 0.0


def test_grudge_fades_with_recency() -> None:
    w, _a, v = _bite_pair(hue_victim=0.25, hue_attacker=0.25, affect_enabled=True)
    step.phase_perceive(w)
    g0 = float(w.sensor_buf[v, 36])
    for _ in range(50):
        step.phase_perceive(w)
    g50 = float(w.sensor_buf[v, 36])
    assert 0.0 < g50 < g0


def test_off_forces_zero_inputs() -> None:
    w, _a, v = _bite_pair()  # default: affect off
    w.stress[v] = 3.0
    w.attacker_hue[v] = 0.25
    w.attacker_age[v] = 0
    step.phase_perceive(w)
    assert w.sensor_buf[v, 35] == 0.0
    assert w.sensor_buf[v, 36] == 0.0
    assert w.stress[v] > 0.0  # scratch still tracks (writes are unconditional)


def test_birth_resets_scratches() -> None:
    cfg = Config(max_creatures=8, initial_creatures=2, reproduce_threshold=40.0)
    w = World(cfg, seed=3)
    a, b = np.flatnonzero(w.alive)
    w.energy[a] = 60.0
    w.energy[b] = 60.0
    w.pos[a] = (500.0, 500.0)
    w.pos[b] = (501.0, 500.0)
    step.phase_rebuild_counts(w)  # mate search reads the CSR
    free_before = int(w.free_count)
    assert free_before > 0
    slot = int(w.free_list[free_before - 1])
    w.stress[slot] = 9.0  # recycled-slot garbage the child must not inherit
    w.attacker_hue[slot] = 0.77
    w.attacker_age[slot] = 12
    step.phase_reproduce(w)
    assert int(w.free_count) < free_before  # both parents found each other
    assert w.stress[slot] == 0.0
    assert w.attacker_hue[slot] == 0.0
    assert int(w.attacker_age[slot]) == NO_MEMORY


def test_determinism_500_ticks() -> None:
    cfg = Config(max_creatures=64, initial_creatures=16, affect_enabled=True)

    def run() -> World:
        w = World(cfg, seed=9)
        for _ in range(500):
            step.advance(w)
        return w

    a = run()
    b = run()
    for name in ("pos", "energy", "genome", "sensor_buf", "stress",
                 "attacker_hue", "creature_id"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert np.array_equal(a.attacker_age, b.attacker_age)
    assert a.births == b.births


def test_save_load_continuation_v11(tmp_path) -> None:
    from primordia.genome import get_numba_rng_state, set_numba_rng_state
    from primordia.io import load_world, save_world

    cfg = Config(max_creatures=64, initial_creatures=16, affect_enabled=True)
    w = World(cfg, seed=21)
    for _ in range(60):
        step.advance(w)
    w.stress[0] = 4.0  # scratchy state: v11 does NOT save it (documented)
    path = tmp_path / "affect.npz"
    save_world(w, str(path))
    saved_stream = get_numba_rng_state()

    v = load_world(str(path))
    assert np.array_equal(v.stress, np.zeros_like(v.stress))  # restarts clean

    # arrays that *are* saved continue bit-for-bit: advance the restored
    # world first, then rewind the stream and advance the original
    for _ in range(40):
        step.advance(v)
    set_numba_rng_state(saved_stream)
    for _ in range(40):
        step.advance(w)
    assert np.array_equal(w.genome, v.genome)
    assert np.array_equal(w.pos, v.pos)
    assert np.array_equal(w.dw, v.dw)
    assert w.births == v.births
    # the original kept its (now decayed) scratch; only the load restarted clean
    assert w.stress[0] > 0.0 and v.stress[0] == 0.0
