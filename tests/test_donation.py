"""Stage 16 tests: donation — physical energy transfer, nearest recipient.

The donate gate is an *action*, never a reward: nothing here may assert
that giving survives better than not giving (principle 1: no fitness).
What is tested is the physics — conservation, the nearest-neighbour rule,
clamping, the phase contract, determinism and the layout pins.
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


def layout_pins() -> None:
    cfg = Config()
    assert Config.N_OUTPUTS == 8
    assert cfg.sensor_input_dim == 37
    assert cfg.hidden_size == 8
    assert cfg.brain_params == 440  # 8*38 + 8*9 + 8*8 (W1, W2, W_rec)
    assert cfg.genome_size == 446  # + 6 body traits (stages 3-21)
    assert 100 <= cfg.brain_params <= 500


def two_in_range(donate_rate: float = 0.5, dist: float = 2.0) -> tuple[World, int, int]:
    """Donor at (50, 50) with one recipient `dist` to the right, others far."""
    cfg = make_config(
        max_creatures=8,
        initial_creatures=4,
        width=100.0,
        height=100.0,
        donate_rate=donate_rate,
        donate_range=3.0,
    )
    w = World(cfg, seed=1)
    slots = np.flatnonzero(w.alive)
    i = int(slots[0])
    j = int(slots[1])
    w.pos[i] = (50.0, 50.0)
    w.pos[j] = (50.0 + dist, 50.0)
    for k in slots[2:]:
        w.pos[k] = (5.0, 5.0)
    w.energy[i] = 10.0
    w.energy[j] = 10.0
    w.actions[i, 4] = np.float32(1.0)  # decoded gate (apply_actions output)
    step.phase_rebuild_counts(w)
    return w, i, j


def test_layout_pins() -> None:
    layout_pins()


def test_config_validation() -> None:
    assert Config().donate_rate == 0.0
    assert Config().donate_range == 3.0
    with pytest.raises(ValueError, match="donate_rate"):
        make_config(donate_rate=-0.1)
    with pytest.raises(ValueError, match="donate_range"):
        make_config(donate_range=-1.0)


def test_transfer_moves_exactly_the_gift() -> None:
    w, i, j = two_in_range(donate_rate=0.5)
    before_i, before_j = float(w.energy[i]), float(w.energy[j])
    step.phase_donate(w)
    # gate reads as decoded: amount = donate_rate * gate = 0.5
    assert float(w.energy[i]) == pytest.approx(before_i - 0.5, abs=1e-6)
    assert float(w.energy[j]) == pytest.approx(before_j + 0.5, abs=1e-6)
    assert float(w.energy_donated) == pytest.approx(0.5, abs=1e-6)
    assert int(w.donate_pairs) == 1


def test_donation_conserves_world_energy() -> None:
    w, i, j = two_in_range(donate_rate=0.5)
    # a third creature inside range, gate left at 0 (never forced to give)
    total_before = float(w.energy.sum())
    step.phase_donate(w)
    total_after = float(w.energy.sum())
    assert total_after == pytest.approx(total_before, rel=1e-5)


def test_rate_zero_is_exact_noop() -> None:
    w, i, j = two_in_range(donate_rate=0.0)
    w.donate_pairs = 7  # stale diagnostic from a previous tick
    before = w.energy.copy()
    step.phase_donate(w)
    assert np.array_equal(w.energy, before)
    assert float(w.energy_donated) == 0.0
    assert int(w.donate_pairs) == 0  # scratch reset even on the fast path
    assert float(w.donate_align_sum) == 0.0


def test_out_of_range_donates_nothing() -> None:
    w, i, j = two_in_range(donate_rate=0.5, dist=5.0)  # range is 3.0
    before = float(w.energy[j])
    step.phase_donate(w)
    assert float(w.energy[j]) == before
    assert int(w.donate_pairs) == 0
    assert float(w.energy_donated) == 0.0


def test_nearest_recipient_wins_and_ties_go_to_lower_slot() -> None:
    w, i, j = two_in_range(donate_rate=0.5, dist=2.0)
    # add a second candidate farther right (distance 4 > 2): must lose
    far = int(np.flatnonzero(w.alive)[2])
    w.pos[far] = (54.0, 50.0)
    w.energy[far] = 10.0
    step.phase_rebuild_counts(w)
    before_far = float(w.energy[far])
    step.phase_donate(w)
    assert float(w.energy[far]) == before_far  # nearest (j) got the gift
    assert int(w.donate_pairs) == 1

    # tie: two candidates at exactly the same distance -> lower slot wins
    w2, i2, j2 = two_in_range(donate_rate=0.5, dist=3.0)
    other = int(np.flatnonzero(w2.alive)[2])
    w2.pos[other] = (47.0, 50.0)  # also exactly distance 3
    w2.energy[other] = 10.0
    step.phase_rebuild_counts(w2)
    low, high = sorted((j2, other))
    e_low, e_high = float(w2.energy[low]), float(w2.energy[high])
    step.phase_donate(w2)
    assert float(w2.energy[low]) == pytest.approx(e_low + 0.5, abs=1e-6)
    assert float(w2.energy[high]) == e_high


def test_donor_never_goes_below_zero() -> None:
    w, i, j = two_in_range(donate_rate=0.5)
    w.energy[i] = 0.2  # less than the requested 0.5 gift
    before_j = float(w.energy[j])
    step.phase_donate(w)
    assert float(w.energy[i]) == 0.0
    assert float(w.energy[j]) == pytest.approx(before_j + 0.2, abs=1e-6)
    assert float(w.energy_donated) == pytest.approx(0.2, abs=1e-6)


def test_donate_phase_sits_between_emit_and_eat() -> None:
    names = [name for name, _ in step.PHASES]
    assert names.index("donate") == names.index("emit_signal") + 1
    assert names.index("donate") < names.index("eat")
    assert len(names) == 22  # stage 21 inserted plasticity after think
    # stage 17 contract: gather -> build -> rebuild -> eat (a fire built
    # this tick already cooks this tick; the rebuild sits between them)
    assert names.index("gather") == names.index("donate") + 1
    assert names.index("build") == names.index("gather") + 1
    assert names.index("rebuild_struct_fields") == names.index("build") + 1
    assert names.index("eat") == names.index("rebuild_struct_fields") + 1


def test_gift_map_written_and_cleared() -> None:
    w, i, j = two_in_range(donate_rate=0.5)
    step.phase_donate(w)
    assert int(w.donate_to[i]) == j  # renderer sees donor -> recipient
    assert int(w.donate_to[j]) == -1
    # recipient leaves range: the map resets, no stale line
    w.pos[j] = (95.0, 95.0)
    step.phase_rebuild_counts(w)
    step.phase_donate(w)
    assert int(w.donate_to[i]) == -1
    # the rate=0 fast path clears stale entries planted by a previous run
    w3, i3, _ = two_in_range(donate_rate=0.0)
    w3.donate_to[i3] = 3
    step.phase_donate(w3)
    assert int(w3.donate_to[i3]) == -1


def test_determinism_with_donation_active() -> None:
    def run() -> tuple[np.ndarray, float]:
        # dense world + wide range so gifts are guaranteed within 200 ticks
        cfg = make_config(
            donate_rate=0.5,
            donate_range=15.0,
            width=80.0,
            height=80.0,
            initial_creatures=80,
        )
        w = World(cfg, seed=42)
        for _ in range(200):
            step.advance(w)
        return w.energy.copy(), float(w.energy_donated)

    e1, d1 = run()
    e2, d2 = run()
    assert d1 > 0.0  # the world actually donated something
    assert d1 == d2
    assert np.array_equal(e1, e2)


def test_continuation_with_donation_is_identical(tmp_path) -> None:
    """save/load mid-run with donation active resumes bit for bit (v8)."""
    from pathlib import Path

    from primordia.genome import get_numba_rng_state, set_numba_rng_state
    from primordia.io import load_world, save_world

    cfg = make_config(
        donate_rate=0.05,
        donate_range=15.0,
        width=80.0,
        height=80.0,
        initial_creatures=80,
    )
    a = World(cfg, seed=42)
    for _ in range(60):
        step.advance(a)
    assert a.energy_donated > 0.0
    path = tmp_path / "mid.npz"
    save_world(a, str(path))
    saved_stream = get_numba_rng_state()

    b = load_world(str(path))
    for _ in range(60):
        step.advance(b)

    set_numba_rng_state(saved_stream)
    for _ in range(60):
        step.advance(a)

    assert a.tick == b.tick == 120
    assert np.array_equal(a.energy, b.energy)
    assert a.energy_donated == b.energy_donated
    assert a.energy_donated > 0.0
