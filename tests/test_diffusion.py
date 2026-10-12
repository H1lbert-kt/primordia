"""Stage 14 core tests: the pheromone pass spreads signal as physics.

Jacobi diffusion conserves mass, wraps the torus, stays bounded at the
stability limit and never touches energy — whether creatures *use* the
trails is an experiment question (experiments/trails), not a rule.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.genome import get_numba_rng_state, set_numba_rng_state
from primordia.io import load_world, save_world
from primordia.world import World


def make_config(**overrides: object) -> Config:
    return replace(Config(), **overrides)


def test_signal_diffuse_default_and_validation() -> None:
    """Diffusion is on by default (user decision), bounded by stability."""
    assert Config().signal_diffuse == 0.1
    for good in (0.0, 0.05, 0.1, 0.25):
        assert make_config(signal_diffuse=good).signal_diffuse == good
    for bad in (-0.01, 0.26, 0.5):
        with pytest.raises(ValueError, match="signal_diffuse"):
            replace(Config(), signal_diffuse=bad)


def test_signal_buf_is_dedicated_scratch() -> None:
    cfg = make_config()
    w = World(cfg, seed=0)
    gh, gw = cfg.grid_shape
    assert w.signal_buf.shape == (gh, gw)
    assert w.signal_buf.dtype == np.float32
    assert w.signal_buf is not w.signal


def test_diffusion_conserves_mass() -> None:
    """A Jacobi step only moves mass: the field sum is unchanged."""
    w = World(make_config(signal_decay=0.0), seed=0)
    w.signal[10, 7] = 3.0
    w.signal[50, 61] = 1.25
    m0 = float(w.signal.sum())
    step.phase_diffuse_signal(w)
    assert float(w.signal.sum()) == pytest.approx(m0, rel=1e-5)


def test_diffusion_spreads_to_four_neighbors_and_wraps() -> None:
    """Corner deposit at r=0.25: center empties into its 4 neighbors,
    including the toroidal wrap to the far edges."""
    w = World(make_config(signal_diffuse=0.25, signal_decay=0.0), seed=0)
    w.signal[0, 0] = 1.0
    gh, gw = w.signal.shape
    step.phase_diffuse_signal(w)
    assert float(w.signal[0, 0]) == pytest.approx(0.0, abs=1e-6)
    for iy, ix in ((1, 0), (gh - 1, 0), (0, 1), (0, gw - 1)):
        assert float(w.signal[iy, ix]) == pytest.approx(0.25, abs=1e-6)
    assert float(w.signal.sum()) == pytest.approx(1.0, rel=1e-6)


def test_diffusion_stays_bounded_at_stability_limit() -> None:
    """Monotone scheme for r <= 0.25: values stay inside the initial
    min/max — 100 passes on a random field neither blow up nor go NaN."""
    w = World(make_config(signal_diffuse=0.25, signal_decay=0.0), seed=0)
    rng = np.random.default_rng(0)
    w.signal[...] = rng.random(w.signal.shape).astype(np.float32)
    lo = float(w.signal.min())
    hi = float(w.signal.max())
    for _ in range(100):
        step.phase_diffuse_signal(w)
    assert np.all(np.isfinite(w.signal))
    assert float(w.signal.min()) >= lo - 1e-6
    assert float(w.signal.max()) <= hi + 1e-6


def test_diffusion_after_decay_scales_mass() -> None:
    """decay then diffuse: total mass is exactly decay * (previous)."""
    w = World(make_config(signal_diffuse=0.1, signal_decay=0.1), seed=0)
    w.signal[:, :] = 1.0
    m0 = float(w.signal.sum())
    step.phase_decay_signal(w)
    step.phase_diffuse_signal(w)
    assert float(w.signal.sum()) == pytest.approx(m0 * 0.9, rel=1e-5)


def test_diffusion_zero_is_noop() -> None:
    """signal_diffuse=0 must leave the field untouched (experiment control)."""
    w = World(make_config(signal_diffuse=0.0), seed=0)
    w.signal[3, 4] = 2.0
    before = w.signal.copy()
    step.phase_diffuse_signal(w)
    assert np.array_equal(w.signal, before)


def test_phase_order_runs_diffusion_right_after_decay() -> None:
    names = [name for name, _step in step.PHASES]
    assert len(names) == 22  # stage 21 added plasticity after think
    assert names.index("diffuse_signal") == names.index("decay_signal") + 1
    assert names.index("diffuse_signal") < names.index("age_metabolize")


def test_determinism_with_strong_diffusion() -> None:
    """Two runs, same seed, r=0.25 + a priced channel: bit-identical."""
    cfg = make_config(
        max_creatures=32,
        initial_creatures=16,
        signal_diffuse=0.25,
        signal_emit_cost=0.3,
        signal_decay=0.1,
    )
    a = World(cfg, seed=11)
    b = World(cfg, seed=11)
    for _ in range(50):
        step.advance(a)
        step.advance(b)
    assert np.array_equal(a.signal, b.signal)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.energy, b.energy)
    assert a.tick == b.tick


def test_continuation_identical_with_diffusion(tmp_path: Path) -> None:
    """save/load mid-run with r=0.25 resumes bit for bit (v6, config-only
    field: the loaded Config inherits signal_diffuse from the JSON when
    present, else the default)."""
    cfg = make_config(
        max_creatures=64,
        initial_creatures=32,
        reproduce_threshold=60.0,
        max_age=140,
        metabolic_cost=0.0,
        move_cost=0.0,
        turn_cost=0.0,
        food_growth_rate=0.2,
        mate_range=1e6,
        signal_diffuse=0.25,
        signal_emit_cost=0.3,
        signal_decay=0.1,
    )
    a = World(cfg, seed=42)
    for _ in range(60):
        step.advance(a)
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
    assert np.array_equal(a.signal, b.signal)
    assert np.array_equal(a.energy, b.energy)
    assert np.array_equal(a.genome, b.genome)
    assert np.array_equal(a.pos, b.pos)
    assert a.signal_emitted == b.signal_emitted
