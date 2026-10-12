"""Stage-21 tests: local Oja plasticity — Baldwin, bounded, paid for.

The rule writes only into ``dw`` (the genome is immutable inside a life,
so speciation metrics keep measuring heritable divergence alone), the
effective synapses are ``genome + dw``, and nothing here may assert that
plasticity improves survival (principle 1: no fitness — the survival
question is the stage-21 experiment's, not the unit tests'). What is
tested is the contract: the phase order, the ``use_dw`` fast path being
bit-identical to zero-delta, clones diverging in ``dw`` while their
genomes stay equal, boundedness/no-NaN under a strong eta, determinism,
save/load continuation (v10), the energy cost proportional to |dw| moved,
the gene mode starting dormant, and the delta-E credit modulation.
"""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest

from primordia import step
from primordia.brain import think
from primordia.config import Config
from primordia.genome import get_numba_rng_state, set_numba_rng_state
from primordia.io import load_world, save_world
from primordia.world import World


def make_config(**overrides: object) -> Config:
    return replace(Config(), **overrides)


def small_config(**overrides: object) -> Config:
    base: dict[str, object] = dict(
        max_creatures=32, initial_creatures=16, width=200.0, height=200.0
    )
    base.update(overrides)
    return make_config(**base)


def alive_arrays(w: World) -> dict[str, np.ndarray]:
    """Arrays the equality checks care about (state, not scratch)."""
    return {
        "pos": w.pos,
        "vel": w.vel,
        "angle": w.angle,
        "energy": w.energy,
        "genome": w.genome,
        "hidden_prev": w.hidden_prev,
        "dw": w.dw,
        "e_last_plastic": w.e_last_plastic,
    }


def test_phase_order_plasticity_right_after_think() -> None:
    names = [name for name, _ in step.PHASES]
    assert len(names) == 22  # stage 21 inserted plasticity
    assert names.index("plasticity") == names.index("think") + 1
    # the commit must run before anything that reads hidden_prev next tick,
    # and the Oja pass must see the sensors produced this tick
    assert names.index("plasticity") < names.index("apply_actions")


def test_use_dw_false_is_bit_identical_to_zero_delta() -> None:
    """Skipping the dw branch == adding an exactly-zero delta.

    Guards the float corner (w + 0.0 and a skipped add must agree for the
    omega=0 sweep arm to be bit-identical to the baseline).
    """
    cfg = small_config()
    w = World(cfg, seed=3)
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    w.dw[:] = np.float32(0.0)

    think(
        w.sensor_buf, w.hidden_buf, w.hidden_prev, w.actions, w.alive,
        w.genome, w.dw, False,
        np.int32(cfg.sensor_input_dim), np.int32(cfg.hidden_size),
        np.int32(cfg.N_OUTPUTS),
    )
    ref_actions = w.actions.copy()
    ref_hidden = w.hidden_buf.copy()

    think(
        w.sensor_buf, w.hidden_buf, w.hidden_prev, w.actions, w.alive,
        w.genome, w.dw, True,
        np.int32(cfg.sensor_input_dim), np.int32(cfg.hidden_size),
        np.int32(cfg.N_OUTPUTS),
    )
    assert np.array_equal(w.actions, ref_actions)
    assert np.array_equal(w.hidden_buf, ref_hidden)


def test_mode_off_and_fixed_zero_runs_are_identical() -> None:
    """plasticity_mode='off' and fixed eta=0 evolve the same world bit-for-bit."""
    cfg_off = small_config(plasticity_mode="off")
    cfg_zero = small_config(plasticity_mode="fixed", plasticity_rate=0.0)
    a = World(cfg_off, seed=5)
    b = World(cfg_zero, seed=5)
    for _ in range(300):
        step.advance(a)
    for _ in range(300):  # sequential: the Numba RNG stream is process-global
        step.advance(b)
    for name, arr in alive_arrays(a).items():
        assert np.array_equal(arr, getattr(b, name)), name


def test_clones_diverge_in_dw_not_genome() -> None:
    """Same genetics, two environments: dw diverges, genome does not.

    The unit-level version of the stage-21 experiment figure: mutation and
    reproduction are off, so any genome difference would mean the rule
    leaked into the blueprint (it must not — Baldwin, not Lamarck).
    """
    common = dict(
        plasticity_mode="fixed",
        plasticity_rate=1e-2,
        plasticity_interval=1,
        mutation_rate=0.0,
        reproduce_threshold=1e9,  # no births: isolate dw from inheritance
        food_patch_scale=40.0,
    )
    env_a = small_config(food_patch_amplitude=0.85, **common)
    env_b = small_config(food_patch_amplitude=0.45, **common)
    a = World(env_a, seed=7)
    b = World(env_b, seed=7)
    assert np.array_equal(a.genome, b.genome)  # identical start
    for _ in range(300):
        step.advance(a)
    for _ in range(300):
        step.advance(b)
    assert np.array_equal(a.genome, b.genome), "the rule wrote into the genome"
    assert float(np.abs(a.dw - b.dw).max()) > 0.0, "clones learned nothing apart"


def test_plasticity_never_writes_the_genome() -> None:
    cfg = small_config(
        plasticity_mode="fixed", plasticity_rate=1e-2, plasticity_interval=1
    )
    w = World(cfg, seed=9)
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    before = w.genome.copy()
    for _ in range(50):
        step.phase_think(w)
        step.phase_plasticity(w)
        step.phase_perceive(w)
    assert np.array_equal(w.genome, before)
    assert float(np.abs(w.dw).max()) > 0.0  # but the delta did move


def test_dw_is_bounded_and_finite_under_strong_eta() -> None:
    """Stress: fast interval, eta at the top of the sweep, credit on.

    No NaN anywhere, |dw| <= clip, and the live stay above zero energy.
    """
    clip = 3.0
    cfg = small_config(
        plasticity_mode="fixed",
        plasticity_rate=1e-2,
        plasticity_interval=1,
        plasticity_clip=clip,
        plasticity_credit=True,
        plasticity_credit_strength=3.0,
        plasticity_cost=1.0,
    )
    w = World(cfg, seed=13)
    for _ in range(500):
        step.advance(w)
    assert np.isfinite(w.dw).all()
    assert np.isfinite(w.genome).all()
    assert np.isfinite(w.energy).all()
    assert float(np.abs(w.dw).max()) <= clip
    assert (w.energy[w.alive] > 0.0).all()


def test_determinism_with_plasticity() -> None:
    """Two sequential runs with learning on: bit-identical worlds."""
    cfg = small_config(
        plasticity_mode="fixed", plasticity_rate=1e-3, plasticity_interval=3
    )
    a = World(cfg, seed=42)
    b = World(cfg, seed=42)
    for _ in range(250):
        step.advance(a)
    for _ in range(250):
        step.advance(b)
    for name, arr in alive_arrays(a).items():
        assert np.array_equal(arr, getattr(b, name)), name


def test_cost_is_proportional_to_dw_movement() -> None:
    """energy drain == plasticity_cost * sum(|dw moved|); never created."""
    cfg = small_config(
        plasticity_mode="fixed",
        plasticity_rate=1e-2,
        plasticity_interval=1,
        plasticity_cost=1.0,
    )
    w = World(cfg, seed=17)
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    w.sensor_buf[w.alive] = np.float32(0.7)  # a strong, nonzero presynaptic side
    step.phase_think(w)

    dw_before = w.dw.copy()
    energy_before = w.energy.copy()
    step.phase_plasticity(w)

    moved = float(np.abs(w.dw - dw_before).sum())
    assert moved > 0.0
    drain = energy_before - w.energy
    expected = cfg.plasticity_cost * moved
    # only the alive slots pay (dead slots: no update, no commit)
    assert float(drain[w.alive].sum()) == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert (drain[w.alive] >= 0.0).all()  # cost only ever drains
    assert (w.energy[w.alive] > 0.0).all()  # nothing died in the phase


def test_gene_mode_starts_dormant_and_wakes_with_trait() -> None:
    cfg = small_config(plasticity_mode="gene", plasticity_interval=1)
    w = World(cfg, seed=19)
    assert np.all(w.genome[w.alive, cfg.plasticity_gene_offset] == 0.0)
    for _ in range(100):
        step.advance(w)
    assert np.all(w.dw == 0.0), "dormant gene must not learn"

    w.genome[w.alive, cfg.plasticity_gene_offset] = 0.05
    for _ in range(20):
        step.advance(w)
    assert float(np.abs(w.dw).max()) > 0.0, "an expressed eta must learn"


def test_negative_eta_is_clamped_at_use() -> None:
    """A negative eta (possible only by hand-crafting the genome) learns nothing."""
    cfg = small_config(plasticity_mode="gene", plasticity_interval=1)
    w = World(cfg, seed=23)
    w.genome[w.alive, cfg.plasticity_gene_offset] = -0.5
    for _ in range(30):
        step.advance(w)
    assert np.all(w.dw == 0.0)


def test_eta_trait_is_inherited_not_erased() -> None:
    """With mutation off, children carry the parents' eta exactly (crossover)."""
    cfg = small_config(
        plasticity_mode="gene",
        mutation_rate=0.0,
        reproduce_threshold=60.0,
        mate_range=1000.0,  # the test pins inheritance, not mate search
    )
    w = World(cfg, seed=29)
    eta_off = cfg.plasticity_gene_offset
    w.genome[w.alive, eta_off] = 0.04
    w.energy[w.alive] = 80.0  # above the threshold: reproduction is possible
    step.phase_rebuild_counts(w)
    free0 = w.free_count
    step.phase_reproduce(w)
    assert w.free_count < free0, "expected at least one birth"
    kids = w.free_list[w.free_count : free0]
    assert np.all(w.genome[kids, eta_off] == np.float32(0.04))
    assert np.all(w.genome[w.alive, eta_off] >= 0.0)


def test_eta_mutation_uses_its_own_std() -> None:
    """The eta gene mutates with plasticity_eta_std, not trait_std.

    trait_std (0.10) would fling eta an order of magnitude above the
    viable sweep band {1e-4..1e-2} on every hit, which the stage-21 pilot
    showed selection purging wholesale. With eta_std = 0.001, a forced
    mutation (rate 1.0) must keep every child's eta below 0.01 — that
    fails loudly if the code ever uses trait_std again.
    """
    cfg = small_config(
        plasticity_mode="gene",
        plasticity_eta_std=0.001,
        mutation_rate=1.0,
        reproduce_threshold=60.0,
        mate_range=1000.0,
    )
    w = World(cfg, seed=41)
    eta_off = cfg.plasticity_gene_offset
    w.genome[w.alive, eta_off] = 0.0
    w.energy[w.alive] = 80.0
    step.phase_rebuild_counts(w)
    free0 = w.free_count
    step.phase_reproduce(w)
    kids = w.free_list[w.free_count : free0]
    assert kids.size >= 5, "need enough forced mutations for the bound to bite"
    etas = w.genome[kids, eta_off]
    assert (etas >= 0.0).all()
    assert float(etas.max()) < 0.01, "eta mutation used trait_std (0.10) again?"


def test_credit_boosts_when_gaining_and_freezes_when_losing() -> None:
    """The delta-E ablation: gain -> bigger steps, loss -> eta clamped to 0."""
    def run(energy: float, e_last: float, credit: bool) -> World:
        cfg = small_config(
            plasticity_mode="fixed",
            plasticity_rate=1e-2,
            plasticity_interval=1,
            plasticity_credit=credit,
            plasticity_credit_strength=2.0,
        )
        w = World(cfg, seed=31)
        step.phase_rebuild_counts(w)
        step.phase_perceive(w)
        w.sensor_buf[w.alive] = np.float32(0.5)
        w.energy[w.alive] = energy
        w.e_last_plastic[w.alive] = e_last
        step.phase_think(w)
        step.phase_plasticity(w)
        return w

    plain = run(60.0, 40.0, credit=False)
    gaining = run(60.0, 40.0, credit=True)
    losing = run(20.0, 40.0, credit=True)

    base = float(np.abs(plain.dw).sum())
    up = float(np.abs(gaining.dw).sum())
    down = float(np.abs(losing.dw).sum())
    assert up > base > 0.0, "positive dE must scale eta above the plain rate"
    assert down == 0.0, "1 + strength*tanh(dE) < 0 clamps eta to 0"


def test_plasticity_save_load_continuation() -> None:
    """v10 roundtrip keeps dw/credit, and the continuation stays bit-identical."""
    cfg = small_config(
        plasticity_mode="fixed", plasticity_rate=1e-3, plasticity_interval=2
    )
    w = World(cfg, seed=37)
    for _ in range(120):
        step.advance(w)

    buf = BytesIO()
    save_world(w, buf)
    saved_stream = get_numba_rng_state()
    buf.seek(0)
    restored = load_world(buf)
    for name, arr in alive_arrays(w).items():
        assert np.array_equal(arr, getattr(restored, name)), name
    assert restored.tick == w.tick

    # sequential legs with an explicit stream restore between them (the
    # Numba RNG is process-global — same rule as tests/test_io.py)
    for _ in range(60):
        step.advance(restored)
    set_numba_rng_state(saved_stream)
    for _ in range(60):
        step.advance(w)
    for name, arr in alive_arrays(w).items():
        assert np.array_equal(arr, getattr(restored, name)), f"continuation {name}"


def test_rate_outside_fixed_mode_is_rejected() -> None:
    """A rate under 'off' would silently charge the control; under 'gene'
    it would be ignored while looking configured. Config forbids both."""
    with pytest.raises(ValueError, match="plasticity_rate"):
        replace(Config(), plasticity_mode="off", plasticity_rate=1e-3)
    with pytest.raises(ValueError, match="plasticity_rate"):
        replace(Config(), plasticity_mode="gene", plasticity_rate=1e-3)
    replace(Config(), plasticity_mode="fixed", plasticity_rate=1e-3)  # ok
    replace(Config(), plasticity_mode="off")  # rate defaults to 0: ok
