"""Stage 15 tests: the lineage tag — heritable hue, circular mutation, sensor.

The hue is a *neutral* marker: it never touches energy, cost or
reproduction. No test here may assert that a colour survives better than
another (principle 1: no fitness, no rewards).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.sensors import input_groups
from primordia.stats import StatsRecorder
from primordia.world import World


def make_config(**overrides: object) -> Config:
    return replace(Config(), **overrides)


def first_alive(w: World) -> int:
    return int(np.flatnonzero(w.alive)[0])


def tag_index(cfg: Config) -> int:
    return 3 * cfg.n_rays + 12


def hue_of(w: World, slot: int) -> float:
    return float(w.genome[slot, w.config.brain_params + 4])


def set_hue(w: World, slot: int, hue: float) -> None:
    w.genome[slot, w.config.brain_params + 4] = np.float32(hue)


# --- the trait itself ------------------------------------------------------


def test_spawn_hue_random_in_unit_interval_and_seed_deterministic() -> None:
    cfg = make_config(max_creatures=8, initial_creatures=8)
    off = cfg.brain_params + 4
    w1 = World(cfg, seed=3)
    w2 = World(cfg, seed=3)
    w3 = World(cfg, seed=4)
    h1 = w1.genome[:, off]
    assert np.array_equal(h1, w2.genome[:, off])  # same seed, same colours
    assert not np.array_equal(h1, w3.genome[:, off])  # different seed differs
    assert h1.min() >= 0.0 and h1.max() < 1.0


def birth_once(cfg: Config, seed: int) -> tuple[World, int, int, int]:
    """One initiator + one nearby partner -> exactly one child (stage-3 style)."""
    w = World(cfg, seed=seed)
    slots = np.flatnonzero(w.alive)
    i, j = int(slots[0]), int(slots[1])
    w.energy[i] = 200.0
    w.energy[j] = 90.0
    w.pos[j] = w.pos[i] + np.float32([1.0, 0.0])
    for k in slots[2:]:
        w.pos[k] = w.pos[i] + np.float32([100.0, 100.0])
    step.phase_rebuild_counts(w)
    step.phase_reproduce(w)
    children = np.flatnonzero(w.alive & (w.parent_id == w.creature_id[i]))
    assert children.size == 1
    return w, i, j, int(children[0])


def test_child_hue_comes_from_a_parent_without_mutation() -> None:
    """mutation_rate=0: uniform crossover, so the child wears a parent's colour."""
    cfg = make_config(
        max_creatures=10, initial_creatures=4, mutation_rate=0.0
    )
    w, i, j, child = birth_once(cfg, seed=5)
    assert hue_of(w, child) in (hue_of(w, i), hue_of(w, j))


def test_hue_mutation_wraps_into_unit_interval() -> None:
    """Large additive noise must wrap, never leave [0,1) or clamp to 0."""
    cfg = make_config(
        max_creatures=10,
        initial_creatures=4,
        mutation_rate=1.0,
        trait_mutation_std=0.7,
    )
    mutated = 0
    for seed in range(20):
        w, i, j, child = birth_once(cfg, seed=seed)
        h = hue_of(w, child)
        assert 0.0 <= h < 1.0, h
        if h not in (hue_of(w, i), hue_of(w, j)):
            mutated += 1
    assert mutated == 20  # rate=1 with std=0.7 changes every child's hue


# --- the similarity sensor -------------------------------------------------


def two_creatures(hue_a: float, hue_b: float) -> tuple[World, np.ndarray]:
    """Two creatures stacked in one cell; returns the world and their slots."""
    cfg = make_config(
        max_creatures=4, initial_creatures=2, width=100.0, height=100.0
    )
    w = World(cfg, seed=1)
    slots = np.flatnonzero(w.alive)
    w.pos[slots[0]] = (50.0, 50.0)
    w.pos[slots[1]] = (50.0, 50.0)
    set_hue(w, int(slots[0]), hue_a)
    set_hue(w, int(slots[1]), hue_b)
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    return w, slots


def test_similarity_alone_is_zero() -> None:
    cfg = make_config(
        max_creatures=4, initial_creatures=2, width=100.0, height=100.0
    )
    w = World(cfg, seed=1)
    slots = np.flatnonzero(w.alive)
    # park the second creature on another corner: outside the smell window
    w.pos[slots[1]] = (0.0, 0.0)
    set_hue(w, int(slots[0]), 0.3)
    step.phase_rebuild_counts(w)
    step.phase_perceive(w)
    solo = float(w.sensor_buf[slots[0], tag_index(cfg)])
    assert abs(solo) < 1e-6


def test_similarity_sign_same_colour_vs_opposite() -> None:
    cfg = make_config(
        max_creatures=4, initial_creatures=2, width=100.0, height=100.0
    )
    idx = tag_index(cfg)

    # identical colours: positive alignment, bounded by the window
    w, slots = two_creatures(0.0, 0.0)
    same = float(w.sensor_buf[slots[0], idx])
    assert 0.0 < same <= 1.0

    # opposite colours (half a wheel apart): dot product is negative
    w, slots = two_creatures(0.0, 0.5)
    opp = float(w.sensor_buf[slots[0], idx])
    assert opp < 0.0
    # exact value: (neighbour unit vector . mine) / window cells = -1/9
    assert opp == pytest.approx(-1.0 / 9.0, abs=1e-5)


def test_rebuild_conserves_tag_vectors_over_the_grid() -> None:
    """Grid sums equal the per-slot (cos, sin) sums — no mass invented."""
    cfg = make_config(max_creatures=8, initial_creatures=6)
    w = World(cfg, seed=2)
    step.phase_rebuild_counts(w)
    for grid, slot_vec in ((w.cell_tag_re, w.hue_re), (w.cell_tag_im, w.hue_im)):
        alive_vec = slot_vec[w.alive]
        assert float(grid.sum()) == pytest.approx(
            float(alive_vec.sum()), abs=1e-3
        )


def test_input_groups_ends_with_tag() -> None:
    cfg = make_config()
    label, group = input_groups(cfg)[-1]
    assert label == "tag"
    assert group == slice(3 * cfg.n_rays + 12, 3 * cfg.n_rays + 13)


# --- stats -----------------------------------------------------------------


def test_stats_hue_columns_are_circular_and_bounded() -> None:
    cfg = make_config(max_creatures=16, initial_creatures=8)
    w = World(cfg, seed=7)
    off = cfg.brain_params + 4
    rec = StatsRecorder(capacity=8, sample_every=1, sample_size=8)

    # all creatures the same colour -> zero spread, mean = that colour
    w.genome[w.alive, off] = np.float32(0.37)
    step.advance(w)
    rec.record(w)
    data = rec.to_dict()
    assert float(data["hue_mean"][0]) == pytest.approx(0.37, abs=1e-3)
    assert float(data["hue_spread"][0]) < 1e-4
    assert -1.0 <= float(data["tag_sim_mean"][0]) <= 1.0
    assert float(data["win_neigh"][0]) >= 0.0

    # spread across the wheel -> high circular spread (linear mean would lie)
    w.genome[w.alive, off] = np.float32([0.0, 0.25, 0.5, 0.75] * 2)
    rec.record(w)
    data = rec.to_dict()
    assert float(data["hue_spread"][1]) > 0.9


# --- render guards ---------------------------------------------------------


def test_hue_lut_and_ring_offsets() -> None:
    pygame = pytest.importorskip("pygame")
    from primordia.render.draw import _ring_offsets, hue_lut

    lut = hue_lut()
    assert lut.shape == (256, 3)
    assert tuple(lut[0]) == pytest.approx((255.0, 0.0, 0.0))  # hue 0 = red
    for radius in (1, 4, 8):
        ox, oy = _ring_offsets(radius)
        d = np.hypot(ox, oy)
        assert d.size > 0
        assert d.min() >= radius + 0.4 - 1e-6
        assert d.max() <= radius + 1.4 + 1e-6


def test_panel_shows_tag_line() -> None:
    pygame = pytest.importorskip("pygame")
    from primordia.render import panel as P
    from primordia.render.settings import Settings

    pygame.font.init()
    w = World(Config(), seed=42)
    for _ in range(5):
        step.advance(w)
    messages: list[str] = []
    original = P._text

    def spy(dst, size, msg, x, y, color):
        messages.append(msg)
        return original(dst, size, msg, x, y, color)

    slot = first_alive(w)
    settings = Settings()
    P._text = spy
    try:
        surf = pygame.Surface((settings.panel_width, settings.window_height))
        hud = P.Hud(fps=60.0, speed=1, paused=False, follow=False, seed=42)
        P.render_panel(surf, w, slot, hud, settings)
    finally:
        P._text = original
    assert any("tag " in m for m in messages)
