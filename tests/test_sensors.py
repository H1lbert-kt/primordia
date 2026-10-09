"""Sensor tests: ray direction, vision range limit, self-exclusion."""

from __future__ import annotations

import numpy as np
import pytest

from primordia import step
from primordia.config import Config
from primordia.world import World


def first_alive(w: World) -> int:
    return int(np.flatnonzero(w.alive)[0])


def test_food_direction() -> None:
    cfg = Config(
        max_creatures=4,
        initial_creatures=1,
        width=100.0,
        height=100.0,
        cell_size=5.0,
        n_rays=5,
        vision_range=50.0,
    )
    w = World(cfg, seed=5)
    slot = first_alive(w)
    w.pos[slot] = (50.0, 50.0)
    w.angle[slot] = 0.0  # facing +x (east)
    w.food[:] = 0.0
    # corridor east of the creature: x in [60,100), y in [45,55)
    w.food[9:11, 12:20] = cfg.food_capacity

    step.phase_rebuild_counts(w)
    step.phase_perceive(w)

    food_feats = w.sensor_buf[slot, 0:5]
    # ray fan = linspace(-pi/2, pi/2, 5); index 2 points straight ahead (east)
    assert food_feats[2] > 0.0
    for other in (0, 1, 3, 4):
        assert food_feats[2] > food_feats[other]


def test_vision_range_limit() -> None:
    base = dict(
        max_creatures=4,
        initial_creatures=1,
        width=100.0,
        height=100.0,
        cell_size=5.0,
        n_rays=5,
    )

    def food_seen(vision_range: float) -> float:
        cfg = Config(vision_range=vision_range, **base)
        w = World(cfg, seed=6)
        w.pos[first_alive(w)] = (50.0, 50.0)
        w.angle[first_alive(w)] = 0.0
        w.food[:] = 0.0
        # food only at x >= 70: more than 20 units from x = 50
        w.food[9:11, 14:20] = cfg.food_capacity
        step.phase_rebuild_counts(w)
        step.phase_perceive(w)
        return float(w.sensor_buf[first_alive(w), 0:5].sum())

    assert food_seen(15.0) == pytest.approx(0.0)  # 3 steps reach only x = 62.5
    assert food_seen(50.0) > 0.0


def test_self_excluded() -> None:
    cfg = Config(max_creatures=4, initial_creatures=1)
    w = World(cfg, seed=7)
    slot = first_alive(w)
    w.pos[slot] = (50.0, 50.0)

    step.phase_rebuild_counts(w)
    step.phase_perceive(w)

    n = cfg.n_rays
    ray_creatures = w.sensor_buf[slot, n : 2 * n]
    smell_creatures = w.sensor_buf[slot, 3 * n + 1]
    assert np.all(ray_creatures == 0.0)
    assert smell_creatures == pytest.approx(0.0, abs=1e-6)


def test_input_groups_tile_the_input_vector() -> None:
    """The panel layout helper covers every input exactly once, in order."""
    from primordia.sensors import input_groups

    for cfg in (Config(), Config(n_rays=5), Config(n_rays=9, smell_radius_cells=2)):
        groups = input_groups(cfg)
        covered: list[int] = []
        labels = set()
        for label, group in groups:
            assert label not in labels
            labels.add(label)
            idx = np.arange(cfg.sensor_input_dim)[group]  # slice or fancy index
            assert idx.size > 0
            assert np.all(np.diff(idx) > 0)  # channel order preserved
            covered.extend(int(v) for v in idx)
        assert sorted(covered) == list(range(cfg.sensor_input_dim))
        assert {"energy", "speed", "smell meat"} <= labels


def test_ray_peak_sees_the_richest_cell() -> None:
    """The peak channel reads the best single cell, not the average."""
    cfg = Config(
        max_creatures=4,
        initial_creatures=1,
        width=100.0,
        height=100.0,
        cell_size=5.0,
        n_rays=5,
        vision_range=50.0,
    )
    w = World(cfg, seed=5)
    slot = first_alive(w)
    w.pos[slot] = (50.0, 50.0)
    w.angle[slot] = 0.0  # facing +x (east); ray 2 points straight ahead
    w.food[:] = 0.0
    w.food[9:11, 14] = cfg.food_capacity  # one full column in the ray's path

    step.phase_rebuild_counts(w)
    step.phase_perceive(w)

    n = cfg.n_rays
    peak = float(w.sensor_buf[slot, 2 * n + 2])
    total = float(w.sensor_buf[slot, 2])
    assert peak == pytest.approx(1.0)  # a full cell normalizes to 1
    assert 0.0 < total < peak  # the sum spreads the same food over the steps


def test_antennas_split_food_by_heading() -> None:
    """Food north of an east-facing creature lands on the left antenna."""
    cfg = Config(
        max_creatures=4,
        initial_creatures=1,
        width=100.0,
        height=100.0,
        cell_size=5.0,
        n_rays=5,
    )
    w = World(cfg, seed=6)
    slot = first_alive(w)
    w.pos[slot] = (50.0, 50.0)
    w.angle[slot] = 0.0  # east
    w.food[:] = 0.0
    # smell window is the 3x3 block around cell (10,10); food in its north row
    w.food[9, 9:12] = cfg.food_capacity

    step.phase_rebuild_counts(w)
    step.phase_perceive(w)

    n = cfg.n_rays
    left = float(w.sensor_buf[slot, 3 * n + 3])
    right = float(w.sensor_buf[slot, 3 * n + 4])
    assert left > 0.0
    assert right == pytest.approx(0.0)
