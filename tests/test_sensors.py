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
    smell_creatures = w.sensor_buf[slot, 2 * n + 1]
    assert np.all(ray_creatures == 0.0)
    assert smell_creatures == pytest.approx(0.0, abs=1e-6)
