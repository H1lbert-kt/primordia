"""Stage-12 tests: diet classes, genome clustering, metric determinism."""

from __future__ import annotations

import numpy as np
import pytest

from primordia.config import Config
from primordia.ecology import (
    DIET_HERBIVORE,
    DIET_OMNIVORE,
    DIET_PREDATOR,
    classify_diets,
    cluster_living,
    genome_scales,
)
from primordia.world import World


def test_classify_diets_boundaries() -> None:
    diet = np.array([0.0, 0.32, 0.33, 0.5, 0.66, 0.67, 1.0], dtype=np.float32)
    labels = classify_diets(diet)
    assert labels.dtype == np.int8
    assert labels.tolist() == [
        DIET_HERBIVORE,
        DIET_HERBIVORE,
        DIET_OMNIVORE,
        DIET_OMNIVORE,
        DIET_OMNIVORE,
        DIET_PREDATOR,
        DIET_PREDATOR,
    ]


def test_genome_scales_follow_config() -> None:
    cfg = Config(n_rays=1, hidden_size=2, max_speed=3.0, vision_range=20.0)
    scale = genome_scales(cfg)
    assert scale.shape == (cfg.genome_size,)
    params = cfg.brain_params
    assert np.all(scale[:params] == cfg.weight_init_std)
    assert scale[params] == 3.0
    assert scale[params + 1] == 1.0
    assert scale[params + 2] == 20.0
    assert scale[params + 3] == 1.0


def test_genome_scales_zero_weight_init() -> None:
    cfg = Config(weight_init_std=0.0)
    scale = genome_scales(cfg)
    assert np.all(scale[: cfg.brain_params] == 1.0)


def _unit_scale_world(n: int, seed: int = 0) -> World:
    """Small world whose genome distances equal raw L2 (all scales 1)."""
    cfg = Config(
        max_creatures=max(n, 2),
        initial_creatures=n,
        weight_init_std=1.0,
        max_speed=1.0,
        vision_range=1.0,
        terrain_amplitude=0.0,
    )
    return World(cfg, seed=seed)


def test_cluster_identical_genomes_one_label() -> None:
    w = _unit_scale_world(5)
    w.genome[:] = 0.5
    _slots, labels = cluster_living(w, threshold=1.0)
    assert labels.shape == (5,)
    assert np.all(labels == 0)


def test_cluster_two_distant_groups() -> None:
    w = _unit_scale_world(6)
    w.genome[:] = 0.0
    w.genome[3:] = 10.0  # far beyond any sane threshold
    _slots, labels = cluster_living(w, threshold=1.0)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_cluster_threshold_boundary_is_inclusive() -> None:
    w = _unit_scale_world(2)
    w.genome[:] = 0.0
    w.genome[1, 0] = 1.0  # L2 distance exactly 1.0 (all scales are 1)
    _, labels = cluster_living(w, threshold=1.0)
    assert np.all(labels == 0)

    w.genome[1, 0] = 1.0001  # just above: new leader
    _, labels = cluster_living(w, threshold=1.0)
    assert labels.tolist() == [0, 1]


def test_cluster_is_deterministic() -> None:
    w = _unit_scale_world(50, seed=3)
    a = cluster_living(w, threshold=0.5)[1]
    b = cluster_living(w, threshold=0.5)[1]
    assert np.array_equal(a, b)


def test_cluster_ignores_dead_slots() -> None:
    w = _unit_scale_world(4)
    w.genome[:] = 0.0
    w.genome[2:] = 9.0
    w.alive[1] = False  # a dead clone of slot 0 must not open a cluster
    w.alive[3] = False  # a dead outlier must not open a cluster
    slots, labels = cluster_living(w, threshold=1.0)
    assert slots.tolist() == [0, 2]
    assert labels.tolist() == [0, 1]


def test_cluster_empty_world() -> None:
    w = _unit_scale_world(2)
    w.alive[:] = False
    slots, labels = cluster_living(w, threshold=1.0)
    assert slots.size == 0 and labels.size == 0


def test_species_threshold_validation() -> None:
    Config(species_threshold=0.01)  # tiny but valid: metric only
    with pytest.raises(ValueError, match="species_threshold"):
        Config(species_threshold=0.0)
    with pytest.raises(ValueError, match="species_threshold"):
        Config(species_threshold=-1.0)
