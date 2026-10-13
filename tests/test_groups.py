"""Stage-23 tests: observer groups, founder scenario, purity and events.

The observer is measurement only (the tick never reads it) and the founder
scenario is an initial condition (bit-identical spawn when off) — both
contracts are what these tests pin down.
"""

from __future__ import annotations

import numpy as np
import pytest

from primordia.config import Config
from primordia.observer import GroupEvent, GroupObserver, GroupSnapshot
from primordia.step import advance
from primordia.world import World


def make_config(**overrides: object) -> Config:
    base: dict = dict(
        max_creatures=40,
        initial_creatures=24,
        width=100.0,
        height=100.0,
        cell_size=5.0,
        vision_range=10.0,
        max_speed=1.0,
        terrain_amplitude=0.0,
        food_patch_amplitude=0.0,
        initial_food=3.0,
        food_capacity=3.0,
        max_age=10_000,
    )
    base.update(overrides)
    return Config(**base)


def _pin_hues(w: World, hues: np.ndarray) -> None:
    """Overwrite tag_hue of the living slots (measurement setup, no RNG)."""
    cfg = w.config
    slots = np.flatnonzero(w.alive)
    w.genome[slots, cfg.brain_params + 4] = hues


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="group_hue_threshold"):
        Config(group_hue_threshold=0.0)
    with pytest.raises(ValueError, match="group_hue_threshold"):
        Config(group_hue_threshold=0.6)
    with pytest.raises(ValueError, match="founder_regions"):
        Config(founder_regions=-1)
    with pytest.raises(ValueError, match="founder_hue_spread"):
        Config(founder_hue_spread=-0.1)
    with pytest.raises(ValueError, match="2 initial_creatures"):
        Config(founder_regions=4, initial_creatures=7)
    Config(founder_regions=4, initial_creatures=8)  # boundary ok


def test_observer_single_hue_band_one_group() -> None:
    cfg = make_config()
    w = World(cfg, seed=1)
    _pin_hues(w, np.full(w.alive.sum(), 0.5, dtype=np.float32))
    snap = GroupObserver(cfg).sample(w)
    assert snap.n_groups == 1
    assert snap.pops == {0: int(w.alive.sum())}


def test_observer_two_hue_bands_two_groups() -> None:
    cfg = make_config()
    w = World(cfg, seed=1)
    slots = np.flatnonzero(w.alive)
    hues = np.where(slots % 2 == 0, 0.1, 0.6).astype(np.float32)
    _pin_hues(w, hues)
    snap = GroupObserver(cfg).sample(w)
    assert snap.n_groups == 2
    assert sorted(snap.pops.values()) == [
        len(slots) // 2,
        len(slots) - len(slots) // 2,
    ]


def test_observer_owner_map_and_borders() -> None:
    """Two interleaved hue halves placed side by side share a frontier."""
    cfg = make_config(width=40.0, height=20.0, initial_creatures=20,
                      max_creatures=20)
    w = World(cfg, seed=2)
    slots = np.flatnonzero(w.alive)
    half = slots.size // 2
    left, right = slots[:half], slots[half:]
    w.pos[left, 0] = np.linspace(2.0, 18.0, left.size)
    w.pos[left, 1] = 10.0
    w.pos[right, 0] = np.linspace(22.0, 38.0, right.size)
    w.pos[right, 1] = 10.0
    _pin_hues(w, np.concatenate([
        np.full(left.size, 0.1, dtype=np.float32),
        np.full(right.size, 0.6, dtype=np.float32),
    ]))
    snap = GroupObserver(cfg).sample(w)
    assert snap.n_groups == 2
    # with creatures packed cell-by-cell along one row, the two clusters
    # are torus-adjacent somewhere only if they touch; here they are 4
    # units apart (x=18 to x=22) = adjacent cells -> frontier exists
    assert snap.border_cells > 0
    # owner map: left cells uid of group A, right cells uid of group B
    occupied = snap.owner >= 0
    assert occupied.any()
    uids_seen = set(np.unique(snap.owner[occupied]).tolist())
    assert uids_seen == set(snap.pops)


def test_observer_separated_groups_no_border() -> None:
    cfg = make_config(width=100.0, height=100.0, initial_creatures=10,
                      max_creatures=10)
    w = World(cfg, seed=3)
    slots = np.flatnonzero(w.alive)
    w.pos[:5, 0], w.pos[:5, 1] = 10.0, 10.0
    w.pos[5:, 0], w.pos[5:, 1] = 60.0, 60.0
    _pin_hues(w, np.concatenate([
        np.full(5, 0.1, dtype=np.float32),
        np.full(5, 0.6, dtype=np.float32),
    ]))
    snap = GroupObserver(cfg).sample(w)
    assert snap.n_groups == 2
    assert snap.border_cells == 0


def test_observer_does_not_mutate_world() -> None:
    cfg = make_config()
    w = World(cfg, seed=5)
    obs = GroupObserver(cfg)
    pos, genome, energy, alive = (
        w.pos.copy(), w.genome.copy(), w.energy.copy(), w.alive.copy(),
    )
    obs.sample(w)
    assert np.array_equal(w.pos, pos)
    assert np.array_equal(w.genome, genome)
    assert np.array_equal(w.energy, energy)
    assert np.array_equal(w.alive, alive)


def test_observer_deterministic() -> None:
    cfg = make_config()
    w = World(cfg, seed=7)
    a = GroupObserver(cfg).sample(w)
    b = GroupObserver(cfg).sample(w)
    assert a.n_groups == b.n_groups
    assert a.border_cells == b.border_cells
    assert np.array_equal(a.owner, b.owner)
    assert a.pops == b.pops
    assert [e.kind for e in a.events] == [e.kind for e in b.events]


def test_observer_uid_stable_across_samples() -> None:
    cfg = make_config()
    w = World(cfg, seed=11)
    obs = GroupObserver(cfg)
    first = obs.sample(w)
    for _ in range(5):
        advance(w)
    second = obs.sample(w)
    # with no major churn, surviving uids persist and ages advance
    shared = set(first.pops) & set(second.pops)
    assert shared, "uids must survive across nearby samples"
    for u in shared:
        assert second.ages[u] >= first.ages[u]


def test_observer_extinction_event() -> None:
    cfg = make_config(initial_creatures=12, max_creatures=12)
    w = World(cfg, seed=13)
    slots = np.flatnonzero(w.alive)
    _pin_hues(w, np.where(slots < 6, 0.1, 0.6).astype(np.float32))
    obs = GroupObserver(cfg)
    obs.sample(w)
    # kill every creature of one hue; recycled slots stay dead
    doomed = slots[:6]
    w.alive[doomed] = False
    w.free_list[w.free_count : w.free_count + doomed.size] = doomed
    w.free_count += doomed.size
    snap = obs.sample(w)
    assert snap.n_groups == 1
    ext = [e for e in snap.events if e.kind == "extinct"]
    assert ext and ext[0].group not in snap.pops


def test_observer_invasion_event_with_hysteresis() -> None:
    # 12x4 cells; group A packs 4 creatures into each of 5 cells on row 2,
    # group B puts 1 creature in each of those cells (A keeps the majority)
    # plus 5 far away. Cross-incursion count = 5.
    cfg = make_config(width=60.0, height=20.0, initial_creatures=30,
                      max_creatures=30)
    w = World(cfg, seed=17)
    slots = np.flatnonzero(w.alive)
    a, b = slots[:20], slots[20:]
    cx = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    ax = np.repeat(cx, 4) * 5.0 + 2.5
    w.pos[a, 0] = ax
    w.pos[a, 1] = 12.5
    w.pos[b[:5], 0] = cx * 5.0 + 2.5
    w.pos[b[:5], 1] = 12.5
    w.pos[b[5:], 0] = np.array([42.5, 47.5, 52.5, 47.5, 42.5], dtype=np.float32)
    w.pos[b[5:], 1] = 2.5
    _pin_hues(w, np.concatenate([
        np.full(a.size, 0.1, dtype=np.float32),
        np.full(b.size, 0.6, dtype=np.float32),
    ]))
    obs = GroupObserver(cfg, invasion_min=5)
    first = obs.sample(w)
    uid_a = min(first.pops, key=lambda u: first.hues[u])
    uid_b = max(first.pops, key=lambda u: first.hues[u])
    # the incursion already holds at the first sample -> event fires there
    inv = [e for e in first.events if e.kind == "invasion"]
    assert inv, first.events
    assert inv[0].group == uid_b and inv[0].other == uid_a
    assert inv[0].count >= 5
    # hysteresis: the pair stays active -> no duplicate on later samples
    second = obs.sample(w)
    third = obs.sample(w)
    assert not [e for e in second.events if e.kind == "invasion"]
    assert not [e for e in third.events if e.kind == "invasion"]


def test_founder_regions_pin_hue_bands() -> None:
    cfg = make_config(founder_regions=4, founder_hue_spread=0.02,
                      initial_creatures=40, max_creatures=40,
                      width=400.0)
    w = World(cfg, seed=42)
    slots = np.flatnonzero(w.alive)
    hue = w.genome[slots, cfg.brain_params + 4]
    strip = cfg.width / 4
    for k in range(4):
        group = slots[k * 10 : (k + 1) * 10]
        assert group.size == 10
        x = w.pos[group, 0]
        assert x.min() >= k * strip and x.max() <= (k + 1) * strip
        # circular distance to the region centre
        d = np.abs(hue[k * 10 : (k + 1) * 10] - k / 4)
        d = np.minimum(d, 1.0 - d)
        assert d.max() <= cfg.founder_hue_spread + 1e-6
    # the observer sees the 4 founding lineages at tick 0
    snap = GroupObserver(cfg).sample(w)
    assert snap.n_groups == 4
    assert sum(snap.pops.values()) == 40


def test_founder_off_is_bit_identical() -> None:
    a = World(make_config(), seed=42)
    b = World(make_config(founder_regions=0), seed=42)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.genome, b.genome)
    assert np.array_equal(a.angle, b.angle)
    for _ in range(50):
        advance(a)
        advance(b)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.genome, b.genome)
    assert np.array_equal(a.energy, b.energy)
    assert np.array_equal(a.alive, b.alive)


def test_founder_world_deterministic() -> None:
    cfg = make_config(founder_regions=3, initial_creatures=30,
                      max_creatures=30)
    a = World(cfg, seed=99)
    b = World(cfg, seed=99)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.genome, b.genome)
    for _ in range(30):
        advance(a)
        advance(b)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.genome, b.genome)


def test_snapshot_shapes() -> None:
    cfg = make_config()
    w = World(cfg, seed=1)
    snap = GroupObserver(cfg).sample(w)
    assert isinstance(snap, GroupSnapshot)
    assert snap.owner.shape == w.food.shape
    assert snap.owner.dtype == np.int16
    assert snap.uid_of_slot.shape == (cfg.max_creatures,)
    assert all(isinstance(e, GroupEvent) for e in snap.events)
    for u in snap.pops:
        assert u in snap.ages and u in snap.hues
