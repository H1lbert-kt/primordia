"""Stage-6 stats: recording, deltas, diversity, roundtrip, plot smoke."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from primordia import stats, step
from primordia.config import Config
from primordia.stats import StatsRecorder
from primordia.world import World

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shapes_dtypes_and_growth() -> None:
    cfg = Config(max_creatures=32, initial_creatures=16)
    w = World(cfg, seed=3)
    with pytest.raises(ValueError):
        StatsRecorder(capacity=0)
    rec = StatsRecorder(capacity=4, sample_every=3, sample_size=8)
    for _ in range(7):  # more rows than the initial capacity
        step.advance(w)
        rec.record(w)

    assert rec.length == 7 and rec.capacity >= 7
    data = rec.to_dict()
    assert set(data) == set(stats.COLUMNS)
    for name in stats.INT_COLUMNS:
        assert data[name].dtype == np.int32, name
    for name in stats.FLOAT_COLUMNS:
        assert data[name].dtype == np.float32, name
    assert np.array_equal(data["tick"], np.arange(1, 8, dtype=np.int32))


def test_recording_does_not_perturb_world() -> None:
    cfg = Config(max_creatures=64, initial_creatures=32)

    def run(with_stats: bool) -> tuple[World, StatsRecorder]:
        w = World(cfg, seed=11)
        rec = StatsRecorder(capacity=40, sample_every=5, sample_size=16)
        for _ in range(40):
            step.advance(w)
            if with_stats:
                rec.record(w)
        return w, rec

    w1, r1 = run(True)
    w2, _ = run(False)
    w3, r3 = run(True)
    for arr in ("pos", "vel", "angle", "energy", "age", "genome", "alive",
                "food"):
        assert np.array_equal(getattr(w1, arr), getattr(w2, arr)), arr
    counters = ("tick", "births", "deaths_famine", "deaths_age",
                "deaths_predation")
    assert tuple(getattr(w1, c) for c in counters) == tuple(
        getattr(w2, c) for c in counters
    )
    # the same seed replays to the same world and the same history
    assert np.array_equal(w1.genome, w3.genome)
    for name in stats.COLUMNS:
        assert np.array_equal(r1.to_dict()[name], r3.to_dict()[name]), name


def test_deltas_match_cumulative_counters() -> None:
    cfg = Config(max_creatures=64, initial_creatures=32)
    w = World(cfg, seed=7)
    rec = StatsRecorder(capacity=64, sample_every=2, sample_size=16)
    for _ in range(60):
        step.advance(w)
        rec.record(w)

    data = rec.to_dict()
    for name in ("births", "deaths_famine", "deaths_age", "deaths_predation"):
        assert np.all(data[name] >= 0), name
        assert int(data[name].sum()) == getattr(w, name), name


def test_trait_diversity_zero_then_positive() -> None:
    # mutation_rate 0.5 + a threshold below spawn energy forces births (and
    # therefore mutations) on the first tick: clones only diverge when
    # children are created, whatever the foraging economy does.
    cfg = Config(max_creatures=64, initial_creatures=32,
                 reproduce_threshold=45.0, mutation_rate=0.5)
    w = World(cfg, seed=5)
    rec = StatsRecorder(capacity=8, sample_every=1, sample_size=16)

    alive = np.flatnonzero(w.alive)
    w.genome[:] = w.genome[alive[0]]  # exact clones: zero diversity
    rec.record(w)
    first = rec.to_dict()
    for trait in ("speed", "size", "vision", "diet"):
        assert first[f"{trait}_std"][0] == 0.0, trait
    assert first["genome_dist"][0] == 0.0

    for _ in range(30):  # reproduction splits the clones apart
        step.advance(w)
        rec.record(w)
    assert w.births > 0
    last = rec.to_dict()
    assert last["genome_dist"][-1] > 0.0
    assert any(last[f"{t}_std"][-1] > 0.0
               for t in ("speed", "size", "vision", "diet"))


def test_genome_dist_forward_fill() -> None:
    cfg = Config(max_creatures=32, initial_creatures=16)
    w = World(cfg, seed=4)
    rec = StatsRecorder(capacity=16, sample_every=5, sample_size=8)

    rec.record(w)  # row 0 always samples
    first = float(rec.to_dict()["genome_dist"][0])
    assert first > 0.0  # spawn brains are independent draws

    # Scaling every living genome is a homothety: every pairwise distance
    # doubles exactly, but the next two records land on non-sample ticks.
    alive = np.flatnonzero(w.alive)
    w.genome[alive] *= 2.0
    w.tick = 1
    rec.record(w)
    w.tick = 4
    rec.record(w)
    filled = rec.to_dict()["genome_dist"][1:3]
    assert np.array_equal(filled, np.full(2, first, dtype=np.float32))

    w.tick = 5  # sample tick again: the fresh value sees the divergence
    rec.record(w)
    assert rec.to_dict()["genome_dist"][3] > first


def test_save_load_roundtrip(tmp_path: Path) -> None:
    cfg = Config(max_creatures=32, initial_creatures=16)
    w = World(cfg, seed=9)
    rec = StatsRecorder(capacity=8, sample_every=3, sample_size=8)
    for _ in range(10):  # also exercises capacity growth
        step.advance(w)
        rec.record(w)

    path = tmp_path / "stats.npz"
    rec.save(str(path), seed=99)
    history = StatsRecorder.load(str(path))

    assert history.seed == 99
    assert history.sample_every == 3 and history.sample_size == 8
    assert len(history) == 10
    assert history.config["bite_rate"] == cfg.bite_rate
    assert history.config["max_creatures"] == cfg.max_creatures
    for name in stats.COLUMNS:
        assert np.array_equal(history[name], rec.to_dict()[name]), name


def test_means_match_manual_at_tick_zero() -> None:
    cfg = Config(max_creatures=32, initial_creatures=16)
    w = World(cfg, seed=0)
    rec = StatsRecorder(capacity=4, sample_every=2, sample_size=8)
    rec.record(w)  # before any tick

    data = rec.to_dict()
    alive = np.flatnonzero(w.alive)
    assert data["tick"][0] == 0
    assert int(data["alive"][0]) == alive.size == w.alive_count
    assert np.isclose(data["mean_energy"][0], w.energy[alive].mean(),
                      rtol=1e-6, atol=0.0)
    assert np.isclose(data["mean_age"][0], w.age[alive].mean(),
                      rtol=1e-6, atol=0.0)


def test_plot_smoke(tmp_path: Path) -> None:
    cfg = Config(max_creatures=32, initial_creatures=16)
    w = World(cfg, seed=2)
    rec = StatsRecorder(capacity=32, sample_every=5, sample_size=8)
    for _ in range(20):
        step.advance(w)
        rec.record(w)

    npz = tmp_path / "stats.npz"
    rec.save(str(npz), seed=2)
    out = tmp_path / "figs"
    proc = subprocess.run(
        [sys.executable, "plot.py", str(npz), "--out", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("population.png", "energy.png", "deaths.png",
                 "diversity.png"):
        png = out / name
        assert png.exists() and png.stat().st_size > 0, name
