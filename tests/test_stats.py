"""Stage-6 stats: recording, deltas, diversity, roundtrip, plot smoke."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
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
                 reproduce_threshold=45.0, mutation_rate=0.5,
                 mate_range=1e6)  # dense mating: pins diversity, not ecology
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


def test_diet_gauges_sum_to_alive() -> None:
    cfg = Config(max_creatures=48, initial_creatures=24)
    w = World(cfg, seed=5)
    rec = StatsRecorder(capacity=40, sample_every=10, sample_size=16,
                        cluster_every=50)
    for _ in range(30):
        step.advance(w)
        rec.record(w)
    data = rec.to_dict()
    total = (data["n_herbivore"] + data["n_omnivore"] + data["n_predator"]
             ).astype(np.int64)
    assert np.array_equal(total, data["alive"].astype(np.int64))


def test_species_columns_forward_fill_between_cluster_ticks() -> None:
    cfg = Config(max_creatures=48, initial_creatures=24)
    w = World(cfg, seed=8)
    rec = StatsRecorder(capacity=40, sample_every=10, sample_size=16,
                        cluster_every=7)
    for _ in range(20):
        step.advance(w)
        rec.record(w)
    data = rec.to_dict()
    n = len(data["tick"])
    # Recompute cluster ticks exactly as the recorder does (i == 0 first).
    for i, tick in enumerate(data["tick"]):
        if i > 0 and tick % 7 != 0:
            continue
        j = i + 1
        while j < n and data["tick"][j] % 7 != 0:
            assert data["n_species"][j] == data["n_species"][i]
            assert data["barrier"][j] == data["barrier"][i]
            j += 1
    assert np.all(data["barrier"] >= 0.0) and np.all(data["barrier"] <= 1.0)
    assert np.all(data["n_species"] >= 0.0)


def test_barrier_counts_cross_cluster_births() -> None:
    """Unit-test _sample_species: hand-placed clusters + hand-written log."""
    cfg = Config(max_creatures=16, initial_creatures=0,
                 species_threshold=0.5)
    w = World(cfg, seed=1)
    slots = w.spawn(3)
    g = cfg.brain_params
    w.genome[:] = 0.0  # identical everything...
    # max_speed trait is normalized by cfg.max_speed=4.0, so a raw 10.0 is a
    # normalized distance of 2.5, well above species_threshold=0.5 -> 2 clusters.
    w.genome[slots[2], g] = 10.0
    w.energy[slots] = 50.0
    w.tick = 5
    step.phase_rebuild_counts(w)

    # Cross-cluster birth in the log: parent = cluster-0 creature,
    # partner = cluster-1 creature (both alive at the sample).
    child_id = w.next_id
    w.next_id += 1
    w.genealogy[0] = (w.tick, child_id, w.creature_id[slots[0]],
                      w.creature_id[slots[2]])
    w.genealogy_used = 1
    w.births = 1
    rec = StatsRecorder(capacity=4, sample_every=1, sample_size=4,
                        cluster_every=1)
    rec.record(w)
    data = rec.to_dict()
    assert float(data["n_species"][0]) == 2.0
    assert int(data["matings_valid"][0]) == 1
    assert float(data["barrier"][0]) == 1.0

    # Same-cluster birth (both parents in cluster 0) -> barrier 0.
    rec2 = StatsRecorder(capacity=4, sample_every=1, sample_size=4,
                         cluster_every=1)
    w.genealogy[0] = (w.tick, child_id + 1, w.creature_id[slots[0]],
                      w.creature_id[slots[1]])
    w.genealogy_used = 1
    w.births = 1
    rec2.record(w)
    data2 = rec2.to_dict()
    assert int(data2["matings_valid"][0]) == 1
    assert float(data2["barrier"][0]) == 0.0

    # Dead partner is skipped: not valid, barrier denominator drops to 0.
    rec3 = StatsRecorder(capacity=4, sample_every=1, sample_size=4,
                         cluster_every=1)
    w.genealogy[0] = (w.tick, child_id + 2, w.creature_id[slots[0]],
                      999999)  # creature id that is not alive
    w.genealogy_used = 1
    w.births = 1
    rec3.record(w)
    data3 = rec3.to_dict()
    assert int(data3["matings_valid"][0]) == 0
    assert float(data3["barrier"][0]) == 0.0


def test_load_npz_without_ecology_columns() -> None:
    """Files saved by stage <= 11 load; new columns are simply absent."""
    cfg = Config(max_creatures=16, initial_creatures=8)
    w = World(cfg, seed=1)
    rec = StatsRecorder(capacity=8, sample_every=2, sample_size=8)
    for _ in range(4):
        step.advance(w)
        rec.record(w)
    old_cols = [c for c in stats.COLUMNS
                if c not in ("n_species", "barrier", "matings_valid",
                             "n_herbivore", "n_omnivore", "n_predator",
                             "diet_max", "meat_eaten",
                             "diet_p50", "diet_p90", "diet_p99",
                             "bite_herb", "bite_omni", "bite_pred",
                             "signal_emitted", "signal_mass", "emit_mean")]
    payload = {name: rec.to_dict()[name] for name in old_cols}
    payload["columns"] = np.array(old_cols)
    payload["seed"] = np.int64(1)
    payload["sample_every"] = np.int32(2)
    payload["sample_size"] = np.int32(8)
    payload["config_json"] = np.array(json.dumps(asdict(cfg), sort_keys=True))
    path = Path("/tmp") / "stats_old_cols.npz"
    np.savez(path, **payload)

    loaded = stats.StatsRecorder.load(str(path))
    assert loaded["alive"].size == 4
    assert "genome_dist" in loaded.data
    assert "n_species" not in loaded.data
    assert "diet_max" not in loaded.data
    assert "meat_eaten" not in loaded.data
    assert "diet_p50" not in loaded.data
    assert "bite_herb" not in loaded.data
    assert "signal_emitted" not in loaded.data
    assert "signal_mass" not in loaded.data


def test_diet_max_tracks_living_diet() -> None:
    """diet_max is the max diet over living slots; 0 when none are alive."""
    cfg = Config(max_creatures=16, initial_creatures=0)
    w = World(cfg, seed=1)
    slots = w.spawn(3)
    g = cfg.brain_params
    w.genome[slots[0], g + 3] = 0.0
    w.genome[slots[1], g + 3] = 0.4
    w.genome[slots[2], g + 3] = 0.9
    w.energy[slots] = 50.0
    rec = StatsRecorder(capacity=4, sample_every=1, sample_size=4,
                        cluster_every=1)
    rec.record(w)
    assert float(rec.to_dict()["diet_max"][0]) == pytest.approx(0.9, abs=1e-6)

    w.alive[slots[2]] = False  # kill the max-diet creature
    rec.record(w)
    assert float(rec.to_dict()["diet_max"][1]) == pytest.approx(0.4, abs=1e-6)

    w.alive[:] = False
    rec.record(w)
    assert float(rec.to_dict()["diet_max"][2]) == 0.0


def test_meat_eaten_is_per_tick_delta() -> None:
    """meat_eaten column sums to the cumulative World counter."""
    cfg = Config(max_creatures=16, initial_creatures=8)
    w = World(cfg, seed=1)
    rec = StatsRecorder(capacity=8, sample_every=2, sample_size=8,
                        cluster_every=1)
    for _ in range(5):
        step.advance(w)
        rec.record(w)
    data = rec.to_dict()
    assert np.all(data["meat_eaten"] >= 0.0)
    total = float(data["meat_eaten"].sum())
    np.testing.assert_allclose(total, w.meat_eaten, rtol=1e-5)


def test_diet_percentiles_ordered_and_bounded() -> None:
    """p50 <= p90 <= p99 <= diet_max on sample ticks; 0 with no living."""
    cfg = Config(max_creatures=16, initial_creatures=0)
    w = World(cfg, seed=1)
    slots = w.spawn(5)
    g = cfg.brain_params
    for slot, diet in zip(slots, (0.0, 0.1, 0.35, 0.7, 0.95)):
        w.genome[slot, g + 3] = diet
        w.energy[slot] = 50.0
    rec = StatsRecorder(capacity=8, sample_every=1, sample_size=8,
                        cluster_every=1)
    rec.record(w)
    data = rec.to_dict()
    p50, p90, p99 = (float(data[k][0]) for k in ("diet_p50", "diet_p90",
                                                 "diet_p99"))
    # numpy percentiles of [0.0, 0.1, 0.35, 0.7, 0.95] (linear interp)
    assert p50 == pytest.approx(0.35, abs=1e-6)
    assert p90 == pytest.approx(0.85, abs=1e-6)
    assert p99 == pytest.approx(0.94, abs=1e-6)
    assert p50 <= p90 <= p99
    assert p99 <= float(data["diet_max"][0]) + 1e-6

    w.alive[:] = False
    rec.record(w)
    data = rec.to_dict()
    assert float(data["diet_p50"][1]) == 0.0
    assert float(data["diet_p90"][1]) == 0.0
    assert float(data["diet_p99"][1]) == 0.0


def test_diet_percentiles_forward_fill() -> None:
    """Percentiles update on sample ticks and forward-fill in between."""
    cfg = Config(max_creatures=16, initial_creatures=0)
    w = World(cfg, seed=2)
    slots = w.spawn(2)
    g = cfg.brain_params
    w.genome[slots[0], g + 3] = 0.2
    w.genome[slots[1], g + 3] = 0.6
    w.energy[slots] = 50.0
    rec = StatsRecorder(capacity=8, sample_every=4, sample_size=8,
                        cluster_every=1)
    rec.record(w)  # row 0 always samples
    tick0 = tuple(float(rec.to_dict()[k][0])
                  for k in ("diet_p50", "diet_p90", "diet_p99"))
    w.tick = 1
    rec.record(w)
    w.tick = 3
    rec.record(w)
    filled = rec.to_dict()
    for k, v in zip(("diet_p50", "diet_p90", "diet_p99"), tick0):
        assert np.array_equal(filled[k][1:3], np.full(2, v, dtype=np.float32))
    w.tick = 4  # sample tick: recompute
    rec.record(w)
    assert float(rec.to_dict()["diet_p50"][3]) == pytest.approx(0.4, abs=1e-6)


def test_bite_buckets_are_per_tick_deltas() -> None:
    """bite_herb/omni/pred deltas sum to World.bite_by_diet."""
    cfg = Config(max_creatures=16, initial_creatures=8, bite_rate=12.0)
    w = World(cfg, seed=1)
    rec = StatsRecorder(capacity=8, sample_every=2, sample_size=8,
                        cluster_every=1)
    for _ in range(5):
        step.advance(w)
        rec.record(w)
    data = rec.to_dict()
    for name in ("bite_herb", "bite_omni", "bite_pred"):
        assert np.all(data[name] >= 0), name
        assert int(data[name].sum()) == int(
            w.bite_by_diet[("bite_herb", "bite_omni", "bite_pred").index(name)]
        ), name


def test_signal_columns_match_world() -> None:
    """signal_emitted deltas sum to the counter; mass/emit_mean are gauges."""
    cfg = Config(
        max_creatures=16,
        initial_creatures=8,
        signal_emit_cost=0.2,
        signal_decay=0.1,
    )
    w = World(cfg, seed=1)
    rec = StatsRecorder(capacity=8, sample_every=2, sample_size=8,
                        cluster_every=1)
    for _ in range(5):
        step.advance(w)
        rec.record(w)
    data = rec.to_dict()
    assert np.all(data["signal_emitted"] >= 0.0)
    assert float(data["signal_emitted"].sum()) == pytest.approx(
        w.signal_emitted, rel=1e-5
    )
    # signal_mass is the live field sum; emit_mean in [0, 1] over living
    assert float(data["signal_mass"][-1]) == pytest.approx(
        float(w.signal.sum(dtype=np.float64)), rel=1e-5
    )
    assert np.all((data["emit_mean"] >= 0.0) & (data["emit_mean"] <= 1.0))
