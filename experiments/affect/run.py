"""Stage-24 experiment: do bite scars (stress) and rancor (grudge) matter?

Pre-registered protocol (decided before the runs, stage-24 plan):

- Question: on the proto-culture world (frequent bites, stage 12.5
  "reach" predation), does turning the affect channel ON
  (``affect_enabled=True``: inputs 35/36 = stress + grudge) change
  population health or show non-trivial internal state, compared with
  OFF (the same scratches are still written, but the brain reads zeros)?
- Variants: ``off`` vs ``on``, only the ``affect_enabled`` knob differs.
  Base: ``configs/proto_culture.toml`` (bite_rate=12, contact_range=2.5,
  emit 0.05, plasticity gene — the world where being bitten is real).
  Seeds 42/7/13, 10 000 ticks, sampled every 100 ticks.
- Honest nulls registered a priori: (H1) selection may not mobilize the
  new inputs in 10k ticks (no reward pressures the scars; the brain must
  discover stress->flee on its own) — then stress/grudge stay near
  chance and the population is identical within seed noise; (H2) the two
  extra inputs may shuffle the Numba RNG stream at spawn (layout bump)
  and produce trajectory noise without any causal channel — then "on"
  vs "off" differences match seed-to-seed variation of "off" alone.
- Hard gate (must hold for the treatment to be runnable at all): no
  extinction, final alive >= 0.9x of the same-seed control, min alive
  > 30. Measured gate (the interesting outcome either way): stress
  variance among the living > 0 after bites happen, and a non-zero
  share of the living carrying a fresh grudge after predation.

Metrics per sample: alive, births, cumulative deaths by cause, stress
mean/var among the living, grudge share (fraction of the living with a
live attacker memory). Sampled outside the tick — the stage-24 scratches
are not part of StatsRecorder (same precedent as the stage-23 observer).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._common import SEEDS, save_fig  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from primordia.config import load_config  # noqa: E402
from primordia.step import advance  # noqa: E402
from primordia.world import NO_MEMORY, World  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
PROTO = Path(__file__).resolve().parents[2] / "configs" / "proto_culture.toml"
SAMPLE_EVERY = 100
TICKS_DEFAULT = 10_000

VARIANTS = {
    "off": {"affect_enabled": False},
    "on": {"affect_enabled": True},
}

_COLUMNS = (
    "tick", "alive", "births",
    "deaths_famine", "deaths_age", "deaths_predation",
    "stress_mean", "stress_var", "grudge_frac",
)


def run_variant(
    overrides: dict, seed: int, ticks: int, npz: Path, force: bool = False
) -> dict[str, np.ndarray]:
    """Run one variant, sampling affect + health metrics every SAMPLE_EVERY.

    The npz is the cache (stale if the length or columns changed). Custom
    columns instead of StatsRecorder: stress/grudge are lifetime scratches
    (stage-24 decision, same precedent as the stage-23 observer) and this
    keeps other experiments' caches untouched.
    """
    n_samples = ticks // SAMPLE_EVERY + 1  # tick 0 included
    if npz.exists() and not force:
        data = np.load(npz)
        if len(data["tick"]) == n_samples and all(c in data for c in _COLUMNS):
            print(f"  seed={seed}: cached {npz.name}", flush=True)
            return {c: data[c] for c in _COLUMNS}
        print(f"  seed={seed}: stale, rerunning")

    base = load_config(str(PROTO))
    cfg = replace(base, **overrides)
    w = World(cfg, seed=seed)
    rows = {c: np.zeros(n_samples, dtype=np.float64) for c in _COLUMNS}

    def record(i: int) -> None:
        rows["tick"][i] = w.tick
        rows["alive"][i] = w.alive_count
        rows["births"][i] = w.births
        rows["deaths_famine"][i] = w.deaths_famine
        rows["deaths_age"][i] = w.deaths_age
        rows["deaths_predation"][i] = w.deaths_predation
        living = w.alive
        if w.alive_count:
            stress = w.stress[living]
            rows["stress_mean"][i] = float(stress.mean())
            rows["stress_var"][i] = float(stress.var())
            rows["grudge_frac"][i] = float(
                (w.attacker_age[living] < NO_MEMORY).mean()
            )
        else:
            rows["stress_mean"][i] = 0.0
            rows["stress_var"][i] = 0.0
            rows["grudge_frac"][i] = 0.0

    start = time.perf_counter()
    record(0)
    i = 1
    for t in range(ticks):
        advance(w)
        if (t + 1) % SAMPLE_EVERY == 0:
            record(i)
            i += 1
    elapsed = time.perf_counter() - start
    print(
        f"  seed={seed}: {ticks / elapsed:.1f} ticks/s ({elapsed:.1f} s), "
        f"alive={w.alive_count}, predation={w.deaths_predation}, "
        f"stress_var={rows['stress_var'][-1]:.3f}, "
        f"grudge={rows['grudge_frac'][-1]:.3f}",
        flush=True,
    )
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **rows)
    return rows


def _stack(histories: list[dict], column: str):
    tick = histories[0]["tick"]
    data = np.stack([h[column] for h in histories])
    return tick, data.mean(axis=0), data.min(axis=0), data.max(axis=0)


def _band(ax, histories: list[dict], column: str, name: str, color=None):
    tick, mean, lo, hi = _stack(histories, column)
    (line,) = ax.plot(tick, mean, label=name, color=color)
    ax.fill_between(tick, lo, hi, color=line.get_color(), alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-24 affect experiment (stress + grudge inputs)"
    )
    parser.add_argument("--ticks", type=int, default=TICKS_DEFAULT)
    parser.add_argument(
        "--force", action="store_true", help="ignore data/ cache and rerun"
    )
    args = parser.parse_args()

    all_hist: dict[str, list[dict]] = {}
    for name, overrides in VARIANTS.items():
        print(f"{name}: {overrides}", flush=True)
        runs = []
        for seed in SEEDS:
            npz = DATA / f"{name}_s{seed}.npz"
            runs.append(run_variant(overrides, seed, args.ticks, npz, args.force))
        all_hist[name] = runs

    FIGS.mkdir(parents=True, exist_ok=True)
    colors = {"off": "0.4", "on": "C3"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for name, runs in all_hist.items():
        _band(axes[0], runs, "alive", name, colors[name])
        _band(axes[1], runs, "stress_var", name, colors[name])
        _band(axes[2], runs, "grudge_frac", name, colors[name])
    axes[0].set_ylabel("alive")
    axes[0].set_title("Population (hard gate)")
    axes[1].set_ylabel("stress variance (living)")
    axes[1].set_title("Bite-scar signal")
    axes[2].set_ylabel("fraction with live grudge")
    axes[2].set_title("Rancor carrier share")
    for ax in axes:
        ax.set_xlabel("tick")
        axes[0].legend()
    fig.suptitle("Stage 24: affect channel off vs on — mean and band, 3 seeds")
    fig.tight_layout()
    save_fig(fig, FIGS / "affect.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, runs in all_hist.items():
        _band(ax, runs, "deaths_predation", name, colors[name])
    ax.set_xlabel("tick")
    ax.set_ylabel("cumulative predation deaths")
    ax.legend()
    ax.set_title("Predation pressure (sanity check)")
    save_fig(fig, FIGS / "predation.png")

    # console summary for the README numbers
    for name, runs in all_hist.items():
        print(f"\n{name}:")
        for col in ("alive", "births", "deaths_predation", "deaths_famine",
                    "deaths_age", "stress_var", "grudge_frac"):
            final = np.array([h[col][-1] for h in runs])
            peak = np.array([h[col].max() for h in runs])
            print(
                f"  {col:>17}: final {final.mean():.1f}"
                f" [{final.min():.0f}, {final.max():.0f}]"
                f"  peak {peak.max():.3f}"
            )
        mins = np.array([h["alive"].min() for h in runs])
        print(f"  {'min alive':>17}: {mins.min():.0f}")


if __name__ == "__main__":
    main()
