"""Stage-23 experiment: do founder lineages produce countries and borders?

Pre-registered protocol (decided before the runs, stage-23 plan):

- Question: with 4 founding lineages pinned to 4 hue bands in 4 torus
  strips (``founder_regions=4``), does the observer (hue-group detector,
  ``primordia.observer``) see stable groups, frontiers and contact events
  — compared with the free run (``founder_regions=0``) where hue groups
  are arbitrary colour bands of a mixed population?
- Variants: ``default`` (off) vs ``founders4`` (regions=4, spread=0.02),
  default Config otherwise (terrain on — water is a free isolation
  mechanism). Seeds 42/7/13, 10 000 ticks, observer sampled every 100
  ticks (pinned; the viewer uses 30 frames).
- Honest nulls registered a priori: (H1) founder lineages may homogenize
  early via interbreeding/dispersal — then n_groups collapses toward the
  free-run level and borders fade; (H2) detection may flicker (uid
  matching is a convention) — then event counts inflate with "new"/
  "extinct" churn while pops stay stable; (H3) free runs may show *many*
  hue bands (random spawn) without any spatial structure (borders ~ 0).

Metrics per sample: n_groups, border_cells, cumulative event counts by
kind (new/extinct/fusion/invasion), alive. Observer output only — the
tick is untouched (the stage-23 bench pinned bit-identical alive_final).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._common import SEEDS, save_fig  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from primordia.config import Config  # noqa: E402
from primordia.observer import GroupObserver  # noqa: E402
from primordia.step import advance  # noqa: E402
from primordia.world import World  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
SAMPLE_EVERY = 100
TICKS_DEFAULT = 10_000

VARIANTS = {
    "default": {"founder_regions": 0},
    "founders4": {"founder_regions": 4, "founder_hue_spread": 0.02},
}

_COLUMNS = (
    "tick", "alive", "n_groups", "border_cells",
    "ev_new", "ev_extinct", "ev_fusion", "ev_invasion",
)


def run_variant(
    overrides: dict, seed: int, ticks: int, npz: Path, force: bool = False
) -> dict[str, np.ndarray]:
    """Run one variant, sampling the observer every SAMPLE_EVERY ticks.

    The npz is the cache (stale if the length changed); columns are the
    observer's — ``StatsRecorder`` is deliberately not involved so no
    other experiment's cache is invalidated by stage 23.
    """
    n_samples = ticks // SAMPLE_EVERY + 1  # tick 0 included
    if npz.exists() and not force:
        data = np.load(npz)
        if len(data["tick"]) == n_samples and all(c in data for c in _COLUMNS):
            print(f"  seed={seed}: cached {npz.name}", flush=True)
            return {c: data[c] for c in _COLUMNS}
        print(f"  seed={seed}: stale, rerunning")

    cfg = Config(**overrides)
    w = World(cfg, seed=seed)
    obs = GroupObserver(cfg)
    rows = {c: np.zeros(n_samples, dtype=np.float64) for c in _COLUMNS}
    cumulative = {k: 0 for k in ("new", "extinct", "fusion", "invasion")}

    def record(i: int, snap) -> None:
        rows["tick"][i] = w.tick
        rows["alive"][i] = w.alive_count
        rows["n_groups"][i] = snap.n_groups
        rows["border_cells"][i] = snap.border_cells
        for ev in snap.events:
            cumulative[ev.kind] += 1
        rows["ev_new"][i] = cumulative["new"]
        rows["ev_extinct"][i] = cumulative["extinct"]
        rows["ev_fusion"][i] = cumulative["fusion"]
        rows["ev_invasion"][i] = cumulative["invasion"]

    import time

    start = time.perf_counter()
    record(0, obs.sample(w))
    i = 1
    for t in range(ticks):
        advance(w)
        if (t + 1) % SAMPLE_EVERY == 0:
            record(i, obs.sample(w))
            i += 1
    elapsed = time.perf_counter() - start
    print(
        f"  seed={seed}: {ticks / elapsed:.1f} ticks/s ({elapsed:.1f} s), "
        f"alive={w.alive_count}, groups={rows['n_groups'][-1]:.0f}, "
        f"borders={rows['border_cells'][-1]:.0f}",
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
        description="Stage-23 founder-scenario experiment (observer groups)"
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
    colors = {"default": "0.4", "founders4": "C3"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, runs in all_hist.items():
        _band(ax, runs, "n_groups", name, colors[name])
    ax.set_xlabel("tick")
    ax.set_ylabel("observer groups (hue countries)")
    ax.legend()
    ax.set_title("Group count — mean and min/max band across seeds")
    save_fig(fig, FIGS / "groups.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, runs in all_hist.items():
        _band(ax, runs, "border_cells", name, colors[name])
    ax.set_xlabel("tick")
    ax.set_ylabel("frontier cells")
    ax.legend()
    ax.set_title("Border length — mean and min/max band across seeds")
    save_fig(fig, FIGS / "borders.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, kind in zip(
        axes.ravel(), ("ev_new", "ev_extinct", "ev_fusion", "ev_invasion")
    ):
        for name, runs in all_hist.items():
            _band(ax, runs, kind, name, colors[name])
        ax.set_title(kind.replace("ev_", ""))
        ax.set_ylabel("cumulative events")
    for ax in axes[1]:
        ax.set_xlabel("tick")
    axes[0][0].legend()
    fig.suptitle("Observer events — mean and band across seeds")
    fig.tight_layout()
    save_fig(fig, FIGS / "events.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, runs in all_hist.items():
        _band(ax, runs, "alive", name, colors[name])
    ax.set_xlabel("tick")
    ax.set_ylabel("alive")
    ax.legend()
    ax.set_title("Population (health check)")
    save_fig(fig, FIGS / "pop.png")

    # console summary for the README numbers
    for name, runs in all_hist.items():
        print(f"\n{name}:")
        for col in ("n_groups", "border_cells", "ev_invasion", "ev_fusion",
                    "ev_extinct", "alive"):
            final = np.array([h[col][-1] for h in runs])
            peak = np.array([h[col].max() for h in runs])
            print(
                f"  {col:>13}: final {final.mean():.1f}"
                f" [{final.min():.0f}, {final.max():.0f}]"
                f"  peak {peak.max():.0f}"
            )


if __name__ == "__main__":
    main()
