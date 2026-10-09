"""Experiment 4 (stage 8): lineage structure from the birth log.

Question: under ``max_age = 2000`` turnover, is the surviving population
descended from a few founder families, how deep does the tree get, and
when do births happen?

Method: defaults, 3 seeds, 5000 ticks. Each run caches the append-only
``genealogy`` log (tick, child_id, parent_id) plus the final alive ids in
``data/`` (gitignored); analysis traces every living creature back to its
founder. Output: figure in ``figs/``, summary table on stdout.

    python experiments/lineage/run.py [--ticks N] [--force]
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments._common import (  # noqa: E402
    SEEDS,
    STAGE8_BASE,
    common_parser,
    save_fig,
)
from primordia.config import Config  # noqa: E402
from primordia.step import advance  # noqa: E402
from primordia.world import World  # noqa: E402

HERE = Path(__file__).resolve().parent
BIN = 100  # ticks per birth histogram bin


def run_lineage(seed: int, ticks: int, npz: Path, force: bool) -> dict:
    """Run one seed and cache (genealogy, alive_ids); loads cache if fresh."""
    if npz.exists() and not force:
        d = np.load(npz)
        if int(d["ticks"]) == ticks:
            print(f"  seed={seed}: cached {npz.name}", flush=True)
            return {
                "genealogy": d["genealogy"],
                "alive_ids": d["alive_ids"],
                "overflow": int(d["overflow"]),
            }
        print(f"  seed={seed}: stale (ticks {int(d['ticks'])} != {ticks}), rerunning")

    w = World(Config(**STAGE8_BASE), seed=seed)
    start = time.perf_counter()
    for _ in range(ticks):
        advance(w)
    elapsed = time.perf_counter() - start
    genealogy = w.genealogy[: w.genealogy_used].copy()
    alive_ids = w.creature_id[w.alive].copy()
    print(
        f"  seed={seed}: {ticks / elapsed:.1f} ticks/s ({elapsed:.1f} s), "
        f"alive={w.alive_count}, births={len(genealogy)}",
        flush=True,
    )
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz,
        genealogy=genealogy,
        alive_ids=alive_ids,
        ticks=np.int64(ticks),
        seed=np.int64(seed),
        overflow=np.int64(w.genealogy_overflow),
    )
    return {
        "genealogy": genealogy,
        "alive_ids": alive_ids,
        "overflow": int(w.genealogy_overflow),
    }


def trace(genealogy: np.ndarray, alive_ids: np.ndarray) -> dict:
    """Family and depth of every living creature via the parent map."""
    parent = {int(c): int(p) for _, c, p in genealogy}
    roots, depths = [], []
    for cid in alive_ids:
        node = int(cid)
        depth = 0
        while node in parent:
            node = parent[node]
            depth += 1
        roots.append(node)
        depths.append(depth)
    return {
        "family_sizes": Counter(roots),
        "depths": np.asarray(depths),
    }


def main() -> None:
    parser = common_parser(__doc__)
    args = parser.parse_args()

    runs: dict = {}
    for seed in SEEDS:
        runs[seed] = run_lineage(
            seed, args.ticks, HERE / "data" / f"lineage_seed{seed}.npz",
            args.force,
        )

    analyses = {s: trace(r["genealogy"], r["alive_ids"]) for s, r in runs.items()}

    # (a) births per bin over time.
    nbins = args.ticks // BIN
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    counts = np.stack([
        np.bincount(
            (runs[s]["genealogy"][:, 0] - 1) // BIN, minlength=nbins
        )[:nbins]
        for s in SEEDS
    ])
    x = np.arange(nbins) * BIN + BIN // 2
    axes[0].plot(x, counts.mean(axis=0), color="C0")
    axes[0].fill_between(
        x, counts.min(axis=0), counts.max(axis=0), color="C0", alpha=0.2
    )
    axes[0].set_title(f"births per {BIN} ticks (mean, min-max band)")
    axes[0].set_xlabel("tick")
    axes[0].set_ylabel("births")
    axes[0].grid(alpha=0.3)

    # (b) cumulative share of the living population by family rank.
    for s in SEEDS:
        sizes = np.sort(
            np.asarray(list(analyses[s]["family_sizes"].values()), dtype=np.int64)
        )[::-1]
        share = np.cumsum(sizes) / sizes.sum()
        axes[1].plot(np.arange(1, len(share) + 1), share, label=f"seed={s}")
    axes[1].set_title("cumulative share of living population by family")
    axes[1].set_xlabel("family rank (sorted)")
    axes[1].set_ylabel("cumulative share")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("lineage structure at the end of the run")
    save_fig(fig, HERE / "figs" / "lineage.png")

    print("\n| seed | births | alive | families | top-10 share "
          "| max depth | median depth | overflow |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        a = analyses[s]
        sizes = np.sort(list(a["family_sizes"].values()))[::-1]
        top10 = float(sizes[:10].sum()) / float(sizes.sum())
        print(
            f"| {s} | {len(runs[s]['genealogy'])} | {len(runs[s]['alive_ids'])} "
            f"| {len(sizes)} | {top10:.3f} | {int(a['depths'].max())} "
            f"| {float(np.median(a['depths'])):.0f} | {runs[s]['overflow']} |"
        )
    print(f"\nfigure -> {HERE / 'figs' / 'lineage.png'}")


if __name__ == "__main__":
    main()
