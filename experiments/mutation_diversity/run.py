"""Experiment 2 (stage 8): trait mutation rate vs. evolutionary diversity.

Question: does ``trait_mutation_std`` control how much diversity the
population maintains, and does a high rate hurt viability?

Method: ``trait_mutation_std {0.05, 0.10, 0.30}`` (0.10 is the default),
3 seeds each, 5000 ticks, per-tick stats recorded. Output: figures in
``figs/``, raw histories in ``data/`` (gitignored), summary table on
stdout for the README.

    python experiments/mutation_diversity/run.py [--ticks N] [--force]
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402

from experiments._common import (  # noqa: E402
    SEEDS,
    STAGE8_BASE,
    band,
    common_parser,
    run_variant,
    save_fig,
)
from primordia.config import Config  # noqa: E402

HERE = Path(__file__).resolve().parent
MUTS = (0.05, 0.10, 0.30)
VARIANTS = [
    (f"mut{m:g}", {"trait_mutation_std": m}) for m in MUTS
]


def main() -> None:
    parser = common_parser(__doc__)
    args = parser.parse_args()

    histories: dict = {}
    for (tag, overrides), m in zip(VARIANTS, MUTS):
        print(f"variant {tag} (trait_mutation_std={m:g})", flush=True)
        cfg = Config(**(asdict(Config()) | STAGE8_BASE | overrides))
        for seed in SEEDS:
            histories[(tag, seed)] = run_variant(
                cfg, seed, args.ticks, HERE / "data" / f"{tag}_seed{seed}.npz",
                args.force,
            )

    def hist(tag: str) -> list:
        return [histories[(tag, s)] for s in SEEDS]

    # Overall neutral diversity (sampled pairwise genome distance).
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for tag, m in zip([t for t, _ in VARIANTS], MUTS):
        band(ax, hist(tag), "genome_dist", f"mutation_std={m:g}")
    ax.set_xlabel("tick")
    ax.set_ylabel("genome distance (sampled pairs)")
    ax.set_title("neutral genome diversity")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, HERE / "figs" / "diversity.png")

    # Per-trait standard deviation (how much the body traits spread).
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    std_cols = ("speed_std", "size_std", "vision_std", "diet_std")
    for ax, col in zip(axes.flat, std_cols):
        for tag, m in zip([t for t, _ in VARIANTS], MUTS):
            band(ax, hist(tag), col, f"{m:g}")
        ax.set_title(col.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("tick")
    fig.suptitle("trait standard deviation over living creatures")
    save_fig(fig, HERE / "figs" / "trait_std.png")

    # Per-trait means (direction of any adaptation).
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    mean_cols = ("speed_mean", "size_mean", "vision_mean", "diet_mean")
    for ax, col in zip(axes.flat, mean_cols):
        for tag, m in zip([t for t, _ in VARIANTS], MUTS):
            band(ax, hist(tag), col, f"{m:g}")
        ax.set_title(col.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("tick")
    fig.suptitle("trait means over living creatures")
    save_fig(fig, HERE / "figs" / "traits.png")

    print("\n| variant | final pop | final mean energy | genome_dist "
          "| vision_std | diet_mean |")
    print("|---|---|---|---|---|---|")
    for tag, _ in VARIANTS:
        hs = hist(tag)

        def final(column: str) -> float:
            return sum(float(h[column][-1]) for h in hs) / len(hs)

        print(
            f"| {tag} | {final('alive'):.0f} | {final('mean_energy'):.0f} "
            f"| {final('genome_dist'):.3f} | {final('vision_std'):.3f} "
            f"| {final('diet_mean'):.4f} |"
        )
    print(f"\nfigures -> {HERE / 'figs'}")


if __name__ == "__main__":
    main()
