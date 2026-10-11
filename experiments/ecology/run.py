"""Experiment (stage 12): do species clusters and a reproductive barrier emerge?

Question: with the stage-11 sexual dynamics on the stage-10 environment,
does the population split into genome clusters that actually *refuse* to
mate (reproductive-barrier metric), and how much of that structure is
maintained by predation?

Method: 2 variants x 3 seeds x 10000 ticks, per-tick stats recorded.
  default      bite_efficiency 0.3 (the calibrated stage-11 value)
  no_predation bite_efficiency 0  (herbivory only; isolates the effect of
               meat on cluster structure and mate choice)
Both run on the *current* Config defaults (terrain, day/night, seasons,
carrion, sexual reproduction) — this experiment asks about HEAD, not about
the pinned stage-8 regime. cluster_every=100 (the species columns are
sampled, not per-tick).

Metrics per tick (stage-12 stats columns):
  n_species   greedy-leader clusters over living genomes (ecology.py)
  barrier     fraction of logged births since the last sample whose two
              *living* parents carried different cluster labels
  matings_valid  denominator of barrier (births with both parents alive)
  n_herbivore/n_omnivore/n_predator  diet-class gauges
  genome_dist  sampled genetic diversity (RMS pairwise distance)
  deaths_predation  predation deaths (0 in no_predation by construction)

Output: figures in ``figs/``, raw histories in ``data/`` (gitignored),
summary table on stdout for the README.

    python experiments/ecology/run.py [--ticks N] [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments._common import (  # noqa: E402
    SEEDS,
    band,
    common_parser,
    run_variant,
    save_fig,
)

HERE = Path(__file__).resolve().parent
# HEAD defaults; only bite_efficiency differs between the two variants.
VARIANTS = [
    ("default", {}),
    ("no_predation", {"bite_efficiency": 0.0}),
]
CLUSTER_EVERY = 100


def main() -> None:
    parser = common_parser(__doc__)
    args = parser.parse_args()

    histories: dict = {}
    for tag, overrides in VARIANTS:
        print(f"variant {tag} {overrides or '(defaults)'}", flush=True)
        for seed in SEEDS:
            histories[(tag, seed)] = run_variant(
                overrides,
                seed,
                args.ticks,
                HERE / "data" / f"{tag}_seed{seed}.npz",
                args.force,
                cluster_every=CLUSTER_EVERY,
            )

    def hist(tag: str) -> list:
        return [histories[(tag, s)] for s in SEEDS]

    # 1. Species count and genome diversity: is the population structured?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for tag, _ in VARIANTS:
        band(axes[0], hist(tag), "n_species", tag)
        band(axes[1], hist(tag), "genome_dist", tag)
    axes[0].set_title("n_species (greedy-leader clusters)")
    axes[1].set_title("genome_dist (RMS pairwise distance)")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("genome clustering over time")
    save_fig(fig, HERE / "figs" / "clusters.png")

    # 2. Reproductive barrier: do clusters actually refuse to mix?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for tag, _ in VARIANTS:
        band(axes[0], hist(tag), "barrier", tag)
        band(axes[1], hist(tag), "matings_valid", tag)
    axes[0].set_title("barrier (cross-cluster fraction, both parents alive)")
    axes[1].set_title("matings_valid (barrier denominator)")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("reproductive barrier")
    save_fig(fig, HERE / "figs" / "barrier.png")

    # 3. Diet classes stacked: does predation reshape the trophic mix?
    fig, ax = plt.subplots(figsize=(8, 4.5))
    classes = (
        ("herbivore", "n_herbivore", "tab:green"),
        ("omnivore", "n_omnivore", "tab:orange"),
        ("predator", "n_predator", "tab:red"),
    )
    x = np.arange(len(VARIANTS))
    width = 0.25
    for k, (name, column, color) in enumerate(classes):
        totals = [
            sum(float(h[column][-1]) for h in hist(tag)) / len(SEEDS)
            for tag, _ in VARIANTS
        ]
        ax.bar(x + (k - 1) * width, totals, width, label=name, color=color,
               alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([tag for tag, _ in VARIANTS])
    ax.set_ylabel("final population (mean of seeds)")
    ax.set_title("diet classes at the final tick")
    ax.legend()
    save_fig(fig, HERE / "figs" / "diet_classes.png")

    # 4. Cross-correlation: does food supply lead population (season forcing)?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for tag, _ in VARIANTS:
        hs = hist(tag)
        # food field is not in the stats columns; use mean_energy as the
        # resource proxy and alive as the response, both standardized.
        for name, column, color in (("mean_energy", "mean_energy", "tab:blue"),
                                    ("alive", "alive", "tab:red")):
            tick, mean, _, _ = _stack_mean(hs, column)
            z = (mean - mean.mean()) / (mean.std() + 1e-9)
            axes[0].plot(tick, z, label=f"{tag}:{name}", color=color,
                         alpha=0.6 if tag == "no_predation" else 1.0)
        lag, xcorr = _cross_corr(hs, "mean_energy", "alive", max_lag=400)
        axes[1].plot(lag, xcorr, label=tag)
    axes[0].set_title("standardized mean_energy vs alive")
    axes[0].set_xlabel("tick")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_title("cross-corr energy->pop (positive = energy leads)")
    axes[1].set_xlabel("lag (ticks)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    save_fig(fig, HERE / "figs" / "cross_corr.png")

    _print_table(histories)
    print(f"\nfigures -> {HERE / 'figs'}")


def _stack_mean(histories, column):
    tick = histories[0]["tick"]
    data = np.stack([h[column] for h in histories])
    return tick, data.mean(axis=0), data.min(axis=0), data.max(axis=0)


def _cross_corr(histories, a: str, b: str, max_lag: int = 400):
    """Mean cross-correlation of standardized series across seeds."""
    lags = np.arange(-max_lag, max_lag + 1)
    acc = np.zeros(lags.size)
    for h in histories:
        x = h[a].astype(np.float64)
        y = h[b].astype(np.float64)
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        n = x.size
        for i, lag in enumerate(lags):
            if lag < 0:
                acc[i] += float(np.dot(x[-lag:], y[:n + lag]) / (n + lag))
            elif lag > 0:
                acc[i] += float(np.dot(x[:n - lag], y[lag:]) / (n - lag))
            else:
                acc[i] += float(np.dot(x, y) / n)
    return lags, acc / len(histories)


def _print_table(histories: dict) -> None:
    tags = [tag for tag, _ in VARIANTS]
    print("\n| variant | final pop | mean n_species | mean barrier "
          "| mean matings_valid | predation deaths | genome_dist (final) |")
    print("|---|---|---|---|---|---|---|")
    for tag in tags:
        hs = [histories[(tag, s)] for s in SEEDS]

        def final(column: str) -> float:
            return sum(float(h[column][-1]) for h in hs) / len(hs)

        def mean(column: str) -> float:
            return sum(float(h[column].mean()) for h in hs) / len(hs)

        def total(column: str) -> float:
            return sum(float(h[column].sum()) for h in hs) / len(hs)

        print(
            f"| {tag} | {final('alive'):.0f} | {mean('n_species'):.1f} "
            f"| {mean('barrier'):.3f} | {mean('matings_valid'):.0f} "
            f"| {total('deaths_predation'):.0f} | {final('genome_dist'):.2f} |"
        )


if __name__ == "__main__":
    main()
