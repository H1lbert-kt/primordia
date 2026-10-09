"""Experiment 3 (stage 8): can predation emerge under adverse conditions?

Question: starting from an all-herbivore population (diet = 0 at spawn),
does any honest pressure — scarce food alone, or scarce food plus a
richer meat reward — produce diet > 0 and predation deaths?

Method: 3 configs, 3 seeds each, 5000 ticks, per-tick stats recorded:
  control      defaults (food_growth_rate 0.05, bite_efficiency 0.7)
  scarce       food_growth_rate 0.01
  scarce_meat  food_growth_rate 0.01, bite_efficiency 0.95
Output: figures in ``figs/``, raw histories in ``data/`` (gitignored),
summary table on stdout for the README.

    python experiments/predation/run.py [--ticks N] [--force]
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
VARIANTS = [
    ("control", {}),
    ("scarce", {"food_growth_rate": 0.01}),
    ("scarce_meat", {"food_growth_rate": 0.01, "bite_efficiency": 0.95}),
]


def main() -> None:
    parser = common_parser(__doc__)
    args = parser.parse_args()

    histories: dict = {}
    for tag, overrides in VARIANTS:
        print(f"variant {tag} {overrides or '(defaults)'}", flush=True)
        cfg = Config(**(asdict(Config()) | STAGE8_BASE | overrides))
        for seed in SEEDS:
            histories[(tag, seed)] = run_variant(
                cfg, seed, args.ticks, HERE / "data" / f"{tag}_seed{seed}.npz",
                args.force,
            )

    def hist(tag: str) -> list:
        return [histories[(tag, s)] for s in SEEDS]

    # The emergence signal: diet trait leaving zero.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for tag, _ in VARIANTS:
        band(axes[0], hist(tag), "diet_mean", tag)
        band(axes[1], hist(tag), "diet_std", tag)
    axes[0].set_title("mean diet (0 = pure herbivore)")
    axes[1].set_title("diet standard deviation")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("diet trait evolution")
    save_fig(fig, HERE / "figs" / "diet.png")

    # Population and energy context (did the pressure actually bite?).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for tag, _ in VARIANTS:
        band(axes[0], hist(tag), "alive", tag)
        band(axes[1], hist(tag), "mean_energy", tag)
    axes[0].set_title("population")
    axes[1].set_title("mean energy")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save_fig(fig, HERE / "figs" / "context.png")

    # Total deaths by cause per variant.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    causes = (
        ("starvation", "deaths_famine", "tab:red"),
        ("old age", "deaths_age", "tab:gray"),
        ("predation", "deaths_predation", "tab:brown"),
    )
    x = range(len(VARIANTS))
    bottoms = [0.0] * len(VARIANTS)
    for name, column, color in causes:
        totals = [
            sum(float(sum(histories[(tag, s)][column])) for s in SEEDS) / len(SEEDS)
            for tag, _ in VARIANTS
        ]
        ax.bar(x, totals, bottom=bottoms, label=name, color=color, alpha=0.85)
        bottoms = [b + t for b, t in zip(bottoms, totals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([tag for tag, _ in VARIANTS])
    ax.set_ylabel("deaths over the run (mean of seeds)")
    ax.set_title("causes of death per variant")
    ax.legend()
    save_fig(fig, HERE / "figs" / "deaths.png")

    print("\n| variant | final pop | max diet_mean | final diet_mean "
          "| starvation | old age | predation |")
    print("|---|---|---|---|---|---|---|")
    for tag, _ in VARIANTS:
        hs = hist(tag)

        def final(column: str) -> float:
            return sum(float(h[column][-1]) for h in hs) / len(hs)

        def total(column: str) -> float:
            return sum(float(h[column].sum()) for h in hs) / len(hs)

        peak = sum(float(h["diet_mean"].max()) for h in hs) / len(hs)
        print(
            f"| {tag} | {final('alive'):.0f} | {peak:.4f} "
            f"| {final('diet_mean'):.4f} "
            f"| {total('deaths_famine'):.0f} | {total('deaths_age'):.0f} "
            f"| {total('deaths_predation'):.0f} |"
        )
    print(f"\nfigures -> {HERE / 'figs'}")


if __name__ == "__main__":
    main()
