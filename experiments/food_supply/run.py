"""Experiment 1 (stage 8): food supply x metabolic cost sweep.

Question: which economy keeps the population regulated instead of
exploding — and which Config defaults should ship?

Method: full factorial ``food_growth_rate {0.01, 0.05, 0.10}`` x
``metabolic_cost {0.05, 0.15}`` (0.05/0.05 are the current defaults),
3 seeds each, 5000 ticks, per-tick stats recorded. Output: figures in
``figs/``, raw histories in ``data/`` (gitignored), summary table on
stdout for the README.

    python experiments/food_supply/run.py [--ticks N] [--force]
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
FOOD = (0.01, 0.05, 0.10)
META = (0.05, 0.15)
VARIANTS = [
    (f"food{fg:g}_meta{mc:g}", {"food_growth_rate": fg, "metabolic_cost": mc})
    for fg in FOOD
    for mc in META
]


def main() -> None:
    parser = common_parser(__doc__)
    args = parser.parse_args()

    histories: dict = {}
    for tag, overrides in VARIANTS:
        print(f"variant {tag}", flush=True)
        cfg = Config(**(asdict(Config()) | STAGE8_BASE | overrides))
        for seed in SEEDS:
            histories[(tag, seed)] = run_variant(
                cfg, seed, args.ticks, HERE / "data" / f"{tag}_seed{seed}.npz",
                args.force,
            )

    def panels(column: str, ylabel: str, title: str, fname: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, mc in zip(axes, META):
            for fg in FOOD:
                tag = f"food{fg:g}_meta{mc:g}"
                band(
                    ax,
                    [histories[(tag, s)] for s in SEEDS],
                    column,
                    f"food_growth={fg:g}",
                )
            ax.set_title(f"metabolic_cost={mc:g}")
            ax.set_xlabel("tick")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title)
        save_fig(fig, HERE / "figs" / fname)

    panels("alive", "population", "population: food supply vs metabolism",
           "population.png")
    panels("mean_energy", "mean energy", "mean energy: food supply vs metabolism",
           "energy.png")

    # Total deaths by cause per variant (sum of per-tick deltas, mean of seeds).
    fig, ax = plt.subplots(figsize=(10, 4.5))
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
    ax.set_xticklabels([tag.replace("_", "\n") for tag, _ in VARIANTS], fontsize=8)
    ax.set_ylabel("deaths over the run (mean of seeds)")
    ax.set_title("causes of death per variant")
    ax.legend()
    save_fig(fig, HERE / "figs" / "deaths.png")

    # Markdown-ready summary for the README.
    print("\n| variant | final pop | final mean energy | starvation | old age "
          "| predation | final diet mean |")
    print("|---|---|---|---|---|---|---|")
    for tag, _ in VARIANTS:
        hs = [histories[(tag, s)] for s in SEEDS]

        def final(column: str) -> float:
            return sum(float(h[column][-1]) for h in hs) / len(hs)

        def total(column: str) -> float:
            return sum(float(h[column].sum()) for h in hs) / len(hs)

        print(
            f"| {tag} | {final('alive'):.0f} | {final('mean_energy'):.0f} "
            f"| {total('deaths_famine'):.0f} | {total('deaths_age'):.0f} "
            f"| {total('deaths_predation'):.0f} | {final('diet_mean'):.4f} |"
        )
    print(f"\nfigures -> {HERE / 'figs'}")


if __name__ == "__main__":
    main()
