"""Experiment (stage 12.5): can Config-only levers break the predation ratchet?

Question: diet is stuck near 0 because bite_rate * gate * diet is tiny at
low diet, so meat never pays and diet never rises (the ratchet documented
in experiments/predation/). Raising bite_rate makes even a small diet
worth something — does that alone produce a predator population?

Hypothesis (stated a priori, before re-running with the stage-12.5
instrumentation): the ratchet is economic, not structural — Config-only
levers that raise bite throughput (bite_rate, contact_range) will push
diet_max past ~0.4 and produce predation in the hundreds per run, but the
strict gate (diet > 0.5 alive in 3/3 seeds at 10k) will not be reached
because dilution-by-crossover keeps regressing lineages toward the
herbivore majority. That prediction is what the gate phase tests.

Method: 8 configs x 3 seeds (42, 7, 13) x 6000 ticks, per-tick stats
recorded on current HEAD defaults (terrain, day/night, seasons, sexual
reproduction all active — this experiment is about where the sim is now,
not the pinned stage-8 regime). Screening phase: cheap 6000-tick runs to
see if ANY config moves diet_max or predation deaths. The gate (3/3 seeds
with diet_max > 0.5 alive at 10k and mean predation >= 100/run) is
evaluated on a follow-up 10000-tick phase B with the top configs plus the
base control (--phase gate).

Variants (only Config overrides; no tick change, no default change):
  base        {}                                          (control)
  bigbite     bite_rate=12
  bigbite+    bite_rate=24
  reach       bite_rate=12, contact_range=2.5
  scarce      bite_rate=12, food_growth_rate=0.06
  meatloop    bite_rate=12, bite_efficiency=0.9, carrion_yield=0.6
  flat        bite_rate=12, terrain_amplitude=0
  aggressive  bite_rate=24, contact_range=2.5, food_growth_rate=0.06,
              bite_efficiency=0.9

Output: figures in ``figs/``, raw histories in ``data/`` (gitignored),
summary table on stdout for the README.

    python experiments/predator_survival/run.py [--ticks N] [--force]
        [--phase screen|gate]
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
# HEAD defaults; only the overrides below differ between variants.
VARIANTS = [
    ("base", {}),
    ("bigbite", {"bite_rate": 12.0}),
    ("bigbite+", {"bite_rate": 24.0}),
    ("reach", {"bite_rate": 12.0, "contact_range": 2.5}),
    ("scarce", {"bite_rate": 12.0, "food_growth_rate": 0.06}),
    ("meatloop", {"bite_rate": 12.0, "bite_efficiency": 0.9,
                  "carrion_yield": 0.6}),
    ("flat", {"bite_rate": 12.0, "terrain_amplitude": 0.0}),
    ("aggressive", {"bite_rate": 24.0, "contact_range": 2.5,
                    "food_growth_rate": 0.06, "bite_efficiency": 0.9}),
]
# Phase B (gate): top screening variants + base control, 10k ticks.
GATE_VARIANTS = [
    ("base", {}),
    ("reach", {"bite_rate": 12.0, "contact_range": 2.5}),
    ("bigbite+", {"bite_rate": 24.0}),
]


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument(
        "--phase", choices=("screen", "gate"), default="screen",
        help="screen = 8 variants (6k default); gate = base/reach/bigbite+ "
             "at 10k with the gate metrics",
    )
    parser.set_defaults(ticks=6000)  # screen default; gate bumps to 10k below
    args = parser.parse_args()
    variants = VARIANTS if args.phase == "screen" else GATE_VARIANTS
    prefix = "" if args.phase == "screen" else "phaseB_"
    if args.phase == "gate" and args.ticks == 6000:
        args.ticks = 10000  # user did not pass --ticks: gate default

    histories: dict = {}
    for tag, overrides in variants:
        print(f"variant {tag} {overrides or '(defaults)'}", flush=True)
        for seed in SEEDS:
            histories[(tag, seed)] = run_variant(
                overrides,
                seed,
                args.ticks,
                HERE / "data" / f"{tag}_seed{seed}.npz",
                args.force,
            )

    def hist(tag: str) -> list:
        return [histories[(tag, s)] for s in SEEDS]

    # 1. The ratchet signal: diet_max leaving zero and climbing, plus the
    # stage-12.5 percentile band (how far the *distribution* moved).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag, _ in variants:
        band(axes[0], hist(tag), "diet_max", tag)
        band(axes[1], hist(tag), "diet_mean", tag)
    axes[0].set_title("diet_max over living slots (>= 0.5 = predator alive)")
    axes[1].set_title("diet_mean (0 = pure herbivore)")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("diet trait evolution")
    save_fig(fig, HERE / "figs" / f"{prefix}diet_max.png")

    # 1b. Diet distribution percentiles (stage 12.5): the ratchet question
    # is not "did one outlier climb" but "did the body of the population
    # move". p50/p90/p99 + max for the strongest variant.
    pick = "reach"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, color in (("diet_max", "tab:red"), ("diet_p99", "tab:orange"),
                        ("diet_p90", "tab:olive"), ("diet_p50", "tab:green")):
        band(ax, hist(pick), name, f"{pick}:{name}", color=color)
    ax.set_title(f"{pick}: diet distribution over living slots "
                 "(p50 <= p90 <= p99 <= max)")
    ax.set_xlabel("tick")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, HERE / "figs" / f"{prefix}diet_dist.png")

    # 2. Predation deaths and omnivore/predator gauges.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag, _ in variants:
        hs = hist(tag)
        tick = hs[0]["tick"]
        # per-tick deltas stacked (3 seeds, T), cumulative over ticks,
        # then mean across seeds
        pred = np.stack([h["deaths_predation"] for h in hs])
        pred = np.cumsum(pred, axis=1).mean(axis=0)
        axes[0].plot(tick, pred, label=tag, alpha=0.8)
        band(axes[1], hs, "n_predator", tag)
    axes[0].set_title("cumulative predation deaths")
    axes[1].set_title("n_predator (diet > 0.66 gauge)")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, HERE / "figs" / f"{prefix}predation.png")

    # 2b. Successful takes grouped by the attacker's diet class
    # (stage 12.5): separates "herbivores nipping" from real predation.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for tag, _ in variants:
        hs = hist(tag)
        tick = hs[0]["tick"]
        bites = np.stack([h["bite_herb"] + h["bite_omni"] + h["bite_pred"]
                          for h in hs])
        bites = np.cumsum(bites, axis=1).mean(axis=0)
        ax.plot(tick, bites, label=tag, alpha=0.8)
    ax.set_title("cumulative successful takes (any attacker diet class)")
    ax.set_xlabel("tick")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, HERE / "figs" / f"{prefix}bites.png")

    # 3. Context: population and meat_eaten (did meat actually flow?).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag, _ in variants:
        band(axes[0], hist(tag), "alive", tag)
        band(axes[1], hist(tag), "meat_eaten", tag)
    axes[0].set_title("population")
    axes[1].set_title("meat_eaten per tick (carrion energy extracted)")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, HERE / "figs" / f"{prefix}context.png")

    # 4. Causes of death (stacked bars, mean of seeds).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    causes = (
        ("famine", "deaths_famine", "tab:red"),
        ("old age", "deaths_age", "tab:gray"),
        ("predation", "deaths_predation", "tab:brown"),
    )
    x = np.arange(len(variants))
    width = 0.25
    for k, (name, column, color) in enumerate(causes):
        totals = [
            sum(float(h[column].sum()) for h in hist(tag)) / len(SEEDS)
            for tag, _ in variants
        ]
        ax.bar(x + (k - 1) * width, totals, width, label=name, color=color,
               alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([tag for tag, _ in variants], rotation=30, ha="right")
    ax.set_ylabel("deaths over the run (mean of seeds)")
    ax.set_title("causes of death per variant")
    ax.legend()
    save_fig(fig, HERE / "figs" / f"{prefix}deaths.png")

    _print_table(histories, variants)
    print(f"\nfigures -> {HERE / 'figs'}")


def _print_table(histories: dict, variants: list) -> None:
    print("\n| variant | final pop | final diet_max | diet p50/p90/p99 "
          "| predation | takes h/o/p | meat_eaten (total) | famine | old age |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag, _ in variants:
        hs = [histories[(tag, s)] for s in SEEDS]

        def final(column: str) -> float:
            return sum(float(h[column][-1]) for h in hs) / len(hs)

        def total(column: str) -> float:
            return sum(float(h[column].sum()) for h in hs) / len(hs)

        takes = tuple(total(c) for c in ("bite_herb", "bite_omni",
                                         "bite_pred"))
        print(
            f"| {tag} | {final('alive'):.0f} | {final('diet_max'):.4f} "
            f"| {final('diet_p50'):.3f}/{final('diet_p90'):.3f}"
            f"/{final('diet_p99'):.3f} "
            f"| {total('deaths_predation'):.0f} "
            f"| {takes[0]:.0f}/{takes[1]:.0f}/{takes[2]:.0f} "
            f"| {total('meat_eaten'):.0f} | {total('deaths_famine'):.0f} "
            f"| {total('deaths_age'):.0f} |"
        )


if __name__ == "__main__":
    main()
