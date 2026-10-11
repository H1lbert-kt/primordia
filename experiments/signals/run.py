"""Experiment (stage 13): does pricing the emit gate change anything?

Question: the signal channel is now wired (field + emit gate + 3 sensors),
but the Config default costs nothing to use. Is that free lunch hiding a
channel that is always "on" (a noise carpet), and does putting a price on
emission change the population, the diet balance or the predation metrics
the proto-culture world (configs/proto_culture.toml: bite_rate=12,
contact_range=2.5) was pinned for in stage 12.5?

Hypothesis (stated a priori, before any run on this code): the price of
the emit gate is an energy tax that the population will route around —
creatures will learn to close the gate (emit_mean drifts toward 0 as
signal_emit_cost rises), so "pricey" runs will show lower signal mass and
no systematic change in population or predation versus "proto". Whether
that is what happens is exactly what the gate phase tests: the honest
outcome here may well be a null result (the channel is wired but unused),
which the README will say in those words.

Method: 3 variants (all on proto_culture.toml, only signal_emit_cost
differs) x 3 seeds (42, 7, 13) x 10000 ticks, per-tick stats recorded.
Screening phase (--phase screen, default): cheap 6k runs to see if the
cost knob moves signal_emitted / emit_mean / signal_mass at all. Gate
phase (--phase gate): the same 3 variants at 10k with the gate metrics
(no extinction in any seed; signal_emitted > 0 in >= 1/3 seeds at the
default cost 0.5).

Variants (only Config overrides on top of proto_culture.toml):
  proto   signal_emit_cost=0.5   (the pinned proto-culture price)
  cheap   signal_emit_cost=0.001 (near-free: does the gate stay open?)
  pricey  signal_emit_cost=2.0   (3x the metabolic cost: close it or starve)

Output: figures in ``figs/``, raw histories in ``data/`` (gitignored),
summary table on stdout for the README.

    python experiments/signals/run.py [--ticks N] [--force]
        [--phase screen|gate]
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402

from experiments._common import (  # noqa: E402
    SEEDS,
    band,
    common_parser,
    run_variant,
    save_fig,
)
from primordia.config import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTO = Path(__file__).resolve().parents[2] / "configs" / "proto_culture.toml"
# signal_emit_cost is the only lever; everything else is the pinned
# predation context from stage 12.5.
VARIANTS = [
    ("proto", {"signal_emit_cost": 0.5}),
    ("cheap", {"signal_emit_cost": 0.001}),
    ("pricey", {"signal_emit_cost": 2.0}),
]


def main() -> None:
    parser = common_parser(__doc__)
    parser.add_argument(
        "--phase", choices=("screen", "gate"), default="screen",
        help="screen = 3 variants (6k default); gate = same 3 at 10k with "
             "the pre-registered gate metrics",
    )
    parser.set_defaults(ticks=6000)  # screen default; gate bumps to 10k below
    args = parser.parse_args()
    prefix = "" if args.phase == "screen" else "phaseB_"
    if args.phase == "gate" and args.ticks == 6000:
        args.ticks = 10000  # user did not pass --ticks: gate default

    base = load_config(str(PROTO))
    histories: dict = {}
    for tag, overrides in VARIANTS:
        cfg = replace(base, **overrides)
        print(f"variant {tag} signal_emit_cost={cfg.signal_emit_cost}", flush=True)
        for seed in SEEDS:
            histories[(tag, seed)] = run_variant(
                cfg,
                seed,
                args.ticks,
                HERE / "data" / f"{tag}_seed{seed}.npz",
                args.force,
            )

    def hist(tag: str) -> list:
        return [histories[(tag, s)] for s in SEEDS]

    # 1. Did anyone emit? signal_emitted (per-tick delta) and the live
    # field mass — the direct read on whether the channel carries anything.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag, _ in VARIANTS:
        band(axes[0], hist(tag), "signal_emitted", tag)
        band(axes[1], hist(tag), "signal_mass", tag)
    axes[0].set_title("signal deposited per tick (emit_rate * gate * alive)")
    axes[1].set_title("signal field mass (deposit_rate / decay equilibrium)")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("signal channel activity")
    save_fig(fig, HERE / "figs" / f"{prefix}signal_activity.png")

    # 2. Did the price close the gate? emit_mean over living slots.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for tag, _ in VARIANTS:
        band(ax, hist(tag), "emit_mean", tag)
    ax.set_title("mean emit-gate fraction over living creatures "
                 "(0 = closed, 0.5 = random-logit baseline)")
    ax.set_xlabel("tick")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, HERE / "figs" / f"{prefix}emit_gate.png")

    # 3. Did the tax change the population or the energy economy?
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag, _ in VARIANTS:
        band(axes[0], hist(tag), "alive", tag)
        band(axes[1], hist(tag), "mean_energy", tag)
    axes[0].set_title("population")
    axes[1].set_title("mean energy")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, HERE / "figs" / f"{prefix}economy.png")

    # 4. Did the predation context change? predation deaths + diet_max.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag, _ in VARIANTS:
        band(axes[0], hist(tag), "deaths_predation", tag)
        band(axes[1], hist(tag), "diet_max", tag)
    axes[0].set_title("predation deaths per tick")
    axes[1].set_title("diet_max over living slots")
    for ax in axes:
        ax.set_xlabel("tick")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    save_fig(fig, HERE / "figs" / f"{prefix}predation.png")

    _print_table(histories)
    print(f"\nfigures -> {HERE / 'figs'}")


def _print_table(histories: dict) -> None:
    print("\n| variant | final pop | emit_mean final | signal_emitted "
          "(total) | signal_mass final | predation | diet_max final |")
    print("|---|---|---|---|---|---|---|")
    for tag, _ in VARIANTS:
        hs = [histories[(tag, s)] for s in SEEDS]

        def final(column: str) -> float:
            return sum(float(h[column][-1]) for h in hs) / len(hs)

        def total(column: str) -> float:
            return sum(float(h[column].sum()) for h in hs) / len(hs)

        print(
            f"| {tag} | {final('alive'):.0f} | {final('emit_mean'):.4f} "
            f"| {total('signal_emitted'):.0f} | {final('signal_mass'):.1f} "
            f"| {total('deaths_predation'):.0f} | {final('diet_max'):.4f} |"
        )


if __name__ == "__main__":
    main()
