"""Experiment (stage 14): does the diffusing field build usable trails?

Question: stage 14 added a Jacobi pass that spreads the signal field over
the toroidal neighbors before each tick's sources emit — physics that
turns point deposits into extended trails (and thins the peaks). Two
questions, two phases:

  1. Price (this is also the stage-13 follow-up the AGENTS asked for
     before any stage-14+ experiment uses the channel): proto_culture's
     pinned signal_emit_cost = 0.5 collapsed the population in stage 13
     (33 alive at 10k, one seed extinct) while 0.001 survived as spam.
     Where is the viable price? Method: signal_emit_cost in
     {0, 0.01, 0.05, 0.15} x 3 seeds x 6000 ticks on proto_culture.toml,
     with signal_diffuse = 0 so the price is isolated from the new
     physics. Taxes at gate~0.7 are {0, 9%, 44%, 131%} of the metabolic
     cost (0.08). Chosen price = highest cost with 3/3 survival.

  2. Gate: at the chosen price, does diffusion change the field's
     structure (signal_cells: trail *area*), and does anything upstream
     move — population, emit gate, predation? Pre-registered outcome:
     physics (signal_cells up in mid/fast) is near-certain; an
     ecological/behavioral difference in 10k ticks is a priori *not*
     expected (stage 13 showed the channel being wired changes nothing
     on its own). If nothing moves, the README says so.

Hypothesis stated a priori before running: price sweep brackets the
viable window between 0.001 and 0.5; diffusion widens the trail area
(signal_cells: nodiff < mid < fast) while signal_mass stays equal
(diffusion only redistributes); ecology deltas null in 10k.

Output: figures in ``figs/``, raw histories in ``data/`` (gitignored),
summary tables on stdout for the README.

    python experiments/trails/run.py --phase price [--ticks N] [--force]
    python experiments/trails/run.py --phase gate --price 0.05
        [--ticks N] [--force]
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

# Price phase: the only lever is the emit tax, diffusion held off so the
# number is a property of the economy, not of the new physics.
PRICE_VARIANTS = [
    ("p0", 0.0),
    ("p01", 0.01),
    ("p05", 0.05),
    ("p15", 0.15),
]
# Gate phase: price pinned to the sweep's choice, the lever is the Jacobi
# rate (0 = control, 0.1 = new default, 0.2 = fast, stability limit 0.25).
GATE_VARIANTS = [
    ("nodiff", 0.0),
    ("mid", 0.1),
    ("fast", 0.2),
]


def main() -> None:
    parser = common_parser(__doc__)
    # common_parser defaults ticks to 5000; the per-phase defaults below
    # must win when the user did not pass --ticks explicitly
    parser.set_defaults(ticks=None)
    parser.add_argument(
        "--phase", choices=("price", "gate"), default="price",
        help="price = cost sweep at signal_diffuse=0 (6k default); "
             "gate = diffusion variants at --price (10k default)",
    )
    parser.add_argument(
        "--price", type=float, default=None,
        help="signal_emit_cost for --phase gate (from the price phase "
             "table; required there)",
    )
    args = parser.parse_args()
    if args.phase == "gate" and args.price is None:
        parser.error("--price is required for --phase gate (run price first)")
    if args.ticks is None:
        args.ticks = 6000 if args.phase == "price" else 10000

    base = load_config(str(PROTO))
    histories: dict = {}
    if args.phase == "price":
        variants = [(tag, {"signal_emit_cost": cost, "signal_diffuse": 0.0})
                    for tag, cost in PRICE_VARIANTS]
        prefix = "price_"
        for tag, overrides in variants:
            cfg = replace(base, **overrides)
            print(f"variant {tag} signal_emit_cost={cfg.signal_emit_cost} "
                  f"signal_diffuse=0", flush=True)
            for seed in SEEDS:
                histories[(tag, seed)] = run_variant(
                    cfg, seed, args.ticks,
                    HERE / "data" / f"{prefix}{tag}_seed{seed}.npz",
                    args.force,
                )
        _figures(histories, PRICE_VARIANTS, prefix)
        _print_price_table(histories)
    else:
        variants = [(tag, {"signal_emit_cost": args.price,
                           "signal_diffuse": rate})
                    for tag, rate in GATE_VARIANTS]
        price_tag = f"{args.price:g}"
        prefix = "gate_"
        for tag, overrides in variants:
            cfg = replace(base, **overrides)
            print(f"variant {tag} signal_diffuse={cfg.signal_diffuse} "
                  f"signal_emit_cost={cfg.signal_emit_cost}", flush=True)
            for seed in SEEDS:
                histories[(tag, seed)] = run_variant(
                    cfg, seed, args.ticks,
                    HERE / "data" / f"{prefix}{price_tag}_{tag}_seed{seed}.npz",
                    args.force,
                )
        _figures(histories, GATE_VARIANTS, prefix)
        _print_gate_table(histories, args.price)

    print(f"\nfigures -> {HERE / 'figs'}")


def _figures(histories: dict, variants: list, prefix: str) -> None:
    def hist(tag: str) -> list:
        return [histories[(tag, s)] for s in SEEDS]

    if prefix == "price_":
        # survival and the economy first: is there a price that does not
        # collapse the world (stage 13's 0.5 did)?
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for tag, _cost in variants:
            band(axes[0], hist(tag), "alive", tag)
            band(axes[1], hist(tag), "mean_energy", tag)
        axes[0].set_title("population (extinction = 0)")
        axes[1].set_title("mean energy")
        # gate response: does the sigmoid learn to close under the tax?
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for tag, _cost in variants:
            band(axes2[0], hist(tag), "emit_mean", tag)
            band(axes2[1], hist(tag), "signal_emitted", tag)
        axes2[0].set_title("mean emit-gate fraction over living creatures")
        axes2[1].set_title("signal deposited per tick")
    else:
        # the physics claim: trail area. Mass must stay flat across
        # variants — diffusion only redistributes, decay decides level.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for tag, _rate in variants:
            band(axes[0], hist(tag), "signal_cells", tag)
            band(axes[1], hist(tag), "signal_mass", tag)
        axes[0].set_title("trail area: cells > 1% of signal_capacity")
        axes[1].set_title("signal field mass (diffusion conserves it)")
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for tag, _rate in variants:
            band(axes2[0], hist(tag), "alive", tag)
            band(axes2[1], hist(tag), "emit_mean", tag)
        axes2[0].set_title("population")
        axes2[1].set_title("mean emit-gate fraction over living creatures")

    fig_names = ("economy", "gate") if prefix == "price_" else ("trails", "pop")
    for figure, name in ((fig, fig_names[0]), (fig2, fig_names[1])):
        for ax in figure.axes:
            ax.set_xlabel("tick")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")
        save_fig(figure, HERE / "figs" / f"{prefix}{name}.png")

    if prefix == "gate_":
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for tag, _rate in variants:
            band(axes3[0], hist(tag), "deaths_predation", tag)
            band(axes3[1], hist(tag), "diet_max", tag)
        axes3[0].set_title("predation deaths per tick")
        axes3[1].set_title("diet_max over living slots")
        for ax in axes3:
            ax.set_xlabel("tick")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")
        save_fig(fig3, HERE / "figs" / f"{prefix}predation.png")


def _final_mean(histories: dict, tag: str, column: str) -> float:
    hs = [histories[(tag, s)] for s in SEEDS]
    return sum(float(h[column][-1]) for h in hs) / len(hs)


def _total_mean(histories: dict, tag: str, column: str) -> float:
    hs = [histories[(tag, s)] for s in SEEDS]
    return sum(float(h[column].sum()) for h in hs) / len(hs)


def _print_price_table(histories: dict) -> None:
    print("\n| variant | cost | final pop (min/mean) | survived "
          "| emit_mean final | emitted total |")
    print("|---|---|---|---|---|---|")
    viable: list[tuple[float, str]] = []
    for tag, cost in PRICE_VARIANTS:
        finals = [float(histories[(tag, s)]["alive"][-1]) for s in SEEDS]
        ok = all(f > 0 for f in finals)
        if ok:
            viable.append((cost, tag))
        print(
            f"| {tag} | {cost:g} | {min(finals):.0f} / "
            f"{sum(finals) / len(finals):.0f} | {'yes' if ok else 'NO'} "
            f"| {_final_mean(histories, tag, 'emit_mean'):.4f} "
            f"| {_total_mean(histories, tag, 'signal_emitted'):.0f} |"
        )
    if viable:
        cost, tag = max(viable)
        print(f"\nchosen price: {cost:g} ({tag}) — highest swept cost with "
              f"3/3 survival; run --phase gate --price {cost:g}")
    else:
        print("\nNO swept price survived 3/3 seeds — report the null.")


def _print_gate_table(histories: dict, price: float) -> None:
    print(f"\nprice = {price:g} (chosen by the price phase)")
    print("\n| variant | diffuse | final pop (min/mean) | signal_cells "
          "| signal_mass | emit_mean final | predation |")
    print("|---|---|---|---|---|---|---|")
    for tag, rate in GATE_VARIANTS:
        finals = [float(histories[(tag, s)]["alive"][-1]) for s in SEEDS]
        print(
            f"| {tag} | {rate:g} | {min(finals):.0f} / "
            f"{sum(finals) / len(finals):.0f} "
            f"| {_final_mean(histories, tag, 'signal_cells'):.0f} "
            f"| {_final_mean(histories, tag, 'signal_mass'):.1f} "
            f"| {_final_mean(histories, tag, 'emit_mean'):.4f} "
            f"| {_total_mean(histories, tag, 'deaths_predation'):.0f} |"
        )


if __name__ == "__main__":
    main()
