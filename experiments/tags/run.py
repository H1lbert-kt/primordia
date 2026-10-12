"""Experiment (stage 15): does a neutral lineage colour build structure?

Question: stage 15 gives every creature a heritable hue (5th body trait,
circular mutation in [0,1); the tick never reads it for energy, cost or
reproduction — neutral by construction) plus one new input: the hue
similarity of the neighbours inside the smell window ("are the creatures
around me my colour?"). Nothing rewards using it.

Pre-registered expectations, stated before the 3-seed batch (a seed-42
10k pilot informed the refinement below; the README says so):

  H1 drift — hue is neutral, so global diversity is decided by drift:
     bottlenecks concentrate the wheel (hue_spread falls). No selection
     acts on the colour; the code never reads tag_hue for survival.

  H2 structure above chance — children are born at the parent's position
     wearing one parent's hue, so families form spatial patches. The
     brain's tag input should sit ABOVE two nested chance levels, both
     computable from the recorded columns:

         null_B = alive * (1 - hue_spread)^2 / n_cells
             (random hues AND random positions: the fully random case)
         null_A = win_neigh * (1 - hue_spread)^2 / 9
             (random hues GIVEN the observed spacing: separates "creatures
             cluster anyway" from "the cluster shares a colour")

     plus the spacing factor win_neigh / (9 * (alive-1) / n_cells), which
     starts at 1 for a random spawn and rises if creatures aggregate.

  H3 sensor use NOT claimed — separating "born together" from "steered
     by the similarity input" needs an ablation control whose genome
     layout differs (not comparable point-wise). Documented as future
     work; this experiment measures structure, not its cause.

Hard gate: every seed finishes the requested ticks with no extinction
(min alive > 0 over the recorded trajectory) and the stage-15 columns
present (``run_variant`` staleness already rejects older caches).

Output: figures in ``figs/``, raw histories in ``data/`` (gitignored),
tables on stdout for the README.

    python experiments/tags/run.py [--ticks 10000] [--force]
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
from primordia.config import Config  # noqa: E402

HERE = Path(__file__).resolve().parent


def _r2(hist):
    """|mean unit hue vector|^2 from the circular spread (spread = 1 - R)."""
    return (1.0 - hist["hue_spread"]) ** 2


def _null_b(hist, n_cells: int):
    """Fully random chance level: random hues AND random positions."""
    return hist["alive"] * _r2(hist) / float(n_cells)


def _null_a(hist):
    """Chance level given the observed spacing: random hues only."""
    return hist["win_neigh"] * _r2(hist) / 9.0


def _spacing(hist, n_cells: int):
    """Observed window crowding vs a random spawn (1.0 = no aggregation)."""
    denom = 9.0 * np.maximum(hist["alive"] - 1.0, 0.0) / float(n_cells)
    return hist["win_neigh"] / np.maximum(denom, 1e-9)


def main() -> None:
    parser = common_parser(__doc__)
    parser.set_defaults(ticks=10000)
    args = parser.parse_args()

    cfg = Config()
    n_cells = int(cfg.grid_shape[0] * cfg.grid_shape[1])
    histories = {}
    for seed in SEEDS:
        print(f"seed {seed} ({args.ticks} ticks, default Config)", flush=True)
        histories[seed] = run_variant(
            cfg, seed, args.ticks, HERE / "data" / f"seed{seed}.npz",
            args.force,
        )

    _figures(histories, n_cells)
    _print_table(histories, n_cells, args.ticks)
    print(f"\nfigures -> {HERE / 'figs'}")


def _figures(histories: dict, n_cells: int) -> None:
    hists = [histories[s] for s in SEEDS]

    # H1: neutral drift of the global colour distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for s in SEEDS:
        h = histories[s]
        axes[0].plot(h["tick"], h["hue_mean"], ".", ms=1, alpha=0.4,
                     label=f"seed {s}")
    band(axes[1], hists, "hue_spread", "hue_spread")
    axes[0].set_title("circular mean hue (neutral drift, wraps at 0/1)")
    axes[1].set_title("hue spread (1 = uniform wheel, 0 = one colour)")

    # H2: the brain's tag input vs two nested chance levels
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for s in SEEDS:
        h = histories[s]
        axes2[0].plot(h["tick"], _null_b(h, n_cells), lw=1, alpha=0.7,
                      label=f"null B seed {s}")
        axes2[0].plot(h["tick"], _null_a(h), lw=1, ls="--", alpha=0.7,
                      label=f"null A seed {s}")
        align = h["tag_sim_mean"] / np.maximum(_null_a(h), 1e-6)
        axes2[1].plot(h["tick"], align, lw=1, alpha=0.7, label=f"seed {s}")
        axes2[2].plot(h["tick"], _spacing(h, n_cells), lw=1, alpha=0.7,
                      label=f"seed {s}")
    band(axes2[0], hists, "tag_sim_mean", "tag_sim_mean (brain input)")
    axes2[0].set_title("tag input vs chance: B = random, A = given spacing")
    axes2[1].set_title("hue alignment beyond spacing (observed / null A)")
    axes2[1].set_yscale("log")
    axes2[2].set_title("spacing factor (1 = random spawn)")
    axes2[2].axhline(1.0, color="k", lw=0.8, alpha=0.6)
    axes2[1].set_yscale("log")

    # health, for honesty: the colour is not allowed to hide a collapse
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    band(axes3[0], hists, "alive", "alive")
    band(axes3[1], hists, "mean_energy", "mean_energy")
    axes3[0].set_title("population (extinction = 0)")
    axes3[1].set_title("mean energy")

    for figure, name in ((fig, "drift"), (fig2, "similarity"),
                         (fig3, "pop")):
        for ax in figure.axes:
            ax.set_xlabel("tick")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        save_fig(figure, HERE / "figs" / f"{name}.png")


def _print_table(histories: dict, n_cells: int, ticks: int) -> None:
    print("\n| seed | final alive | min alive | hue_spread final "
          "| tag_sim mean/final | win_neigh final | spacing final "
          "| align ratio final |")
    print("|---|---|---|---|---|---|---|---|")
    verdict = []
    for s in SEEDS:
        h = histories[s]
        alive = h["alive"]
        align = h["tag_sim_mean"] / np.maximum(_null_a(h), 1e-6)
        verdict.append(bool(alive.min() > 0 and alive[-1] > 0))
        print(
            f"| {s} | {float(alive[-1]):.0f} | {float(alive.min()):.0f} "
            f"| {float(h['hue_spread'][-1]):.3f} "
            f"| {float(h['tag_sim_mean'].mean()):.3f} / "
            f"{float(h['tag_sim_mean'][-1]):.3f} "
            f"| {float(h['win_neigh'][-1]):.2f} "
            f"| {float(_spacing(h, n_cells)[-1]):.1f} "
            f"| {float(align[-1]):.1f} |"
        )
    gate = all(verdict) and ticks > 0
    print(f"\nHARD GATE (no extinction, {ticks} ticks, 3 seeds): "
          f"{'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
