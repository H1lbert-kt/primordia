"""Experiment (stage 16): does a donation channel survive selection?

Question: stage 16 opens a physical gift — the donate gate (output 4)
sends ``donate_rate * gate`` energy to the nearest living neighbour
inside ``donate_range`` every tick. Nothing rewards giving (principle 1):
if gifts persist it must be because recipients return the favour in the
currencies the sim already has (kinship: children are born in place, and
stage-15 clusters are 9-23x chance; or reciprocity), never because the
tick says so.

Variants on ``configs/proto_culture.toml`` (reach + emit 0.05), the
variant axis is the gift price:

  off    donate_rate = 0     control (config identical otherwise)
  cheap  donate_rate = 0.02  ~25% of metabolism/tick at gate=1
  dear   donate_rate = 0.10  ~125% of metabolism/tick at gate=1

Pre-registered outcomes (stated before any run):

  Hard gate: every run completes 10k ticks with min alive > 0.
  H1 (selection against the leak): ``donate_mean`` FALLS from the first
      20% to the last 20% of the run in cheap/dear — the sigmoid's
      floor costs energy with no immediate return, so the gate should
      be selected to close.
  H2 (gifts follow kin): ``donate_align`` above chance. Chance for the
      circular hue similarity of a random pair is ``(1 - hue_spread)^2``
      (stage 15's null); the observed mean over gift pairs sits above it
      iff gifts preferentially reach colour-mates = kin clusters.
      Conservative on purpose: ticks with no gift pair read 0 on the
      observed side only, which biases *against* H2.
  Context: population, energy, predation, barrier vs the control — if
      the world changes shape, H1/H2 are read against that.
  Null is publishable: a channel wired but unused (donate_mean pinned at
      the sigmoid floor) is the stage-13 signal result again.

Phases: ``pilot`` (800 ticks x 3 x 3, fast sanity of the price bracket)
before ``gate`` (3 seeds x 10k; minutes of CPU — announce it).

    python experiments/proto_culture/run.py --phase pilot [--force]
    python experiments/proto_culture/run.py --phase gate [--force]
"""

from __future__ import annotations

import sys
from dataclasses import replace
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
from primordia.config import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTO = Path(__file__).resolve().parents[2] / "configs" / "proto_culture.toml"

VARIANTS = [
    ("off", 0.0),
    ("cheap", 0.02),
    ("dear", 0.10),
]
COLORS = {"off": "#888888", "cheap": "#2a9d8f", "dear": "#e76f51"}
DEFAULT_TICKS = {"pilot": 800, "gate": 10000}


def main() -> None:
    parser = common_parser(__doc__)
    parser.set_defaults(ticks=None)
    parser.add_argument(
        "--phase", choices=("pilot", "gate"), default="pilot",
        help="pilot = 800-tick price sanity (default); "
             "gate = the pre-registered 10k batch",
    )
    args = parser.parse_args()
    if args.ticks is None:
        args.ticks = DEFAULT_TICKS[args.phase]

    base = load_config(str(PROTO))
    histories: dict = {}
    for tag, rate in VARIANTS:
        cfg = replace(base, donate_rate=rate)
        print(f"variant {tag} donate_rate={cfg.donate_rate:g} "
              f"emit_cost={cfg.signal_emit_cost:g}", flush=True)
        for seed in SEEDS:
            histories[(tag, seed)] = run_variant(
                cfg, seed, args.ticks,
                HERE / "data" / f"{args.phase}_{tag}_seed{seed}.npz",
                args.force,
            )

    _figures(histories)
    _print_table(histories, args.phase)
    print(f"\nfigures -> {HERE / 'figs'}")


def _edge_mean(history, column: str, frac: float = 0.2, tail: bool = False) -> float:
    """Mean of a column over the first (or last) ``frac`` of the run."""
    n = len(history[column])
    k = max(1, int(n * frac))
    return float(history[column][-k:].mean() if tail else history[column][:k].mean())


def _null_align(history) -> float:
    """Chance level of the gift-pair hue similarity: (1 - hue_spread)^2."""
    return float(((1.0 - history["hue_spread"]) ** 2).mean())


def _smooth(y, window: int = 201):
    """Rolling mean with edge padding — a few pairs per tick makes the
    raw per-tick alignment unusable as a line; the smoothed series shows
    the trajectory the README quotes."""
    if len(y) < window:
        return y
    half = window // 2
    padded = np.pad(np.asarray(y, dtype=np.float64), (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="same")[half:-half]


class _SmoothedHistory:
    """Read-only shim: donate_align comes back as its 201-tick mean."""

    def __init__(self, history, window: int = 201) -> None:
        self._history = history
        self._window = window

    def __getitem__(self, key):
        if key == "donate_align":
            return _smooth(self._history[key], self._window)
        return self._history[key]


def _figures(histories: dict) -> None:
    def hist(tag: str) -> list:
        return [histories[(tag, s)] for s in SEEDS]

    # does the world actually use the channel, and does selection close it?
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag, _rate in VARIANTS:
        band(axes[0], hist(tag), "donated", tag, color=COLORS[tag])
        band(axes[1], hist(tag), "donate_mean", tag, color=COLORS[tag])
    axes[0].set_title("energy gifted per tick")
    axes[1].set_title("mean donate-gate fraction over living creatures")

    # do gifts follow colour-mates (kin) more than chance? (201-tick mean)
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for tag, _rate in VARIANTS:
        smooth = [_SmoothedHistory(h) for h in hist(tag)]
        band(ax, smooth, "donate_align", f"{tag} (201-tick mean)",
             color=COLORS[tag])
        nulls = [_null_align(h) for h in hist(tag)]
        ax.axhline(
            sum(nulls) / len(nulls), color=COLORS[tag], linestyle="--",
            alpha=0.6,
            label=f"{tag} chance (1-hue_spread)^2",
        )
    ax.set_title("gift-pair hue alignment (201-tick mean) vs chance (dashed)")

    # the context the gate is read against
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag, _rate in VARIANTS:
        band(axes3[0], hist(tag), "alive", tag, color=COLORS[tag])
        band(axes3[1], hist(tag), "mean_energy", tag, color=COLORS[tag])
    axes3[0].set_title("population (extinction = 0)")
    axes3[1].set_title("mean energy")

    for figure, axes_list in (
        (fig, list(fig.axes)), (fig2, [ax]), (fig3, list(axes3))
    ):
        for a in axes_list:
            a.set_xlabel("tick")
            a.grid(alpha=0.3)
            a.legend(fontsize=7, loc="best")
    save_fig(fig, HERE / "figs" / "donation.png")
    save_fig(fig2, HERE / "figs" / "alignment.png")
    save_fig(fig3, HERE / "figs" / "pop.png")


def _print_table(histories: dict, phase: str) -> None:
    print(f"\nphase = {phase}")
    print("\n| variant | final pop (min/mean) | min alive | "
          "donate_mean first->last 20% | donated total | align | chance |")
    print("|---|---|---|---|---|---|---|")
    for tag, rate in VARIANTS:
        hs = [histories[(tag, s)] for s in SEEDS]
        finals = [float(h["alive"][-1]) for h in hs]
        mins = [float(h["alive"].min()) for h in hs]
        first = sum(_edge_mean(h, "donate_mean") for h in hs) / len(hs)
        last = sum(
            _edge_mean(h, "donate_mean", tail=True) for h in hs
        ) / len(hs)
        donated = sum(float(h["donated"].sum()) for h in hs) / len(hs)
        align = sum(float(h["donate_align"].mean()) for h in hs) / len(hs)
        chance = sum(_null_align(h) for h in hs) / len(hs)
        print(
            f"| {tag} ({rate:g}) | {min(finals):.0f} / "
            f"{sum(finals) / len(finals):.0f} | {min(mins):.0f} "
            f"| {first:.4f} -> {last:.4f} "
            f"| {donated:.0f} | {align:.4f} | {chance:.4f} |"
        )
    survived = all(
        float(histories[(tag, s)]["alive"].min()) > 0
        for tag, _ in VARIANTS for s in SEEDS
    )
    print(f"\nhard gate (min alive > 0 in all runs): "
          f"{'PASS' if survived else 'FAIL'}")


if __name__ == "__main__":
    main()
