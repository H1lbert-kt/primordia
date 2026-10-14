"""Experiment (stage 25): does a banked house become a territory worth raiding?

Question: stage 25 adds an economy on top of the stage-22 house — surplus
energy above the reproduce threshold banks itself into the creature's own
complete house (automatic, no gate), the same open mouth that eats also
withdraws from its own bank or raids an unguarded foreign one, the owner
creature alive within guard_range makes the bank drain-proof, a mark ring
appears on the ground over banked houses, and standing on your own mark
trims the metabolic bill. Nothing rewards banking (principle 1): the
question is whether, in a world where houses actually get built (the
proto_culture "reach" context + the stage-22 build knobs), the bank fills
and theft ever happens — or whether the flows are inert / collapse the
population.

Context: configs/proto_culture.toml + the stage-22 house knobs that made
construction viable in the houses pilot (wood 2, energy 3, cover 0.5,
carry 1.0, bias -1.5, rot 0.01, fire_radius 6.0, spread 0). Two variants:

  off    houses on, territory knobs at zero (deposit_rate=0, raid_rate=0,
         comfort=0) — the exact stage-22 baseline; the mark threshold can
         stay at 1.0 because a zero deposit_rate never fills a bank.
  store  the Config defaults: deposit_rate=0.5, raid_rate=0.3,
         guard_range=3.0, comfort=0.3, deposit_threshold_frac=1.1.

Seeds 42/7 x 10k (two seeds, as in the houses gate — banked houses keep
the population small, the third seed belongs to a future price sweep if
this one publishes a null).

Pre-registered outcomes (stated before any run):

  Hard gate (both variants): every run completes its ticks with min
      alive > 0 (extinction = fail — lesson 17/18: a collapsing economy
      is reported, never hacked).
  1. bank fills: total deposited > 0 in >= 2/2 store seeds (if the
      deposit trigger 1.1 x threshold is never reached by a banked
      creature, deposited stays 0 — a calibration null, published).
  2. theft emerges: total raided > 0 in >= 2/2 store seeds (the guard
      is a presence check; whether anybody ever stands next to an
      unguarded foreign bank with an open mouth is the empirical
      question).
  3. no collapse: final pop of store >= 0.8 x off (same seed, same
      ticks) — the bank is a new sink; if it starves the world, the
      price of territory is negative and the null stands.

  Null is publishable: if the bank never fills, or theft never happens,
  or the store variant collapses below 0.8 x control, territory does not
  have a viable regime in this world yet — numbers to the user, no hacks,
  default stays deposit_rate=0 (Config default is the neutral one the
  pre-stage-25 world already ran on).

    python experiments/territory/run.py [--ticks 10000] [--force]
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

# stage-22 house knobs that made construction viable in the houses pilot
# (experiments/houses: wood 2 / energy 3 built 39-97 houses at 3k ticks;
# the shelter did not pay for itself there, but territory does not need
# shelter to pay — it needs the house to EXIST as a bank).
HOUSE_KNOBS: dict[str, object] = {
    "houses_enabled": True,
    "house_wood_cost": 2.0,
    "house_energy_cost": 3.0,
    "cover_reduction": 0.5,
    "carry_weight": 1.0,
    "build_house_bias": -1.5,
    "house_rot_rate": 0.01,
    "house_fire_radius": 6.0,
    "birth_spread": 0.0,
}

OFF_KNOBS: dict[str, object] = {
    "deposit_rate": 0.0,
    "raid_rate": 0.0,
    "comfort": 0.0,
}
STORE_KNOBS: dict[str, object] = {
    "deposit_rate": 0.5,
    "raid_rate": 0.3,
    "guard_range": 3.0,
    "comfort": 0.3,
    "deposit_threshold_frac": 1.1,
}
PILOT_SEEDS = SEEDS[:2]
DEFAULT_TICKS = 10000


def _variants() -> list[tuple[str, dict[str, object]]]:
    return [
        ("off", {**HOUSE_KNOBS, **OFF_KNOBS}),
        ("store", {**HOUSE_KNOBS, **STORE_KNOBS}),
    ]


def main() -> None:
    parser = common_parser(__doc__)
    parser.set_defaults(ticks=DEFAULT_TICKS)
    parser.add_argument(
        "--phase", choices=("run",), default="run",
        help="single pre-registered batch (kept for interface parity)",
    )
    args = parser.parse_args()

    base = load_config(str(PROTO))
    variants = _variants()
    seeds = PILOT_SEEDS
    print(
        f"run: {len(variants)} configs x {len(seeds)} seeds x {args.ticks} ticks "
        f"(~{len(variants) * len(seeds) * args.ticks / 300 / 60:.0f} min at 300 t/s)",
        flush=True,
    )

    histories: dict = {}
    for tag, overrides in variants:
        cfg = replace(base, **overrides)
        print(
            f"variant {tag}: "
            + " ".join(f"{k}={v}" for k, v in sorted(overrides.items())),
            flush=True,
        )
        for seed in seeds:
            histories[(tag, seed)] = run_variant(
                cfg, seed, args.ticks, HERE / "data" / f"{tag}_seed{seed}.npz",
                args.force,
            )

    tags = [tag for tag, _ in variants]
    _figures(histories, tags)
    verdict = _print_table(histories, tags, seeds, args.ticks)
    print(f"\nfigures -> {HERE / 'figs'}")
    if verdict:
        print(verdict)


def _figures(histories: dict, tags: list) -> None:
    seeds = sorted({s for (_, s) in histories})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    colors = {"off": "#888888", "store": "#2a9d8f"}
    for tag in tags:
        hs = [histories[(tag, s)] for s in seeds if (tag, s) in histories]
        if not hs:
            continue
        color = colors.get(tag, "#888888")
        band(axes[0], hs, "alive", tag, color=color)
        band(axes[1], hs, "mean_energy", tag, color=color)
        band(axes2[0], hs, "energy_in_stores", tag, color=color)
        band(axes2[1], hs, "mark_cells", tag, color=color)
        band(axes3[0], hs, "deposited", tag, color=color)
        band(axes3[1], hs, "raided", tag, color=color)
    axes[0].set_title("population (extinction = 0)")
    axes[1].set_title("mean energy")
    axes2[0].set_title("energy banked in houses")
    axes2[1].set_title("marked cells")
    axes3[0].set_title("deposited per tick")
    axes3[1].set_title("raided per tick")
    for axes_group in (axes, axes2, axes3):
        for a in axes_group:
            a.set_xlabel("tick")
            a.grid(alpha=0.3)
            a.legend(fontsize=6, loc="best")
    save_fig(fig, HERE / "figs" / "economy.png")
    save_fig(fig2, HERE / "figs" / "bank.png")
    save_fig(fig3, HERE / "figs" / "flows.png")


def _summary(h) -> dict:
    alive = np.asarray(h["alive"], dtype=np.float64)
    return {
        "final": float(alive[-1]),
        "min": float(alive.min()),
        "deposited": float(np.sum(h["deposited"])),
        "raided": float(np.sum(h["raided"])),
        "bank_fin": float(h["energy_in_stores"][-1]),
        "bank_max": float(np.max(h["energy_in_stores"])),
        "mark_fin": int(h["mark_cells"][-1]),
        "mark_max": int(np.max(h["mark_cells"])),
        "n_house_fin": float(h["n_house"][-1]),
        "births": float(np.sum(h["births"])),
    }


def _print_table(histories: dict, tags: list, seeds: list, ticks: int) -> str:
    print(f"\nticks = {ticks}, seeds = {list(seeds)}")
    print("\n| variant | final pop mean/min | deposited | raided | bank fin/max | "
          "mark fin/max | n_house fin | births |")
    print("|---|---|---|---|---|---|---|---|")
    sums = {
        tag: [_summary(histories[(tag, s)]) for s in seeds if (tag, s) in histories]
        for tag in tags
    }

    def mean(tag: str, key: str) -> float:
        vals = [d[key] for d in sums[tag]]
        return float(np.mean(vals)) if vals else 0.0

    for tag in tags:
        ds = sums[tag]
        if not ds:
            continue
        finals = [d["final"] for d in ds]
        print(
            f"| {tag} | {np.mean(finals):.0f}/{min(finals):.0f} "
            f"| {mean(tag, 'deposited'):.0f} | {mean(tag, 'raided'):.0f} "
            f"| {mean(tag, 'bank_fin'):.0f}/{mean(tag, 'bank_max'):.0f} "
            f"| {mean(tag, 'mark_fin'):.0f}/{mean(tag, 'mark_max'):.0f} "
            f"| {mean(tag, 'n_house_fin'):.0f} | {mean(tag, 'births'):.0f} |"
        )

    # verdict against the four pre-registered criteria
    ctrl_final = {s: float(histories[("off", s)]["alive"][-1]) for s in seeds}
    ctrl_alive = all(
        float(histories[("off", s)]["alive"].min()) > 0 for s in seeds
    )
    ds = sums["store"]
    if len(ds) != len(seeds):
        return "\nVERDICT: store runs incomplete — no verdict."
    alive_all = all(d["min"] > 0 for d in ds)
    bank_ok = sum(d["deposited"] > 0 for d in ds) >= 2
    raid_ok = sum(d["raided"] > 0 for d in ds) >= 2
    pop_ok = all(
        d["final"] >= 0.8 * ctrl_final[s] for d, s in zip(ds, seeds)
    )
    flags = (alive_all, bank_ok, raid_ok, pop_ok)
    print(
        f"\n  store: alive={alive_all} bank_fills={bank_ok} "
        f"theft_emerges={raid_ok} no_collapse(>=0.8xoff)={pop_ok} "
        f"(control stable across seeds: {ctrl_alive})"
    )
    if not ctrl_alive:
        return (
            "\nVERDICT: hard gate fails for the CONTEXT, not the treatment — "
            "the off control itself goes extinct on a seed, so the pre-"
            "registered gate cannot pass for any territory variant. The "
            "mechanism numbers (bank/raids/marks on the surviving seed) are "
            "still measured and reported; a stable context needs a price or "
            "economy sweep with the user (lesson 17/18), no hacks."
        )
    if all(flags):
        return (
            "\nVERDICT: store passes all four criteria — the bank fills and "
            "theft emerges without collapsing the population. Territory has a "
            "viable regime in this world."
        )
    return (
        "\nVERDICT: store fails at least one criterion — null publishable, "
        "default stays neutral (deposit_rate=0)."
    )


if __name__ == "__main__":
    main()
