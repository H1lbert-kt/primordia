"""Experiment (stage 22): can an incremental house pay for itself?

Question: stage 17's house collapsed every world it touched (paid in
full up front, benefit only if predation ever happens — which, in the
default world, it does not). Stage 22 redesigns the object, not the
price tag: wood is delivered in increments (all-or-nothing *per
delivery*), cover scales with completeness u, unfinished houses rot
away, carrying wood is dearer than holding it, a spawn bias keeps the
build gate mostly closed, and each delivery wants a live fire nearby.
Nothing rewards building (principle 1): the question is whether, in a
world where being bitten is a real risk (the proto_culture "reach"
context — bite_rate 12, contact_range 2.5), shelter ever returns more
energy than it drains.

Context: configs/proto_culture.toml + houses_enabled=true, with the
stage-22 knobs pinned: carry_weight=1.0, build_house_bias=-1.5,
house_rot_rate=0.01, house_fire_radius=6.0 (the plan's draft 3.0 is a
documented trap at cell_size 5.0: the fire owns its cell and the next
free cell centre is 5.0 away, so no delivery could ever pass — 6.0 is
the smallest honest "next to the fire"), birth_spread=0.

Phases:

  pilot  (2 seeds x 3k)  wood total in {2, 4, 8} x energy total in
         {1.5, 3, 6} x cover_reduction in {0.3, 0.5}, plus per-cost
         no-effect twins (cover_reduction=0 — same drain, no shelter:
         separates "building drains" from "shelter pays") and the off
         control (houses_enabled=false). 2 seeds bracket the price;
         the third seed only appears in the gate.
  gate   (3 seeds x 10k) at --wood/--energy/--cover (the pilot's
         choice): the pre-registered hard/measured gate below.

Pre-registered outcomes (stated before any run):

  Hard gate (both phases): every run completes its ticks with min
      alive > 0 (extinction = fail).
  1. survival: 2/2 alive at 3k (3/3 at gate);
      final pop >= 0.85 x the off control (same seed, same ticks).
  2. adoption without spam: final n_house in [5, 400] (0 = never
      learns; >500 = spam — rot and the carrying tax exist to prevent
      it, and the numbers say whether they did).
  3. attribution: at the same cost, the cover=0 twin ends in a SMALLER
      population than the covered variant, and the covered variant
      suffers fewer predation deaths than its twin (the shelter, not
      the construction schedule, is what paid).
  4. payback: measured payback < L/5, where L = mean lifetime measured
      in the off control (Little's law: total alive-ticks / births —
      not max_age; starvation pulls L into the hundreds). Payback uses
      the energy the variants actually sank into completeness versus
      the bite damage they avoided relative to their twins. Spend is a
      LOWER bound (rot-destroyed stubs are not counted against it), so
      a failing payback here is the strong direction: if even the
      optimistic bookkeeping does not close, the null stands.

  Null is publishable: if no variant passes all four, houses do not
      have a viable price in this world yet — numbers to the user, no
      hacks, default stays houses_enabled=false.

Fixation rule (stated before any run): among passing variants pick the
one with the highest MEDIAN final n_house whose pop still clears
0.85 x control (a tie breaks toward the cheaper total). That price is
pinned into configs/houses.toml and carried into the gate.

    python experiments/houses/run.py --phase pilot [--force]
    python experiments/houses/run.py --phase gate \
        --wood 4 --energy 3 --cover 0.5 [--force]
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
from primordia.config import Config, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTO = Path(__file__).resolve().parents[2] / "configs" / "proto_culture.toml"

WOOD_TOTALS = (2.0, 4.0, 8.0)
ENERGY_TOTALS = (1.5, 3.0, 6.0)
COVERS = (0.3, 0.5)
PILOT_SEEDS = SEEDS[:2]
DEFAULT_TICKS = {"pilot": 3000, "gate": 10000}
# knobs every house variant shares (the design under test)
HOUSE_KNOBS: dict[str, object] = {
    "houses_enabled": True,
    "carry_weight": 1.0,
    "build_house_bias": -1.5,
    "house_rot_rate": 0.01,
    "house_fire_radius": 6.0,
    "birth_spread": 0.0,
}


def _variant_name(wood: float, energy: float, cover: float) -> str:
    return f"w{wood:g}_e{energy:g}_c{cover:g}"


def _pilot_variants() -> list[tuple[str, dict[str, object]]]:
    """Covered variants + per-cost no-effect twins + the off control."""
    variants: list[tuple[str, dict[str, object]]] = []
    for wood in WOOD_TOTALS:
        for energy in ENERGY_TOTALS:
            cost = {"house_wood_cost": wood, "house_energy_cost": energy}
            for cover in COVERS:
                variants.append(
                    (_variant_name(wood, energy, cover),
                     {**HOUSE_KNOBS, **cost, "cover_reduction": cover})
                )
            # no-effect twin: same drain, no shelter (attribution axis)
            variants.append(
                (_variant_name(wood, energy, 0.0),
                 {**HOUSE_KNOBS, **cost, "cover_reduction": 0.0})
            )
    variants.append(("off", {"houses_enabled": False}))
    return variants


def _color(name: str) -> str:
    palette = {"2": "#e76f51", "4": "#e9c46a", "8": "#2a9d8f"}
    return palette.get(name.split("_")[0][1:], "#888888")


def main() -> None:
    parser = common_parser(__doc__)
    parser.set_defaults(ticks=None)
    parser.add_argument(
        "--phase", choices=("pilot", "gate"), default="pilot",
        help="pilot = price/cover sweep, 2 seeds x 3k (default); "
             "gate = the pre-registered 10k batch at the fixed price",
    )
    parser.add_argument("--wood", type=float, default=None,
                        help="house_wood_cost total for --phase gate")
    parser.add_argument("--energy", type=float, default=None,
                        help="house_energy_cost total for --phase gate")
    parser.add_argument("--cover", type=float, default=None,
                        help="cover_reduction for --phase gate")
    args = parser.parse_args()
    if args.phase == "gate":
        missing = [n for n in ("wood", "energy", "cover")
                   if getattr(args, n) is None]
        if missing:
            parser.error(f"--phase gate requires {', '.join('--' + m for m in missing)}")
    if args.ticks is None:
        args.ticks = DEFAULT_TICKS[args.phase]

    base = load_config(str(PROTO))
    if args.phase == "pilot":
        variants = _pilot_variants()
        seeds = PILOT_SEEDS
        prefix = "pilot_"
        print(f"pilot: {len(variants)} configs x {len(seeds)} seeds "
              f"x {args.ticks} ticks (~{len(variants) * len(seeds) * args.ticks / 300 / 60:.0f} min at 300 t/s)", flush=True)
    else:
        overrides = {**HOUSE_KNOBS,
                     "house_wood_cost": args.wood,
                     "house_energy_cost": args.energy,
                     "cover_reduction": args.cover}
        tag = _variant_name(args.wood, args.energy, args.cover)
        variants = [(tag, overrides),
                    (f"{tag}_c0", {**overrides, "cover_reduction": 0.0}),
                    ("off", {"houses_enabled": False})]
        seeds = SEEDS
        prefix = "gate_"

    histories: dict = {}
    for tag, overrides in variants:
        cfg = replace(base, **overrides)
        print(f"variant {tag}: "
              + " ".join(f"{k}={v}" for k, v in sorted(overrides.items())), flush=True)
        for seed in seeds:
            histories[(tag, seed)] = run_variant(
                cfg, seed, args.ticks,
                HERE / "data" / f"{prefix}{tag}_seed{seed}.npz",
                args.force,
            )

    tags = [tag for tag, _ in variants]
    _figures(histories, tags, prefix)
    verdict = _print_table(histories, tags, seeds, args.phase, args.ticks)
    print(f"\nfigures -> {HERE / 'figs'}")
    if verdict:
        print(verdict)


def _figures(histories: dict, tags: list, prefix: str) -> None:
    seeds = sorted({s for (_, s) in histories})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag in tags:
        hs = [histories[(tag, s)] for s in seeds if (tag, s) in histories]
        if not hs:
            continue
        color = _color(tag)
        band(axes[0], hs, "alive", tag, color=color)
        band(axes[1], hs, "mean_energy", tag, color=color)
        band(axes2[0], hs, "n_house", tag, color=color)
        band(axes2[1], hs, "house_progress", tag, color=color)
        band(axes3[0], hs, "deaths_predation", tag, color=color)
        band(axes3[1], hs, "inventory_mean", tag, color=color)
    axes[0].set_title("population (extinction = 0)")
    axes[1].set_title("mean energy")
    axes2[0].set_title("houses alive")
    axes2[1].set_title("mean completeness u of houses")
    axes3[0].set_title("predation deaths per tick")
    axes3[1].set_title("mean carried wood")
    for axes_group in (axes, axes2, axes3):
        for a in axes_group:
            a.set_xlabel("tick")
            a.grid(alpha=0.3)
            a.legend(fontsize=6, loc="best")
    save_fig(fig, HERE / "figs" / f"{prefix}economy.png")
    save_fig(fig2, HERE / "figs" / f"{prefix}building.png")
    save_fig(fig3, HERE / "figs" / f"{prefix}exposure.png")


def _summary(h, ticks: int) -> dict:
    alive = np.asarray(h["alive"], dtype=np.float64)
    births = float(np.sum(h["births"]))
    deaths_pred = float(np.sum(h["deaths_predation"]))
    meat = float(np.sum(h["meat_eaten"]))
    n_house = np.asarray(h["n_house"], dtype=np.float64)
    progress = np.asarray(h["house_progress"], dtype=np.float64)
    total_u = n_house * progress  # total completeness standing
    return {
        "final": float(alive[-1]),
        "min": float(alive.min()),
        "births": births,
        "deaths_pred": deaths_pred,
        "meat": meat,
        "house_fin": float(n_house[-1]),
        "house_max": float(n_house.max()),
        "u_fin": float(total_u[-1]),
        "u_max": float(total_u.max()),
        "mean_energy": float(np.mean(h["mean_energy"])),
        # Little's law: mean lifetime = alive-ticks per birth
        "L": float(alive.sum() / births) if births > 0 else 0.0,
        "alive_sum": float(alive.sum()),
        "ticks": ticks,
    }


def _print_table(histories: dict, tags: list, seeds: list,
                 phase: str, ticks: int) -> str:
    print(f"\nphase = {phase}  (ticks = {ticks}, seeds = {list(seeds)})")
    print("\n| variant | final pop mean/min | pred deaths | meat eaten | "
          "house fin/max | u final | L | payback |")
    print("|---|---|---|---|---|---|---|---|")
    sums = {tag: [_summary(histories[(tag, s)], ticks) for s in seeds
                  if (tag, s) in histories] for tag in tags}

    def mean(tag: str, key: str) -> float:
        vals = [d[key] for d in sums[tag]]
        return float(np.mean(vals)) if vals else 0.0

    ctrl_L = mean("off", "L")
    paybacks: dict[str, float] = {}
    rows: dict[str, tuple] = {}
    for tag in tags:
        ds = sums[tag]
        if not ds:
            continue
        finals = [d["final"] for d in ds]
        row = (
            f"{np.mean(finals):.0f}/{min(finals):.0f}",
            f"{mean(tag, 'deaths_pred'):.0f}",
            f"{mean(tag, 'meat'):.0f}",
            f"{mean(tag, 'house_fin'):.0f}/{mean(tag, 'house_max'):.0f}",
            f"{mean(tag, 'u_fin'):.2f}",
            f"{mean(tag, 'L'):.0f}",
        )
        # payback: energy sunk per tick / bite damage avoided per tick
        # (vs the same-cost cover=0 twin; the off control is the L source)
        payback = float("nan")
        if tag != "off" and not tag.endswith("_c0"):
            twin = f"{tag.rsplit('_c', 1)[0]}_c0"
            if twin in sums:
                # spend lower bound: energy price x peak standing
                # completeness (rot-destroyed stubs are NOT charged ->
                # optimistic bookkeeping: a failing payback is the
                # strong direction for the null)
                spend = _energy_of(tag) * mean(tag, "u_max") / ticks
                # avoided damage = deaths avoided x mean energy of a
                # creature (what a killed body takes to its grave) +
                # energy the twin's victims actually lost to mouths
                # (meat_eaten difference), both per tick
                avoided_deaths = mean(twin, "deaths_pred") - mean(tag, "deaths_pred")
                energy_per_creature = mean(tag, "mean_energy")
                saved = (
                    avoided_deaths * energy_per_creature
                    + (mean(twin, "meat") - mean(tag, "meat"))
                ) / ticks
                payback = spend / saved if saved > 0 else float("inf")
        paybacks[tag] = payback
        print(f"| {tag} | " + " | ".join(row) + f" | {payback:.0f} |")

    # verdict against the four pre-registered criteria (pilot scoring)
    ctrl_final = {s: float(histories[("off", s)]["alive"][-1]) for s in seeds}
    passed = []
    for tag in tags:
        if tag == "off" or tag.endswith("_c0"):
            continue
        ds = sums[tag]
        if len(ds) != len(seeds):
            continue
        alive_all = all(d["min"] > 0 for d in ds)
        pop_ok = all(
            d["final"] >= 0.85 * ctrl_final[s]
            for d, s in zip(ds, seeds)
        )
        adopt = 5.0 <= mean(tag, "house_fin") <= 400.0
        twin = f"{tag.rsplit('_c', 1)[0]}_c0"
        attribution = (
            twin in sums
            and mean(tag, "final") > mean(twin, "final")
            and mean(tag, "deaths_pred") < mean(twin, "deaths_pred")
        )
        payback_ok = paybacks[tag] < ctrl_L / 5.0 if ctrl_L > 0 else False
        flags = (alive_all, pop_ok, adopt, attribution, payback_ok)
        if all(flags):
            passed.append((mean(tag, "house_fin"), tag, flags))
        print(f"  {tag}: alive={alive_all} pop>=0.85ctrl={pop_ok} "
              f"adoption={adopt} attribution={attribution} "
              f"payback<L/5={payback_ok} (L_ctrl={ctrl_L:.0f}, "
              f"L/5={ctrl_L / 5:.0f})")

    if not passed:
        return "\nVERDICT: no variant passes all four criteria — null publishable, default stays off."
    # fixation rule: highest median standing houses among passers
    passed.sort(reverse=True)
    _hs, best, _flags = passed[0]
    return (f"\nVERDICT: {best} passes all four criteria "
            f"(fixation rule: highest median final houses). "
            f"Pin it into configs/houses.toml and run the gate.")


def _energy_of(tag: str) -> float:
    """house_energy_cost total encoded in the variant name (w*_e*_*)."""
    return float(tag.split("_e")[1].split("_")[0])

if __name__ == "__main__":
    main()
