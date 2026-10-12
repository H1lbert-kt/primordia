"""Experiment (stage 17): does building pay, or is a house a trap?

Question: stage 17 gives the brain three new decisions — carry wood
(gather), raise a fire (build_fire), raise a house (build_house). Nothing
rewards building (principle 1): a fire pays only if cooking its food
outweighs the energy its fuel cost, a house pays only if shelter ever
matters (in proto_culture predation is rare, so cover may be worthless).
Whether builders outlive non-builders is selection's answer, not a rule.

But before any of that can be asked, the world must stay alive. The
stage-17 plan shipped ``house_wood_cost=6.0`` against a
``max_inventory`` of 4.0 (unbuildable — fixed to 4.0 with a Config
guard), and at the default ``house_energy_cost=20.0`` seed 42 built 370
houses in 3k ticks and collapsed (500 -> 37 alive, births 259 -> 26)
while the same world with unbuildable houses recovered to 201. Houses
drain the reproduction budget faster than any shelter return exists.
(Stage 18 subsequently priced houses out of the *default* Config —
1e6 — and swept the fire price; see the README's postscript. This
pre-registration is unchanged: its overrides still apply.)

Phases:

  pilot  (2 seeds x 5k)  house_energy_cost in {20, 12, 8} — does a
         cheaper house stop the birth collapse? Costs stay in Config;
         the sweep *is* the calibration the stage-17 plan asked for.
  gate   (3 seeds x 10k) at ``--house-energy`` (the pilot's choice):
         the pre-registered hard/measured gate below.

Pre-registered outcomes (stated before any run):

  H1 (cost drives the collapse): final pop and births sum rise as
      house_energy_cost falls (monotone over 20 > 12 > 8). If 8 is no
      better than 20, the house price is not the cause and the README
      says so.
  H2 (the cheap house must not kill the fire): n_fire max > 0 and
      energy_cooked total > 0 in every variant — construction economy
      is shared (wood, pack), so a house fix that extinguishes campfires
      fails.
  Hard gate (phase gate): every run completes 10k with min alive > 0;
      mean final pop < 0.95 * max_creatures; frac of ticks with
      mean_energy >= reproduce_threshold < 0.5 (documented proxy: stats
      record the mean, not the per-creature distribution).
  Measured (phase gate): n_fire max > 0 in >= 2/3 seeds;
      energy_cooked total > 0; wood_gathered total > 0;
      n_house max >= 1 in >= 2/3 seeds (fail = report honestly).
  Null is publishable: if even the cheap house collapses, proto_culture
      cannot host houses yet — numbers to the user, no hacks.

    python experiments/civilization/run.py --phase pilot [--force]
    python experiments/civilization/run.py --phase gate --house-energy 8
        [--force]
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
from primordia.config import Config, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTO = Path(__file__).resolve().parents[2] / "configs" / "proto_culture.toml"

# pilot: the only lever is the house price (wood cost is pinned at 4.0 by
# the affordability guard; energy is the knob that drained the births).
# `off` = a price no one can pay (payment is all-or-nothing), which makes
# houses impossible exactly like the stage-16 world did — the baseline
# that separates "houses collapse the world" from "proto is at its limit".
PILOT_VARIANTS = [
    ("off", 1.0e6),
    ("h20", 20.0),
    ("h12", 12.0),
    ("h8", 8.0),
]
COLORS = {"off": "#888888", "h20": "#e76f51", "h12": "#e9c46a",
          "h8": "#2a9d8f"}
DEFAULT_TICKS = {"pilot": 5000, "gate": 10000}
PILOT_SEEDS = SEEDS[:2]  # the plan's pilot: 2 seeds


def main() -> None:
    parser = common_parser(__doc__)
    parser.set_defaults(ticks=None)
    parser.add_argument(
        "--phase", choices=("pilot", "gate"), default="pilot",
        help="pilot = house-price sweep, 2 seeds x 5k (default); "
             "gate = the pre-registered 10k batch at --house-energy",
    )
    parser.add_argument(
        "--house-energy", type=float, default=None,
        help="house_energy_cost for --phase gate (from the pilot table; "
             "required there)",
    )
    args = parser.parse_args()
    if args.phase == "gate" and args.house_energy is None:
        parser.error("--house-energy is required for --phase gate")
    if args.ticks is None:
        args.ticks = DEFAULT_TICKS[args.phase]

    base = load_config(str(PROTO))
    histories: dict = {}
    if args.phase == "pilot":
        variants = [(tag, e) for tag, e in PILOT_VARIANTS]
        seeds = PILOT_SEEDS
        prefix = "pilot_"
    else:
        variants = [(f"h{args.house_energy:g}", args.house_energy)]
        seeds = SEEDS
        prefix = "gate_"

    for tag, house_energy in variants:
        # stage 22 turned houses off in Config defaults; this experiment
        # studies them, so it re-enables them. Note: stage 22 also made
        # payment incremental (per delivery, not per house) — the cached
        # npz files preserve the stage-17 lump results, but a --force
        # re-run today measures the new mechanics at the same totals.
        cfg = replace(base, house_energy_cost=house_energy,
                      houses_enabled=True)
        print(f"variant {tag} house_energy_cost={cfg.house_energy_cost:g} "
              f"house_wood_cost={cfg.house_wood_cost:g}", flush=True)
        for seed in seeds:
            histories[(tag, seed)] = run_variant(
                cfg, seed, args.ticks,
                HERE / "data" / f"{prefix}{tag}_seed{seed}.npz",
                args.force,
            )

    tags = [tag for tag, _ in variants]
    _figures(histories, tags, prefix)
    _print_table(histories, tags, base, args.phase)
    print(f"\nfigures -> {HERE / 'figs'}")


def _figures(histories: dict, tags: list, prefix: str) -> None:
    seeds_by_tag = {tag: [s for (t, s) in histories if t == tag] for tag in tags}

    # survival first: the collapse H1 is about
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag in tags:
        hs = [histories[(tag, s)] for s in seeds_by_tag[tag]]
        band(axes[0], hs, "alive", tag, color=COLORS.get(tag))
        band(axes[1], hs, "mean_energy", tag, color=COLORS.get(tag))
    axes[0].set_title("population (extinction = 0)")
    axes[1].set_title("mean energy")

    # what is being built (the mechanism H1 claims to see)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag in tags:
        hs = [histories[(tag, s)] for s in seeds_by_tag[tag]]
        band(axes2[0], hs, "n_fire", tag, color=COLORS.get(tag))
        band(axes2[1], hs, "n_house", tag, color=COLORS.get(tag))
    axes2[0].set_title("campfires alive")
    axes2[1].set_title("houses alive")

    # the wood economy behind both
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for tag in tags:
        hs = [histories[(tag, s)] for s in seeds_by_tag[tag]]
        band(axes3[0], hs, "wood_gathered", tag, color=COLORS.get(tag))
        band(axes3[1], hs, "energy_cooked", tag, color=COLORS.get(tag))
    axes3[0].set_title("wood harvested per tick")
    axes3[1].set_title("cooking bonus paid per tick (from fuel)")

    for figure, axes_list in (
        (fig, list(fig.axes)), (fig2, list(axes2)), (fig3, list(axes3))
    ):
        for a in axes_list:
            a.set_xlabel("tick")
            a.grid(alpha=0.3)
            a.legend(fontsize=7, loc="best")
    save_fig(fig, HERE / "figs" / f"{prefix}economy.png")
    save_fig(fig2, HERE / "figs" / f"{prefix}building.png")
    save_fig(fig3, HERE / "figs" / f"{prefix}wood.png")


def _seeds_of(histories: dict, tag: str) -> list:
    return [s for (t, s) in histories if t == tag]


def _print_table(histories: dict, tags: list, base: Config, phase: str) -> None:
    cap = base.max_creatures
    print(f"\nphase = {phase}")
    print("\n| variant | final pop (mean/min) | births sum | "
          "fire max/final | house max/final | gathered tot | cooked tot |")
    print("|---|---|---|---|---|---|---|")
    summary = {}
    for tag in tags:
        hs = [histories[(tag, s)] for s in _seeds_of(histories, tag)]
        finals = [float(h["alive"][-1]) for h in hs]
        births = sum(float(h["births"].sum()) for h in hs) / len(hs)
        fire_max = max(float(h["n_fire"].max()) for h in hs)
        fire_fin = sum(float(h["n_fire"][-1]) for h in hs) / len(hs)
        house_max = max(float(h["n_house"].max()) for h in hs)
        house_fin = sum(float(h["n_house"][-1]) for h in hs) / len(hs)
        gathered = sum(float(h["wood_gathered"].sum()) for h in hs) / len(hs)
        cooked = sum(float(h["energy_cooked"].sum()) for h in hs) / len(hs)
        summary[tag] = dict(finals=finals, births=births, fire_max=fire_max,
                            house_max=house_max, fire_fin=fire_fin,
                            house_fin=house_fin, gathered=gathered,
                            cooked=cooked, hs=hs)
        print(
            f"| {tag} | {sum(finals) / len(finals):.0f} / "
            f"{min(finals):.0f} | {births:.0f} | "
            f"{fire_max:.0f} / {fire_fin:.0f} | "
            f"{house_max:.0f} / {house_fin:.0f} | "
            f"{gathered:.0f} | {cooked:.0f} |"
        )

    survived = all(min(s["finals"]) > 0 for s in summary.values())
    print(f"\nhard gate (min alive > 0 in every run): "
          f"{'PASS' if survived else 'FAIL'}")

    if phase != "gate":
        return
    # the pre-registered gate, printed only where it was registered
    threshold = base.reproduce_threshold
    ok_pop = all(
        sum(s["finals"]) / len(s["finals"]) < 0.95 * cap
        for s in summary.values()
    )
    frac = {}
    for tag, s in summary.items():
        f = [
            float((h["mean_energy"] >= threshold).mean()) for h in s["hs"]
        ]
        frac[tag] = sum(f) / len(f)
    ok_frac = all(v < 0.5 for v in frac.values())
    # measured: existence of the structures and the wood economy
    n_runs = sum(len(summary[t]["hs"]) for t in tags)
    fire_runs = sum(
        1 for t in tags for h in summary[t]["hs"]
        if float(h["n_fire"].max()) > 0
    )
    house_runs = sum(
        1 for t in tags for h in summary[t]["hs"]
        if float(h["n_house"].max()) >= 1
    )
    cooked_ok = all(summary[t]["cooked"] > 0 for t in tags)
    gathered_ok = all(summary[t]["gathered"] > 0 for t in tags)
    print(f"pop < 0.95*cap ({0.95 * cap:.0f}): {'PASS' if ok_pop else 'FAIL'}")
    print("frac(ticks mean_energy >= reproduce_threshold) "
          + " ".join(f"{t}={frac[t]:.3f}" for t in tags)
          + f" -> {'PASS' if ok_frac else 'FAIL'} (<0.5)")
    print(f"n_fire > 0 in >= 2/3 runs ({fire_runs}/{n_runs}): "
          f"{'PASS' if fire_runs * 3 >= n_runs * 2 else 'FAIL'}")
    print(f"n_house >= 1 in >= 2/3 runs ({house_runs}/{n_runs}): "
          f"{'PASS' if house_runs * 3 >= n_runs * 2 else 'FAIL'}")
    print(f"energy_cooked > 0: {'PASS' if cooked_ok else 'FAIL'}; "
          f"wood_gathered > 0: {'PASS' if gathered_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
