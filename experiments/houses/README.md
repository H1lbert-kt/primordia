# Houses that pay for themselves? (stage 22)

## Question

Stage 17's house collapsed every world it touched: paid in full up
front, benefit only if predation ever happens — and in the default
world it barely does. Stage 22 redesigned the *object*, not the price
tag. Wood is now delivered in increments (all-or-nothing **per
delivery**, never per house), cover scales with completeness `u`,
unfinished houses rot away, carrying wood is dearer than holding it
(`carry_weight`), a spawn bias keeps the build gate mostly closed, and
each delivery wants a live fire within `house_fire_radius`. Nothing
rewards building (principle 1). The question is whether, in a world
where being bitten is a real risk, shelter ever returns more energy
than it drains.

## Method

- Context: `configs/proto_culture.toml` (`bite_rate=12`,
  `contact_range=2.5` — the "reach" ratchet where ~1179 predation
  deaths/run make cover meaningful) + the stage-22 knobs pinned:
  `carry_weight=1.0`, `build_house_bias=-1.5`, `house_rot_rate=0.01`,
  `house_fire_radius=6.0`, `birth_spread=0`.
- Sweep (pre-registered in `run.py` before any run): wood total
  {2, 4, 8} × energy total {1.5, 3, 6} × `cover_reduction` {0.3, 0.5},
  plus a **no-effect twin** per cost (`cover_reduction=0` — same drain,
  no shelter) and an **off** control (`houses_enabled=false`). Seeds
  42 and 7 × 3000 ticks.
- Pre-registered success (all four): every run survives; final pop ≥
  0.85 × off-control; final standing houses ∈ [5, 400]; the covered
  variant ends in a larger population than its same-cost no-effect
  twin *and* suffers fewer predation deaths; measured payback < L/5
  (L = mean lifetime from the off control, Little's law). Null is
  publishable; no gate runs unless a variant passes.

`house_fire_radius` is 6.0, not the plan's draft 3.0: at `cell_size=5`
the fire owns its own cell, so the nearest free cell centre is 5.0 away
and a radius of 3 makes *every* house impossible — a documented trap,
not a price. 6.0 reaches a builder in the adjacent cell: the smallest
honest "next to the fire" gate.

## Result — null, and the default stays off

No variant passed all four criteria. `experiments/houses/figs/`
(economy / building / exposure) carries the trajectories; the pilot
table is reproducible with the command below.

| variant | final pop mean/min | pred deaths | houses fin/max | L | payback |
|---|---|---|---|---|---|
| off (control) | 214 / 189 | 62 | 0 | 1350 | — |
| w2_e3_c0.5 | 79 / 54 | 2 | 94 / 96 | 2351 | ~0 |
| w2_e3_c0 (twin) | 86 / 58 | 8 | 92 / 95 | 2183 | — |
| w2_e6_c0.3 | 74 / 47 | 0 | 96 / 98 | 2385 | ~1 |
| w4_e3_c0.5 | 56 / 36 | 0 | 14 / 22 | 3233 | 873 |
| w8_* (all) | 36–50 | 0–6 | 0–2 | 3580–14473 | inf |

What the numbers say:

1. **The kill is reproductive, not mortal.** Every run survives (hard
   gate never trips), and the off control ends at 189–239 with 287–410
   births. The cheapest surviving house world (`w2_e3`) ends at 54–113
   with **69–153 births** — less than half. Creatures are not dying of
   houses; they are failing to fund reproduction once wood-hauling and
   construction drain the budget. Pop of every house variant is ~0.3×
   the control, far under the 0.85× bar.
2. **Adoption works exactly as designed — and is not the problem.** At
   the cheap total (wood 2) the world builds 39–97 standing houses and
   `house_progress` shows them finishing (`u→1`); at wood 8 almost none
   (0–2) rot before completing. Rot, the carrying tax and the fire gate
   successfully stop spam: no variant exceeds ~100 houses, and the
   wood price monotonically throttles construction. The stage-17
   failure mode ("spam 591 houses, then collapse") is gone.
3. **Shelter does reduce bites — measurably, per cost.** `w2_e3`
   suffers 8 → 4 → 2 predation deaths as cover goes 0 → 0.3 → 0.5, and
   `meat_eaten` falls 123 → 74 → 57. The attribution axis fires: the
   covered world is strictly better than its no-effect twin on the
   predation axis. Payback closes easily (≈0–1 ticks ≪ L/5 = 270).
   **But the energy saved by not being bitten is smaller than the
   energy spent hauling and building**, so the population still ends
   2.5× below control. The benefit is real; it is just not enough.
4. **The cheap-energy corner starves differently.** `w2_e1.5` (total
   energy 1.5) builds fine but ends at 22–61 with 19–65 births and
   *zero* predation — the construction drain pushes the world into a
   low-energy regime where being bitten never becomes a live risk, so
   cover can never pay. A house priced to be affordable can delete the
   very threat it exists to defend against.

## Limitations

- **Payback is optimistic and incomplete.** Spend is bookkept as
  `energy_price × peak standing completeness / ticks`; rot-destroyed
  stubs are not charged against it, and the *carrying* tax
  (`carry_weight × inventory` on movement) is invisible to a
  completeness-based spend — it shows up only as the missing births.
  A payback that "closes" here therefore overstates the case for
  houses; the population criterion is the honest one, and it fails.
- **L is censored at 3k ticks** (Little's law over a short window over
  a truncated life); it is used only as the payback yardstick, not as
  a demographic claim.
- **Two seeds, 3k ticks.** Enough to bracket a price and reject the
  grid; not enough to claim evolution had time to *discover* a
  profitable house. The spawn bias (−1.5) plus mutation could in
  principle route around the drain over tens of thousands of ticks;
  this pilot deliberately asked whether the *mechanics* have a window,
  not whether a long evolution finds it.
- **Identical trajectories across cover values are a feature, not a
  cache bug.** `cover_reduction` enters the world only through the bite
  shield; when a run records zero predation (e.g. `w2_e1.5`, `w8_*`),
  the 0.3 / 0.5 / 0 variants are bit-identical by determinism. The
  divergence in `w2_e3` / `w2_e6` is where bites actually happen.
- The gate (10k × 3 seeds) was **not run**: the pre-registered
  fixation rule found no passer, and running a gate on a price the
  pilot already rejected would only manufacture a longer null.

## Reproduce

```bash
python experiments/houses/run.py --phase pilot          # 2 seeds x 3k, ~20 min
python experiments/houses/run.py --phase gate \
    --wood 4 --energy 3 --cover 0.5                    # only if a price passes
```

Caches in `data/` are invalidated automatically when `stats.COLUMNS`
gains a column (e.g. the stage-22 `house_progress` / `inventory_mean`),
so a re-run refreshes stale npz files. Defaults in
`primordia/config.py` are untouched: `houses_enabled=false`, so the
default world reproduces every pre-stage-22 result bit-for-bit
(`alive_final` identical against the stage-21 HEAD).
