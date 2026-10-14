# Stage 14 — pheromone diffusion: does the field build usable trails?

## Question

Stage 14 added a Jacobi pass over the signal field: every tick, after
geometric decay and *before* the tick's own deposits, each cell absorbs
`signal_diffuse * (N + S + E + W - 4 * self)` from its four toroidal
neighbors (stable for `signal_diffuse <= 0.25`, mass-conserving). Physics
only — the tick still never rewards communicating. Two questions:

1. **Price (the stage-13 follow-up):** `configs/proto_culture.toml` pins
   `signal_emit_cost = 0.5`, which stage 13 showed is economically lethal
   (33 alive at 10k, one seed extinct) while 0.001 survives as spam. What
   is the viable price?
2. **Trails:** does diffusion turn point deposits into extended
   structures (`signal_cells` — cells above 1% of `signal_capacity`), and
   does anything upstream of the physics move — population, emit gate,
   predation?

## Hypothesis (stated a priori, before any run on this code)

> The price sweep brackets a viable window between 0.001 and 0.5.
> Diffusion widens the trail area (`signal_cells`: nodiff < mid < fast)
> while total mass stays equal (diffusion only redistributes, decay sets
> the level). Ecological deltas are **null in 10k** — stage 13 already
> showed that wiring a channel changes nothing on its own.

## Method

Two phases, canonical seeds (42, 7, 13), on top of `proto_culture.toml`
(stage-12.5 "reach" predation context).

**Price phase** — 6000 ticks, `signal_diffuse = 0` so the price is a
property of the economy alone. Only `signal_emit_cost` differs:

| variant | cost | tax at gate 0.7 (metabolism = 0.08) |
|---|---|---|
| `p0` | 0 | 0 (spam control) |
| `p01` | 0.01 | ~9% |
| `p05` | 0.05 | ~44% |
| `p15` | 0.15 | ~131% |

**Gate phase** — 10000 ticks at the chosen price **0.05** (user decision:
the strict rule "highest cost with 3/3 survival" selected 0.15, but 0.15
ended the screen at 99 alive — 45% of control, min 71 — with real 10k
extinction risk; 0.05 kept 209 alive ≈ control's 219 with selection
already visible: `emit_mean` 0.62 vs 0.72, emitted −50%). Only
`signal_diffuse` differs:

| variant | rate |
|---|---|
| `nodiff` | 0 (control) |
| `mid` | 0.1 (the new default) |
| `fast` | 0.2 (near the 0.25 stability limit) |

Pre-registered gate: **hard** = 3/3 survival at the chosen price;
**physics** = `signal_cells(nodiff) < signal_cells(mid/fast)` at 10k;
**behavior** = delta in `emit_mean` / predation, a priori *null*.

## Results

### Price phase (6k, diffusion off)

| variant | cost | final pop (min / mean) | survived | `emit_mean` final | emitted total |
|---|---|---|---|---|---|
| `p0` | 0 | 138 / 219 | 3/3 | 0.717 | 1 390 368 |
| `p01` | 0.01 | 105 / 212 | 3/3 | 0.699 | 1 116 423 |
| `p05` | 0.05 | 199 / 209 | 3/3 | 0.622 | 700 047 |
| `p15` | 0.15 | 71 / 99 | 3/3 | 0.579 | 324 737 |

Selection acts on the gate: `emit_mean` falls monotonically with the
price and total emission drops 4.3× from `p0` to `p15`. All four prices
survive 6k — the window stage 13 could only bracket (0.001 survives,
0.5 collapses) is real and **0.05 sits inside it**. Note the 6k endpoint
lands at the bottom of a population cycle (the control itself fell from
~495 at 5k to 219 at 6k), so read the rows as ratios, not absolute
health.

### Gate phase (10k, price 0.05)

| variant | diffuse | final pop (min / mean) | `signal_cells` mean final (max) | `signal_mass` final | `emit_mean` | predation |
|---|---|---|---|---|---|---|
| `nodiff` | 0 | 547 / 974 | 7 740 (11 7k) | 4 363 | 0.596 | 469 |
| `mid` | 0.1 | 690 / 859 | 15 439 (34.4k) | 5 589 | 0.646 | 417 |
| `fast` | 0.2 | 290 / 900 | 16 529 (37.7k) | 5 270 | 0.616 | 485 |

### Verdict against the pre-registered gate

1. **Hard gate: PASS.** All 9 runs alive at 10k (min 290). The chosen
   price 0.05 is viable through 10k — the stage-13 open question
   ("find an affordable price") is closed.
2. **Physics: PASS, as predicted.** Trail area with diffusion is ~2× the
   control at the end of the run and ~3.4× at mid-run (11.9–13.3k vs
   3.2–4.1k cells at 5k). The ceiling moves too: nodiff trails never
   exceed ~11.7k cells (29% of the 40k grid — paths creatures actually
   walked), while `mid`/`fast` reach 34–37k cells (85–94%) — a diffusing
   field saturates the world, a point field stays where feet went.
   `signal_cells` is doing exactly the job it was added for.
3. **Behavior: NULL, as predicted.** Population means 974 / 859 / 900
   with seed ranges that overlap (`fast` has both the second best seed,
   1203, and the worst, 290); predation totals 469 / 417 / 485;
   `emit_mean` 0.596 / 0.646 / 0.616; `diet_max` endpoints overlap
   across variants (0.22–0.70 nodiff vs 0.26–0.42 mid vs 0.24–0.43
   fast). Nothing in this world has a *use* for a wider trail yet —
   the channel is physically trail-shaped and semantically uninterpreted
   in 10k ticks. Stages 15 (tag/color) and 16 (donation) shipped later
   and neither consumes the field — the channel still has no reader.
4. **`signal_mass` differs across variants (4.4k–5.6k) without mass
   creation.** The Jacobi step conserves the field sum exactly (unit
   test `test_diffusion_conserves_mass`); what differs is the *source* —
   each variant runs a different population with a different emit gate
   (deposits = `emit_rate * gate * alive`), and decay pulls the level
   toward `emission / signal_decay`. Diffusion redistributes; the
   population writes the level.

## Limitations

- **Population confound.** Variant means differ (290–1358 alive across
  seeds); the ecology panels' bands overlap, but every ecological
  comparison is between density regimes, not pure signal regimes.
- **10k ticks is short for behavior.** Same honest reading as stage 13:
  "no evidence of use", not "evidence of no use" — there is no task in
  this world where reading a wider trail helps (yet).
- **The price was calibrated with diffusion off.** The screen isolated
  the economy; the gate then re-validated survival *with* diffusion at
  the same price (it held: 3/3). An interaction smaller than that was
  not looked for.
- **`signal_cells` has an epsilon** (1% of `signal_capacity`); trail
  area is threshold-dependent and the threshold was not swept.
- **One price in the gate.** 0.05 was chosen by a documented user
  decision over the strict rule's 0.15 (pop-health vs rule-fidelity);
  0.15's 10k behavior remains unmeasured.
- **Ecosystem still predation-poor.** Proto-culture predation fires
  (417–485 deaths) but no variant reaches predator-class `diet_max` in
  every seed — trails changed nothing here, and neither did the price
  beyond slowing the channel down.

## Reproduce

```bash
python experiments/trails/run.py --phase price              # 6k sweep, data/ cache
python experiments/trails/run.py --phase gate --price 0.05  # 10k gate, the table above
python experiments/trails/run.py --phase gate --price 0.05 --force  # recompute
```

Raw per-tick histories land in `data/` (gitignored); figures in `figs/`
(`price_economy.png`, `price_gate.png`, `gate_trails.png`,
`gate_pop.png`, `gate_predation.png`). The 3-seed set is the canonical
(42, 7, 13).
