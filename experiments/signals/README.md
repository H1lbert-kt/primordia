# Stage 13 — pricing the emit gate: does the channel survive contact with the economy?

## Question

The signal channel is now wired end to end: every creature owns a scalar
emit gate (brain output 3) that deposits `emit_rate * gate` units of signal
on its own cell at an energy cost of `signal_emit_cost * gate` per tick,
and three new sensors (`signal`, `signal left`, `signal right`) smell the
field over the same cheiro window the food antennas use. The field decays
geometrically (`signal_decay = 0.05`/tick; no diffusion yet — that is
stage 14). Nothing in the tick rewards communicating: whether anyone uses
the channel is an experiment question, not a rule.

`Config.signal_emit_cost` defaults to **0.0** on purpose: the emit gate is
a sigmoid that never reaches exactly 0, so a nonzero default would tax
every creature `~cost/2` per tick from birth and silently change the whole
economy. The price is a *treatment*, applied by this experiment on top of
the pinned proto-culture world (`configs/proto_culture.toml`:
`bite_rate = 12`, `contact_range = 2.5`, the stage-12.5 "reach" variant).

## Hypothesis (stated a priori, before any run on this code)

> The price of the emit gate is an energy tax the population will route
> around — creatures will learn to close the gate (`emit_mean` drifts
> toward 0 as `signal_emit_cost` rises), so "pricey" runs will show lower
> signal mass and **no systematic change in population or predation**
> versus "proto".

The second half of that prediction is what the gate phase was designed to
falsify. It did.

## Method

3 variants (only `signal_emit_cost` differs; everything else is the
`proto_culture.toml` predation context) x 3 seeds (42, 7, 13):

| variant | `signal_emit_cost` | rationale |
|---|---|---|
| `proto` | 0.5 | the pinned proto-culture price (~6x metabolic cost at gate=1) |
| `cheap` | 0.001 | near-free: does the gate just stay open and spam? |
| `pricey` | 2.0 | 25x metabolic cost: close it or starve |

Screening phase: 6000 ticks. Gate phase: 10000 ticks, same variants,
pre-registered metrics (hard: no extinction in any seed; measured:
`signal_emitted > 0` in >= 1/3 seeds at the default price).

## Results

### Gate phase (10k ticks)

| variant | final pop (mean of 3 seeds) | `emit_mean` final | `signal_emitted` total | `signal_mass` final | predation deaths | `diet_max` final |
|---|---|---|---|---|---|---|
| `proto` | 33 (32 / 67 / **0**) | 0.335 | 40 463 | 4.6 | 4 | 0.034 |
| `cheap` | 728 (1258 / 261 / 664) | 0.704 | 2 059 625 | 7 490 | 788 | 0.518 |
| `pricey` | **0** (0 / 0 / 0) | 0.000 | 11 497 | 0.0 | 0 | 0.000 |

Screening phase (6k) told the same story earlier: `proto` 9 alive,
`cheap` 269, `pricey` 2.

### Verdict against the pre-registered gate

- **Hard gate: FAIL for `proto` and `pricey`, PASS for `cheap`.**
  `proto` lost seed 13 to extinction by 10k (32 / 67 / **0**); `pricey`
  went extinct in 3/3 seeds. Determinism and save/load v6 are covered by
  the unit suite, not by these runs.
- **Measured gate: technically met** (`signal_emitted > 0` in 3/3 seeds
  at cost 0.5 — 40 463 units deposited) **but hollow**: the world paying
  for those units was dying while it deposited them.

### What the trajectories actually show

1. **The gate does evolve under price — the first half of the hypothesis
   survives.** `emit_mean` over living slots falls monotonically with the
   cost: 0.70 (`cheap`) -> 0.33 (`proto`) -> 0.17 at 6k, then 0.00 by 10k
   (`pricey`). Selection is acting on output 3 within a few thousand
   ticks. Nobody "wants" to pay.

2. **But the response is too slow against a tax that bites from birth —
   the second half is refuted.** A freshly spawned creature inherits a
   random-init gate around 0.7 and pays `0.7 * 0.5 = 0.35` energy/tick
   immediately: ~4x the metabolic cost (0.08) and on top of movement and
   senescence. The population is already deep in the red on the ticks
   where selection would need thousands of generations to pull the emit
   logit down through crossover and mutation. At cost 2.0 the arithmetic
   is fatal: 3/3 seeds extinct, no survivors left to carry a closed gate.

3. **`cheap` is a spam ocean, and the channel is provably load-bearing
   for nothing (yet).** Near-zero price keeps the gate open (`emit_mean`
   0.70, flat) and pumps 2.06 million units into the field — a
   signal-mass equilibrium around `deposit / decay` ~ 7 500. The world
   survives, predation thrives (788 deaths, `diet_max` 0.52 — the only
   variant where the proto-culture predation context actually fires), but
   nothing here shows anyone *reads* the field. This is a wiring result,
   not a semantics result: the channel is cheap, loud, and uninterpreted
   in 10k ticks.

4. **The viable-price window is narrow and was not explored.** Between
   0.001 (survives, spam) and 0.5 (collapses) there is at least one
   threshold where the tax is affordable *and* the gate starts to close.
   This experiment deliberately brackets it from both sides instead of
   hunting it; finding it is a follow-up sweep, not a stage-13 result.

## Limitations

- **Population size confounds everything downstream.** `cheap` has 728
  alive and `proto` has 33; comparing predation (788 vs 4) between them
  compares two different density regimes, not two signal regimes. The
  predation number for `cheap` is close to the stage-12.5 `reach` result
  (~1 179 at the same bite parameters but the pre-stage-13 genome), which
  is consistent with "population size, not signal, drives encounter rate".
- **10k ticks is short for a channel nobody is selected to use.** The
  honest reading of `cheap` is "no evidence of use", not "evidence of no
  use": there is no task in this world that reading signal would help
  with yet (stage 15 tag / stage 16 donation are the planned consumers).
- **One price point per class.** 0.5 and 2.0 bracket the collapse; 0.001
  brackets the spam floor. The actual affordable-price band is unknown.
- **The `proto_culture.toml` price of 0.5 is now known to be economically
  lethal** for this world and should not be used as a default for stages
  14+ without a recalibration sweep first.

## Reproduce

```bash
python experiments/signals/run.py              # screening, 6k, uses data/ cache
python experiments/signals/run.py --force      # recompute
python experiments/signals/run.py --phase gate # 10k, the table above
```

Raw per-tick histories land in `data/` (gitignored); figures in `figs/`
(`signal_activity.png`, `emit_gate.png`, `economy.png`, `predation.png`
plus `phaseB_*` for the gate phase). The 3-seed set is the canonical
(42, 7, 13).
