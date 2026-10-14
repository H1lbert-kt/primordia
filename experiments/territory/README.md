# Territory: does a banked house become a territory worth raiding?

Stage 25 adds an economy on top of the stage-22 house: surplus energy
above the reproduce threshold banks itself into the creature's own
complete house (automatic, no gate), the same open mouth that eats also
withdraws from its own bank or raids an unguarded foreign one, the owner
creature alive within `guard_range` makes the bank drain-proof, a mark
ring appears on the ground over banked houses, and standing on your own
mark trims the metabolic bill. Nothing rewards banking (principle 1).

## Method

Context: `configs/proto_culture.toml` (bite_rate 12, contact_range 2.5)
+ the stage-22 house knobs that made construction viable in the houses
pilot (wood 2, energy 3, cover 0.5, carry 1.0, bias −1.5, rot 0.01,
fire_radius 6.0, spread 0). Two variants × seeds 42/7 × 10k ticks
(2 seeds, as in the houses gate — banked houses keep the population
small; the third seed belongs to a future sweep if this one needs one):

| variant | knobs |
|---|---|
| `off` | houses on, `deposit_rate=0, raid_rate=0, comfort=0` (stage-22 baseline) |
| `store` | `deposit_rate=0.5, raid_rate=0.3, guard_range=3.0, comfort=0.3, deposit_threshold_frac=1.1` |

Pre-registered gate: hard (all runs alive at 10k) + bank fills (≥2/2)
+ theft emerges (≥2/2) + no collapse (store ≥ 0.8× off, same seed).

## Result

**Hard gate fails for the context, not the treatment.** The `off`
control itself goes extinct on seed 7 (tick 5248, 522 famine deaths,
22 births — the house economy at this price is famine-dominated and
marginal), so no territory variant can pass the registered gate. On the
surviving seed 42 the mechanism demonstrably works:

| seed 42 | off | store |
|---|---|---|
| final pop | 29 | **116** |
| min pop | 2 | 25 |
| deposited (total) | 0 | **2296** |
| raided (total) | 0 | **61** (265 ticks, first @434, peak 0.57/tick) |
| bank standing (final/max) | 0 | 161 / 317 |
| mark cells (final/max) | 0 | 26 / 45 |
| births | 155 | **385** |
| famine deaths | 595 | 672 |

- **Bank fills** (deposited > 0 in 2/2 store seeds — even the dying
  seed 7 banks 321 energy before collapsing at 6616, later than the
  control's 5248).
- **Theft emerges on the surviving seed** (61 energy raided over 265
  ticks), but only 1/2 seeds — fails the ≥2/2 registration.
- **No collapse — the opposite direction**: on seed 42 the store
  variant ends with **4× the control's population** (116 vs 29) and
  2.5× the births. The comfort discount (metabolic × 0.7 on own mark)
  plus the withdrawable bank are a real survival advantage in a
  famine-dominated world; this is trajectory, not a randomized
  treatment effect (same seed, but the dynamics diverge from tick one).

## Verdict

Null as pre-registered (the hard gate is unconditional), but the null
is a **context-stability null, not an inert-mechanism null**: the
deposit/withdraw/raid economy demonstrably moves energy (2296 in, 61
raided, 317 standing at peak, 45 marked cells), and on the one stable
seed it correlates with a substantially *healthier* population. A
clean causal claim needs a stable context — next step is an economy /
price sweep with the user (lesson 17/18: report, never hack).

## Limitations

1. The off control goes extinct on seed 7 — the house-price context
   itself is marginal at 10k; the registered gate cannot separate
   treatment from context instability.
2. n = 2 seeds (banked houses keep pops small; a third seed belongs to
   a follow-up sweep, not to this null).
3. The 4× population gap on seed 42 is trajectory, not a randomized
   effect — comfort + bank change energy flows from tick one.
4. Raids are rare (265 of 10k ticks) — the guard (presence within
   3.0 of the house) is a cheap defense; whether a *costly* guard
   (patrolling, not just standing) would change the ratio is untested.
5. `deposit_threshold_frac=1.1` (bank only above 110 energy) was never
   swept; a lower trigger would bank earlier and more often.
6. Figures are min–max bands across 2 seeds; on seed 7 the store
   trajectory ends at the extinction tick.

## Reproduce

```bash
python experiments/territory/run.py --ticks 10000        # cached in data/
python experiments/territory/run.py --ticks 10000 --force
```

Figures: `figs/economy.png` (pop, energy), `figs/bank.png` (energy in
stores, mark cells), `figs/flows.png` (deposited, raided per tick).
