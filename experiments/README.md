# Experiments

One directory per experiment, each with the same four parts:

- **`run.py`** — reproducible script (`--seed`, cached npz in `data/`,
  figures in `figs/`, pre-registration noted in the module docstring);
- **`data/`** — raw run outputs, **gitignored** (regenerate with `run.py`);
- **`figs/`** — generated figures, committed so the READMEs render on GitHub;
- **`README.md`** — question, method, numbers, limitations.

Raw caches carry staleness checks: if a recorded column or the current layout
differs from what produced the npz, the variant re-runs automatically.

| Experiment | Question | Result | Stage |
|---|---|---|---|
| [`food_supply/`](food_supply/) | Do the obvious economy knobs fix energy growing without a ceiling? | No — they only change the slope; led to `max_age` 5000 → 2000 | 8 |
| [`mutation_diversity/`](mutation_diversity/) | Does mutation rate control diversity? | Yes: genome distance 3.34 / 7.60 / 12.77 for std 0.05/0.10/0.30 — spread, not direction | 8 |
| [`predation/`](predation/) | Does predation emerge from an all-herbivore spawn? | **Negative:** 0/3/2 kills per run; the `bite_rate × diet` lock | 8 |
| [`lineage/`](lineage/) | Who survives? | 49–56 of 500 families remain; top 10 = 74–77% of the population | 8 |
| [`predator_survival/`](predator_survival/) | Can a ratchet make predation real? | Yes mechanically (`reach`: 1179 kills/run vs 43) — but no obligate predator (diet > 0.5 never alive) | 12.5 |
| [`ecology/`](ecology/) | Does meat reward shape genome isolation? | Speciation-lite either way; the meat reward **doubles** the reproductive barrier (0.100 vs 0.057) | 12 |
| [`signals/`](signals/) | Does price discipline the emit gate? | Yes: emission falls monotonically with cost; too pricey goes extinct 3/3, default `signal_emit_cost = 0` | 13 |
| [`trails/`](trails/) | Is pheromone diffusion physical? Does it change behaviour? | Physics confirmed (trail area ~2× with diffusion); behaviour null at 10k | 14 |
| [`tags/`](tags/) | Does a neutral lineage hue structure space? | Yes: 9–23× denser than a random spawn, alignment 1.5–19× beyond spacing; H3 (sensor use) not claimed | 15 |
| [`proto_culture/`](proto_culture/) | Does donation survive its own price? | **GATE PASS**; gifts align 2.6–4.7× beyond chance toward kin; cheap gate doesn't close | 16 |
| [`civilization/`](civilization/) | Does a wood/fire/house economy survive? | Fire/cooking survives 3/3; houses collapse the population ~10× | 17 |
| [`plasticity/`](plasticity/) | Does Oja learning pay for itself? | Control never beaten; η=1e-2 extinct 3/3; evolvable η sits *below* neutral drift — selection against | 21 |
| [`houses/`](houses/) | Which house prices pass a pre-registered gate? | **Null:** 0/18 pass; shelter cuts bites but the drain is reproductive | 22 |
| [`groups/`](groups/) | Do founder lineages stay as countries? | 4 exact groups 3/3, hold 2–5k ticks, 4× fewer fusions — then drift to ~9 bands by 10k | 23 |
| [`affect/`](affect/) | Do stress/grudge bite scars help? | **HARD GATE FAIL:** causally read, cuts predation 296 → 83, halves the population 3/3; default off | 24 |
| [`territory/`](territory/) | Does the house bank/raid/mark mechanism work and pay? | Mechanism works (2296 banked, 61 raided); gate fails for the *context* — the control arm goes extinct | 25 |

Conventions shared by every experiment (and by the project at large):

- **Pre-registered gates.** The pass/fail rule is written down *before* the
  run; FAILs are published as-is with their limitations.
- **Multiple seeds, honest variance.** 3 seeds × 5k–10k is the house style;
  n is always stated.
- **Old caches date back.** Figures produced before stage 9 describe the old
  regime; bit-exact reproduction needs the old commit (see the
  reproducibility note in the [main README](../README.md)).
