# Plasticity (stage 21): does local Oja learning pay for itself?

**Question.** Synapses can now change during a life by a purely local rule
with no reward — Oja's rule written into a separate `dw` scratch
(`W_eff = genome + dw`), decaying back to the evolved blueprint, clamped,
and charged to the creature's energy in proportion to `|dw|` moved. The
genome is immutable inside a life (Baldwin, not Lamarck). Nothing rewards
learning. Does it help? Does selection turn it on when the rate itself is
an evolvable trait?

**Method.** 6 variants × seeds 42/7/13 × 10 000 ticks (default `Config`,
current HEAD economy), pre-registered in `run.py` *before* the batch.
The learning price `plasticity_cost=1.0` is **held fixed everywhere** —
this stage does not sweep it; a price sensitivity is a separate
experiment. Modes are set explicitly per variant, so the Config default
(which is `off` since the post-gate decision, with `gene` pinned in
`configs/proto_culture.toml`) does not affect these runs:

| variant | rule |
|---|---|
| `off` | `plasticity_mode="off"` — free control, `dw` frozen at 0 |
| `oja1e-4/1e-3/1e-2` | fixed η ∈ {1e-4, 1e-3, 1e-2}, `plasticity_cost=1.0` |
| `oja_de` | fixed η=1e-3 **+ delta-E credit** (local credit assignment, off by default) |
| `gene` | evolvable η trait, spawned at 0, mutation at `plasticity_eta_std=0.01` |

A 2-seed × 5k pilot (disclosed) informed `plasticity_eta_std`: with the
default `trait_std=0.10`, η jumps to ~0.1 (10× the sweep's top arm) on
every mutation hit and selection purged the gene to η=0 in all survivors
(pilot: 0/442 and 0/380 with η>0) before ever exploring a viable rate.
With η₀=0.001 the same pilot keeps η>0 standing (17/380 alive, `|dw|` up
to 0.9) — the finer mutation scale is what makes the gene question
measurable at all.

## Results (10k gate, real numbers)

Final alive / min alive / cumulative births (seeds 42 / 7 / 13):

| variant | final | min | births | verdict |
|---|---|---|---|---|
| `off` | 1053 / 1531 / 1124 | 96 / 122 / 136 | 2755 / 3769 / 2880 | control |
| `oja1e-4` | 832 / 1155 / 834 | 48 / 112 / 123 | 1433 / 3286 / 2107 | below control |
| `oja1e-3` | 926 / 959 / 839 | 104 / 115 / 113 | 2507 / 2981 / 2069 | below control |
| `oja1e-2` | **0 / 0 / 0** | 0 | 11 / 8 / 0 | **extinct ~5k** |
| `oja_de` | 65 / 868 / 466 | 17 / 95 / 76 | 546 / 1982 / 1175 | destabilised |
| `gene` | **1141 / 1265 / 1461** | 96 / 122 / 122 | 3079 / 3143 / 3023 | alive, engaged |

Figures: `figs/gate.png` (survival + engagement), `figs/economy.png`
(energy, births, η share), `figs/clones.png` (clone divergence).

### H1 — pure Oja does not improve survival: **confirmed**

Every fixed-η arm sits *below* the control (mean final alive 940 @1e-4 and
908 @1e-3 vs 1236 for `off`), with roughly half the births at 1e-4 seed 42
(1433 vs 2755). The rule learns the *statistics of the inputs*, not what
is good for survival, and every unit of `|dw|` moved is billed at
`plasticity_cost=1.0`. η=1e-2 pays the bill until the world is empty:
extinction at ticks **4999 / 5351 / 4999**, ~500 famine deaths each.

### H2 — with a cost, selection keeps η near 0: **confirmed, and it is suppression**

In `gene`, the share of survivors with η>0 peaks at **0.9 / 5.4 / 1.9 %**
mid-run and settles at **0.61 / 0.71 / 0.75 %** at 10k; η itself sits at
p50 ≈ 0.003–0.006 with p90 ≈ 0.013–0.017 — *inside* the sweep's viable
band.

The selection test set before reading the data: the result counts as
selection *for* learning only if mean η stayed **above** the
neutral-drift control (the `off` arm still carries the unused gene,
drifting at `trait_std`) in 3/3 seeds. **It was run, and it fails that
test in every seed** — mean η under plasticity is *below* drift
throughout:

| seed | gene η̄ final / mid-run | off (drift) η̄ final / mid-run | gene η_max / off η_max |
|---|---|---|---|
| 42 | 5e-5 / 3e-5 | 3.6e-4 / 2.2e-4 | 0.017 / 0.018 |
| 7 | 2e-5 / 7e-5 | 9.6e-4 / 5.4e-4 | 0.008 / 0.019 |
| 13 | 4e-5 / 2e-5 | 8.8e-4 / 7.3e-4 | 0.015 / 0.028 |

(and likewise for the η>0 share: 0.6–0.75 % under plasticity vs 5.4–14.1 %
drifting unused). So the honest reading is **negative selection**: the
cost is being paid for, the distribution is suppressed an order of
magnitude below what mutation alone would produce, and the small
standing polymorphism of learners is mutation–selection balance — not
evidence that selection *turned learning on*. What the experiment does
show is the separation the stage was built for: when the rate is
evolvable, the population never walks into the lethal η=1e-2 band on its
own.

### H3 — delta-E credit is the most divergent: **confirmed**

`oja_de` is the only variant with the same η as `oja1e-3` but wildly
different outcomes: seed 42 collapses to **65 alive (min 17)** while
`oja1e-3` stays at 926; cross-seed spread (65/868/466) dwarfs every other
variant. Reading the individual's own energy change as credit destabilises
what the pure local rule leaves alone — exactly why it is off by default
and declared per principle 1 (energy is a sensed state variable, never a
fitness score).

### Clones: open mind, inherited blueprint

Two worlds, bit-identical genetics (`mutation_rate=0`, no reproduction),
two food environments: after 2000 ticks `max|genome_A − genome_B| =
0.000000` while `max|dw_A − dw_B| = 0.5465`; the plasticity-off control
stays at exactly 0. The learned delta is lifetime state; the heritable
unit is untouched (`genome_dist`/speciation keep measuring genes alone).

## Gate as registered: FAIL (disclosed)

The pre-registered hard gate said *every run of every variant* must
survive. `oja1e-2` extinct 3/3 → **GATE: FAIL**, reported as-is. The
failure is the result: it brackets the viable price window at η ≤ 1e-3
under `plasticity_cost=1.0` and is why the shipped default is not a fixed
rate at all. The *default-variant* sub-gate (`gene`) passes: 3/3 alive,
min 96–122, no NaN.

## Limitations

1. **One cost price, by design.** `plasticity_cost=1.0` is held fixed
   through the whole stage (including the extinct arm); a price sweep is
   a separate sensitivity experiment, deliberately not mixed in here.
2. **Oja learns input statistics, not tasks.** A supervised or
   reward-predictive local rule is a different experiment (and a
   different principle conversation).
3. **n=3 seeds**, current-HEAD economy; predation deaths are still
   small (1–302) so "learning helps hunt" is not testable yet.
4. **The gene's mutation scale is itself a choice** (`eta_std=0.01`,
   pilot-calibrated): a different scale moves the equilibrium share.
5. **Clones use fixed η=1e-3**, not the gene — clone divergence of the
   evolvable arm would confound mutation with environment (mutation is
   off by construction there).

## Reproduce

```bash
python experiments/plasticity/run.py --ticks 10000        # gate (~15 min)
python experiments/plasticity/run.py --phase clones       # clone pair
python experiments/plasticity/run.py --ticks 10000 --force
```

`data/*.npz` (per-tick histories) is gitignored; figures regenerate from it.
