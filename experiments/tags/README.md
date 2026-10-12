# Stage 15 — lineage colour: does a neutral tag build structure?

## Question

Stage 15 gives every creature a heritable hue (the 5th body trait,
`tag_hue ∈ [0,1)`, circular mutation, uniform spawn) and one new brain
input: the hue similarity of the neighbours inside the smell window
("are the creatures around me my colour?"). The tick never reads
`tag_hue` for energy, cost, mating or death — the colour is **neutral
by construction** (principle 1: nothing rewards a behaviour). Sprites
keep hue = diet; a thin ring under each sprite shows the lineage colour.

Does structure appear anyway, from birth mechanics alone?

## Hypotheses (stated a priori in `run.py`; a seed-42 10k pilot informed
## the refinement below — disclosed here, not hidden)

- **H1 — drift only.** A neutral, nearly-uniform trait is decided by
  drift: bottlenecks concentrate the wheel (`hue_spread` falls, seed
  dependent, not monotone), and the circular mean hue wanders.
- **H2 — structure above two nested chance levels.** Children are born
  at the parent's position wearing one parent's hue, so families should
  form spatial patches. The brain's tag input should exceed:
  - `null_B = alive·R²/n_cells` — random hues *and* random positions;
  - `null_A = win_neigh·R²/9` — random hues *given the observed
    spacing*, where `R = 1 − hue_spread` is the mean unit vector and
    `win_neigh` is the recorded mean neighbour count in the window.
  The spacing factor `win_neigh / (9·(alive−1)/n_cells)` must start at
  1 (random spawn) and rise if creatures aggregate.
- **H3 — sensor use NOT claimed.** Separating "born together" from
  "steered by the similarity input" needs an ablation control with a
  different genome layout (not comparable point-wise). Future work.

## Method

`python experiments/tags/run.py` — 3 canonical seeds (42, 7, 13) ×
10 000 ticks, default `Config`, stats every tick, ~214–253 ticks/s.
`win_neigh` reuses the brain's own crowd sensor (`9 × smell-creatures`
input), so it costs one extra mean per recorded tick.

## Results

**Hard gate PASS** — 3/3 seeds finish 10k with no extinction
(min alive 55 / 54 / 141; final 270 / 469 / 939; mean 178 / 189 / 363).

| seed | final alive | min alive | hue_spread final | tag_sim mean/final | win_neigh final | spacing final | align ratio final |
|---|---|---|---|---|---|---|---|
| 42 | 270 | 55 | 0.175 | 0.046 / 0.150 | 1.35 | 22.4 | 1.5 |
| 7 | 469 | 54 | 0.773 | 0.054 / 0.132 | 1.84 | 17.5 | 12.5 |
| 13 | 939 | 141 | 0.796 | 0.042 / 0.171 | 1.98 | 9.4 | 18.8 |

- **H1 confirmed (drift).** `hue_spread` ends 0.175 (seed 42 — the wheel
  concentrated around one lineage) vs 0.773/0.796 (seeds 7/13 — still
  near-uniform). The circular mean hue wanders per seed and wraps the
  wheel (visible in `drift.png`). The spread band widens after the ~6500
  population bottleneck — classic drift, and no tick code reads the hue.
- **H2 confirmed (structure).** `tag_sim_mean` averages 0.042–0.054 over
  the whole run and ends 0.132–0.171, while `null_B` ends ≈0.0005–0.005
  and `null_A` ≈0.009–0.10 (`similarity.png`, left). Decomposed:
  - **spacing:** creatures sit 9–23× denser than a random spawn
    (factor starts at 1.0, mean after tick 1000 = 17.9/23.1/10.1,
    spikes to ~140 at the bottleneck); `win_neigh` final 1.35–1.98 vs
    chance ≈0.06;
  - **hue alignment beyond spacing:** 1.5× (seed 42) to 12.5×/18.8×
    (seeds 7/13) at the final tick.
- **Health intact** (`pop.png`): population cycles 55–939 without
  collapse, mean energy 30→65–90, no extinction, predation columns
  unchanged from earlier stages.

## Limitations (read before quoting the numbers)

1. **Structure ≠ sensor use (H3).** The tag input rises because kin are
   *born* next to each other; whether any lineage *acts* on the input is
   not measured here. It needs an ablation whose genome layout differs
   — documented as future work, not asserted.
2. **The alignment ratio is noisy early.** With random spawn hues,
   `R ≈ 1/√N ≈ 0.05`, so `null_A ≈ 0` and the ratio spikes by 10³–10⁴
   before tick ~1500 (`similarity.png`, middle, log scale). Read it only
   once `R²` stabilises; the table reports the final tick.
3. **Seed 42 ends at 1.5× only.** Its wheel concentrated (`hue_spread`
   0.175), so `null_A` is large and the ratio is diluted — absolute
   `tag_sim` (0.150) is as high as the others. Ratio and absolute level
   tell slightly different stories; both are reported.
4. **`null_A` is an approximation** — it treats each creature's window
   count as independent of its hue, which kin clustering violates in the
   conservative direction (real null would be ≥ computed null_A).
5. **One pilot seed (42, 10k) refined H2** into the two-null version
   before the 3-seed batch; stated here so the "pre-registered" claim
   is honest.

## Reproduce

```bash
python experiments/tags/run.py --ticks 10000     # ~2.5 min, 3 seeds
python experiments/tags/run.py --ticks 500       # smoke, seconds
```

Raw histories in `data/` (gitignored), figures in `figs/`:
`drift.png` (H1), `similarity.png` (H2 decomposition), `pop.png`
(health).
