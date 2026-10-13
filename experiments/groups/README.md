# Groups — founder lineages, countries and borders (stage 23)

**Question.** If the initial population is split into 4 founding lineages
pinned to 4 `tag_hue` bands in 4 torus strips (`founder_regions=4`), does
the *observer* (hue-group detector, `primordia.observer` — the tick never
sees any of it) see stable groups, frontiers and contact events? Control:
the free run (`founder_regions=0`), where hue groups are arbitrary colour
bands of one mixed population.

**Method.** Default Config (terrain on — water is a free isolation
mechanism), seeds 42/7/13, 10 000 ticks, observer sampled every 100
ticks. Metrics: `n_groups`, `border_cells` (torus-adjacent occupied cells
of different groups), cumulative event counts (`new`/`extinct`/`fusion`/
`invasion` — geometric conventions documented in `primordia/observer.py`),
`alive`. Pre-registered nulls: (H1) lineages homogenize early;
(H2) detection flicker inflates event counts; (H3) free runs show many
bands with no spatial structure.

```bash
python experiments/groups/run.py --force      # 6 runs, ~4 min
```

## Results (real numbers, mean [min, max] across 3 seeds)

| metric (final @10k)         | default        | founders4      |
|-----------------------------|----------------|----------------|
| alive                       | 1236 [1053, 1531] | 1437 [1230, 1701] |
| observer groups             | 9.3 [9, 10]    | 9.0 [9, 9]     |
| frontier cells              | 149 [61, 210]  | 100 [44, 152]  |
| fusion events (cumulative)  | 112 [94, 124]  | 25 [15, 36]    |
| extinction events           | 26 [22, 33]    | 12 [7, 14]     |
| invasion events             | 3.3 [0, 8]     | 2.0 [0, 4]     |

**Founding works — for a few thousand ticks.** 3/3 founders4 seeds start
at *exactly* 4 groups (the pinned lineages) and hold 4–5 groups through
2k ticks (seed 7 holds 4 until 5k). Group count then drifts up and both
variants converge to ~9 bands by 10k (fig `groups.png`). The drift is
mutation, not war: the hue gene mutates only through the per-gene gate
(`mutation_rate=0.01` per birth, `trait_mutation_std=0.10`), so bands
persist for many generations and split slowly. The ~9-band equilibrium is
a property of (`group_hue_threshold=0.05`, mutation rates, population
size) — not a universal constant.

**Coherence is measurable in the event log** (`events.png`): founders4
accumulates 4× fewer fusions (25 vs 112) and 2× fewer extinctions than
the free run, whose 15–16 random spawn bands immediately churn and merge.
H2 (flicker) is real but mild: ~12 extinction events in founders4 over
10k ticks against a stable population.

**Borders exist in both** (`borders.png`): frontier cells grow with
population density (from ~0 at the sparse start to 34–152 at 10k).
Invasions (≥8 crossers in foreign-majority cells, hysteresis per pair)
are rare under this strict geometric definition — 0–8 per run, appearing
only at high density near the end (2/3 founders4 seeds).

**H1 partially confirmed:** the founder identity blurs by ~5–10k ticks
(mutation + interbreeding across the strip seams); the observer is honest
about it — it reports drift, it does not preserve the "country" by fiat.

## Limitations

1. **Groups are observer constructs.** Identity = slot-overlap matching
   between samples; uids flicker when detection flickers. "Fusion",
   "extinction" and "invasion" are the geometric definitions in
   `observer.py`, not simulated politics — no creature knows a group
   exists.
2. **Hue bands are not genealogy.** Crossover inherits one parent's hue
   and mutation drifts it; after enough generations a hue group is a
   colour neighbourhood, not a family tree.
3. The invasion threshold (8 crossers) is a convention; a looser one
   fires earlier, a stricter one never.
4. Border counts undercount sparse worlds: frontiers require *occupied*
   cells to touch, so two groups separated by empty cells share no
   measurable border.
5. 3 seeds; default Config's harsh early famine (pop crashes 500→~100
   before recovering) shapes the density curves. The final-population
   difference between variants is a trajectory effect of the changed
   spawn RNG, not a treatment effect.
6. Equilibrium band count was not swept against `group_hue_threshold`;
   only the default 0.05 was measured.

## Files

- `data/*.npz` — per-sample observer metrics (gitignored, regenerable)
- `figs/groups.png`, `figs/borders.png`, `figs/events.png`, `figs/pop.png`
