# Stage 24 — Affect channel (stress + grudge): HARD GATE FAIL, channel is causally read

## Question

On the proto-culture world (frequent bites, stage-12.5 "reach" predation),
does turning the stage-24 affect channel **ON** (`affect_enabled=True`:
input 35 = bite scar `stress`, input 36 = rancor `grudge` toward the last
attacker's hue, faded by recency) change population health compared with
**OFF** (same scratches written, brain reads exact zeros)?

## Method

- Base: `configs/proto_culture.toml` (`bite_rate=12`, `contact_range=2.5`,
  emit 0.05, plasticity gene). Only `affect_enabled` differs between
  variants. Seeds 42/7/13, 10 000 ticks, sampled every 100 ticks.
- Metrics (sampled outside the tick — the scratches are lifetime state,
  not `StatsRecorder` columns, same precedent as the stage-23 observer):
  alive, births, deaths by cause, stress variance among the living,
  grudge carrier share.
- Pre-registered hard gate: no extinction, final alive ≥ 0.9× of the
  same-seed control, min alive > 30.

Reproduce:

```bash
python experiments/affect/run.py            # uses data/ as cache
python experiments/affect/run.py --force    # recompute everything
```

## Result

**HARD GATE FAIL in 3/3 seeds** — final population drops ~50% with the
channel on:

| seed | off final | on final | ratio | predation off/on | min alive off/on |
|------|-----------|----------|-------|------------------|------------------|
| 42   | 571       | 206      | 0.36  | 296 / 83         | 59 / 36          |
| 7    | 625       | 450      | 0.72  | 251 / 240        | 131 / 103        |
| 13   | 164       | 32       | 0.20  | 22 / 0           | 13 / 16          |

- **The channel is causally read (not stream noise).** The two variants
  are bit-identical until the first bite (samples 13–36, tick ~1300–3600)
  and diverge only after stress/grudge become non-zero. Same layout, same
  spawn stream — divergence is the brain acting on the new inputs.
- **Measured gate (signal exists):** stress variance among the living peaks
  at 300 (off) / 141 (on); grudge carrier share peaks at 0.69–0.93. The
  scratches fill under proto-culture predation as designed.
- **Predation deaths fall in 2/3 seeds with the channel on** (296→83,
  22→0) — consistent with stress-driven avoidance — but the population
  still ends smaller: whatever the scars teach (flee? stop foraging near
  conspecifics?), it costs more energy than the bites it avoids.
- Seed 13's *control* already bottoms at 13 alive (pre-existing
  proto-culture bottleneck on that seed, seen in earlier experiments);
  the "min alive > 30" criterion was calibrated for healthier runs and
  fails on the control too. The ≥0.9× ratio criterion fails independently
  and decisively.

## Conclusion

As wired (stress = landed damage with exponential decay, grudge = circular
hue similarity to the last attacker), the affect channel is **read and
behaviorally potent but not free** — on this world it halves the
population. The Config default stays **off** (`affect_enabled=False`).
A softer treatment (stress_scale ≫ 1.0 so the input rarely saturates, or
a decay fast enough that only very fresh scars register) is a follow-up
hypothesis, not a hot-fix — per AGENTS.md, balance is reported, not
patched.

## Limitations

1. **n=3 seeds** — the direction is consistent but the magnitude
   (0.20–0.72×) is seed-dependent; no claim about the variance of the
   effect size.
2. **Mechanism not isolated** — stress and grudge move together (both fill
   on every bite); the experiment toggles the pair, not each input. Which
   one drives the population drop is unknown.
3. **One world** — proto-culture predation is frequent by design; the
   default world (bite_rate=4) barely bites, so "on" there is nearly
   identical to "off" (few scars to read).
4. **Fixed prices** — `stress_decay=0.02`, `stress_scale=1.0`,
   `grudge_decay=0.01` were not swept; the negative may be a price
   problem, not a channel problem.
5. **Trajectory divergence confounds late-run comparisons** — after the
   first bite the two variants live in different worlds (different
   populations, different lineages); final-population ratios mix the
   channel's effect with chaotic amplification. The bit-identical prefix
   is what makes the early divergence attributable.
6. The 500-tick pilot (identical pops, zero predation) is cached under
   the same names; `--force` at 10k overwrote it. The identity claim
   above comes from the full-run arrays, not the pilot.
