"""Shared helpers for the stage-8 experiments.

Conventions every ``experiments/<name>/run.py`` follows:

    python experiments/<name>/run.py            # uses data/ as cache
    python experiments/<name>/run.py --force    # recompute everything

Runs are **sequential** (each ``World`` reseeds the process-global Numba RNG
at construction — never interleave worlds). ``data/*.npz`` holds the raw
per-tick stats (gitignored: regenerable, AGENTS.md forbids committing long
run outputs); ``figs/*.png`` are the committed figures. ``SEEDS`` is the
canonical 3-seed set so every experiment reports mean and spread honestly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot: experiments run headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from primordia.config import Config  # noqa: E402
from primordia.stats import StatsHistory, StatsRecorder  # noqa: E402
from primordia.step import advance  # noqa: E402
from primordia.world import World  # noqa: E402

SEEDS = (42, 7, 13)
DPI = 120

# Config defaults of stage 8, pinned so cached runs and README numbers stay
# reproducible after later stages change Config defaults. Merge order in the
# experiment scripts: asdict(Config()) | STAGE8_BASE | variant overrides.
# Stage 9.5 changed the economy (food_capacity 100->3, eat_rate 4->1, ...);
# stage 10 added terrain, day/night and seasons (off here) — pin the whole
# old regime, not just one knob, so --force reruns stay documented (exact
# bit-reproduction still needs the stage-8 commit: brains gained recurrence
# in stage 9).
STAGE8_BASE = {
    "food_patch_amplitude": 0.0,  # stage 8 ran on a uniform field
    "food_capacity": 100.0,
    "food_growth_rate": 0.05,
    "initial_food": 60.0,
    "eat_rate": 4.0,
    "metabolic_cost": 0.05,
    "move_cost": 0.05,
    "max_turn": 0.6,
    "turn_cost": 0.0,  # did not exist in stage 8: free steering
    "terrain_amplitude": 0.0,  # stage 10 environment, off for stage-8 runs
    "season_amp": 0.0,
    "day_amp": 0.0,
}


def run_variant(
    cfg: Config, seed: int, ticks: int, npz: Path, force: bool = False
) -> StatsHistory:
    """Run one (config, seed) variant for ``ticks`` ticks, recording stats.

    Returns the history from ``npz`` (already on disk — it doubles as the
    cache, so an interrupted batch resumes where it stopped unless
    ``force``). Sequential by contract: never call from two threads.
    """
    if npz.exists() and not force:
        history = StatsRecorder.load(str(npz))
        if len(history) == ticks:
            print(f"  seed={seed}: cached {npz.name}", flush=True)
            return history
        print(f"  seed={seed}: stale ({len(history)} != {ticks} ticks), rerunning")

    rec = StatsRecorder(capacity=ticks)
    w = World(cfg, seed=seed)
    start = time.perf_counter()
    for _ in range(ticks):
        advance(w)
        rec.record(w)
    elapsed = time.perf_counter() - start
    print(
        f"  seed={seed}: {ticks / elapsed:.1f} ticks/s ({elapsed:.1f} s), "
        f"alive={w.alive_count}, mean_energy="
        f"{float(w.energy[w.alive].mean()) if w.alive_count else 0.0:.1f}",
        flush=True,
    )
    npz.parent.mkdir(parents=True, exist_ok=True)
    rec.save(str(npz), seed=seed)
    return StatsRecorder.load(str(npz))


def label(overrides: dict) -> str:
    """Legend label for a variant: ``food=0.01, metab=0.15``."""
    return ", ".join(f"{k}={v:g}" for k, v in overrides.items())


def stacked(histories: list[StatsHistory], column: str) -> tuple:
    """Stack one column across seeds: returns (tick, mean, lo, hi) arrays.

    ``lo``/``hi`` are the min/max across seeds — honest for n=3 (no fake
    confidence intervals). All histories of an experiment share the same
    tick grid, which is what makes the stack valid.
    """
    tick = histories[0]["tick"]
    data = np.stack([h[column] for h in histories])  # noqa: N806 (local)
    return tick, data.mean(axis=0), data.min(axis=0), data.max(axis=0)


def band(ax, histories: list[StatsHistory], column: str, name: str, color=None):
    """Plot mean trajectory with a min-max band across seeds."""
    tick, mean, lo, hi = stacked(histories, column)
    (line,) = ax.plot(tick, mean, label=name, color=color)
    ax.fill_between(tick, lo, hi, color=line.get_color(), alpha=0.2)


def save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ticks", type=int, default=5000)
    parser.add_argument(
        "--force", action="store_true", help="ignore data/ cache and rerun"
    )
    return parser
