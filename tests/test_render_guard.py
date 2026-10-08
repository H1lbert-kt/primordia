"""Stage-4 rendering guards: the core never imports pygame, and the window
runs headlessly (SDL dummy driver) producing the same world as --headless."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from primordia.config import Config
from primordia.step import advance
from primordia.world import World

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_core_never_imports_pygame() -> None:
    """Importing the simulation must not pull pygame or matplotlib."""
    code = (
        "import sys\n"
        "import primordia.config, primordia.world, primordia.step, "
        "primordia.genome, primordia.brain, primordia.sensors, primordia.bench, "
        "primordia.stats\n"
        "assert 'pygame' not in sys.modules, 'pygame leaked into the core'\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib leaked into the core'\n"
        "print('core-clean')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr
    assert "core-clean" in out.stdout


def test_windowed_run_matches_headless(tmp_path: Path) -> None:
    """100 windowed ticks on the dummy driver == 100 headless ticks.

    Rendering reads state and draws no random numbers, so the viewer cannot
    perturb the simulation (determinism contract).
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    cfg = replace(Config(), max_creatures=120, initial_creatures=120)

    # Worlds run sequentially: the Numba RNG is a process-global stream.
    w1 = World(cfg, seed=42)
    for _ in range(100):
        advance(w1)

    from primordia.render import run_app

    shot = tmp_path / "frame.png"
    w2 = World(cfg, seed=42)
    summary = run_app(
        w2, seed=42, max_ticks=100, select_slot=0, screenshot_path=str(shot)
    )

    assert summary["ticks"] == 100
    assert summary["frames"] > 0
    assert summary["mean_fps"] > 0.0
    assert shot.exists() and shot.stat().st_size > 0

    assert np.array_equal(w1.pos, w2.pos)
    assert np.array_equal(w1.vel, w2.vel)
    assert np.array_equal(w1.angle, w2.angle)
    assert np.array_equal(w1.energy, w2.energy)
    assert np.array_equal(w1.age, w2.age)
    assert np.array_equal(w1.genome, w2.genome)
    assert np.array_equal(w1.alive, w2.alive)
    assert np.array_equal(w1.food, w2.food)
    assert w1.tick == w2.tick
    assert (w1.births, w1.deaths_famine, w1.deaths_age) == (
        w2.births,
        w2.deaths_famine,
        w2.deaths_age,
    )
