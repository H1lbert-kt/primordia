"""Stage-8 recording: frames/GIF are produced without touching the world.

A recorded run must be bit-identical to the same run without recording
(rendering reads state and draws no random numbers), and the pipeline must
emit exactly one PNG per capture plus a readable multi-frame GIF.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from primordia.config import Config
from primordia.render.record import WorldRecorder
from primordia.step import advance
from primordia.world import World

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # windowless even locally

TICKS = 100
EVERY = 25


def _snapshot(world: World) -> dict:
    return {
        "pos": world.pos.copy(),
        "vel": world.vel.copy(),
        "angle": world.angle.copy(),
        "energy": world.energy.copy(),
        "age": world.age.copy(),
        "genome": world.genome.copy(),
        "alive": world.alive.copy(),
        "creature_id": world.creature_id.copy(),
        "births": world.births,
        "deaths_famine": world.deaths_famine,
        "deaths_age": world.deaths_age,
        "deaths_predation": world.deaths_predation,
        "tick": world.tick,
        "genealogy_used": world.genealogy_used,
        "genealogy": world.genealogy[: world.genealogy_used].copy(),
    }


def test_recording_is_read_only_and_deterministic(tmp_path) -> None:
    cfg = Config()

    plain = World(cfg, seed=42)
    for _ in range(TICKS):
        advance(plain)

    recorded = World(cfg, seed=42)
    rec = WorldRecorder(recorded, tmp_path / "rec", seed=42, speed=EVERY)
    rec.capture(recorded)  # tick 0
    for _ in range(TICKS):
        advance(recorded)
        if recorded.tick % EVERY == 0:
            rec.capture(recorded)

    a, b = _snapshot(plain), _snapshot(recorded)
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert np.array_equal(a[key], b[key]), f"field diverged: {key}"
        else:
            assert a[key] == b[key], f"field diverged: {key}"

    frames = rec.frames
    assert len(frames) == 1 + TICKS // EVERY  # tick 0 plus one per window
    assert all(p.stat().st_size > 0 for p in frames)


def test_gif_is_assembled_from_all_frames(tmp_path) -> None:
    world = World(Config(), seed=7)
    rec = WorldRecorder(world, tmp_path / "rec", seed=7, speed=10)
    for _ in range(40):
        advance(world)
        rec.capture(world)  # age ticks: every frame differs

    gif = rec.save_gif()
    assert gif.exists()
    assert gif.stat().st_size < 4 * 1024 * 1024  # README budget
    img = Image.open(gif)
    assert img.n_frames == 40
    assert img.width <= 960
