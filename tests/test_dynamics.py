"""Stage-3 render dynamics: birth/death flash diff, aging/cap, sparklines.

These trackers live on the render side and only read the world; the tests
pin their behavior so recording/headless equivalence keeps holding.
"""

from __future__ import annotations

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from primordia import step  # noqa: E402
from primordia.config import Config  # noqa: E402
from primordia.step import advance  # noqa: E402
from primordia.world import World  # noqa: E402


def _settings(**overrides):
    from dataclasses import replace

    from primordia.render.settings import Settings

    return replace(Settings(), **overrides)


@pytest.fixture(scope="module")
def ready() -> None:
    pygame.font.init()


def test_birth_flash_is_green(ready) -> None:
    from primordia.render.draw import EventTracker

    cfg = Config(
        max_creatures=8,
        initial_creatures=4,
        reproduce_threshold=10.0,
        max_age=10_000,
    )
    w = World(cfg, seed=5)
    settings = _settings()
    tracker = EventTracker(w)
    w.energy[w.alive] = 50.0  # way above threshold, free slots exist
    advance(w)
    assert w.births > 0
    tracker.update(w, settings)
    assert tracker.flashes
    assert all(f.color == settings.flash_birth for f in tracker.flashes)


def test_death_flash_uses_famine_color(ready) -> None:
    from primordia.render.draw import EventTracker

    cfg = Config(max_creatures=8, initial_creatures=8, reproduce_threshold=1e9)
    w = World(cfg, seed=5)
    settings = _settings()
    tracker = EventTracker(w)
    # kill directly: going through `advance` would let them eat first
    w.energy[w.alive] = 0.0
    step.phase_deaths(w)
    assert w.deaths_famine == 8 and w.alive_count == 0
    tracker.update(w, settings)
    assert len(tracker.flashes) == 8
    assert all(f.color == settings.flash_famine for f in tracker.flashes)


def test_flashes_age_out_and_are_capped(ready) -> None:
    from primordia.render.draw import EventTracker

    settings = _settings(flash_duration=5, flash_max=10)
    cfg = Config(max_creatures=64, initial_creatures=64, reproduce_threshold=1e9)
    w = World(cfg, seed=5)
    tracker = EventTracker(w)
    # 64 simultaneous deaths, cap keeps only the newest 10
    w.energy[w.alive] = 0.0
    step.phase_deaths(w)
    tracker.update(w, settings)
    assert len(tracker.flashes) == settings.flash_max
    # no further events: every flash ages out within flash_duration frames
    for _ in range(settings.flash_duration + 1):
        tracker.update(w, settings)
    assert not tracker.flashes


def test_sparkline_samples_track_world(ready) -> None:
    from primordia.render.draw import SparkTracker

    settings = _settings(sparkline_samples=100)
    w = World(Config(max_creatures=32, initial_creatures=32), seed=5)
    sparks = SparkTracker(settings)
    for _ in range(10):
        advance(w)
        sparks.sample(w)
    assert len(sparks.alive) == 10
    assert sparks.alive[-1] == w.alive_count
    assert sparks.energy[-1] == pytest.approx(float(w.energy[w.alive].mean()))
    for _ in range(200):  # ring buffer never exceeds its maxlen
        sparks.sample(w)
    assert len(sparks.alive) == settings.sparkline_samples


def test_render_flashes_writes_alpha(ready) -> None:
    from primordia.render.camera import Camera
    from primordia.render.draw import Flash, render_flashes
    from primordia.render.settings import Settings

    settings = Settings()
    # surface matches the camera's viewport so world_to_screen lands inside
    surf = pygame.Surface(
        (settings.viewport_width, settings.viewport_height), pygame.SRCALPHA
    )
    surf.fill((0, 0, 0, 0))
    w = World(Config(), seed=5)
    camera = Camera(
        settings,
        w.config.width,
        w.config.height,
        float(w.cell_w),
        float(w.cell_h),
        gw=w.food.shape[1],
        gh=w.food.shape[0],
    )
    cx, cy = w.config.width / 2.0, w.config.height / 2.0  # on-screen centre
    flashes = [Flash(x=cx, y=cy, size=1.0, color=(10, 200, 30), age=0)]
    render_flashes(surf, flashes, camera, settings)
    sx, sy = camera.world_to_screen(
        np.array([[cx, cy]], dtype=np.float32)
    )
    arr = np.frombuffer(pygame.image.tobytes(surf, "RGBA"), dtype=np.uint8)
    arr = arr.reshape(settings.viewport_height, settings.viewport_width, 4)
    center = arr[int(sy[0]), int(sx[0])]
    assert center[3] > 0  # alpha written where the flash sits
    assert tuple(center[:3]) == (10, 200, 30)
