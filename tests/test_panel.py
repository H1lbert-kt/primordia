"""Panel guard: renders without a selection and with one, and fits the window.

The panel draws onto a fixed 720-px surface; content taller than the window
silently clips (the key help used to disappear), so the layout height is an
invariant, not a style preference.
"""

from __future__ import annotations

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from primordia.config import Config  # noqa: E402
from primordia.step import advance  # noqa: E402
from primordia.world import World  # noqa: E402


@pytest.fixture(scope="module")
def ready_world() -> World:
    """Minimal headless world + a few ticks so sensors carry real values."""
    pygame.font.init()  # plain surfaces + fonts, no display needed
    w = World(Config(), seed=42)
    for _ in range(20):
        advance(w)
    return w


def _render_height(world: World, slot: int, settings) -> int:
    """Render the panel and report the lowest y any text reached."""
    from primordia.render import panel as P

    original = P._text
    max_y = 0

    def spy(dst, size, msg, x, y, color):
        nonlocal max_y
        max_y = max(max_y, y + size)
        return original(dst, size, msg, x, y, color)

    P._text = spy
    try:
        surf = pygame.Surface((settings.panel_width, settings.window_height))
        hud = P.Hud(fps=60.0, speed=1, paused=False, follow=False, seed=42)
        P.render_panel(surf, world, slot, hud, settings)
    finally:
        P._text = original
    return max_y


def test_panel_fits_window(ready_world: World) -> None:
    from primordia.render.settings import Settings

    settings = Settings()
    alive = np.flatnonzero(ready_world.alive)
    assert alive.size > 0
    for slot in (int(alive[0]), -1):
        height = _render_height(ready_world, slot, settings)
        assert height <= settings.window_height, (
            f"panel content overflows: y={height} > {settings.window_height}"
        )


def test_panel_shows_all_sensor_groups(ready_world: World) -> None:
    """The selected panel must display every group from input_groups."""
    from primordia.render import panel as P
    from primordia.render.settings import Settings
    from primordia.sensors import input_groups

    messages: list[str] = []
    original = P._text

    def spy(dst, size, msg, x, y, color):
        messages.append(msg)
        return original(dst, size, msg, x, y, color)

    slot = int(np.flatnonzero(ready_world.alive)[0])
    settings = Settings()
    P._text = spy
    try:
        surf = pygame.Surface((settings.panel_width, settings.window_height))
        hud = P.Hud(fps=60.0, speed=1, paused=False, follow=False, seed=42)
        P.render_panel(surf, ready_world, slot, hud, settings)
    finally:
        P._text = original

    for label, _group in input_groups(ready_world.config):
        assert label in messages, f"sensor group {label!r} not shown in panel"
