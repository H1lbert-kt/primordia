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


def test_panel_shows_cluster_when_cached(ready_world: World) -> None:
    """Selected panel shows 'cluster #k' when the app cached labels.

    Without the cache (None, as in the recorder) it shows 'cluster n/a' —
    neither path may crash or leave the old vestigial 'species 0'.
    """
    from primordia.render import panel as P
    from primordia.render.settings import Settings

    settings = Settings()
    slot = int(np.flatnonzero(ready_world.alive)[0])

    def collect(hud) -> list[str]:
        messages: list[str] = []
        original = P._text

        def spy(dst, size, msg, x, y, color):
            messages.append(msg)
            return original(dst, size, msg, x, y, color)

        P._text = spy
        try:
            surf = pygame.Surface((settings.panel_width, settings.window_height))
            P.render_panel(surf, ready_world, slot, hud, settings)
        finally:
            P._text = original
        return messages

    lut = np.full(ready_world.config.max_creatures, -1, dtype=np.int32)
    lut[slot] = 3
    with_cache = collect(P.Hud(
        fps=60.0, speed=1, paused=False, follow=False, seed=42,
        species_k=5, cluster_label_of_slot=lut,
    ))
    without = collect(P.Hud(
        fps=60.0, speed=1, paused=False, follow=False, seed=42,
    ))

    assert any("cluster #3" in m for m in with_cache), with_cache
    assert any("cluster n/a" in m for m in without), without
    # The inspection line must not keep the vestigial "species 0" id label;
    # the WORLD HUD's "species K" gauge line is a different, intended string.
    assert not any("species 0" in m for m in with_cache + without)


def test_panel_shows_all_gates_and_w2_layout(ready_world: World) -> None:
    """Stage 17: the outputs list shows all 8 outputs and W2 is 8 wide."""
    from primordia.render import panel as P
    from primordia.render.settings import Settings

    settings = Settings()
    slot = int(np.flatnonzero(ready_world.alive)[0])
    messages: list[str] = []
    original = P._text
    # stage-17 double-sigmoid fix: gates arrive decoded from apply_actions,
    # so a forced 0.5 gate must print +0.50 (the old second sigmoid printed
    # +0.62) — set the scratch before rendering, restore after
    saved_gates = ready_world.actions[slot, 2:8].copy()
    ready_world.actions[slot, 2:8] = 0.5

    def spy(dst, size, msg, x, y, color):
        messages.append(msg)
        return original(dst, size, msg, x, y, color)

    P._text = spy
    try:
        surf = pygame.Surface((settings.panel_width, settings.window_height))
        hud = P.Hud(fps=60.0, speed=1, paused=False, follow=False, seed=42)
        P.render_panel(surf, ready_world, slot, hud, settings)
    finally:
        P._text = original
        ready_world.actions[slot, 2:8] = saved_gates

    for label in ("gate", "emit", "don", "gather", "fire", "house"):
        assert label in messages, f"output {label!r} missing from the panel"
    assert "+0.50" in messages, "gate must be shown decoded, not sigmoid(x) again"
    assert any("W2|b2 (8x10)" in m for m in messages), messages
    assert any("W_rec (9x9)" in m for m in messages), messages
    assert any("wood " in m and "energy" in m for m in messages), "inventory line"
    assert any("fire " in m and "house " in m for m in messages), "structure counts"
