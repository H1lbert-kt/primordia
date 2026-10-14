"""Pygame application: window, input, per-frame loop and selection overlays.

The viewer only *reads* the world and draws no random numbers, so a windowed
run advances the simulation exactly like ``--headless`` with the same seed.
One frame = clock tick, ``speed_steps`` calls to ``advance`` (0 while paused),
the world surface (food + creature scatter written directly into its packed
pixel buffer), an alpha overlay (selected trail + birth/death flashes), one
selection overlay, the side panel, and two blits.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pygame

from ..cycles import light_level
from ..ecology import cluster_living
from ..observer import GroupObserver, GroupSnapshot
from ..step import advance
from ..world import World
from .camera import Camera
from .draw import (
    EventTracker,
    FrameBuffers,
    SparkTracker,
    draw_overlay,
    draw_trail,
    draw_world,
    hue_lut,
    make_food_lut,
    make_terrain_lut,
    make_wood_lut,
    pick_creature,
    pixel_format,
    render_flashes,
    render_raids,
    render_transfers,
    stamp_creatures,
    stamp_deposit_bars,
    stamp_structures,
)
from .panel import Hud, clear_font_cache, draw_event_log, render_panel
from .settings import Settings


def _save_screenshot(screen: pygame.Surface, path: str) -> None:
    pygame.image.save(screen, path)


def _rebuild_group_overlay(
    snapshot: GroupSnapshot,
    group_rgb: np.ndarray,
    group_alpha: np.ndarray,
    settings: Settings,
) -> None:
    """Fill the country fill/alpha grids from the latest observer sample.

    Each owned cell gets its group's circular-mean hue (so the overlay
    colour is the lineage colour the creatures already wear as rings);
    frontier cells — torus-adjacent to a different living group — get the
    brighter border alpha. Observer output only; the world is untouched.
    """
    owner = snapshot.owner
    occupied = owner >= 0
    group_rgb.fill(0.0)
    group_alpha.fill(0.0)
    if not occupied.any():
        return
    lut = hue_lut()
    uids = np.unique(owner[occupied])
    idx = np.zeros(int(uids.max()) + 1, dtype=np.int32)
    for u in uids:
        idx[int(u)] = int(snapshot.hues.get(int(u), 0.0) * 255.0) & 255
    group_rgb[occupied] = lut[idx[owner[occupied]]]
    group_alpha[occupied, 0] = settings.group_overlay_alpha
    diff = np.zeros(owner.shape, dtype=bool)
    for axis in (0, 1):
        for shift in (1, -1):
            other = np.roll(owner, shift, axis=axis)
            diff |= occupied & (other >= 0) & (other != owner)
    group_alpha[occupied & diff, 0] = settings.group_border_alpha


def _rebuild_territory_overlay(
    world,
    mark_rgb: np.ndarray,
    mark_alpha: np.ndarray,
    settings: Settings,
) -> None:
    """Fill the territorial wash grids from this tick's mark scratch.

    Each marked cell is tinted with its owner's tag hue (the same circular
    colour creatures already wear as rings, so the ring on the ground and
    the ring on the sprite match); a mark whose owner has since died falls
    back to a dim neutral gray (the deposit persists, the creature does
    not). ``mark``/``mark_owner`` are tick-fresh scratch, so this must run
    every frame while the overlay is on — unlike the 30-frame-cached group
    observer. Viewer-only; the world is untouched.
    """
    mark = world.mark
    owner = world.mark_owner
    occupied = mark > 0.5
    mark_rgb.fill(0.0)
    mark_alpha.fill(0.0)
    if not occupied.any():
        return
    lut = hue_lut()
    alive = world.alive
    ids = world.creature_id[alive]
    hues = world.genome[alive, world.config.brain_params + 4]
    hue_of = {int(i): float(h) for i, h in zip(ids, hues) if i >= 0}
    gray = np.array([0.45, 0.45, 0.48], dtype=np.float32)
    for oid in np.unique(owner[occupied]):
        cells = owner == oid
        hue = hue_of.get(int(oid))
        mark_rgb[cells] = (
            lut[(int(hue * 255.0) & 255)] if hue is not None else gray
        )
    mark_alpha[occupied, 0] = settings.territory_overlay_alpha


def run_app(
    world: World,
    seed: int,
    max_ticks: int | None = None,
    screenshot_path: str | None = None,
    select_slot: int = -1,
    settings: Settings | None = None,
) -> dict:
    """Open the viewer and run until the window closes (or ``max_ticks``).

    Returns a summary dict (frames, ticks, fps) for the CLI and tests.
    """
    settings = settings or Settings()
    pygame.init()
    screen = pygame.display.set_mode(
        (settings.window_width, settings.window_height)
    )
    pygame.display.set_caption("Primordia")
    clock = pygame.time.Clock()
    cfg = world.config

    vw, vh = settings.viewport_width, settings.viewport_height
    # Owned (never display-) surfaces: a display subsurface has a padded
    # pitch, which makes pixels3d/pixels2d access strided and slower.
    world_surface = pygame.Surface((vw, vh))
    panel_surface = pygame.Surface((settings.panel_width, vh))
    shifts = pixel_format(world_surface)
    food_lut = make_food_lut(settings)
    terrain_lut = make_terrain_lut(settings)
    wood_lut = make_wood_lut(settings)
    buffers = FrameBuffers.create(vw, vh)
    camera = Camera(
        settings,
        cfg.width,
        cfg.height,
        float(world.cell_w),
        float(world.cell_h),
        gw=world.food.shape[1],
        gh=world.food.shape[0],
    )

    selected = select_slot if 0 <= select_slot < cfg.max_creatures else -1
    sel_age = int(world.age[selected]) if selected >= 0 else -1
    # Stage 12: species clustering is O(alive * K) pure-Python, so it is
    # recomputed every CLUSTER_REFRESH_FRAMES frames and reused in between
    # (both for the HUD K and the selected creature's cluster label).
    # Stage 23: the same timer refreshes the hue-group observer (cheap 1-D
    # clustering + per-cell majority) behind the optional country overlay.
    CLUSTER_REFRESH_FRAMES = 30
    cluster_k: int | None = None
    cluster_label_of_slot: np.ndarray | None = None
    observer = GroupObserver(cfg)
    show_groups = settings.show_groups
    show_territory = settings.show_territory
    snapshot: GroupSnapshot | None = None
    gh, gw = world.food.shape
    group_rgb = np.zeros((gh, gw, 3), dtype=np.float32)
    group_alpha = np.zeros((gh, gw, 1), dtype=np.float32)
    mark_rgb = np.zeros((gh, gw, 3), dtype=np.float32)
    mark_alpha = np.zeros((gh, gw, 1), dtype=np.float32)
    paused = False
    speed_index = 0
    follow = selected >= 0  # a preset selection starts tracking its creature
    drag_pos: tuple[int, int] | None = None
    drag_moved = 0
    start_tick = world.tick
    frames = 0
    fps_samples: list[float] = []
    event_tracker = EventTracker(world)
    sparks = SparkTracker(settings)
    trail: deque = deque(maxlen=settings.trail_points)
    trail_slot = -1  # trail resets whenever the selection changes
    overlay = pygame.Surface((vw, vh), pygame.SRCALPHA)
    running = True
    t0 = time.perf_counter()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_PERIOD and paused:
                    advance(world)  # single manual tick while paused
                elif pygame.K_1 <= event.key <= pygame.K_4:
                    speed_index = min(
                        event.key - pygame.K_1, len(settings.speed_steps) - 1
                    )
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed_index = min(speed_index + 1, len(settings.speed_steps) - 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_index = max(speed_index - 1, 0)
                elif event.key == pygame.K_f:
                    if selected >= 0:
                        follow = not follow
                elif event.key == pygame.K_g:
                    show_groups = not show_groups
                elif event.key == pygame.K_t:
                    show_territory = not show_territory
                elif event.key == pygame.K_ESCAPE:
                    selected, follow = -1, False
                elif event.key == pygame.K_F12 and screenshot_path:
                    _save_screenshot(screen, screenshot_path)
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx < vw and event.y != 0:
                    camera.zoom_at(float(mx), float(my), 1.1**event.y)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] < vw:
                    drag_pos = (event.pos[0], event.pos[1])
                    drag_moved = 0
            elif event.type == pygame.MOUSEMOTION and drag_pos is not None:
                if event.buttons[0]:
                    dx = event.pos[0] - drag_pos[0]
                    dy = event.pos[1] - drag_pos[1]
                    drag_moved += abs(dx) + abs(dy)
                    if drag_moved >= settings.pan_drag_px:
                        camera.pan(float(dx), float(dy))
                        drag_pos = (event.pos[0], event.pos[1])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if drag_pos is not None:
                    if (
                        drag_moved < settings.pan_drag_px
                        and event.pos[0] < vw
                        and running
                    ):
                        selected = pick_creature(
                            world,
                            camera,
                            float(event.pos[0]),
                            float(event.pos[1]),
                            settings,
                        )
                        follow = False
                        if selected >= 0:
                            sel_age = int(world.age[selected])
                drag_pos = None

        dt = clock.tick(settings.target_fps)
        steps = 0 if paused else settings.speed_steps[speed_index]
        for _ in range(steps):
            advance(world)

        # Drop a selection whose slot died and was recycled (age resets to 0).
        if selected >= 0:
            if not world.alive[selected] or int(world.age[selected]) < sel_age:
                selected, follow, sel_age = -1, False, -1
            else:
                sel_age = int(world.age[selected])
        if follow and selected < 0:
            follow = False
        if follow and selected >= 0:
            camera.follow(world.pos[selected])

        # render-side dynamics (read-only w.r.t. the world, no RNG)
        event_tracker.update(world, settings)
        sparks.sample(world)
        if selected != trail_slot:
            trail.clear()
            trail_slot = selected
        if selected >= 0:
            trail.append(world.pos[selected].copy())

        pixels = pygame.surfarray.pixels2d(world_surface)
        light = float(light_level(cfg, world.tick))
        if frames % CLUSTER_REFRESH_FRAMES == 0:
            snapshot = observer.sample(world)
        if show_groups and snapshot is not None:
            _rebuild_group_overlay(snapshot, group_rgb, group_alpha, settings)
            g_rgb: np.ndarray | None = group_rgb
            g_alpha: np.ndarray | None = group_alpha
        else:
            g_rgb = g_alpha = None
        if show_territory:
            _rebuild_territory_overlay(world, mark_rgb, mark_alpha, settings)
            m_rgb: np.ndarray | None = mark_rgb
            m_alpha: np.ndarray | None = mark_alpha
        else:
            m_rgb = m_alpha = None
        draw_world(pixels, world, camera, food_lut, terrain_lut, wood_lut, light, shifts, buffers, settings.meat_color, settings.signal_color, g_rgb, g_alpha, m_rgb, m_alpha)
        stamp_structures(pixels, world, camera, settings, shifts, light)
        stamp_creatures(pixels, world, camera, settings, shifts, light)
        del pixels
        overlay.fill((0, 0, 0, 0))
        if len(trail) >= 2:
            draw_trail(
                overlay,
                np.asarray(trail, dtype=np.float32),
                camera,
                settings,
            )
        render_flashes(overlay, event_tracker.flashes, camera, settings)
        render_transfers(overlay, event_tracker.transfers, camera, settings)
        render_raids(overlay, event_tracker.raid_lines, camera, settings)
        stamp_deposit_bars(overlay, world, camera, settings)
        world_surface.blit(overlay, (0, 0))
        draw_overlay(world_surface, world, camera, selected, settings)
        if show_groups:
            draw_event_log(world_surface, observer.events, settings)
        if frames % CLUSTER_REFRESH_FRAMES == 0:
            slots, labels = cluster_living(world, cfg.species_threshold)
            cluster_k = int(np.unique(labels).size) if slots.size else 0
            lut = np.full(cfg.max_creatures, -1, dtype=np.int32)
            lut[slots] = labels
            cluster_label_of_slot = lut
        render_panel(
            panel_surface,
            world,
            selected,
            Hud(
                fps=clock.get_fps(),
                speed=steps,
                paused=paused,
                follow=follow,
                seed=seed,
                alive_hist=sparks.alive,
                energy_hist=sparks.energy,
                species_k=cluster_k,
                cluster_label_of_slot=cluster_label_of_slot,
                groups_k=snapshot.n_groups if snapshot is not None else None,
                border_cells=(
                    snapshot.border_cells if snapshot is not None else None
                ),
                group_uid_of_slot=(
                    snapshot.uid_of_slot if snapshot is not None else None
                ),
                group_info=(
                    {
                        u: (snapshot.pops[u], snapshot.areas.get(u, 0), snapshot.ages[u])
                        for u in snapshot.pops
                    }
                    if snapshot is not None
                    else None
                ),
            ),
            settings,
        )
        screen.blit(world_surface, (0, 0))
        screen.blit(panel_surface, (vw, 0))
        pygame.display.flip()

        frames += 1
        if frames > 60 and dt > 0:
            fps_samples.append(1000.0 / dt)
        if max_ticks is not None and (world.tick - start_tick) >= max_ticks:
            running = False

    elapsed = time.perf_counter() - t0
    if screenshot_path and frames:
        _save_screenshot(screen, screenshot_path)
    pygame.quit()
    clear_font_cache()  # pygame.quit() invalidates the cached Font objects
    return {
        "frames": frames,
        "ticks": world.tick - start_tick,
        "mean_fps": frames / elapsed if elapsed > 0 else 0.0,
        "min_fps": min(fps_samples) if fps_samples else 0.0,
        "selected": selected,
    }
