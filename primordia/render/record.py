"""Headless frame capture: world state -> PNG frames -> demo GIF.

Recording renders the world *read-only* into offscreen surfaces (SDL dummy
driver, no window) and writes numbered PNG frames to disk; the GIF is
assembled afterwards with Pillow, outside the simulation loop. The viewer
pipeline (food pass, creature scatter, side panel) is reused unchanged, so
a recorded frame looks exactly like the app. Rendering draws no random
numbers and never writes world state: a recorded run is bit-identical to
the same run without recording (``tests/test_record.py``).

Frames land in ``<out_dir>/frames/frame_%05d.png``; ``save_gif`` assembles
them into ``<out_dir>/demo.gif`` downscaled to at most ``max_width`` px
wide with a 256-color palette, to stay README-sized (< 4 MB).
"""

from __future__ import annotations

import os
from pathlib import Path

import pygame
from PIL import Image

from ..cycles import light_level
from .camera import Camera
from .draw import (
    EventTracker,
    FrameBuffers,
    SparkTracker,
    draw_world,
    make_food_lut,
    make_terrain_lut,
    pixel_format,
    render_flashes,
    stamp_creatures,
)
from .panel import Hud, render_panel
from .settings import Settings


def assemble_gif(
    frames: list[Path],
    gif_path: str | Path,
    duration_ms: int = 100,
    max_width: int = 960,
    colors: int = 256,
) -> Path:
    """Assemble PNG frames into an animated GIF (Pillow).

    Frames wider than ``max_width`` are downscaled and quantized against a
    *single* palette built from the first frame. One palette per frame (the
    naive ``convert("P")``) makes every pixel look changed to Pillow, which
    then cannot crop frames to their moved region; a patchy world stops
    compressing (40 frames: 9.1 MB -> 1.6 MB). Dithering is off so equal
    pixels map to equal palette indices, which the frame differ relies on.
    Pillow merges runs of pixel-identical frames by extending their duration,
    which is correct GIF behavior (identical consecutive frames are
    redundant).
    """
    if not frames:
        raise RuntimeError("no frames to assemble")
    gif = Path(gif_path)
    rgb = []
    for frame in frames:
        img = Image.open(frame)
        if img.width > max_width:
            height = round(img.height * max_width / img.width)
            img = img.resize((max_width, height), Image.Resampling.LANCZOS)
        rgb.append(img.convert("RGB"))
    palette = rgb[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)
    images = [palette] + [
        im.quantize(palette=palette, dither=Image.Dither.NONE) for im in rgb[1:]
    ]
    gif.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        gif,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return gif


class WorldRecorder:
    """Capture the world into PNG frames and assemble them into a GIF."""

    def __init__(
        self,
        world,
        out_dir: str | Path,
        seed: int,
        speed: int = 1,
        settings: Settings | None = None,
    ) -> None:
        """Allocate the offscreen surfaces for ``world`` under ``out_dir``.

        ``speed`` is shown in the side panel as ticks per captured frame.
        Surfaces and the camera are created once; ``capture`` re-renders
        them from the current state without touching it.
        """
        self.settings = settings or Settings()
        self.seed = int(seed)
        self.speed = int(speed)
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._frame_index = 0

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no window
        pygame.init()

        s = self.settings
        vw, vh = s.viewport_width, s.viewport_height
        self._screen = pygame.Surface((s.window_width, s.window_height))
        self._world_surface = pygame.Surface((vw, vh))
        self._panel_surface = pygame.Surface((s.panel_width, vh))
        self._shifts = pixel_format(self._world_surface)
        self._food_lut = make_food_lut(s)
        self._terrain_lut = make_terrain_lut(s)
        self._buffers = FrameBuffers.create(vw, vh)
        self._event_tracker = EventTracker(world)
        self._sparks = SparkTracker(s)
        self._overlay = pygame.Surface((vw, vh), pygame.SRCALPHA)
        cfg = world.config
        self._camera = Camera(
            s,
            cfg.width,
            cfg.height,
            float(world.cell_w),
            float(world.cell_h),
            gw=world.food.shape[1],
            gh=world.food.shape[0],
        )

    @property
    def frames(self) -> list[Path]:
        """Captured frame paths, in order."""
        return sorted(self.frames_dir.glob("frame_*.png"))

    def capture(self, world) -> Path:
        """Render the current world state; write the next PNG frame."""
        s = self.settings
        pixels = pygame.surfarray.pixels2d(self._world_surface)
        light = float(light_level(world.config, world.tick))
        draw_world(
            pixels, world, self._camera,
            self._food_lut, self._terrain_lut, light, self._shifts, self._buffers,
        )
        stamp_creatures(pixels, world, self._camera, s, self._shifts, light)
        del pixels
        # Read-only dynamics: flash diff + sparkline sample (no RNG, so the
        # recorded run stays bit-identical to the headless one).
        self._event_tracker.update(world, s)
        self._sparks.sample(world)
        self._overlay.fill((0, 0, 0, 0))
        render_flashes(self._overlay, self._event_tracker.flashes, self._camera, s)
        self._world_surface.blit(self._overlay, (0, 0))
        render_panel(
            self._panel_surface,
            world,
            -1,
            Hud(
                fps=0.0,
                speed=self.speed,
                paused=False,
                follow=False,
                seed=self.seed,
                alive_hist=self._sparks.alive,
                energy_hist=self._sparks.energy,
            ),
            s,
        )
        self._screen.blit(self._world_surface, (0, 0))
        self._screen.blit(self._panel_surface, (s.viewport_width, 0))
        path = self.frames_dir / f"frame_{self._frame_index:05d}.png"
        pygame.image.save(self._screen, str(path))
        self._frame_index += 1
        return path

    def save_gif(
        self,
        path: str | Path | None = None,
        duration_ms: int = 100,
        max_width: int = 960,
        colors: int = 256,
    ) -> Path:
        """Assemble captured frames into an animated GIF (Pillow)."""
        gif = Path(path) if path is not None else self.out_dir / "demo.gif"
        return assemble_gif(
            self.frames, gif, duration_ms=duration_ms,
            max_width=max_width, colors=colors,
        )
