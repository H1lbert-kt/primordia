"""World rendering: direct writes into the world surface's pixel memory.

Hot paths avoid Python loops over creatures and avoid slow strided writes:
the food grid is gathered through a camera-cached flat index and expanded to
packed ``uint32`` pixels with one ``np.take``, then living creatures are
scattered into the same packed buffer (one vectorized pass per disc radius).
``pygame.surfarray.pixels2d`` of an owned surface is contiguous, so every
large copy here is contiguous. Only the selection overlay (rings, smell
circle, seven rays) uses ``pygame.draw``, and only for the selected creature.

Body-trait layout (see ``Config``): ``brain_params + 0`` max_speed,
``+1`` size, ``+2`` vision, ``+3`` diet — the pixel radius comes from size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from .camera import Camera
from .settings import Settings

# Per-channel bit shifts of the R, G and B bytes in the surface pixel format.
PixelFormat = tuple[int, int, int]

# Per-radius disc offsets (ox, oy with ox^2 + oy^2 <= r^2), built on demand.
_DISC_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _disc(radius: int) -> tuple[np.ndarray, np.ndarray]:
    if radius not in _DISC_CACHE:
        oy = np.arange(-radius, radius + 1, dtype=np.int32)
        dy, dx = np.meshgrid(oy, oy, indexing="ij")
        keep = dx * dx + dy * dy <= radius * radius
        _DISC_CACHE[radius] = (dx[keep], dy[keep])
    return _DISC_CACHE[radius]


def pixel_format(surface: pygame.Surface) -> PixelFormat:
    """Probe where the R, G and B bytes sit in this surface's pixels."""
    masks = (
        surface.map_rgb(255, 0, 0),
        surface.map_rgb(0, 255, 0),
        surface.map_rgb(0, 0, 255),
    )
    shifts: list[int] = []
    for m in masks:
        if m == 0:
            raise ValueError("surface has no RGB channel mask")
        shift = (m & -m).bit_length() - 1  # lowest set bit
        if m != (0xFF << shift):
            raise ValueError(f"unexpected RGB pixel layout: {hex(m)}")
        shifts.append(shift)
    return (shifts[0], shifts[1], shifts[2])


def rgb_to_pixel(rgb: np.ndarray, shifts: PixelFormat) -> np.ndarray:
    """Pack an (..., 3) uint8 RGB array into uint32 surface pixels."""
    out = np.zeros(rgb.shape[:-1], dtype=np.uint32)
    for c, shift in enumerate(shifts):
        out |= rgb[..., c].astype(np.uint32) << shift
    return out


@dataclass
class FrameBuffers:
    """Pre-allocated scratch for the food pass (image-major layout)."""

    values: np.ndarray  # (viewport_w, viewport_h) float32
    indices: np.ndarray  # (viewport_w, viewport_h) intp — np.take casts
    # other integer dtypes to intp internally (a hidden multi-MB copy), so
    # both the camera cache and this buffer stay in the native index dtype.
    packed: np.ndarray  # (viewport_w, viewport_h) uint32 contiguous pixels

    @classmethod
    def create(cls, viewport_width: int, viewport_height: int) -> FrameBuffers:
        shape = (viewport_width, viewport_height)
        return cls(
            values=np.zeros(shape, dtype=np.float32),
            indices=np.zeros(shape, dtype=np.intp),
            packed=np.zeros(shape, dtype=np.uint32),
        )


def make_food_lut(settings: Settings) -> np.ndarray:
    """256-entry RGB lookup: food energy (0..capacity) -> color."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    ramp = np.arange(256, dtype=np.float64)
    stops = np.array([0.0, 140.0, 255.0])
    for c in range(3):
        values = (settings.food_low[c], settings.food_mid[c], settings.food_high[c])
        lut[:, c] = np.interp(ramp, stops, values).astype(np.uint8)
    return lut


def creature_colors(world, settings: Settings) -> np.ndarray:
    """RGB per living creature: hue from diet, brightness from energy."""
    slots = np.flatnonzero(world.alive)
    if slots.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    diet = world.genome[slots, world.config.brain_params + 3]
    energy01 = np.clip(
        world.energy[slots] / (2.0 * world.config.initial_energy), 0.0, 1.0
    )
    brightness = 0.35 + 0.65 * energy01
    herb = np.asarray(settings.creature_herbivore, dtype=np.float64)
    pred = np.asarray(settings.creature_predator, dtype=np.float64)
    col = herb[None, :] * (1.0 - diet)[:, None] + pred[None, :] * diet[:, None]
    return (col * brightness[:, None]).astype(np.uint8)


def draw_food(
    pixels: np.ndarray,
    world,
    camera: Camera,
    lut: np.ndarray,
    buffers: FrameBuffers,
) -> None:
    """Write the food grid into ``pixels`` ((w, h) uint32, packed RGB).

    The gather/quantize/expand path runs on contiguous buffers; the final
    copy into ``pixels`` (the surface's transposed pixel view) is one
    contiguous-to-strided assignment instead of a strided gather.
    """
    np.take(world.food.ravel(), camera.flat_indices(), out=buffers.values)
    np.multiply(
        buffers.values, 255.0 / float(world.config.food_capacity), out=buffers.values
    )
    np.copyto(buffers.indices, buffers.values, casting="unsafe")
    # Invariant: 0 <= food <= capacity, so indices stay in [0, 255]; mode
    # "clip" only guards against floating-point overshoot at the edges.
    np.take(lut, buffers.indices, axis=0, out=buffers.packed, mode="clip")
    pixels[:] = buffers.packed


def stamp_creatures(
    pixels: np.ndarray,
    world,
    camera: Camera,
    settings: Settings,
    shifts: PixelFormat,
) -> None:
    """Scatter every living creature onto ``pixels`` ((w, h) uint32)."""
    vh = pixels.shape[1]
    vw = pixels.shape[0]
    slots = np.flatnonzero(world.alive)
    if slots.size == 0:
        return
    sx, sy = camera.world_to_screen(world.pos[slots])
    ix = np.rint(sx).astype(np.int32)
    iy = np.rint(sy).astype(np.int32)
    size_trait = world.config.brain_params + 1
    radius = np.rint(
        world.genome[slots, size_trait] * camera.zoom * settings.size_scale
    ).astype(np.int32)
    np.clip(radius, 1, settings.max_radius_px, out=radius)
    colors = rgb_to_pixel(creature_colors(world, settings), shifts)

    order = np.argsort(radius, kind="stable")
    radius = radius[order]
    ix = ix[order]
    iy = iy[order]
    colors = colors[order]
    unique, starts = np.unique(radius, return_index=True)
    ends = np.append(starts[1:], radius.size)

    for k, r in enumerate(unique):
        a, b = int(starts[k]), int(ends[k])
        dxs, dys = _disc(int(r))
        xs = (ix[a:b, None] + dxs[None, :]).ravel()
        ys = (iy[a:b, None] + dys[None, :]).ravel()
        cols = np.repeat(colors[a:b], dxs.size)
        ok = (xs >= 0) & (xs < vw) & (ys >= 0) & (ys < vh)
        if ok.all():
            pixels[xs, ys] = cols
        else:
            pixels[xs[ok], ys[ok]] = cols[ok]


def pick_creature(
    world, camera: Camera, mouse_x: float, mouse_y: float, settings: Settings
) -> int:
    """Slot under the cursor (shortest screen distance), or -1."""
    slots = np.flatnonzero(world.alive)
    if slots.size == 0:
        return -1
    sx, sy = camera.world_to_screen(world.pos[slots])
    d2 = (sx - mouse_x) ** 2 + (sy - mouse_y) ** 2
    j = int(np.argmin(d2))
    if d2[j] <= settings.pick_radius_px**2:
        return int(slots[j])
    return -1


def draw_overlay(
    screen: pygame.Surface,
    world,
    camera: Camera,
    slot: int,
    settings: Settings,
) -> None:
    """Rings, smell radius and ray fan for the one selected creature."""
    if slot < 0 or not world.alive[slot]:
        return
    pos = world.pos[slot : slot + 1]
    sx, sy = camera.world_to_screen(pos)
    cx, cy = int(sx[0]), int(sy[0])
    vw, vh = camera.viewport

    traits = world.config.brain_params
    cell = 0.5 * (float(world.cell_w) + float(world.cell_h))
    smell = (world.config.smell_radius_cells + 0.5) * cell * camera.zoom
    pygame.draw.circle(screen, settings.smell_ring, (cx, cy), int(smell), 1)

    angle = float(world.angle[slot])
    c = np.cos(angle)
    s = np.sin(angle)
    unit = world.ray_unit
    dirx = c * unit[:, 0] - s * unit[:, 1]
    diry = s * unit[:, 0] + c * unit[:, 1]
    vision = float(world.genome[slot, traits + 2])
    ends = np.empty((unit.shape[0], 2), dtype=np.float64)
    ends[:, 0] = (pos[0, 0] + dirx * vision) % world.config.width
    ends[:, 1] = (pos[0, 1] + diry * vision) % world.config.height
    ex, ey = camera.world_to_screen(ends)

    food_ray = np.asarray(settings.ray_food)
    creature_ray = np.asarray(settings.ray_creature)
    ray_values = world.sensor_buf[slot]
    n_rays = world.config.n_rays
    for r in range(n_rays):
        # Skip rays whose torus-wrapped endpoint jumps across the viewport.
        if abs(float(ex[r]) - cx) > vw or abs(float(ey[r]) - cy) > vh:
            continue
        f = float(ray_values[r])
        cr = float(ray_values[n_rays + r])
        t = float(np.clip(cr / (f + cr + 1e-6), 0.0, 1.0))
        color = tuple(int(v) for v in (food_ray * (1.0 - t) + creature_ray * t))
        pygame.draw.line(screen, color, (cx, cy), (int(ex[r]), int(ey[r])), 1)

    radius = max(int(np.rint(float(world.genome[slot, traits + 1]) * camera.zoom)), 3)
    pygame.draw.circle(screen, settings.selection, (cx, cy), radius + 2, 1)
