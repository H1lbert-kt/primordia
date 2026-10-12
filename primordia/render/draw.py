"""World rendering: direct writes into the world surface's pixel memory.

Hot paths avoid Python loops over creatures and avoid slow strided writes:
the world pass (terrain + food, stage 10) blends both grids' LUT colors per
*cell* (200x200 once per frame), dims the result by the ambient light, and
expands to packed ``uint32`` pixels with one gather through the
camera-cached flat index; living creatures are then scattered into the same
packed buffer (one vectorized pass per ``(radius, heading)`` sprite pose,
also dimmed by the light). ``pygame.surfarray.pixels2d`` of an owned
surface is contiguous, so every large copy here is contiguous. Only the
selection overlay (rings, smell circle, seven rays), the selected creature's
trail and the event flashes use ``pygame.draw`` (those stay at full
brightness so the UI reads against a night-dark world).

Body-trait layout (see ``Config``): ``brain_params + 0`` max_speed,
``+1`` size, ``+2`` vision, ``+3`` diet — the pixel radius comes from size;
``+4`` tag_hue — the lineage ring color (stage 15).

Render-side state trackers (``EventTracker``, ``SparkTracker``) diff
successive frames into flashes/sparklines; they only *read* the world and
draw no random numbers, so recording stays bit-identical to headless runs.
"""

from __future__ import annotations

import colorsys
from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame

from .camera import Camera
from .settings import Settings

# Per-channel bit shifts of the R, G and B bytes in the surface pixel format.
PixelFormat = tuple[int, int, int]

# Humanoid sprite masks per (radius, heading bucket, bucket count).
_SPRITE_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _humanoid_offsets(
    radius: int, bucket: int, buckets: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel offsets of a pre-rendered humanoid pose (cached per key).

    The figure (head + torso) is drawn upright facing north, rotated so it
    faces the bucket's heading, and reduced to an alpha mask — the scatter
    paints the per-creature color onto these offsets. The rotation angle is
    ``heading + 90``: measured on pygame-ce, ``rotate(+90)`` turns the
    north-facing sprite east. Canvas side ~2.9r keeps the diagonal poses
    unclipped.
    """
    key = (radius, bucket, buckets)
    if key not in _SPRITE_CACHE:
        side = int(np.ceil(2.9 * radius)) + 3
        if side % 2 == 0:
            side += 1
        surf = pygame.Surface((side, side), pygame.SRCALPHA)
        c = side // 2
        head_r = max(1, int(round(radius * 0.38)))
        head_cy = c - int(round(radius * 0.55))
        pygame.draw.circle(surf, (255, 255, 255, 255), (c, head_cy), head_r)
        bw = max(2, int(round(radius * 1.1)))
        bh = max(2, int(round(radius * 1.0)))
        body_top = head_cy + head_r - max(1, radius // 4)
        pygame.draw.rect(surf, (255, 255, 255, 255), (c - bw // 2, body_top, bw, bh))
        heading = (bucket + 0.5) * (360.0 / buckets)
        rot = pygame.transform.rotate(surf, heading + 90.0)
        alpha = pygame.surfarray.pixels_alpha(rot)
        ox, oy = np.nonzero(alpha > 127)
        ox = ox.astype(np.int32) - rot.get_width() // 2
        oy = oy.astype(np.int32) - rot.get_height() // 2
        _SPRITE_CACHE[key] = (ox, oy)
    return _SPRITE_CACHE[key]


# Lineage-ring masks per sprite radius (stage 15).
_RING_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

# Rainbow LUT: tag_hue in [0,1) -> full-saturation RGB (cached, 256 bins).
_HUE_LUT: np.ndarray | None = None


def hue_lut() -> np.ndarray:
    """(256, 3) uint8 lookup from tag_hue (index/255) to rainbow RGB."""
    global _HUE_LUT
    if _HUE_LUT is None:
        _HUE_LUT = np.array(
            [
                np.multiply(colorsys.hsv_to_rgb(i / 256.0, 1.0, 1.0), 255.0)
                for i in range(256)
            ],
            dtype=np.float64,
        )
    return _HUE_LUT


def _ring_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel offsets of a thin halo hugging the sprite of ``radius``.

    The band sits just outside the humanoid silhouette (which spans up to
    ~0.93r), so the sprite always paints over the inner edge. Cached per
    radius; callers scatter the per-creature tag colour onto these offsets.
    """
    if radius not in _RING_CACHE:
        inner = radius + 0.4
        outer = radius + 1.4
        side = int(np.ceil(2.0 * outer)) + 3
        if side % 2 == 0:
            side += 1
        c = side // 2
        yy, xx = np.mgrid[0:side, 0:side]
        d = np.hypot(xx - c, yy - c)
        ox, oy = np.nonzero((d >= inner) & (d <= outer))
        _RING_CACHE[radius] = (
            ox.astype(np.int32) - c,
            oy.astype(np.int32) - c,
        )
    return _RING_CACHE[radius]


def _scatter_groups(
    pixels: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    key: np.ndarray,
    cols: np.ndarray,
    viewport_width: int,
    viewport_height: int,
    offsets_fn,
) -> None:
    """Paint each ``key`` group with its ``offsets_fn(k)`` pixel mask.

    Shared by the humanoid pass and the lineage ring: one stable sort and
    one vectorized copy per group, no Python loop over creatures.
    """
    order = np.argsort(key, kind="stable")
    key = key[order]
    ix = ix[order]
    iy = iy[order]
    cols = cols[order]
    unique, starts = np.unique(key, return_index=True)
    ends = np.append(starts[1:], key.size)
    for k, kv in enumerate(unique):
        a, b = int(starts[k]), int(ends[k])
        ox, oy = offsets_fn(int(kv))
        if ox.size == 0:
            continue
        xs = (ix[a:b, None] + ox[None, :]).ravel()
        ys = (iy[a:b, None] + oy[None, :]).ravel()
        cc = np.repeat(cols[a:b], ox.size)
        ok = (xs >= 0) & (xs < viewport_width) & (ys >= 0) & (
            ys < viewport_height
        )
        if ok.all():
            pixels[xs, ys] = cc
        else:
            pixels[xs[ok], ys[ok]] = cc[ok]


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
    """Pre-allocated scratch for the world pass (image-major layout)."""

    packed: np.ndarray  # (viewport_w, viewport_h) uint32 contiguous pixels

    @classmethod
    def create(cls, viewport_width: int, viewport_height: int) -> FrameBuffers:
        return cls(packed=np.zeros((viewport_width, viewport_height), dtype=np.uint32))


def make_food_lut(settings: Settings) -> np.ndarray:
    """256-entry RGB lookup: food energy (0..capacity) -> color.

    ``food_gamma`` bends the ramp before interpolation: values < 1 lift the
    midtones so persistent patches read against the depleted background
    (which now shows the terrain through, stage 10).
    """
    ramp = np.arange(256, dtype=np.float64)
    if settings.food_gamma != 1.0:
        ramp = 255.0 * (ramp / 255.0) ** settings.food_gamma
    stops = np.array([0.0, 140.0, 255.0])
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        values = (settings.food_low[c], settings.food_mid[c], settings.food_high[c])
        lut[:, c] = np.interp(ramp, stops, values).astype(np.uint8)
    return lut


def make_terrain_lut(settings: Settings) -> np.ndarray:
    """256-entry RGB lookup: terrain fertility (0..1) -> ground color.

    Index 0 is water (exact 0.0 in the terrain array), everything else ramps
    shore -> fertile green. Drawn *under* the food pass: rich food covers
    the ground, depletion reveals it (lakes, shorelines, poor soil).
    """
    ramp = np.arange(256, dtype=np.float64)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(
            ramp, [0.0, 128.0, 255.0],
            [settings.terrain_shore[c], settings.terrain_mid[c],
             settings.terrain_high[c]],
        ).astype(np.uint8)
    lut[0] = settings.terrain_water
    return lut


def make_wood_lut(settings: Settings) -> np.ndarray:
    """256-entry RGB lookup: wood in cell (0..wood_capacity) -> brown ramp.

    Drawn over the terrain and under the food (stage 17): a full cell reads
    as litter/twigs, depletion shows the ground again.
    """
    ramp = np.arange(256, dtype=np.float64)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(
            ramp, [0.0, 255.0],
            [settings.wood_low[c], settings.wood_high[c]],
        ).astype(np.uint8)
    return lut


def creature_colors(world, settings: Settings, light: float) -> np.ndarray:
    """RGB per living creature: hue from diet, brightness from energy x light."""
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
    scaled = col * (brightness * light)[:, None]
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def draw_world(
    pixels: np.ndarray,
    world,
    camera: Camera,
    food_lut: np.ndarray,
    terrain_lut: np.ndarray,
    wood_lut: np.ndarray,
    light: float,
    shifts: PixelFormat,
    buffers: FrameBuffers,
    meat_color: tuple[int, int, int] = (140, 40, 36),
    signal_color: tuple[int, int, int] = (150, 90, 220),
) -> None:
    """Write terrain + wood + food + carrion + signal into ``pixels``.

    Per cell: ``rgb = lerp(lerp(lerp(lerp(terrain, wood, a_w), food, a_f),
    meat, a_m), signal, a_s) * light`` with ``a_w = wood / wood_capacity``
    (stage 17 — the brown litter sits over the ground and under the food,
    so harvesting reveals it the way depletion reveals terrain), then
    ``a_f = food / capacity``, ``a_m = clip(meat / capacity, 0, 1)`` and
    ``a_s = clip(signal / signal_capacity, 0, 1)``. The blend runs on the
    200x200 cell grid once per frame; the viewport step is the same single
    gather as before.
    """
    cap = float(world.config.food_capacity)
    f_idx = np.clip(world.food * (255.0 / cap), 0.0, 255.0).astype(np.intp)
    t_idx = np.clip(world.terrain * 255.0, 0.0, 255.0).astype(np.intp)
    alpha = (f_idx * np.float32(1.0 / 255.0))[..., None]
    food_rgb = food_lut[f_idx].astype(np.float32)
    terrain_rgb = terrain_lut[t_idx].astype(np.float32)
    w_cap = float(world.config.wood_capacity)
    w_idx = np.clip(world.wood * (255.0 / w_cap), 0.0, 255.0).astype(np.intp)
    wood_a = (w_idx * np.float32(1.0 / 255.0))[..., None]
    wood_rgb = wood_lut[w_idx].astype(np.float32)
    rgb = terrain_rgb + (wood_rgb - terrain_rgb) * wood_a
    rgb = rgb + (food_rgb - rgb) * alpha
    meat_a = np.clip(world.meat * np.float32(1.0 / cap), 0.0, 1.0)[..., None]
    meat_rgb = np.asarray(meat_color, dtype=np.float32)[None, None, :]
    rgb = rgb + (meat_rgb - rgb) * meat_a
    sig_cap = float(world.config.signal_capacity)
    sig_a = np.clip(world.signal * np.float32(1.0 / sig_cap), 0.0, 1.0)[..., None]
    sig_rgb = np.asarray(signal_color, dtype=np.float32)[None, None, :]
    rgb = rgb + (sig_rgb - rgb) * sig_a
    rgb *= float(light)
    np.clip(rgb, 0.0, 255.0, out=rgb)
    cell_pixels = rgb_to_pixel(rgb.astype(np.uint8), shifts)
    np.take(cell_pixels.ravel(), camera.flat_indices(), out=buffers.packed)
    pixels[:] = buffers.packed


def stamp_creatures(
    pixels: np.ndarray,
    world,
    camera: Camera,
    settings: Settings,
    shifts: PixelFormat,
    light: float,
) -> None:
    """Scatter every living creature as an oriented humanoid sprite.

    The lineage ring (stage 15) paints first — a thin halo in each
    creature's tag hue, grouped by radius only — then the humanoid pass
    (grouping key ``(radius, heading bucket)``, at most
    ``max_radius_px * sprite_buckets`` poses) paints each group with its
    pose's alpha mask over the ring's inner edge. ``light`` dims both with
    the world (the selection overlay stays bright).
    """
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

    # lineage ring: identity colour only — brightness follows the light,
    # never the energy, so the tag reads the same at any charge level
    lut = hue_lut()
    hue_idx = np.clip(
        world.genome[slots, world.config.brain_params + 4] * 255.0, 0, 255
    ).astype(np.int32)
    ring_rgb = (lut[hue_idx] * light).astype(np.uint8)
    ring_cols = rgb_to_pixel(ring_rgb, shifts)
    _scatter_groups(pixels, ix, iy, radius, ring_cols, vw, vh, _ring_offsets)

    buckets = int(settings.sprite_buckets)
    two_pi = np.float32(2.0 * np.pi)
    bucket = (
        (world.angle[slots] % two_pi) / np.float32(two_pi / buckets)
    ).astype(np.int32) % buckets
    key = radius * buckets + bucket
    colors = rgb_to_pixel(creature_colors(world, settings, light), shifts)

    def humanoid_offsets(kv: int) -> tuple[np.ndarray, np.ndarray]:
        r, ib = divmod(kv, buckets)
        return _humanoid_offsets(r, ib, buckets)

    _scatter_groups(pixels, ix, iy, key, colors, vw, vh, humanoid_offsets)


# Structure sprite masks per (kind, radius) — kind 0 = fire (disc),
# kind 1 = house (square). Cached like the humanoid poses.
_STRUCT_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _struct_offsets(kind: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel offsets of one structure sprite, centered on (0, 0)."""
    key = (kind, radius)
    if key not in _STRUCT_CACHE:
        side = 2 * radius + 1
        surf = pygame.Surface((side, side), pygame.SRCALPHA)
        c = side // 2
        if kind == 0:
            pygame.draw.circle(surf, (255, 255, 255, 255), (c, c), radius)
        else:
            pygame.draw.rect(
                surf,
                (255, 255, 255, 255),
                (c - radius, c - radius, 2 * radius + 1, 2 * radius + 1),
            )
        alpha = pygame.surfarray.pixels_alpha(surf)
        ox, oy = np.nonzero(alpha > 127)
        _STRUCT_CACHE[key] = (
            ox.astype(np.int32) - c,
            oy.astype(np.int32) - c,
        )
    return _STRUCT_CACHE[key]


def stamp_structures(
    pixels: np.ndarray,
    world,
    camera: Camera,
    settings: Settings,
    shifts: PixelFormat,
    light: float,
) -> None:
    """Scatter fires and houses into the frame (stage 17).

    Called between the cell blend and :func:`stamp_creatures`, so walking
    creatures paint over the objects. A fire flickers **deterministically**
    — a pure function of ``(tick, slot)``, never RNG — and dims with its
    remaining fuel (a dying fire glows weak); a house is flat. Both dim
    with the ambient light like the rest of the world. Nothing here reads
    scratch that the simulation depends on, so recording stays bit-identical.
    """
    slots = np.flatnonzero(world.struct_alive)
    if slots.size == 0:
        return
    vh = pixels.shape[1]
    vw = pixels.shape[0]
    sx, sy = camera.world_to_screen(world.struct_pos[slots])
    ix = np.rint(sx).astype(np.int32)
    iy = np.rint(sy).astype(np.int32)
    radius = max(2, int(round(settings.struct_world_size * camera.zoom)))
    kinds = world.struct_type[slots].astype(np.int32)
    key = kinds * 4096 + radius  # radius << 12 (zoom is clamped to 10)

    rgb = np.empty((slots.size, 3), dtype=np.float64)
    house_mask = kinds != 0
    rgb[house_mask] = settings.house_color
    fire = ~house_mask
    if np.any(fire):
        fuel_cap = float(world.config.fire_initial_fuel)
        fuel01 = (
            world.struct_fuel[slots[fire]] / fuel_cap
            if fuel_cap > 0.0
            else np.ones(int(fire.sum()))
        )
        np.clip(fuel01, 0.0, 1.0, out=fuel01)
        flick = 0.78 + 0.22 * np.sin(
            np.float64(world.tick) * 0.9 + slots[fire] * 1.7
        )
        dim = (0.45 + 0.55 * fuel01) * flick * float(light)
        rgb[fire] = np.asarray(settings.fire_color, dtype=np.float64) * dim[:, None]
    rgb[house_mask] = rgb[house_mask] * float(light)
    np.clip(rgb, 0.0, 255.0, out=rgb)

    colors = rgb_to_pixel(rgb.astype(np.uint8), shifts)

    def struct_offsets_fn(kv: int) -> tuple[np.ndarray, np.ndarray]:
        kind, r = divmod(kv, 4096)
        return _struct_offsets(kind, r)

    _scatter_groups(pixels, ix, iy, key, colors, vw, vh, struct_offsets_fn)


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


@dataclass
class Flash:
    """One birth/death flash: world position, body size, color, frame age."""

    x: float
    y: float
    size: float  # body size trait; pixel radius recomputed per frame (zoom)
    color: tuple[int, int, int]
    age: int = 0


@dataclass
class Transfer:
    """One donation gift (stage 16): donor->recipient endpoints, frame age.

    Rebuilt from ``world.donate_to`` every frame, so the line follows the
    live positions of both creatures; ``age`` counts frames since the
    donor's last gift and drives the fade in :func:`render_transfers`.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    age: int = 0


class EventTracker:
    """Render-side birth/death flashes from successive alive masks.

    Diffing frames (plus the ``deaths_*`` counter deltas for the death
    color) keeps the simulation untouched: recording a run stays
    bit-identical to running it headless.
    """

    def __init__(self, world) -> None:
        self.prev_alive = world.alive.copy()
        self.prev_counters = self._counters(world)
        self.flashes: deque[Flash] = deque()
        # donation gift lines (stage 16), rebuilt each update from the
        # per-tick world.donate_to map: donor -> [recipient, frames_since_last_gift]
        self._gifts: dict[int, list[int]] = {}
        self.transfers: list[Transfer] = []

    @staticmethod
    def _counters(world) -> tuple[int, int, int]:
        return (
            int(world.deaths_famine),
            int(world.deaths_age),
            int(world.deaths_predation),
        )

    def update(self, world, settings: Settings) -> None:
        """Age existing flashes and append this frame's births/deaths."""
        for f in self.flashes:
            f.age += 1
        while self.flashes and self.flashes[0].age >= settings.flash_duration:
            self.flashes.popleft()

        alive = world.alive
        born = np.flatnonzero(alive & ~self.prev_alive)
        died = np.flatnonzero(self.prev_alive & ~alive)
        if born.size or died.size:
            size_trait = world.config.brain_params + 1
            for slot in born:
                self.flashes.append(
                    Flash(
                        float(world.pos[slot, 0]),
                        float(world.pos[slot, 1]),
                        float(world.genome[slot, size_trait]),
                        settings.flash_birth,
                    )
                )
            if died.size:
                color = self._death_color(world, settings)
                for slot in died:
                    self.flashes.append(
                        Flash(
                            float(world.pos[slot, 0]),
                            float(world.pos[slot, 1]),
                            float(world.genome[slot, size_trait]),
                            color,
                        )
                    )
            while len(self.flashes) > settings.flash_max:
                self.flashes.popleft()

        self.prev_alive = alive.copy()
        self.prev_counters = self._counters(world)
        self._update_transfers(world, settings)

    def _update_transfers(self, world, settings: Settings) -> None:
        """Rebuild the donor->recipient gift lines from the tick's map.

        An active donor resets to age 0 every frame (the line persists
        while it keeps giving, tracking both live positions); once the
        gifts stop, the line fades over ``flash_duration`` frames. A
        dead donor expires immediately — its slot may already belong to
        somebody else. Reading the render-only scratch never perturbs
        the simulation, same contract as the rest of this class.
        """
        gift_map = world.donate_to
        alive = world.alive
        active = {int(d): int(gift_map[d]) for d in np.flatnonzero(gift_map >= 0)}
        for d, r in active.items():
            self._gifts[d] = [r, 0]
        for d in list(self._gifts):
            if d in active:
                continue
            if d >= alive.size or not alive[d]:
                del self._gifts[d]
                continue
            self._gifts[d][1] += 1
            if self._gifts[d][1] >= settings.flash_duration:
                del self._gifts[d]
        pos = world.pos
        self.transfers = [
            Transfer(
                float(pos[d, 0]),
                float(pos[d, 1]),
                float(pos[r, 0]),
                float(pos[r, 1]),
                age,
            )
            for d, (r, age) in self._gifts.items()
        ]

    def _death_color(self, world, settings: Settings) -> tuple[int, int, int]:
        """Dominant cause of this frame's deaths (per-slot causes are not
        stored; a frame is almost always one cohort dying together)."""
        now = self._counters(world)
        delta = [now[i] - self.prev_counters[i] for i in range(3)]
        if delta[2] > 0 and delta[2] >= delta[0] and delta[2] >= delta[1]:
            return settings.flash_predation
        if delta[0] >= delta[1]:
            return settings.flash_famine
        return settings.flash_age


class SparkTracker:
    """Ring buffers feeding the panel's world sparkline (one sample/frame)."""

    def __init__(self, settings: Settings) -> None:
        n = settings.sparkline_samples
        self.alive: deque[int] = deque(maxlen=n)
        self.energy: deque[float] = deque(maxlen=n)

    def sample(self, world) -> None:
        self.alive.append(int(world.alive_count))
        alive = world.alive
        self.energy.append(
            float(world.energy[alive].mean()) if world.alive_count else 0.0
        )


def draw_trail(
    overlay: pygame.Surface,
    points: np.ndarray,
    camera: Camera,
    settings: Settings,
) -> None:
    """Fade polyline of the selected creature's recent path (world coords).

    Segments that jump across a torus wrap are skipped so the trail never
    streaks across the whole viewport.
    """
    n = points.shape[0]
    if n < 2:
        return
    vw, vh = camera.viewport
    sx, sy = camera.world_to_screen(points)
    base = np.asarray(settings.trail_color, dtype=np.float64)
    for i in range(1, n):
        x0, y0 = float(sx[i - 1]), float(sy[i - 1])
        x1, y1 = float(sx[i]), float(sy[i])
        if abs(x1 - x0) > 0.5 * vw or abs(y1 - y0) > 0.5 * vh:
            continue
        t = i / (n - 1)
        alpha = int(40 + 180 * t)
        color = (int(base[0]), int(base[1]), int(base[2]), alpha)
        pygame.draw.line(
            overlay, color, (int(x0), int(y0)), (int(x1), int(y1)), 1
        )


def render_flashes(
    overlay: pygame.Surface,
    flashes: deque[Flash],
    camera: Camera,
    settings: Settings,
) -> None:
    """Semi-transparent circles over fresh births/deaths (RGBA on overlay)."""
    for f in flashes:
        t = f.age / settings.flash_duration
        if t >= 1.0:
            continue
        alpha = int(200.0 * (1.0 - t))
        if alpha <= 0:
            continue
        radius = int(np.rint(f.size * camera.zoom * settings.size_scale)) + 2
        radius = max(3, min(radius, settings.max_radius_px + 3))
        sx, sy = camera.world_to_screen(
            np.array([[f.x, f.y]], dtype=np.float32)
        )
        cx, cy = int(sx[0]), int(sy[0])
        color = (f.color[0], f.color[1], f.color[2], alpha)
        pygame.draw.circle(overlay, color, (cx, cy), radius)


def render_transfers(
    overlay: pygame.Surface,
    transfers: list[Transfer],
    camera: Camera,
    settings: Settings,
) -> None:
    """Cyan donor->recipient gift lines (stage 16), fading with age.

    Fresh gifts (age 0, the ones the tick just made) draw slightly
    thicker; older lines thin out and fade as ``age`` approaches
    ``flash_duration``. Skip entirely when nothing donated — the default
    donate_rate is 0, so this is a no-op in vanilla runs.
    """
    if not transfers:
        return
    color = settings.flash_donation
    duration = max(1, settings.flash_duration)
    half_w = camera.world_width * 0.5
    half_h = camera.world_height * 0.5
    for t in transfers:
        fade = 1.0 - t.age / duration
        if fade <= 0.0:
            continue
        # torus seam: a "nearest" gift across the wrap would draw a line
        # over the whole map — skip it, same rule as draw_trail
        if abs(t.x1 - t.x2) > half_w or abs(t.y1 - t.y2) > half_h:
            continue
        alpha = int(190.0 * fade)
        sx, sy = camera.world_to_screen(
            np.array([[t.x1, t.y1], [t.x2, t.y2]], dtype=np.float32)
        )
        p1 = (int(sx[0]), int(sy[0]))
        p2 = (int(sx[1]), int(sy[1]))
        pygame.draw.line(
            overlay,
            (color[0], color[1], color[2], alpha),
            p1,
            p2,
            2 if t.age == 0 else 1,
        )
