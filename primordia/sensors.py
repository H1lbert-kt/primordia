"""Ray sensors: what a creature perceives each tick.

Input vector layout (``Config.sensor_input_dim`` floats, written into
``World.sensor_buf``), in fixed order:

    [0 : n_rays)            food energy summed along each ray / (capacity * steps)
    [n_rays : 2*n_rays)     creatures in cells crossed by each ray / steps
                            (own cell subtracted, so no self-detection)
    [2n : 2n+2)             smell: food and creatures in (2r+1)^2 cells around
                            the creature (own creature subtracted)
    [2n+2 : 2n+4)           internal: energy / (2 * initial_energy), |vel| / max_speed

Rays march in steps of one cell up to the creature's ``vision_range`` body
trait, wrapping toroidally like the rest of the world. Creature counts come
from ``World.cell_counts`` (rebuilt every tick): O(cells + N), never O(N^2).
No random numbers are drawn in this module.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def perceive(
    pos: np.ndarray,
    angle: np.ndarray,
    vel: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    food: np.ndarray,
    cell_counts: np.ndarray,
    ray_unit: np.ndarray,
    sensor_buf: np.ndarray,
    n_rays: np.int32,
    smell_r: np.int32,
    traits_off: np.int32,
    inv_cell_w: np.float32,
    inv_cell_h: np.float32,
    step_len: np.float32,
    gw: np.int32,
    gh: np.int32,
    width: np.float32,
    height: np.float32,
    food_capacity: np.float32,
    energy_scale: np.float32,
) -> None:
    """Fill ``sensor_buf[i]`` for every living creature (parallel over slots)."""
    inv_cap = np.float32(1.0) / food_capacity
    inv_e = np.float32(1.0) / energy_scale
    smell_side = 2 * smell_r + 1
    inv_smell = np.float32(1.0) / (smell_side * smell_side)
    base = 2 * n_rays

    for i in prange(alive.size):
        if not alive[i]:
            continue

        px0 = pos[i, 0]
        py0 = pos[i, 1]
        # multiply by the reciprocal instead of dividing (two divisions per sample)
        ix0 = int(px0 * inv_cell_w)
        iy0 = int(py0 * inv_cell_h)
        if ix0 < 0:
            ix0 = 0
        elif ix0 >= gw:
            ix0 = gw - 1
        if iy0 < 0:
            iy0 = 0
        elif iy0 >= gh:
            iy0 = gh - 1

        c = np.cos(angle[i])
        s = np.sin(angle[i])
        vision = genome[i, traits_off + 2]
        max_speed_trait = genome[i, traits_off]
        nsteps = int(vision / step_len)

        if nsteps > 0:
            inv_n = np.float32(1.0) / nsteps
            for r in range(n_rays):
                # rotate the precomputed unit vector by the creature heading
                dx = c * ray_unit[r, 0] - s * ray_unit[r, 1]
                dy = s * ray_unit[r, 0] + c * ray_unit[r, 1]
                fs = np.float32(0.0)
                cs = np.float32(0.0)
                for k in range(nsteps):
                    d = (k + 0.5) * step_len
                    px = px0 + dx * d
                    py = py0 + dy * d
                    while px < 0.0:
                        px = px + width
                    while px >= width:
                        px = px - width
                    while py < 0.0:
                        py = py + height
                    while py >= height:
                        py = py - height
                    ix = int(px * inv_cell_w)
                    iy = int(py * inv_cell_h)
                    if ix < 0:
                        ix = 0
                    elif ix >= gw:
                        ix = gw - 1
                    if iy < 0:
                        iy = 0
                    elif iy >= gh:
                        iy = gh - 1
                    fs = fs + food[iy, ix]
                    cnt = cell_counts[iy, ix]
                    if ix == ix0 and iy == iy0:
                        cnt = cnt - 1
                    cs = cs + np.float32(cnt)
                sensor_buf[i, r] = fs * inv_cap * inv_n
                sensor_buf[i, n_rays + r] = cs * inv_n
        else:
            for r in range(base):
                sensor_buf[i, r] = 0.0

        # smell: everything in the neighbourhood, self excluded
        sf = np.float32(0.0)
        sc = np.float32(0.0)
        for dy in range(-smell_r, smell_r + 1):
            for dx in range(-smell_r, smell_r + 1):
                iy = iy0 + dy
                while iy < 0:
                    iy = iy + gh
                while iy >= gh:
                    iy = iy - gh
                ix = ix0 + dx
                while ix < 0:
                    ix = ix + gw
                while ix >= gw:
                    ix = ix - gw
                sf = sf + food[iy, ix]
                sc = sc + np.float32(cell_counts[iy, ix])
        sensor_buf[i, base] = sf * inv_cap * inv_smell
        sensor_buf[i, base + 1] = (sc - np.float32(1.0)) * inv_smell

        # internal sensors
        speed = math.sqrt(vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1])
        sensor_buf[i, base + 2] = energy[i] * inv_e
        if max_speed_trait > 0.0:
            sensor_buf[i, base + 3] = np.float32(speed) / max_speed_trait
        else:
            sensor_buf[i, base + 3] = 0.0
