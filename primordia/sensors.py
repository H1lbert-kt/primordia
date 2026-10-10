"""Ray sensors: what a creature perceives each tick.

Input vector layout (``Config.sensor_input_dim`` floats, written into
``World.sensor_buf``), in fixed order:

    [0 : n_rays)             food energy summed along each ray / (capacity * steps)
    [n_rays : 2*n_rays)      creatures in cells crossed by each ray / steps
                             (own cell subtracted, so no self-detection)
    [2n : 3n)                peak food cell along each ray / capacity — a rich
                             path reads high even when its sum looks like haze
    [3n : 3n+4)              smell: food, creatures, prey (sum of the diet
                             trait per cell, i.e. "how much prey is near") and
                             carrion (meat grid, stage 11) in (2r+1)^2 cells
                             around the creature (self excluded)
    [3n+4 : 3n+6)            antennas: food smell on the left / right of the
                             heading (cross product of heading and cell offset)
    [3n+6 : 3n+9)            internal: energy / (2 * initial_energy), |vel| /
                             max_speed, and ambient light L(t) (stage 10's
                             circadian cue: what "night" feels like from inside)

Rays march in steps of one cell up to the creature's ``vision_range`` body
trait times the current light level ``L(t)`` (seeing in the dark is seeing
less far) divided by the wear factor ``exp(senescence_rate * age)`` (stage
11: old eyes see less), wrapping toroidally like the rest of the world.
Creature counts come from ``World.cell_counts`` and the carrion smell from
``World.meat`` (both current): O(cells + N), never O(N^2).
No random numbers are drawn in this module.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .config import Config


def input_groups(config: Config) -> tuple[tuple[str, slice | np.ndarray], ...]:
    """Labels and index groups of the input vector, in ``sensor_buf`` order.

    Single source of truth for consumers that display inputs (the viewer
    panel): indices are derived from Config here, never hardcoded twice.
    The ray-food group gathers the sum and peak channels (non-contiguous)
    so the panel shows them as one row.
    """
    n = config.n_rays
    smell = 3 * n
    return (
        ("ray food  sum|peak", np.r_[0:n, 2 * n : 3 * n]),
        ("ray creatures", slice(n, 2 * n)),
        ("smell food", slice(smell, smell + 1)),
        ("smell creatures", slice(smell + 1, smell + 2)),
        ("smell prey", slice(smell + 2, smell + 3)),
        ("smell carrion", slice(smell + 3, smell + 4)),
        ("smell left", slice(smell + 4, smell + 5)),
        ("smell right", slice(smell + 5, smell + 6)),
        ("energy", slice(smell + 6, smell + 7)),
        ("speed", slice(smell + 7, smell + 8)),
        ("light", slice(smell + 8, smell + 9)),
    )


@njit(cache=True, parallel=True)
def perceive(
    pos: np.ndarray,
    angle: np.ndarray,
    vel: np.ndarray,
    energy: np.ndarray,
    age: np.ndarray,
    alive: np.ndarray,
    genome: np.ndarray,
    food: np.ndarray,
    meat: np.ndarray,
    cell_counts: np.ndarray,
    cell_diet: np.ndarray,
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
    light: np.float32,
    senescence_rate: np.float32,
) -> None:
    """Fill ``sensor_buf[i]`` for every living creature (parallel over slots)."""
    inv_cap = np.float32(1.0) / food_capacity
    inv_e = np.float32(1.0) / energy_scale
    smell_side = 2 * smell_r + 1
    inv_smell = np.float32(1.0) / (smell_side * smell_side)
    inv_side = np.float32(2.0) * inv_smell  # antennas see ~half the window
    base = 3 * n_rays

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
        # effective reach shrinks with light (stage 10) and with wear
        # (stage 11): at L=1, age=0 this is the stage-9 vision
        vision = genome[i, traits_off + 2] * light
        if senescence_rate > 0.0:
            vision = vision / np.exp(senescence_rate * np.float32(age[i]))
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
                fmax = np.float32(0.0)
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
                    fv = food[iy, ix]
                    fs = fs + fv
                    if fv > fmax:
                        fmax = fv
                    cnt = cell_counts[iy, ix]
                    if ix == ix0 and iy == iy0:
                        cnt = cnt - 1
                    cs = cs + np.float32(cnt)
                sensor_buf[i, r] = fs * inv_cap * inv_n
                sensor_buf[i, n_rays + r] = cs * inv_n
                sensor_buf[i, 2 * n_rays + r] = fmax * inv_cap
        else:
            for r in range(base):
                sensor_buf[i, r] = 0.0

        # smell: everything in the neighbourhood, self excluded; the antennas
        # split the same window by the cross product of heading and offset
        sf = np.float32(0.0)
        sc = np.float32(0.0)
        sp = np.float32(0.0)  # prey: sum of diet traits (how much meat-walker)
        scarrion = np.float32(0.0)  # carrion: the meat grid itself (stage 11)
        sf_left = np.float32(0.0)
        sf_right = np.float32(0.0)
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
                fv = food[iy, ix]
                sf = sf + fv
                sc = sc + np.float32(cell_counts[iy, ix])
                sp = sp + cell_diet[iy, ix]
                scarrion = scarrion + meat[iy, ix]
                cross = c * np.float32(dy) - s * np.float32(dx)
                if cross < 0.0:
                    sf_left = sf_left + fv
                elif cross > 0.0:
                    sf_right = sf_right + fv
        sensor_buf[i, base] = sf * inv_cap * inv_smell
        sensor_buf[i, base + 1] = (sc - np.float32(1.0)) * inv_smell
        # prey smell excludes my own diet contribution (like the creature count)
        sensor_buf[i, base + 2] = (
            sp - genome[i, traits_off + 3]
        ) * inv_smell
        sensor_buf[i, base + 3] = scarrion * inv_cap * inv_smell
        sensor_buf[i, base + 4] = sf_left * inv_cap * inv_side
        sensor_buf[i, base + 5] = sf_right * inv_cap * inv_side

        # internal sensors
        speed = math.sqrt(vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1])
        sensor_buf[i, base + 6] = energy[i] * inv_e
        if max_speed_trait > 0.0:
            sensor_buf[i, base + 7] = np.float32(speed) / max_speed_trait
        else:
            sensor_buf[i, base + 7] = 0.0
        sensor_buf[i, base + 8] = light  # ambient light: identical for all
