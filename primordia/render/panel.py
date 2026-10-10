"""Side panel: the selected creature's brain, sensors, traits and world HUD.

The panel is fixed-size content (a few hundred primitives per frame), so it
uses plain ``pygame.draw``; the thousands-of-creatures rule applies to the
world viewport, which is pure NumPy (see ``draw.py``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame

from ..cycles import light_level
from ..sensors import input_groups
from .settings import Settings

_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _font(size: int) -> pygame.font.Font:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]


def clear_font_cache() -> None:
    """Drop cached Font objects (pygame.quit() invalidates the font module)."""
    _FONT_CACHE.clear()


@dataclass(frozen=True)
class Hud:
    """App-side values shown alongside the world state."""

    fps: float
    speed: int  # ticks per frame (0 while paused)
    paused: bool
    follow: bool
    seed: int
    alive_hist: deque | None = None  # sparkline samples (see SparkTracker)
    energy_hist: deque | None = None


def _text(dst: pygame.Surface, size: int, msg: str, x: int, y: int, color) -> int:
    surf = _font(size).render(msg, True, color)
    dst.blit(surf, (x, y))
    return y + surf.get_height()


def _hbar(
    dst: pygame.Surface,
    x: int,
    y: int,
    w: int,
    h: int,
    frac: float,
    settings: Settings,
    signed: bool,
) -> None:
    pygame.draw.rect(dst, settings.bar_bg, (x, y, w, h))
    frac = max(-1.0, min(1.0, frac))
    if signed:
        mid = x + w // 2
        if frac >= 0.0:
            pygame.draw.rect(dst, settings.bar_pos, (mid, y, int((w / 2) * frac), h))
        else:
            pygame.draw.rect(
                dst, settings.bar_neg, (mid + int((w / 2) * frac), y, int(-(w / 2) * frac), h)
            )
        pygame.draw.line(dst, settings.panel_border, (mid, y), (mid, y + h))
    else:
        pygame.draw.rect(dst, settings.bar_pos, (x, y, int(w * frac), h))


def _vbar_row(
    dst: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    values: np.ndarray,
    settings: Settings,
    signed: bool,
) -> None:
    """One row of small vertical bars sharing a baseline."""
    n = values.size
    slot_w = width // n
    base = y + height
    pygame.draw.line(dst, settings.panel_border, (x, base), (x + width, base))
    for i, v in enumerate(values):
        v = float(max(-1.0, min(1.0, v)))
        bh = int(abs(v) * height)
        bx = x + i * slot_w + 2
        bw = max(2, slot_w - 4)
        if signed and v < 0.0:
            pygame.draw.rect(dst, settings.bar_neg, (bx, base, bw, bh))
        else:
            pygame.draw.rect(dst, settings.bar_pos, (bx, base - bh, bw, bh))


def _heatmap(
    dst: pygame.Surface,
    x: int,
    y: int,
    cell_w: int,
    cell_h: int,
    grid: np.ndarray,
    settings: Settings,
) -> tuple[int, int]:
    """Diverging blue/red heatmap of a small weight matrix; returns size."""
    zero = np.asarray(settings.heat_zero)
    pos = np.asarray(settings.heat_pos)
    neg = np.asarray(settings.heat_neg)
    rows, cols = grid.shape
    t = np.clip(grid.astype(np.float64) / 2.0, -1.0, 1.0)
    rgb = np.empty((rows, cols, 3), dtype=np.int32)
    up = t >= 0.0
    rgb[up] = (zero + (pos - zero) * t[up, None]).astype(np.int32)
    rgb[~up] = (zero + (neg - zero) * (-t[~up, None])).astype(np.int32)
    for j in range(cols):
        for i in range(rows):
            pygame.draw.rect(
                dst,
                tuple(int(c) for c in rgb[i, j]),
                (x + j * cell_w, y + i * cell_h, cell_w - 1, cell_h - 1),
            )
    return cols * cell_w, rows * cell_h


def _sparkline(
    dst: pygame.Surface,
    x: int,
    y: int,
    width: int,
    height: int,
    series: list[tuple[deque | None, tuple[int, int, int]]],
    settings: Settings,
) -> None:
    """Overlay of independent min..max polylines (alive + mean energy)."""
    pygame.draw.rect(dst, settings.bar_bg, (x, y, width, height))
    for values, color in series:
        if values is None or len(values) < 2:
            continue
        a = np.asarray(values, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        xs = x + np.linspace(0.0, width - 1.0, a.size)
        if hi - lo < 1e-9:
            # flat series: draw a centered line instead of hugging an edge
            ys = np.full(a.shape, y + height // 2, dtype=np.float64)
        else:
            ys = y + height - 1.0 - (a - lo) / (hi - lo) * (height - 2.0)
        pts = list(zip(xs.astype(np.int32), ys.astype(np.int32)))
        pygame.draw.lines(dst, color, False, pts, 1)


def _legend_row(dst: pygame.Surface, x: int, y: int, settings: Settings) -> int:
    """Color key for sparkline series and creature diets (one dim line each)."""
    pygame.draw.circle(dst, settings.spark_alive, (x + 5, y + 6), 4)
    pygame.draw.circle(dst, settings.spark_energy, (x + 78, y + 6), 4)
    _text(dst, 14, "alive", x + 13, y, settings.text_dim)
    _text(dst, 14, "energy", x + 86, y, settings.text_dim)
    y += 15
    pygame.draw.circle(dst, settings.creature_herbivore, (x + 5, y + 6), 4)
    pygame.draw.circle(dst, settings.creature_predator, (x + 108, y + 6), 4)
    _text(dst, 14, "herbivore", x + 13, y, settings.text_dim)
    _text(dst, 14, "predator  brighter = energy", x + 116, y, settings.text_dim)
    return y + 15


def render_panel(
    dst: pygame.Surface,
    world,
    slot: int,
    hud: Hud,
    settings: Settings,
) -> None:
    """Draw the whole side panel onto ``dst`` (panel-sized surface)."""
    dst.fill(settings.panel_bg)
    pygame.draw.line(
        dst, settings.panel_border, (0, 0), (0, dst.get_height()), 2
    )
    pad = 10
    width = settings.panel_width - 2 * pad
    x = pad
    y = 10
    cfg = world.config

    selected = slot >= 0 and world.alive[slot]
    if selected:
        y = _text(dst, 26, f"CREATURE  #{slot}", x, y, settings.text)
        y = _text(
            dst,
            18,
            f"age {int(world.age[slot])}   energy {float(world.energy[slot]):.1f}",
            x,
            y,
            settings.text_dim,
        )
        frac = float(world.energy[slot]) / (cfg.reproduce_threshold * 2.0)
        _hbar(dst, x, y, width, 10, frac, settings, signed=False)
        y += 14
        y = _text(
            dst,
            18,
            f"pos ({float(world.pos[slot, 0]):.0f}, {float(world.pos[slot, 1]):.0f})"
            f"   tick {world.tick}",
            x,
            y,
            settings.text_dim,
        )
        y += 6

        # sensors: groups from the single layout in sensors.input_groups
        y = _text(dst, 20, "SENSORS", x, y, settings.text)
        inputs = world.sensor_buf[slot]
        for label, group in input_groups(cfg):
            vals = inputs[group]
            if vals.size > 1:
                y = _text(dst, 16, label, x, y, settings.text_dim)
                _vbar_row(dst, x, y, width, 22, vals, settings, signed=False)
                y += 26
            else:
                val = float(vals[0])
                _text(dst, 16, label, x, y - 1, settings.text_dim)
                _hbar(dst, x + 110, y, width - 160, 9, val, settings, signed=False)
                _text(
                    dst,
                    16,
                    f"{val:.2f}",
                    x + width - 46,
                    y - 1,
                    settings.text_dim,
                )
                y += 10  # stage 11: 8 single-sensor rows must fit the 720px panel
        y += 4

        # brain: hidden activations and raw outputs
        y = _text(dst, 20, "BRAIN", x, y, settings.text)
        _text(dst, 16, "hidden (tanh)", x, y, settings.text_dim)
        y += 16
        _vbar_row(
            dst, x, y, width, 22, world.hidden_buf[slot], settings, signed=True
        )
        y += 26
        logits = world.actions[slot]
        outs = (
            ("accel", float(np.tanh(logits[0])), True),
            ("turn", float(np.tanh(logits[1])), True),
            ("gate", float(0.5 * (1.0 + np.tanh(logits[2]))), False),
        )
        for label, val, signed in outs:
            _text(dst, 16, label, x, y - 1, settings.text_dim)
            _hbar(dst, x + 110, y, width - 160, 9, val, settings, signed)
            _text(dst, 16, f"{val:+.2f}", x + width - 46, y - 1, settings.text_dim)
            y += 12
        y += 4

        # weights: W1|b1, W2|b2 and the recurrent W_rec (layout in brain.py)
        traits = cfg.brain_params
        w1_size = cfg.hidden_size * (cfg.sensor_input_dim + 1)
        w1 = world.genome[slot, :w1_size].reshape(
            cfg.hidden_size, cfg.sensor_input_dim + 1
        )
        w2_size = cfg.N_OUTPUTS * (cfg.hidden_size + 1)
        w2 = world.genome[slot, w1_size : w1_size + w2_size].reshape(
            cfg.N_OUTPUTS, cfg.hidden_size + 1
        )
        wrec = world.genome[slot, w1_size + w2_size : traits].reshape(
            cfg.hidden_size, cfg.hidden_size
        )
        y = _text(dst, 18, f"W1|b1  ({w1.shape[0]}x{w1.shape[1]})", x, y, settings.text_dim)
        _, h1 = _heatmap(dst, x, y, 10, 6, w1, settings)
        y += h1 + 6
        # W2 and W_rec share one band (fixed panel height budget)
        y = _text(
            dst,
            18,
            f"W2|b2 ({w2.shape[0]}x{w2.shape[1]})"
            f"    W_rec ({wrec.shape[0]}x{wrec.shape[1]})",
            x,
            y,
            settings.text_dim,
        )
        _, h2 = _heatmap(dst, x, y, 10, 6, w2, settings)
        _, hr = _heatmap(dst, x + 120, y, 10, 6, wrec, settings)
        y += max(h2, hr) + 6

        # body traits
        g = world.genome[slot]
        y = _text(
            dst,
            18,
            f"speed {float(g[traits]):.2f}   size {float(g[traits + 1]):.2f}",
            x,
            y,
            settings.text,
        )
        y = _text(
            dst,
            18,
            f"vision {float(g[traits + 2]):.1f}   diet {float(g[traits + 3]):.2f}",
            x,
            y,
            settings.text,
        )
        cid = int(world.creature_id[slot])
        parent = int(world.parent_id[slot])
        y = _text(
            dst,
            16,
            f"id #{cid}   parent #{parent}   species {int(world.species_id[slot])}",
            x,
            y,
            settings.text_dim,
        )
        y += 6
    else:
        y = _text(dst, 26, "NO SELECTION", x, y, settings.text)
        y = _text(dst, 16, "click a creature to inspect its brain", x, y, settings.text_dim)
        y += 8

    # world stats (always visible)
    y = _text(dst, 20, "WORLD", x, y, settings.text)
    alive = world.alive_count
    mean_e = float(world.energy[world.alive].mean()) if alive else 0.0
    meat_total = float(world.meat.sum(dtype=np.float64))
    stats = (
        f"tick {world.tick}   alive {alive}/{world.config.max_creatures}",
        f"mean energy {mean_e:.1f}   meat {meat_total:.0f}",
        f"births {world.births}   famine {world.deaths_famine}",
        f"age deaths {world.deaths_age}   predation {world.deaths_predation}",
        f"fps {hud.fps:.0f}   sim x{hud.speed}{'  [PAUSED]' if hud.paused else ''}"
        f"{'  [FOLLOW]' if hud.follow else ''}",
        f"seed {hud.seed}   light {float(light_level(world.config, world.tick)):.2f}",
    )
    for line in stats:
        y = _text(dst, 16, line, x, y, settings.text)
    y += 4

    if not selected:
        # Live dynamics in overview mode only: the inspect view spends the
        # height budget on sensors and weight heatmaps.
        _sparkline(
            dst,
            x,
            y,
            width,
            30,
            [
                (hud.alive_hist, settings.spark_alive),
                (hud.energy_hist, settings.spark_energy),
            ],
            settings,
        )
        y += 34
        y = _legend_row(dst, x, y, settings)
        y += 4

    # key help (fixed budget: must fit the window height)
    for line in (
        "keys: space pause  . step",
        "1-4 / +-  sim speed",
        "wheel zoom  drag pan  click select",
        "esc clear   f follow",
    ):
        _text(dst, 16, line, x, y, settings.text_dim)
        y += 13
