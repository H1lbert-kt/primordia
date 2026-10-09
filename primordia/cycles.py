"""Environmental cycles: light and season as pure functions of (Config, tick).

Stage 10's astronomical laws. No state, no RNG: the same tick always
produces the same values, which is what lets save/load resume mid-cycle
bit-for-bit (the tick is saved) and lets the renderer read the *same*
formulas the simulation uses — one source of truth for day/night in both
core and viewer.

    light_level(cfg, tick)        L(t) in [1 - day_amp, 1], noon at 0
    season_multiplier(cfg, tick)  S(t) in [1 - amp, 1 + amp], mean 1
    growth_multiplier(cfg, tick)  S(t) * L(t), >= 0 (can exceed 1: a strong
                                  season at noon refills faster than nominal;
                                  the applied *rate* is clamped to <= 1 in
                                  step.phase_grow_food so food stays in [0, K])

With the amplitudes at 0 every function collapses to 1.0 (the identity),
which is how experiments pin the pre-stage-10 regime.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config


def light_level(cfg: Config, tick: int) -> np.float32:
    """Ambient light L(t): 1.0 at noon, 1 - day_amp at midnight."""
    if cfg.day_amp == 0.0:
        return np.float32(1.0)
    phase = 2.0 * math.pi * (tick % cfg.day_period) / cfg.day_period
    level = 1.0 - cfg.day_amp * 0.5 * (1.0 - math.cos(phase))
    return np.float32(max(level, 0.0))


def season_multiplier(cfg: Config, tick: int) -> np.float32:
    """Seasonal growth multiplier S(t): mean 1, peaks at season peaks."""
    if cfg.season_amp == 0.0:
        return np.float32(1.0)
    phase = 2.0 * math.pi * (tick % cfg.season_period) / cfg.season_period
    return np.float32(1.0 + cfg.season_amp * math.sin(phase))


def growth_multiplier(cfg: Config, tick: int) -> np.float32:
    """Effective food-growth scale for this tick: S(t) * L(t), >= 0.

    Not clamped above 1 — the boom is real; ``phase_grow_food`` clamps the
    resulting *rate* instead so the refill stays a proper fraction of the
    deficit (no overshoot, no negative food).
    """
    product = float(season_multiplier(cfg, tick)) * float(light_level(cfg, tick))
    return np.float32(max(product, 0.0))
