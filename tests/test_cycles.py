"""Stage 10 cycle tests: light and season as pure functions of (cfg, tick)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from primordia.config import Config
from primordia.cycles import growth_multiplier, light_level, season_multiplier


def make_config(**overrides: object) -> Config:
    return replace(Config(), **overrides)


def test_cycles_off_are_identity() -> None:
    cfg = make_config(day_amp=0.0, season_amp=0.0)
    for tick in (0, 1, 199, 400, 12345):
        assert float(light_level(cfg, tick)) == 1.0
        assert float(season_multiplier(cfg, tick)) == 1.0
        assert float(growth_multiplier(cfg, tick)) == 1.0


def test_light_level_phases() -> None:
    cfg = make_config()  # day_amp 0.5, period 400
    assert float(light_level(cfg, 0)) == 1.0  # noon
    assert float(light_level(cfg, 200)) == 0.5  # midnight = 1 - amp
    assert float(light_level(cfg, 100)) == 0.75  # quarter day
    values = [float(light_level(cfg, t)) for t in range(cfg.day_period)]
    assert min(values) >= 0.5 - 1e-6 and max(values) <= 1.0 + 1e-6
    # periodic: the law depends only on tick % period
    assert float(light_level(cfg, 7)) == float(light_level(cfg, 7 + 400))


def test_light_full_amplitude_reaches_zero() -> None:
    cfg = make_config(day_amp=1.0)
    assert float(light_level(cfg, 200)) == 0.0


def test_season_multiplier_phases() -> None:
    cfg = make_config()  # season_amp 0.5, period 8000
    assert float(season_multiplier(cfg, 0)) == 1.0
    assert float(season_multiplier(cfg, 2000)) == 1.5  # season peak
    assert float(season_multiplier(cfg, 6000)) == 0.5  # season trough
    values = np.array(
        [float(season_multiplier(cfg, t)) for t in range(cfg.season_period)]
    )
    # mean 1: the cycle redistributes food in time, it does not add any
    assert abs(values.mean() - 1.0) < 1e-3
    assert float(season_multiplier(cfg, 999)) == float(
        season_multiplier(cfg, 999 + 8000)
    )


def test_growth_multiplier_is_the_product() -> None:
    cfg = make_config(season_amp=0.5, day_amp=0.5)
    tick = 123
    expected = float(season_multiplier(cfg, tick)) * float(light_level(cfg, tick))
    assert float(growth_multiplier(cfg, tick)) == float(np.float32(expected))

    # S * L can exceed 1 (strong season at noon): the multiplier is the raw
    # product — clamping it here would kill the seasonal *boom* and leave
    # only the bust. The applied-rate clamp lives in phase_grow_food
    # (pinned by test_environment.test_growth_rate_never_overshoots).
    hot = make_config(season_amp=1.0, day_amp=1.0, season_period=4, day_period=16)
    assert float(growth_multiplier(hot, 1)) == pytest.approx(
        2.0 * float(light_level(hot, 1)), rel=1e-5
    )
