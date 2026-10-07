"""Presentation settings for the pygame viewer.

These are window geometry, colors and input feel — never simulation balance
numbers (those live in ``primordia.config.Config``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Everything the viewer needs that is not simulation balance."""

    window_width: int = 1280
    window_height: int = 720
    panel_width: int = 320
    target_fps: int = 60

    min_zoom: float = 0.1  # pixels per world unit
    max_zoom: float = 10.0
    initial_zoom: float = 0.85
    pick_radius_px: float = 14.0  # click tolerance when selecting
    pan_drag_px: float = 4.0  # movement before a drag counts as pan

    # palette (RGB)
    background: tuple[int, int, int] = (10, 12, 18)
    panel_bg: tuple[int, int, int] = (22, 26, 34)
    panel_border: tuple[int, int, int] = (60, 68, 84)
    text: tuple[int, int, int] = (210, 214, 222)
    text_dim: tuple[int, int, int] = (130, 136, 150)
    bar_bg: tuple[int, int, int] = (40, 46, 58)
    bar_pos: tuple[int, int, int] = (90, 200, 160)
    bar_neg: tuple[int, int, int] = (220, 110, 90)
    selection: tuple[int, int, int] = (255, 255, 255)
    ray_food: tuple[int, int, int] = (240, 210, 90)
    ray_creature: tuple[int, int, int] = (200, 120, 230)
    smell_ring: tuple[int, int, int] = (90, 150, 230)
    # Blue reads clearly against the green food palette and against the
    # red predator tint (hue distance, not energy-dependent).
    creature_herbivore: tuple[int, int, int] = (60, 140, 255)
    creature_predator: tuple[int, int, int] = (230, 90, 70)
    food_low: tuple[int, int, int] = (8, 16, 10)
    food_mid: tuple[int, int, int] = (28, 110, 55)
    food_high: tuple[int, int, int] = (150, 245, 150)

    # creature scatter: pixel radius = clamp(round(size * zoom * size_scale), 1, max)
    size_scale: float = 2.4
    max_radius_px: int = 6

    # simulation speed: ticks advanced per rendered frame
    speed_steps: tuple[int, ...] = (1, 2, 4, 8)

    heat_pos: tuple[int, int, int] = (230, 90, 80)
    heat_neg: tuple[int, int, int] = (80, 140, 235)
    heat_zero: tuple[int, int, int] = (32, 36, 46)

    @property
    def viewport_width(self) -> int:
        """World viewport width in pixels (window minus the side panel)."""
        return self.window_width - self.panel_width

    @property
    def viewport_height(self) -> int:
        return self.window_height
