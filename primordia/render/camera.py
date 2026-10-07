"""Camera: world <-> screen mapping for the torus, with a cached cell lookup.

Screen positions use the shortest toroidal delta (``(p - c + m/2) % m - m/2``)
so creatures and food appear seamlessly across world edges. The screen->cell
index arrays used by the food layer are cached and only recomputed when the
camera moves.
"""

from __future__ import annotations

import numpy as np

from .settings import Settings


class Camera:
    """Viewport camera: center (world coords), zoom (pixels per world unit)."""

    def __init__(
        self,
        settings: Settings,
        world_width: float,
        world_height: float,
        cell_w: float,
        cell_h: float,
        gw: int,
        gh: int,
    ) -> None:
        self.settings = settings
        self.world_width = float(world_width)
        self.world_height = float(world_height)
        self.cell_w = float(cell_w)
        self.cell_h = float(cell_h)
        self.gw = int(gw)
        self.gh = int(gh)
        self.viewport = (settings.viewport_width, settings.viewport_height)
        self.center = np.array(
            [self.world_width / 2.0, self.world_height / 2.0], dtype=np.float64
        )
        self.zoom = settings.initial_zoom
        self._cell_x: np.ndarray | None = None
        self._cell_y: np.ndarray | None = None
        self._flat_idx: np.ndarray | None = None

    def invalidate(self) -> None:
        """Drop the cached screen->cell lookup (camera moved)."""
        self._cell_x = None
        self._cell_y = None
        self._flat_idx = None

    def world_to_screen(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(M,2) world coords -> viewport pixel coordinates (floats)."""
        vw, vh = self.viewport
        w, h = self.world_width, self.world_height
        dx = (xy[:, 0] - self.center[0] + w / 2.0) % w - w / 2.0
        dy = (xy[:, 1] - self.center[1] + h / 2.0) % h - h / 2.0
        return vw / 2.0 + dx * self.zoom, vh / 2.0 + dy * self.zoom

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Viewport pixel -> world coords, wrapped into [0, world)."""
        vw, vh = self.viewport
        wx = (self.center[0] + (sx - vw / 2.0) / self.zoom) % self.world_width
        wy = (self.center[1] + (sy - vh / 2.0) / self.zoom) % self.world_height
        return float(wx), float(wy)

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """Multiply zoom by ``factor`` keeping the world point under the cursor."""
        vw, vh = self.viewport
        lo, hi = self.settings.min_zoom, self.settings.max_zoom
        new_zoom = min(max(self.zoom * factor, lo), hi)
        if new_zoom == self.zoom:
            return
        wx = self.center[0] + (sx - vw / 2.0) / self.zoom
        wy = self.center[1] + (sy - vh / 2.0) / self.zoom
        self.zoom = new_zoom
        self.center[0] = (wx - (sx - vw / 2.0) / new_zoom) % self.world_width
        self.center[1] = (wy - (sy - vh / 2.0) / new_zoom) % self.world_height
        self.invalidate()

    def pan(self, dsx: float, dsy: float) -> None:
        """Shift the center so the dragged content follows the pointer."""
        self.center[0] = (self.center[0] - dsx / self.zoom) % self.world_width
        self.center[1] = (self.center[1] - dsy / self.zoom) % self.world_height
        self.invalidate()

    def follow(self, xy: np.ndarray) -> None:
        """Center on a world position (used by follow-cam)."""
        self.center[0] = float(xy[0]) % self.world_width
        self.center[1] = float(xy[1]) % self.world_height
        self.invalidate()

    def cell_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached 1D cell indices per viewport row and column (torus-wrapped)."""
        if self._cell_x is None:
            vw, vh = self.viewport
            sx = np.arange(vw, dtype=np.float64)
            sy = np.arange(vh, dtype=np.float64)
            wx = (self.center[0] + (sx - vw / 2.0) / self.zoom) % self.world_width
            wy = (self.center[1] + (sy - vh / 2.0) / self.zoom) % self.world_height
            cx = (wx / self.cell_w).astype(np.int32)
            cy = (wy / self.cell_h).astype(np.int32)
            np.clip(cx, 0, self.gw - 1, out=cx)
            np.clip(cy, 0, self.gh - 1, out=cy)
            self._cell_x = cx
            self._cell_y = cy
        return self._cell_y, self._cell_x

    def flat_indices(self) -> np.ndarray:
        """Cached flattened food-grid index per viewport pixel.

        Layout is image-major ``(viewport_w, viewport_h)`` in the native
        ``intp`` index dtype (``np.take`` would cast smaller integers and
        copy megabytes per frame). Recomputed only when the camera moves.
        """
        if self._flat_idx is None:
            cy, cx = self.cell_indices()
            self._flat_idx = cx[:, None].astype(np.intp) + (
                cy.astype(np.intp) * np.intp(self.gw)
            )[None, :]
        return self._flat_idx
