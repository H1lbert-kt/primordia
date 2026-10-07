"""Pygame rendering layer — never imported by the headless core."""

from __future__ import annotations

from .app import run_app
from .settings import Settings

__all__ = ["run_app", "Settings"]
