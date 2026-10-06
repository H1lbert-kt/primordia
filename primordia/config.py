"""All balance parameters live here; the rest of the code must not contain magic numbers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Immutable balance parameters for a world.

    Every tunable number of the simulation is a field here so experiments can
    pass a modified Config instead of editing code.
    """

    # --- world geometry ---
    width: float = 1000.0
    height: float = 1000.0
    cell_size: float = 5.0  # edge length of one food cell (world units)

    # --- population ---
    max_creatures: int = 2000  # fixed array capacity (SoA allocation)
    initial_creatures: int = 500
    initial_energy: float = 50.0
    max_age: int = 5000  # ticks; death when age >= max_age

    # --- metabolism ---
    metabolic_cost: float = 0.05  # energy drained per tick while alive

    # --- food (energy per grid cell) ---
    food_capacity: float = 100.0  # max energy stored in one cell
    food_growth_rate: float = 0.05  # fraction of the deficit refilled per tick
    initial_food: float = 60.0  # energy per cell at t=0
    eat_rate: float = 4.0  # max energy a creature extracts per tick

    # --- genome ---
    genome_size: int = 128  # placeholder rows per creature; exact layout (brain
    # weights + body traits) is defined in stages 2-3

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("world dimensions must be positive")
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if not 0 <= self.initial_creatures <= self.max_creatures:
            raise ValueError("initial_creatures must be in [0, max_creatures]")
        if self.max_creatures < 1 or self.genome_size < 1 or self.max_age < 1:
            raise ValueError("max_creatures, genome_size and max_age must be >= 1")
        if self.initial_energy < 0.0 or self.metabolic_cost < 0.0:
            raise ValueError("initial_energy and metabolic_cost must be >= 0")
        if self.food_capacity < 0.0 or self.initial_food > self.food_capacity:
            raise ValueError("need 0 <= initial_food <= food_capacity")
        if not 0.0 <= self.food_growth_rate <= 1.0:
            raise ValueError("food_growth_rate must be in [0, 1]")
        if self.eat_rate <= 0.0:
            raise ValueError("eat_rate must be positive")

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Food grid dimensions as (height, width) in cells."""
        return (
            int(math.ceil(self.height / self.cell_size)),
            int(math.ceil(self.width / self.cell_size)),
        )
