"""All balance parameters live here; the rest of the code must not contain magic numbers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Config:
    """Immutable balance parameters for a world.

    Every tunable number of the simulation is a field here so experiments can
    pass a modified Config instead of editing code. Derived sizes (input
    dimension, parameter count, genome rows) are properties so the layout has
    a single source of truth.
    """

    N_OUTPUTS: ClassVar[int] = 3  # accel, turn, eat-gate
    N_BODY_TRAITS: ClassVar[int] = 4  # max_speed, size, vision_range, diet

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

    # --- sensors (ray fan + smell) ---
    n_rays: int = 7
    field_of_view: float = math.pi  # radians covered by the ray fan
    vision_range: float = 50.0  # default genome trait: how far rays see
    smell_radius_cells: int = 1  # (2r+1)^2 cells sensed around the creature

    # --- brain ---
    hidden_size: int = 12
    weight_init_std: float = 0.5  # std of N(0, std) weights drawn at spawn

    # --- locomotion (cinematic: vel = dir(angle) * max_speed * accel) ---
    max_speed: float = 4.0  # default genome trait: world units per tick
    max_turn: float = 0.6  # radians per tick at |turn| = 1

    # --- reproduction and mutation (stage 3) ---
    reproduce_threshold: float = 100.0  # energy needed to split in two
    child_fraction: float = 0.5  # share of parent's energy given to child
    mutation_rate: float = 0.01  # probability per gene of being mutated
    mutation_std: float = 0.05  # additive std for brain genes
    trait_mutation_std: float = 0.10  # relative std for multiplicative traits

    # --- movement cost ---
    move_cost: float = 0.05  # energy per tick at speed 1 (quadratic in |vel|)

    # --- predation (stage 5): capability comes from the diet trait ---
    bite_rate: float = 4.0  # max energy an attacker drains per tick (budget)
    bite_efficiency: float = 0.7  # share of drained energy kept (rest dissipates)
    contact_range: float = 1.5  # contact = this * (size_a + size_v), world units
    max_size: float = 4.0  # clamp on the size trait (spawn starts at 1.0)

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("world dimensions must be positive")
        if self.cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if not 0 <= self.initial_creatures <= self.max_creatures:
            raise ValueError("initial_creatures must be in [0, max_creatures]")
        if self.max_creatures < 1 or self.max_age < 1:
            raise ValueError("max_creatures and max_age must be >= 1")
        if self.initial_energy < 0.0 or self.metabolic_cost < 0.0:
            raise ValueError("initial_energy and metabolic_cost must be >= 0")
        if self.food_capacity < 0.0 or self.initial_food > self.food_capacity:
            raise ValueError("need 0 <= initial_food <= food_capacity")
        if not 0.0 <= self.food_growth_rate <= 1.0:
            raise ValueError("food_growth_rate must be in [0, 1]")
        if self.eat_rate <= 0.0:
            raise ValueError("eat_rate must be positive")
        if self.n_rays < 1 or self.hidden_size < 1:
            raise ValueError("n_rays and hidden_size must be >= 1")
        if not 0.0 < self.field_of_view <= 2.0 * math.pi:
            raise ValueError("field_of_view must be in (0, 2*pi]")
        if self.vision_range <= 0.0 or self.max_speed <= 0.0:
            raise ValueError("vision_range and max_speed must be positive")
        if not 0.0 < self.max_turn <= math.pi:
            raise ValueError("max_turn must be in (0, pi]")
        if self.weight_init_std < 0.0 or self.smell_radius_cells < 0:
            raise ValueError("weight_init_std and smell_radius_cells must be >= 0")
        if self.reproduce_threshold <= 0.0:
            raise ValueError("reproduce_threshold must be positive")
        if not 0.0 < self.child_fraction < 1.0:
            raise ValueError("child_fraction must be in (0, 1)")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.mutation_std < 0.0 or self.trait_mutation_std < 0.0:
            raise ValueError("mutation_std and trait_mutation_std must be >= 0")
        if self.move_cost < 0.0:
            raise ValueError("move_cost must be >= 0")
        if self.bite_rate < 0.0:
            raise ValueError("bite_rate must be >= 0")
        if not 0.0 <= self.bite_efficiency <= 1.0:
            raise ValueError("bite_efficiency must be in [0, 1]")
        if self.contact_range <= 0.0:
            raise ValueError("contact_range must be positive")
        if self.max_size < 1.0:
            raise ValueError("max_size must be >= 1.0 (spawn initializes size=1.0)")

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Food grid dimensions as (height, width) in cells."""
        return (
            int(math.ceil(self.height / self.cell_size)),
            int(math.ceil(self.width / self.cell_size)),
        )

    @property
    def sensor_input_dim(self) -> int:
        """Per-creature input vector: ray food + ray creatures + smell + internal."""
        return self.n_rays * 2 + 3 + 2

    @property
    def brain_params(self) -> int:
        """MLP weights and biases: W1 (hid, in) + b1 + W2 (out, hid) + b2."""
        return self.hidden_size * (self.sensor_input_dim + 1) + self.N_OUTPUTS * (
            self.hidden_size + 1
        )

    @property
    def genome_size(self) -> int:
        """Exact genome rows: brain parameters followed by body traits."""
        return self.brain_params + self.N_BODY_TRAITS
