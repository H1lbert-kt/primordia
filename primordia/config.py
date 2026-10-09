"""All balance parameters live here; the rest of the code must not contain magic numbers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
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
    max_age: int = 2000  # ticks; death when age >= max_age

    # --- metabolism ---
    metabolic_cost: float = 0.08  # energy drained per tick while alive

    # --- food (energy per grid cell) ---
    # Stage 9.5 calibration (seeds 42/7 x 6000 ticks): supply
    # growth*K*cells ~= demand lets famine regulate the population below
    # the cap instead of saturating it. energy per cell is small on purpose:
    # a cell is a meal, not a bank.
    # Stage 10 recalibration: mean light (0.75) x mean terrain fertility
    # (~0.43) cut effective supply ~60%, so growth went 0.02 -> 0.12
    # (sweep 0.04-0.27, seeds 42/7 x 6000 ticks; 0.12 restored the stage
    # 9.5 pop band without pinning the cap).
    food_capacity: float = 3.0  # max energy stored in one cell
    food_growth_rate: float = 0.12  # fraction of the deficit refilled per tick
    initial_food: float = 2.0  # energy per cell at t=0
    eat_rate: float = 1.0  # max energy a creature extracts per tick

    # --- food patches (stage 9: persistent foraging landscape) ---
    # Local carrying capacity K per cell: value noise in
    # [1 - amplitude, 1] * capacity. Growth refills toward K, not toward the
    # global capacity, so rich and poor patches persist instead of the world
    # flattening into a uniform sea of food. amplitude = 0 restores the
    # uniform field exactly.
    food_patch_amplitude: float = 0.85  # 0 = uniform K at food_capacity
    food_patch_scale: float = 60.0  # world units between noise lattice points

    # --- terrain (stage 10: geography creates niches) ---
    # Value noise in [0, 1]; amplitude is the land share: cells whose raw
    # noise falls below 1 - amplitude become water (stored as exact 0.0),
    # land fertility is rescaled to (0, 1]. Local growth target is
    # food_field * terrain, so water never grows food and drains what it
    # has. amplitude = 0 turns terrain off (flat fertility 1.0, no water).
    terrain_amplitude: float = 0.85  # 0 = terrain disabled, ~15% water at 0.85
    terrain_scale: float = 80.0  # world units between noise lattice points
    # Swimming is not free: move cost multiplier on water cells (>= 1).
    # A physical law, same framing as move_cost/turn_cost — the amphibious
    # niche (cheap empty water vs rich costly land) has to emerge.
    water_move_cost: float = 3.0

    # --- season (stage 10: slow boom/bust on food growth) ---
    # growth multiplier S(t) = 1 + season_amp * sin(2*pi*t / season_period),
    # mean 1, so the cycle redistributes food in time instead of adding it.
    season_period: int = 8000  # ticks between season peaks
    season_amp: float = 0.5  # 0 = no seasons

    # --- day/night (stage 10: light drives photosynthesis and vision) ---
    # L(t) = 1 - day_amp * 0.5 * (1 - cos(2*pi*t / day_period)) in
    # [1 - day_amp, 1]: noon at t = 0 (mod period). Food growth and the
    # effective ray reach are both proportional to L; creatures also sense
    # it (the "light" brain input), so circadian behavior can evolve.
    day_period: int = 400  # ticks per full day
    day_amp: float = 0.5  # 0 = constant full light

    # --- sensors (ray fan + smell) ---
    n_rays: int = 7
    field_of_view: float = math.pi  # radians covered by the ray fan
    vision_range: float = 50.0  # default genome trait: how far rays see
    smell_radius_cells: int = 1  # (2r+1)^2 cells sensed around the creature

    # --- brain ---
    # 10 keeps brain_params at 433 (with the 29 inputs + W_rec), inside the
    # 100-500 parameter budget; 12 would push it well beyond.
    hidden_size: int = 10
    weight_init_std: float = 0.5  # std of N(0, std) weights drawn at spawn

    # --- locomotion (cinematic: vel = dir(angle) * max_speed * accel) ---
    max_speed: float = 4.0  # default genome trait: world units per tick
    max_turn: float = 0.3  # radians per tick at |turn| = 1 (inertia)

    # --- reproduction and mutation (stage 3) ---
    reproduce_threshold: float = 100.0  # energy needed to split in two
    child_fraction: float = 0.5  # share of parent's energy given to child
    mutation_rate: float = 0.01  # probability per gene of being mutated
    mutation_std: float = 0.05  # additive std for brain genes
    trait_mutation_std: float = 0.10  # relative std for multiplicative traits

    # --- genealogy (stage 7): pre-allocated birth log buffer ---
    genealogy_capacity: int = 100_000  # rows of (tick, child_id, parent_id); 0 = off

    # --- movement cost ---
    move_cost: float = 0.05  # energy per tick at speed 1 (quadratic in |vel|)
    # Steering is not free (stage 9.5): energy drained per radian actually
    # turned, i.e. turn_cost * |turn| * max_turn. Same physical-law framing
    # as move_cost — perpetual spinning starves.
    turn_cost: float = 0.3

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
        if not 0.0 <= self.food_patch_amplitude <= 1.0:
            raise ValueError("food_patch_amplitude must be in [0, 1]")
        if self.food_patch_scale <= 0.0:
            raise ValueError("food_patch_scale must be positive")
        if not 0.0 <= self.terrain_amplitude <= 1.0:
            raise ValueError("terrain_amplitude must be in [0, 1]")
        if self.terrain_scale <= 0.0:
            raise ValueError("terrain_scale must be positive")
        if self.water_move_cost < 1.0:
            raise ValueError("water_move_cost must be >= 1 (water never cheaper)")
        if self.season_period < 2 or self.day_period < 2:
            raise ValueError("season_period and day_period must be >= 2 ticks")
        if not 0.0 <= self.season_amp <= 1.0:
            raise ValueError("season_amp must be in [0, 1]")
        if not 0.0 <= self.day_amp <= 1.0:
            raise ValueError("day_amp must be in [0, 1]")
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
        if self.turn_cost < 0.0:
            raise ValueError("turn_cost must be >= 0")
        if self.bite_rate < 0.0:
            raise ValueError("bite_rate must be >= 0")
        if not 0.0 <= self.bite_efficiency <= 1.0:
            raise ValueError("bite_efficiency must be in [0, 1]")
        if self.contact_range <= 0.0:
            raise ValueError("contact_range must be positive")
        if self.max_size < 1.0:
            raise ValueError("max_size must be >= 1.0 (spawn initializes size=1.0)")
        if self.genealogy_capacity < 0:
            raise ValueError("genealogy_capacity must be >= 0")

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Food grid dimensions as (height, width) in cells."""
        return (
            int(math.ceil(self.height / self.cell_size)),
            int(math.ceil(self.width / self.cell_size)),
        )

    @property
    def sensor_input_dim(self) -> int:
        """Inputs: 3 ray channels (food sum, creatures, food peak), 3 smells,
        2 food antennas (left/right of heading) and 3 internal sensors
        (energy, speed, ambient light — stage 10's circadian cue)."""
        return self.n_rays * 3 + 8

    @property
    def brain_params(self) -> int:
        """MLP + recurrence: W1 (hid, in) + b1 + W2 (out, hid) + b2
        + W_rec (hid, hid) — the Elman recurrent weights, no other state."""
        return (
            self.hidden_size * (self.sensor_input_dim + 1)
            + self.N_OUTPUTS * (self.hidden_size + 1)
            + self.hidden_size * self.hidden_size
        )

    @property
    def genome_size(self) -> int:
        """Exact genome rows: brain parameters followed by body traits."""
        return self.brain_params + self.N_BODY_TRAITS


def load_config(path: str, base: Config | None = None) -> Config:
    """Build a Config from a JSON overrides file applied on top of ``base``.

    The file is a flat object of Config field names (stage-8 ``--config``);
    fields it does not mention keep their value from ``base`` (or the
    defaults). Unknown names fail with :class:`ValueError` so a typo cannot
    silently create a dead option, and the merged result still goes through
    ``Config.__post_init__`` validation.
    """
    with open(path, encoding="utf-8") as fh:
        overrides = json.load(fh)
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: expected a JSON object of Config fields")
    allowed = {f.name for f in fields(Config)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown Config fields {unknown}")
    merged = asdict(base or Config()) | overrides
    try:
        return Config(**merged)
    except TypeError as exc:
        raise ValueError(f"{path}: bad override value ({exc})") from exc
