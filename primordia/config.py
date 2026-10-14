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

    N_OUTPUTS: ClassVar[int] = 8  # accel, turn, eat, emit, donate, gather,
    # build_fire, build_house (stage 17 added outputs 5-7)
    N_BODY_TRAITS: ClassVar[int] = 6  # max_speed, size, vision_range, diet,
    # tag_hue, plasticity_eta (stage 21: evolvable learning rate)

    # --- world geometry ---
    width: float = 1000.0
    height: float = 1000.0
    cell_size: float = 5.0  # edge length of one food cell (world units)

    # --- population ---
    max_creatures: int = 2000  # fixed array capacity (SoA allocation)
    initial_creatures: int = 500
    initial_energy: float = 50.0
    # Stage 11: senescence (wear exp(senescence_rate * age)) is the real
    # limit on lifespan now; max_age stays as a high safety ceiling so the
    # death-age histogram reflects wear/famine, not this cliff.
    max_age: int = 5000  # ticks; death when age >= max_age

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
    # Stage 25: 38 inputs (the signed store reading appended) fit the
    # 100-500 budget only at hidden 8 — brain_params = 8*39 + 8*9 + 8*8 =
    # 448, genome = 454. Hidden 9 would cost 9*39 + 8*10 + 9*9 = 512, over
    # budget.
    hidden_size: int = 8
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

    # --- sex and senescence (stage 11: real diversity, disposable soma) ---
    # Reproduction is sexual: an initiator above the threshold needs a
    # partner alive within mate_range (world units); each parent pays half
    # of the child's energy and the child's genome is a per-gene uniform
    # crossover plus mutation. No partner -> no birth: mobility and
    # aggregation get selected without being programmed.
    mate_range: float = 3.0
    # Wear factor w = exp(senescence_rate * age), applied at *use*, never
    # inherited: effective speed and vision shrink by 1/w while the
    # metabolic bill grows by w. 0 = no senescence (stage-10 behavior).
    # Fase-E sweep (seeds 42/7, 6000 ticks): 0.002 -> extinction, 0.001 ->
    # collapse to ~73-126 alive, 0.0005 -> stable 277-417 with predation.
    senescence_rate: float = 0.0005

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

    # --- carrion (stage 11: nutrients return to the ground) ---
    # Death deposits meat on its cell: predation converts the bite share
    # that would dissipate as heat (yield * (1 - bite_efficiency) * take),
    # old age deposits the leftover body energy (yield * max(energy, 0)),
    # famine deposits ~0 (a starved body has nothing to leave). Eaters
    # drain food first, then meat; meat decays by meat_decay per tick.
    carrion_yield: float = 0.3
    meat_decay: float = 0.02

    # --- signal channel (stage 13: one common scalar channel, no reward) ---
    # A creature may deposit signal on its own cell (emit gate, output 3)
    # and smell the sum over the cheiro window. The tick never rewards
    # communicating — whether anyone uses the channel is an experiment
    # question, not a rule. Stage 14 (pheromone): each tick the field also
    # spreads over its 4 toroidal neighbors (Jacobi pass) before the
    # sources emit, turning point deposits into extended trails.
    # Default signal_emit_cost is 0: the emit gate is a sigmoid that never
    # reaches exactly 0, so a nonzero default would tax every creature
    # ~cost/2 per tick and silently change the whole economy. The cost is
    # a treatment applied by experiments (configs/proto_culture.toml).
    signal_decay: float = 0.05  # per-tick multiplicative decay in [0, 1)
    emit_rate: float = 1.0  # signal units deposited per tick at gate=1
    signal_emit_cost: float = 0.0  # energy drained per tick at gate=1
    signal_capacity: float = 1.0  # perception normalizer (like food_capacity)
    # Jacobi weight: new = old + rate * (N+S+E+W - 4*old). Mass-conserving;
    # stable only for rate <= 0.25 (the highest Laplacian mode is critically
    # damped there, 1 + rate*(-4) = 0). 0 disables diffusion exactly.
    signal_diffuse: float = 0.1

    # --- donation (stage 16: physical energy transfer, never a reward) ---
    # The donate gate (output 4) is a sigmoid that never reaches exactly 0,
    # so the default rate is 0 — same reasoning as signal_emit_cost: a
    # nonzero default would drain every creature from birth. Experiments
    # set the price; the recipient is the nearest living neighbour inside
    # donate_range (the mate search's rule), chosen by physics, not by us.
    donate_rate: float = 0.0  # energy gifted per tick at gate=1
    donate_range: float = 3.0  # world units, nearest living recipient

    # --- civilization base (stage 17: wood, fire, houses) ---
    # Wood is a second ground field like food: value-noise patches, harvest
    # depletes, growth refills toward K (and never on water — terrain gates
    # it like food). Nothing rewards building: the bonus lives in the object
    # (a campfire cooks, a house shelters) and every step is paid in energy
    # or carried wood, so whether constructing pays is selection's question.
    wood_capacity: float = 2.0  # max wood stored in one cell
    wood_growth_rate: float = 0.01  # fraction of the deficit refilled per tick
    initial_wood: float = 0.0  # wood per cell at t=0 (0 <= this <= capacity)
    wood_patch_amplitude: float = 1.0  # 0 = uniform K at wood_capacity
    wood_patch_scale: float = 60.0  # world units between noise lattice points

    # gather gate (output 5): wood harvested per tick at gate=1, clamped by
    # both what the cell holds and the carry capacity — hauling wood costs
    # move_cost like everything else, so transport is physics, not a rule
    gather_rate: float = 0.5
    max_inventory: float = 4.0  # wood a creature can carry at once

    # structures (one per cell; a fire burns out without fuel, a house
    # persists). Units: 1 wood = 1 fuel = 1 energy (house price calibrated
    # in the stage-17 pilot, fire price in the stage-18 default sweep;
    # documented so the cook accounting reads honestly)
    fire_wood_cost: float = 3.0  # wood consumed to build or refuel a fire
    # Stage-18 default sweep (seeds 42/7 x 5k, houses priced out): mean
    # final pop 408 @5.0, 145 @10.0, 60 @20.0 — cheaper fires drain less
    # per build while the cook bonus returns more, and the gate (3 seeds
    # x 10k) ended 1163/1170/821 alive vs 3/86/70 at the old 10.0/20.0.
    fire_energy_cost: float = 5.0  # energy consumed to build a fire
    # a house is raised in INCREMENTS (stage 22): the totals below are
    # paid in wood deliveries (all-or-nothing per delivery, never per
    # house) and the shelter scales with completeness u in [0, 1]
    # (stored in struct_store). Stage 17's lump price is gone.
    house_wood_cost: float = 4.0  # total wood of a complete house
    # Total energy of a complete house, charged as energy_cost * du per
    # delivery. The stage-18 prohibitive 1e6 default died with the lump
    # design: houses are off by default through `houses_enabled` now.
    house_energy_cost: float = 3.0
    # Off by default (stage 22): the default Config is the baseline the
    # older experiments measured; presets (configs/houses.toml) enable
    # houses. False = the build_house gate is a no-op.
    houses_enabled: bool = False
    # Incomplete houses rot: u -= house_rot_rate * (1 - u) per tick; at
    # u <= 0 the structure dies and its slot is recycled. Complete
    # houses (u = 1) never decay (stage-17 rule, now scoped to finished
    # houses only).
    house_rot_rate: float = 0.01
    # If > 0, every house delivery requires a live fire within this
    # radius (world units, toroidal) of the building cell — the fire is
    # the step before the house. 0 = no requirement.
    house_fire_radius: float = 0.0
    # Hauling tax (stage 22): movement costs
    # move_cost * |v|^2 * (1 + carry_weight * inventory/max_inventory).
    # 0 = carrying is free, bit-identical to the pre-stage-22 world.
    carry_weight: float = 0.0
    # Spawn-time prior on the build_house output logit (b2[7]): a
    # negative value makes attempting a house rare at birth; mutation
    # can evolve it up (selection decides, nothing is hard-coded).
    # 0 = the old N(0, weight_init_std) prior, bit-identical worlds.
    build_house_bias: float = 0.0
    # Offspring placement: child = parent + N(0, birth_spread) with
    # toroidal wrap; 0 = born on the parent's cell (pre-stage-22
    # behaviour, no RNG draw). Local parentage means a house cluster can
    # become a multi-generational neighbourhood.
    birth_spread: float = 0.0
    fire_initial_fuel: float = 3.0  # fuel charged when a fire is built
    fire_fuel_decay: float = 0.002  # fuel burned per tick; fire dies at 0
    fire_radius: float = 6.0  # world units the heat reaches (cook + glow)
    house_radius: float = 5.0  # world units of bite cover (shelter)
    # Cooking: food eaten on a heated cell yields cook_bonus x energy, and
    # the extra is drained 1:1 from the covering fire's fuel — the bonus is
    # burned wood, never created energy (stage-17 conservation law).
    cook_bonus: float = 1.5
    cover_reduction: float = 0.5  # share of bite efficiency lost at full cover
    max_structures: int = 1024  # fixed structure capacity (SoA allocation)

    # --- species clustering (stage 12: metric only, never read by the tick) ---
    # Normalized genome distance at or below which two living creatures are
    # counted as the same cluster by ecology.cluster_living. Calibrated at
    # 3000 ticks (seeds 42/7/123, alive 814-1294): pairwise distances are
    # bimodal, median ~7.5 with a far tail out to ~370, so K is flat over
    # [8, 50] and lands 5-15 inside the target band. 1.0 would return K ~= N
    # (every creature its own cluster). Greedy-leader (not single-linkage):
    # a big central cluster plus distant outliers each get their own label,
    # so K tracks the outliers, not a transitive chain.
    species_threshold: float = 10.0

    # --- observer groups (stage 23: measurement only, never read by the tick) ---
    # Circular distance in tag_hue at or below which two living creatures
    # count as the same observer group (viewer "countries", experiment
    # metrics). Hue is the lineage colour of stage 15: inherited from one
    # parent, mutated slowly around the wheel, pinned per founding region
    # by founder_regions below — so hue groups track lineages, while
    # genome-distance species clustering stays its own metric (with random
    # brains at spawn it returns ~one cluster per creature). Greedy-leader
    # on the circle, same first-match convention as species_threshold.
    group_hue_threshold: float = 0.05

    # --- founder scenario (stage 23: initial condition only, never a rule) ---
    # 0 = off: spawn draws the pre-stage-23 RNG stream bit-for-bit. K > 0
    # splits the initial population into K vertical strips of the torus;
    # each strip gets tag_hue centred on k/K with +- founder_hue_spread
    # jitter, so the observer's hue groups start as the founding lineages
    # and contact between strips produces borders. Every region needs >= 2
    # creatures: sexual reproduction only finds a mate inside mate_range.
    founder_regions: int = 0
    founder_hue_spread: float = 0.02

    # --- synaptic plasticity (stage 21: local Oja, no reward, Baldwin) ---
    # Weights never mutate inside a life: the rule writes only into the
    # scratch ``dw``, zeroed at birth and NOT inherited, so genome_dist and
    # the speciation metrics keep measuring heritable divergence alone
    # (Lamarck inheritance would blur them; it stays a future opt-in flag).
    # Effective synapses are W_eff = genome + dw; an explicit decay pulls dw
    # back to 0 and a clamp bounds |dw|, so evolved weights are never erased.
    # The Oja term -h^2 * W_eff only acts while the postsynaptic neuron is
    # active (h ~ 0 => step ~ 0) — it is not idle-synapse atrophy.
    # mode: "off" (no plasticity, forward skips dw -> bit-identical to the
    # pre-stage-21 baseline), "fixed" (rate below), "gene" (per-creature eta
    # read from the last body trait, spawned at 0 — selection decides
    # whether to pay plasticity_cost to turn learning on).
    # Lifetime plasticity (stage 21). Default "off": the pre-stage-21 world
    # is the baseline old experiments were measured on; "gene" (the
    # evolvable eta) is pinned in configs/proto_culture.toml.
    plasticity_mode: str = "off"
    # Fixed-mode learning rate; experiments sweep {0, 1e-4, 1e-3, 1e-2}
    # instead of trusting a single default (stage-21 decision).
    plasticity_rate: float = 0.0
    plasticity_interval: int = 10  # apply every K ticks (>= 1)
    plasticity_clip: float = 5.0  # max |dw| per synapse (10 sigma of init)
    # Mutation scale of the evolvable eta gene. Its own parameter on
    # purpose: trait_std (0.10) would jump eta straight to ~0.1, ten times
    # the top of the viable sweep, and selection would purge the gene
    # before it ever explored {1e-4..1e-2} (stage-21 pilot: every alive
    # creature sat at eta=0 after 5k ticks). 0.01 centers mutations on the
    # sweep band so "does selection turn learning on?" is measurable.
    plasticity_eta_std: float = 0.01
    # Memory fades back to the blueprint when not reinforced: dw decays by
    # this fraction at every application (0 = only Oja's own term acts).
    plasticity_decay: float = 0.01
    # Energy drained per unit of |dw| moved in one application. The price
    # makes learning a real trade-off for the evolvable-eta experiment:
    # a gene that learns must earn its keep in survival, not for free.
    # 1.0 x typical |dw| move (~0.4 at eta=1e-2) ~ half the per-tick
    # metabolic bill amortized over plasticity_interval; calibrate in pilot.
    plasticity_cost: float = 1.0
    # Ablation (stage 21, default off): scale eta by the creature's own
    # energy change since the last tick, 1 + strength * tanh(dE). This is
    # LOCAL CREDIT ASSIGNMENT — the first signal in the sim that modulates
    # learning by payoff; kept off by default and declared as such.
    plasticity_credit: bool = False
    plasticity_credit_strength: float = 1.0

    # --- affect (stage 24: stress + grudge as pure information, no reward) ---
    # Being bitten fills two lifetime scratches the brain can read (stage-24
    # inputs 35/36): ``stress`` accumulates the landed energy and decays
    # exponentially; ``grudge`` is the circular hue similarity to the last
    # attacker faded by how long ago the bite was. Neither is inherited,
    # neither is saved (lifetime state, zeroed at birth like dw). There is
    # NO punishment/reward wiring: if fleeing or vengeance never pays for
    # itself in energy, the channel stays silent and the experiment reports
    # a null — that is the honest outcome, same contract as stage 16's
    # donation and stage 21's plasticity.
    # Default off: the pre-stage-24 default world is the baseline every
    # older experiment measured against (forward skips the two inputs,
    # they stay exact zeros). The stage-24 experiment turns it on.
    affect_enabled: bool = False
    # Per-tick multiplicative decay of the stress scratch: remaining stress
    # *= exp(-stress_decay). 0 = a bite scar never fades (documented
    # degenerate case; the input then only grows).
    stress_decay: float = 0.02
    # Normalizer of the stress input: min(stress, scale) / scale in [0, 1].
    # A full initial_energy (50) bite = stress 50 saturates any scale <= 50.
    stress_scale: float = 1.0
    # Per-tick decay of the grudge recency factor exp(-grudge_decay * ticks
    # since the bite). 0 = the last attacker is remembered perfectly forever
    # (the other degenerate case). Hue similarity is a physical fact of the
    # tag (stage 15): the brain sees who hit it, not a scoreboard.
    grudge_decay: float = 0.01

    # --- territory (stage 25: possession, savings and comfort — no reward) ---
    # A complete house (u = 1, immortal since stage 22) is a bank: energy
    # above ``reproduce_threshold * deposit_threshold_frac`` is moved into
    # ``struct_deposit`` automatically at ``deposit_rate`` per tick, floored
    # at the reproduce threshold (the creature keeps enough to split). The
    # eat gate (output 2) withdraws from the creature's OWN house and raids
    # a FOREIGN one at ``raid_rate`` — the flows are independent of the food
    # budget (an open mouth eats AND banks/raids; conservation holds). A
    # house whose owner creature is alive within ``guard_range`` cannot be
    # drained from outside: guarding costs opportunity, never energy.
    # Deposits above ``mark_threshold`` stamp a territorial mark ring
    # (binary 1.0 within house_radius, scratch rebuilt every tick); standing
    # on YOUR OWN mark discounts the metabolic bill by ``comfort`` — a
    # physical fact of place, never a reward function. Defaults keep the
    # pre-stage-25 economy: deposit_rate = 0 means no deposits ever exist,
    # so every territory reading is an exact zero (bit-identical world).
    deposit_rate: float = 0.5  # energy stored/withdrawn per tick at gate=1
    raid_rate: float = 0.3  # energy stolen per tick at gate=1 (unguarded)
    guard_range: float = 3.0  # world units: owner alive within blocks raids
    comfort: float = 0.3  # metabolic discount on own mark, in [0, 1]
    mark_threshold: float = 1.0  # deposit above which a house stamps its ring
    deposit_threshold_frac: float = 1.1  # bank only above thr * frac energy
    store_scale: float = 100.0  # store-input normalizer (saturates at thr)

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
        if self.mate_range <= 0.0:
            raise ValueError("mate_range must be positive")
        if self.senescence_rate < 0.0:
            raise ValueError("senescence_rate must be >= 0")
        if not 0.0 <= self.carrion_yield <= 1.0:
            raise ValueError("carrion_yield must be in [0, 1]")
        if not 0.0 <= self.meat_decay < 1.0:
            raise ValueError("meat_decay must be in [0, 1)")
        if not 0.0 <= self.signal_decay < 1.0:
            raise ValueError("signal_decay must be in [0, 1)")
        if self.emit_rate < 0.0:
            raise ValueError("emit_rate must be >= 0")
        if self.signal_emit_cost < 0.0:
            raise ValueError("signal_emit_cost must be >= 0")
        if self.signal_capacity <= 0.0:
            raise ValueError("signal_capacity must be positive")
        if not 0.0 <= self.signal_diffuse <= 0.25:
            raise ValueError("signal_diffuse must be in [0, 0.25]")
        if self.donate_rate < 0.0:
            raise ValueError("donate_rate must be >= 0")
        if self.donate_range < 0.0:
            raise ValueError("donate_range must be >= 0")
        if self.wood_capacity <= 0.0:
            raise ValueError("wood_capacity must be positive (smell normalizer)")
        if not 0.0 <= self.wood_growth_rate <= 1.0:
            raise ValueError("wood_growth_rate must be in [0, 1]")
        if self.initial_wood < 0.0 or self.initial_wood > self.wood_capacity:
            raise ValueError("need 0 <= initial_wood <= wood_capacity")
        if not 0.0 <= self.wood_patch_amplitude <= 1.0:
            raise ValueError("wood_patch_amplitude must be in [0, 1]")
        if self.wood_patch_scale <= 0.0:
            raise ValueError("wood_patch_scale must be positive")
        if self.gather_rate < 0.0:
            raise ValueError("gather_rate must be >= 0")
        if self.max_inventory < 0.0:
            raise ValueError("max_inventory must be >= 0")
        if min(self.fire_wood_cost, self.fire_energy_cost,
               self.house_wood_cost, self.house_energy_cost) < 0.0:
            raise ValueError("construction costs must be >= 0")
        if self.house_wood_cost <= 0.0:
            raise ValueError("house_wood_cost must be positive")
        if self.fire_wood_cost > self.max_inventory:
            # the fire is still an all-or-nothing lump: inventory is the
            # only wood depot, so a price above the pack can never be paid
            raise ValueError("fire wood cost must fit in max_inventory")
        if not 0.0 <= self.house_rot_rate < 1.0:
            raise ValueError("house_rot_rate must be in [0, 1)")
        if self.house_fire_radius < 0.0:
            raise ValueError("house_fire_radius must be >= 0")
        if self.carry_weight < 0.0:
            raise ValueError("carry_weight must be >= 0")
        if self.build_house_bias > 4.0 or self.build_house_bias < -4.0:
            raise ValueError("build_house_bias must be in [-4, 4]")
        if self.birth_spread < 0.0:
            raise ValueError("birth_spread must be >= 0")
        if self.fire_initial_fuel < 0.0:
            raise ValueError("fire_initial_fuel must be >= 0")
        if self.fire_fuel_decay < 0.0:
            raise ValueError("fire_fuel_decay must be >= 0")
        if self.fire_radius < 0.0 or self.house_radius < 0.0:
            raise ValueError("fire_radius and house_radius must be >= 0")
        if self.cook_bonus < 1.0:
            raise ValueError("cook_bonus must be >= 1.0 (1.0 = no effect)")
        if not 0.0 <= self.cover_reduction <= 1.0:
            raise ValueError("cover_reduction must be in [0, 1]")
        if self.max_structures < 0:
            raise ValueError("max_structures must be >= 0")
        if self.species_threshold <= 0.0:
            raise ValueError("species_threshold must be positive")
        if not 0.0 < self.group_hue_threshold <= 0.5:
            raise ValueError("group_hue_threshold must be in (0, 0.5]")
        if self.founder_regions < 0:
            raise ValueError("founder_regions must be >= 0")
        if self.founder_hue_spread < 0.0:
            raise ValueError("founder_hue_spread must be >= 0")
        if (
            self.founder_regions > 0
            and self.initial_creatures < 2 * self.founder_regions
        ):
            raise ValueError(
                "founder_regions needs >= 2 initial_creatures per region"
                " (sexual reproduction requires a mate)"
            )
        if self.genealogy_capacity < 0:
            raise ValueError("genealogy_capacity must be >= 0")
        if self.plasticity_mode not in ("off", "fixed", "gene"):
            raise ValueError("plasticity_mode must be 'off', 'fixed' or 'gene'")
        if self.plasticity_rate < 0.0:
            raise ValueError("plasticity_rate must be >= 0")
        if self.plasticity_rate > 0.0 and self.plasticity_mode != "fixed":
            raise ValueError(
                "plasticity_rate > 0 only applies to plasticity_mode='fixed'"
                " ('off' must stay a free control and 'gene' reads the"
                " per-creature eta trait)"
            )
        if self.plasticity_interval < 1:
            raise ValueError("plasticity_interval must be >= 1")
        if self.plasticity_clip <= 0.0:
            raise ValueError("plasticity_clip must be positive")
        if not 0.0 <= self.plasticity_decay < 1.0:
            raise ValueError("plasticity_decay must be in [0, 1)")
        if self.plasticity_cost < 0.0:
            raise ValueError("plasticity_cost must be >= 0")
        if self.plasticity_eta_std < 0.0:
            raise ValueError("plasticity_eta_std must be >= 0")
        if self.plasticity_credit_strength < 0.0:
            raise ValueError("plasticity_credit_strength must be >= 0")
        if self.stress_decay < 0.0:
            raise ValueError("stress_decay must be >= 0")
        if self.stress_scale <= 0.0:
            raise ValueError("stress_scale must be positive")
        if self.grudge_decay < 0.0:
            raise ValueError("grudge_decay must be >= 0")
        if self.deposit_rate < 0.0:
            raise ValueError("deposit_rate must be >= 0")
        if self.raid_rate < 0.0:
            raise ValueError("raid_rate must be >= 0")
        if self.guard_range < 0.0:
            raise ValueError("guard_range must be >= 0")
        if not 0.0 <= self.comfort <= 1.0:
            raise ValueError("comfort must be in [0, 1]")
        if self.mark_threshold < 0.0:
            raise ValueError("mark_threshold must be >= 0")
        if self.deposit_threshold_frac <= 0.0:
            raise ValueError("deposit_threshold_frac must be positive")
        if self.store_scale <= 0.0:
            raise ValueError("store_scale must be positive")

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Food grid dimensions as (height, width) in cells."""
        return (
            int(math.ceil(self.height / self.cell_size)),
            int(math.ceil(self.width / self.cell_size)),
        )

    @property
    def sensor_input_dim(self) -> int:
        """Inputs: 3 ray channels (food sum, creatures, food peak), 4 smells
        (food, creatures, prey-diet, carrion — stage 11), 2 food antennas
        (left/right of heading), 3 internal sensors (energy, speed,
        ambient light — stage 10's circadian cue), 3 signal slots (smell +
        left/right antennas over the signal field, stage 13), 1 tag slot
        (hue similarity of the neighbours, stage 15), 1 wood slot (smell of
        the wood field — stage 17), the affect pair (stress, grudge —
        stage 24) and, since stage 25, the signed store reading: the
        deposit of the house underfoot (+ mine, - a foreign complete house,
        0 elsewhere), normalised by store_scale."""
        return self.n_rays * 3 + 17

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

    @property
    def plastic_params(self) -> int:
        """Synapses plastic under Oja (stage 21): W1 (hid, in) plus
        W_rec (hid, hid). The dw scratch has exactly this many columns."""
        return (
            self.hidden_size * self.sensor_input_dim
            + self.hidden_size * self.hidden_size
        )

    @property
    def plasticity_gene_offset(self) -> int:
        """Genome column of the evolvable eta trait (always the last trait)."""
        return self.brain_params + self.N_BODY_TRAITS - 1

    @property
    def plasticity_active(self) -> bool:
        """True when some creature can write ``dw`` during this run.

        ``mode='off'`` and ``mode='fixed'`` with rate 0 both keep ``dw``
        permanently zero, so the forward skips it entirely and the run is
        bit-identical to the pre-stage-21 baseline (the sweep's eta=0 arm).
        """
        return self.plasticity_mode == "gene" or (
            self.plasticity_mode == "fixed" and self.plasticity_rate > 0.0
        )


def load_config(path: str, base: Config | None = None) -> Config:
    """Build a Config from an overrides file applied on top of ``base``.

    ``.toml`` files are parsed with the stdlib :mod:`tomllib` (stage 12.5,
    ``configs/proto_culture.toml``); any other extension is JSON (stage-8
    ``--config``). The file is a flat mapping of Config field names; fields
    it does not mention keep their value from ``base`` (or the defaults).
    Unknown names fail with :class:`ValueError` so a typo cannot silently
    create a dead option, and the merged result still goes through
    ``Config.__post_init__`` validation.
    """
    if path.endswith(".toml"):
        import tomllib

        with open(path, "rb") as fh:
            overrides = tomllib.load(fh)
    else:
        with open(path, encoding="utf-8") as fh:
            overrides = json.load(fh)
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: expected an object of Config fields")
    allowed = {f.name for f in fields(Config)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown Config fields {unknown}")
    merged = asdict(base or Config()) | overrides
    try:
        return Config(**merged)
    except TypeError as exc:
        raise ValueError(f"{path}: bad override value ({exc})") from exc
