"""Ecological metrics: diet classes and genome clustering (stage 12).

Pure read-only functions over world state — never called inside the tick,
never drawing random numbers, never importing pygame or matplotlib. The
core stays headless; the recorder, the HUD and the experiment scripts
share the same definitions so every reported number is reproducible.

Diet classes (``HERB_MAX``/``PRED_MIN``) are *measurement* boundaries, not
rules: the sim only ever sees the continuous ``diet`` trait in [0, 1].
``classify_diets`` buckets it into herbivore / omnivore / predator so the
stats recorder can count how many of each are alive; no code path branches
on the label.

Species clustering (``cluster_living``) is a deterministic greedy-leader
pass over the living genomes, normalized by the same spawn scales
``genome_dist`` uses. Two creatures share a label when their normalized
distance is ≤ ``Config.species_threshold`` from the *first* leader that
fits (not single-linkage: a chain of near neighbours does not merge two
far-apart leaders). Slots are scanned ascending and ties go to the lower
leader index, so the labels are bit-reproducible for a given world state.
This is a *metric*: the tick never reads the labels and nothing in the
simulation is conditioned on them.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .world import World

# Measurement boundaries for the diet trait (stage 12). Below HERB_MAX a
# creature is counted as herbivore; above PRED_MIN as predator; between the
# two, omnivore. Chosen at the thirds of [0, 1]; purely for counting.
HERB_MAX: float = 0.33
PRED_MIN: float = 0.66

DIET_HERBIVORE = 0
DIET_OMNIVORE = 1
DIET_PREDATOR = 2


def genome_scales(cfg: Config) -> np.ndarray:
    """Per-gene spawn scales so distances mix dimensionless genes.

    Brain genes are scaled by ``weight_init_std`` (or 1.0 if that is 0);
    body traits by ``max_speed``, 1.0, ``vision_range``, 1.0, 1.0 (the
    tag_hue hue is already in [0, 1), stage 15). Shared by
    ``stats._genome_dist`` and ``cluster_living`` so both metrics live in
    the same normalized space.
    """
    params = cfg.brain_params
    scale = np.full(cfg.genome_size, cfg.weight_init_std, dtype=np.float32)
    if cfg.weight_init_std <= 0.0:
        scale[:params] = 1.0
    scale[params] = cfg.max_speed
    scale[params + 1] = 1.0
    scale[params + 2] = cfg.vision_range
    scale[params + 3] = 1.0
    scale[params + 4] = 1.0
    return scale


def classify_diets(diet: np.ndarray) -> np.ndarray:
    """Bucket the continuous diet trait into 0/1/2 (herb/omni/pred).

    Input is the raw trait in [0, 1]; output is int8 with the class
    constants above. Vectorized, allocation-light (one bool + one int8).
    """
    labels = np.full(diet.shape, DIET_OMNIVORE, dtype=np.int8)
    labels[diet < HERB_MAX] = DIET_HERBIVORE
    labels[diet > PRED_MIN] = DIET_PREDATOR
    return labels


def cluster_living(
    world: World, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy-leader clustering of the living genomes.

    Returns ``(slots, labels)``: the living slots in ascending order and an
    int32 label per slot, 0..K-1, assigned in first-seen leader order.
    A creature joins the first leader whose normalized distance is
    ``<= threshold``; otherwise it opens a new leader. Deterministic:
    ascending slots, first-match wins, no RNG.
    """
    slots = np.flatnonzero(world.alive)
    if slots.size == 0:
        return slots, np.zeros(0, dtype=np.int32)
    scale = genome_scales(world.config)
    x = world.genome[slots] * scale
    thr2 = np.float32(threshold * threshold)
    labels = np.empty(slots.size, dtype=np.int32)
    leaders: list[np.ndarray] = []
    for i in range(slots.size):
        xi = x[i]
        assigned = -1
        for k, lk in enumerate(leaders):
            d = xi - lk
            if float(np.dot(d, d)) <= thr2:
                assigned = k
                break
        if assigned < 0:
            assigned = len(leaders)
            leaders.append(xi)
        labels[i] = assigned
    return slots, labels
