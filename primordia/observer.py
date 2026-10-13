"""Observer-side group detection (stage 23): countries by lineage colour.

Pure measurement over world state — the tick never reads any of this, no
random number is drawn, and no creature knows a group exists (the same
contract as the stage-12 ecological metrics). A *group* here is a
greedy-leader cluster of the circular ``tag_hue`` trait: the hue is
inherited from one parent (stage-15 crossover), mutates slowly around the
wheel, and the founder scenario (stage 23) pins one hue band per founding
region — so hue groups *are* the founding lineages until mutation and
interbreeding blur them. Genome-distance species clustering
(``ecology.cluster_living``) stays a separate metric: with random brains at
spawn it would return ~one cluster per creature, which is useless as a
notion of "country".

Group *identity between samples* is a viewer-side convention, not a fact of
the simulation: uid matching is slot-overlap greedy (below), uids flicker
whenever detection flickers, and every event kind is defined here — an
"invasion" is geometry, not war. Callers that need reproducible numbers
must pin the sampling period (the viewer refreshes every 30 frames, the
experiment every 100 ticks).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass(frozen=True)
class GroupEvent:
    """One observer-detected event between two consecutive samples."""

    tick: int
    kind: str  # "new" | "extinct" | "fusion" | "invasion"
    group: int  # primary uid (winner of the match, invaded-into, or new)
    other: int  # fusion partner / invaded group; -1 when not applicable
    count: int  # population involved / cross-incursion count


@dataclass(frozen=True)
class GroupSnapshot:
    """Observer state at one sample; buffer views live until the next."""

    tick: int
    n_groups: int
    border_cells: int
    events: tuple[GroupEvent, ...]  # events raised by THIS sample only
    owner: np.ndarray  # (gh, gw) int16 uid per cell, -1 = no group present
    uid_of_slot: np.ndarray  # (max_creatures,) int16, -1 = not in a group
    pops: dict[int, int]  # uid -> living count
    areas: dict[int, int]  # uid -> owned cells
    ages: dict[int, int]  # uid -> ticks since first detection
    hues: dict[int, float]  # uid -> circular-mean hue of the members


class GroupObserver:
    """Deterministic hue-group detector with stable uids and an event log.

    uid matching between samples: for each new hue label, count slot
    overlap with each previous uid; a candidate pair needs overlap >=
    max(1, 0.25 * min(prev_pop, new_pop)); labels claim their best
    (largest overlap, ties -> smaller uid) unclaimed uid, otherwise the
    group is new. A label claiming two or more candidates raises a fusion
    (the winner inherits the biggest partner's uid). Extinct uids are the
    previous pops that claimed nothing. Invasions count living slots of
    group g standing in cells majority-owned by a different, still-living
    group h; a pair (g, h) raises at most one event until the count falls
    back under the threshold (hysteresis).
    """

    #: Fraction of the smaller group that must overlap for a match.
    MATCH_FRAC = 0.25
    #: Circular hue convention: the same value for every observer.
    HUE_WRAP = 1.0

    def __init__(
        self,
        cfg: Config,
        *,
        invasion_min: int = 8,
        event_log: int = 200,
    ) -> None:
        if cfg.group_hue_threshold <= 0.0:
            raise ValueError("group_hue_threshold must be positive")
        self.threshold = float(cfg.group_hue_threshold)
        self.invasion_min = int(invasion_min)
        self._hue_off = cfg.brain_params + 4
        self._max_creatures = cfg.max_creatures
        self._next_uid = 0
        self._first_seen: dict[int, int] = {}
        self._prev_pops: dict[int, int] = {}
        self._active_invasions: set[tuple[int, int]] = set()
        self._slot_uid = np.full(cfg.max_creatures, -1, dtype=np.int16)
        self._owner: np.ndarray | None = None
        self.events: deque[GroupEvent] = deque(maxlen=event_log)

    def _cluster_hue(self, hue: np.ndarray) -> np.ndarray:
        """Greedy-leader clustering on the circle (first-match, no RNG)."""
        thr = self.threshold
        labels = np.empty(hue.size, dtype=np.int32)
        leaders: list[float] = []
        for i in range(hue.size):
            h = float(hue[i])
            assigned = -1
            for k, lk in enumerate(leaders):
                d = abs(h - lk)
                if d > 0.5:
                    d = 1.0 - d
                if d <= thr:
                    assigned = k
                    break
            if assigned < 0:
                assigned = len(leaders)
                leaders.append(h)
            labels[i] = assigned
        return labels

    def _match(
        self,
        slots: np.ndarray,
        labels: np.ndarray,
        tick: int,
        out_events: list[GroupEvent],
    ) -> np.ndarray:
        """Assign a stable uid to each label; raise new/fusion events."""
        prev_uid = self._slot_uid  # last sample's assignment (read-only here)
        n_labels = int(labels.max()) + 1 if labels.size else 0
        per_label: list[Counter] = [Counter() for _ in range(n_labels)]
        pop_of = np.zeros(n_labels, dtype=np.int64)
        for i in range(slots.size):
            u = int(prev_uid[slots[i]])
            if u >= 0:
                per_label[int(labels[i])][u] += 1
            pop_of[int(labels[i])] += 1

        def qualifies(k: int, u: int, overlap: int) -> bool:
            need = max(1, int(self.MATCH_FRAC * min(self._prev_pops.get(u, 0), int(pop_of[k]))))
            return overlap >= need

        # candidates per label, best first (overlap desc, uid asc)
        cands: list[list[tuple[int, int]]] = []
        for k in range(n_labels):
            pairs = [
                (u, c) for u, c in per_label[k].items() if qualifies(k, u, c)
            ]
            pairs.sort(key=lambda p: (-p[1], p[0]))
            cands.append(pairs)

        uid_of_label = np.full(n_labels, -1, dtype=np.int32)
        taken: set[int] = set()
        for k in range(n_labels):
            winner = -1
            for u, _overlap in cands[k]:
                if u not in taken:
                    winner = u
                    break
            if winner >= 0:
                uid_of_label[k] = winner
                taken.add(winner)
                if len(cands[k]) >= 2:
                    partner = next(
                        u for u, _ in cands[k] if u != winner
                    )
                    out_events.append(
                        GroupEvent(tick, "fusion", winner, partner, int(pop_of[k]))
                    )
            else:
                uid_of_label[k] = self._next_uid
                self._next_uid += 1
                self._first_seen[int(uid_of_label[k])] = tick
                out_events.append(
                    GroupEvent(tick, "new", int(uid_of_label[k]), -1, int(pop_of[k]))
                )

        for u, pop in self._prev_pops.items():
            if pop > 0 and u not in taken:
                out_events.append(GroupEvent(tick, "extinct", u, -1, 0))
                self._first_seen.pop(u, None)
        return uid_of_label

    def _majority_owner(
        self, world, slots: np.ndarray, uid: np.ndarray
    ) -> np.ndarray:
        """Per-cell majority uid (ties -> smaller label/uid), -1 if empty."""
        gh, gw = world.food.shape
        if self._owner is None or self._owner.shape != (gh, gw):
            self._owner = np.full((gh, gw), -1, dtype=np.int16)
        owner = self._owner
        owner.fill(-1)
        if slots.size == 0:
            return owner
        ix = (world.pos[slots, 0] * (1.0 / float(world.cell_w))).astype(np.intp)
        iy = (world.pos[slots, 1] * (1.0 / float(world.cell_h))).astype(np.intp)
        np.clip(ix, 0, gw - 1, out=ix)
        np.clip(iy, 0, gh - 1, out=iy)
        # label = rank of uid in this sample's uid list (ties deterministic)
        uniq = np.unique(uid)
        rank = np.searchsorted(uniq, uid)
        cell = iy * gw + ix
        key = cell * np.int64(uniq.size) + rank
        counts = np.bincount(key, minlength=gh * gw * uniq.size).reshape(
            gh * gw, uniq.size
        )
        best = counts.argmax(axis=1)
        has = counts.max(axis=1) > 0
        flat = np.where(has, uniq[best].astype(np.int16), np.int16(-1))
        owner[:] = flat.reshape(gh, gw)
        return owner

    @staticmethod
    def _border_cells(owner: np.ndarray) -> int:
        """Torus-adjacent cells with different non-empty owners."""
        occupied = owner >= 0
        if not occupied.any():
            return 0
        diff = np.zeros(owner.shape, dtype=bool)
        for axis in (0, 1):
            for shift in (1, -1):
                other = np.roll(owner, shift, axis=axis)
                diff |= occupied & (other >= 0) & (other != owner)
        return int(np.count_nonzero(occupied & diff))

    def sample(self, world) -> GroupSnapshot:
        """Detect groups in the current world state; raise events."""
        tick = world.tick
        slots = np.flatnonzero(world.alive)
        out_events: list[GroupEvent] = []

        if slots.size:
            hue = np.mod(world.genome[slots, self._hue_off], 1.0)
            labels = self._cluster_hue(hue)
        else:
            labels = np.zeros(0, dtype=np.int32)
        uid_of_label = self._match(slots, labels, tick, out_events)
        uid = uid_of_label[labels] if slots.size else np.zeros(0, dtype=np.int32)

        new_slot_uid = np.full(self._max_creatures, -1, dtype=np.int16)
        if slots.size:
            new_slot_uid[slots] = uid.astype(np.int16)
        self._slot_uid = new_slot_uid

        owner = self._majority_owner(world, slots, new_slot_uid[slots])
        border_cells = self._border_cells(owner)

        pops: dict[int, int] = {}
        if slots.size:
            for u, c in zip(*np.unique(uid, return_counts=True)):
                pops[int(u)] = int(c)
        areas: dict[int, int] = {}
        hues: dict[int, float] = {}
        if slots.size:
            flat = owner[owner >= 0]
            if flat.size:
                for u, c in zip(*np.unique(flat, return_counts=True)):
                    areas[int(u)] = int(c)
            # circular mean hue per uid (overlay colour = lineage colour)
            ang = 2.0 * np.pi * np.mod(world.genome[slots, self._hue_off], 1.0)
            re = np.bincount(uid, weights=np.cos(ang), minlength=self._next_uid)
            im = np.bincount(uid, weights=np.sin(ang), minlength=self._next_uid)
            for u in pops:
                h = np.arctan2(im[u], re[u]) / (2.0 * np.pi)
                hues[u] = float(h - np.floor(h))

        # invasions: slots of g standing in majority-cells of a living h
        if slots.size:
            ix = (world.pos[slots, 0] * (1.0 / float(world.cell_w))).astype(np.intp)
            iy = (world.pos[slots, 1] * (1.0 / float(world.cell_h))).astype(np.intp)
            np.clip(ix, 0, owner.shape[1] - 1, out=ix)
            np.clip(iy, 0, owner.shape[0] - 1, out=iy)
            here = owner[iy, ix]
            cross: Counter = Counter()
            for g, h in zip(uid.tolist(), here.tolist()):
                if h >= 0 and int(h) != int(g):
                    cross[(int(g), int(h))] += 1
            active = {p: c for p, c in cross.items() if c >= self.invasion_min}
            for (g, h), c in sorted(active.items()):
                if (g, h) not in self._active_invasions:
                    out_events.append(GroupEvent(tick, "invasion", g, h, c))
            self._active_invasions = set(active)

        for ev in out_events:
            self.events.append(ev)
        self._prev_pops = pops

        ages = {
            u: tick - self._first_seen.get(u, tick) for u in pops
        }
        return GroupSnapshot(
            tick=tick,
            n_groups=len(pops),
            border_cells=border_cells,
            events=tuple(out_events),
            owner=owner,
            uid_of_slot=new_slot_uid,
            pops=pops,
            areas=areas,
            ages=ages,
            hues=hues,
        )
