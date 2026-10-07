"""Benchmark: ticks/s for several population sizes.

Usage:
    python -m primordia.bench --ticks 1000
    python -m primordia.bench --profile          # microseconds per phase

The JIT is warmed up on a small throwaway world before timing so compilation
does not pollute the measurement. ``alive_final`` is printed alongside ticks/s
because without reproduction (stage 3) the population decays during the run
and later ticks get cheaper; the number is honest only with that context.

``--profile`` times each tick phase in isolation (200 back-to-back calls):
relative costs are representative, but isolated repeats distort state slightly
(e.g. eating drains cells inside its own window), so treat it as a hotspot
map, not an absolute budget.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from .config import Config
from .step import PHASES, advance
from .world import World

DEFAULT_POPULATIONS = (500, 2000, 5000)


def warm_up_jit() -> None:
    """Compile all kernels on a throwaway world so timing stays clean."""
    cfg = replace(Config(), max_creatures=64, initial_creatures=64)
    w = World(cfg, seed=0)
    for _ in range(30):
        advance(w)


def measure(n: int, ticks: int, seed: int) -> tuple[float, int, float]:
    """Run ``ticks`` at population ``n``; return (ticks/s, alive_final, mean_energy)."""
    cfg = replace(Config(), max_creatures=n, initial_creatures=n)
    w = World(cfg, seed=seed)
    start = time.perf_counter()
    for _ in range(ticks):
        advance(w)
    elapsed = time.perf_counter() - start
    mean_energy = float(w.energy[w.alive].mean()) if w.alive_count else 0.0
    return ticks / elapsed, w.alive_count, mean_energy


def profile(n: int, iters: int, seed: int) -> None:
    """Print microseconds per tick phase at population ``n``."""
    cfg = replace(Config(), max_creatures=n, initial_creatures=n)
    w = World(cfg, seed=seed)
    rows: list[tuple[str, float]] = []
    for name, phase in PHASES:
        start = time.perf_counter()
        for _ in range(iters):
            phase(w)
        rows.append((name, (time.perf_counter() - start) * 1e6 / iters))

    start = time.perf_counter()
    for _ in range(iters):
        advance(w)
    full_us = (time.perf_counter() - start) * 1e6 / iters

    total = sum(us for _, us in rows)
    print(f"primordia profile — N={n}, {iters} calls/phase, seed={seed}")
    print(f"{'phase':>16} {'us/tick':>10} {'share':>7}")
    for name, us in rows:
        print(f"{name:>16} {us:>10.1f} {100.0 * us / total:>6.1f}%")
    print(f"{'sum(phases)':>16} {total:>10.1f} {100.0:>6.1f}%")
    print(f"{'advance()':>16} {full_us:>10.1f}   (includes Python orchestration)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=1000, help="ticks per measurement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--populations", type=int, nargs="+", default=list(DEFAULT_POPULATIONS)
    )
    parser.add_argument(
        "--profile", action="store_true", help="time each tick phase instead"
    )
    parser.add_argument("--iters", type=int, default=200, help="calls per phase in --profile")
    args = parser.parse_args()

    warm_up_jit()
    if args.profile:
        n = args.populations[1] if len(args.populations) > 1 else args.populations[0]
        profile(n, args.iters, args.seed)
        return

    print(f"primordia bench — {args.ticks} ticks/point, seed={args.seed} (JIT warmed up)")
    header = f"{'N':>6} {'ticks/s':>10} {'time_s':>8} {'alive_final':>12} {'mean_energy':>12}"
    print(header)
    for n in args.populations:
        rate, alive, mean_energy = measure(n, args.ticks, args.seed)
        print(
            f"{n:>6} {rate:>10.1f} {args.ticks / rate:>8.2f} "
            f"{alive:>12} {mean_energy:>12.1f}"
        )


if __name__ == "__main__":
    main()
