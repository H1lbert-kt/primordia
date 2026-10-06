"""Benchmark: ticks/s for several population sizes.

Usage: ``python -m primordia.bench --ticks 1000``

The JIT is warmed up on a small throwaway world before timing so compilation
does not pollute the measurement. ``alive_final`` is printed alongside ticks/s
because without reproduction (stage 3) the population decays during the run
and later ticks get cheaper; the number is honest only with that context.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from .config import Config
from .step import advance
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=1000, help="ticks per measurement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--populations", type=int, nargs="+", default=list(DEFAULT_POPULATIONS)
    )
    args = parser.parse_args()

    warm_up_jit()
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
