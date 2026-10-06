"""Primordia CLI.

Stage 1 is headless only; windowed rendering arrives in stage 4.

    python run.py --seed 42 --headless --ticks 1000
"""

from __future__ import annotations

import argparse
import time

from primordia.bench import warm_up_jit
from primordia.config import Config
from primordia.step import advance
from primordia.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--pop", type=int, default=500, help="initial population")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if not args.headless:
        raise SystemExit("interactive rendering arrives in stage 4; pass --headless")

    cfg = Config(max_creatures=max(args.pop, 1), initial_creatures=args.pop)
    warm_up_jit()
    w = World(cfg, seed=args.seed)
    start = time.perf_counter()
    for _ in range(args.ticks):
        advance(w)
    elapsed = time.perf_counter() - start

    mean_energy = float(w.energy[w.alive].mean()) if w.alive_count else 0.0
    print(
        f"seed={args.seed} ticks={args.ticks} pop={args.pop} -> "
        f"{args.ticks / elapsed:.1f} ticks/s ({elapsed:.2f} s), "
        f"alive={w.alive_count}, mean_energy={mean_energy:.1f}"
    )


if __name__ == "__main__":
    main()
