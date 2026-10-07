"""Primordia CLI.

    python run.py --seed 42 --headless --ticks 1000   # no window
    python run.py --seed 42                           # pygame viewer
    python run.py --seed 42 --ticks 300 --screenshot shot.png --select 0

Windowed mode imports ``primordia.render`` lazily so the headless core never
pulls in pygame (guarded by tests/test_render_guard.py).
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
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="ticks to run (default: 1000 headless, unlimited windowed)",
    )
    parser.add_argument("--pop", type=int, default=500, help="initial population")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--screenshot", type=str, default=None, help="save the final window frame"
    )
    parser.add_argument(
        "--select", type=int, default=-1, help="preselect a creature slot"
    )
    args = parser.parse_args()

    # --pop is the *initial* population; keep room to grow (reproduction
    # needs free slots) unless the user asks for a bigger starting world.
    cfg = Config(
        max_creatures=max(args.pop, Config().max_creatures),
        initial_creatures=args.pop,
    )
    warm_up_jit()
    w = World(cfg, seed=args.seed)

    if not args.headless:
        from primordia.render import run_app  # lazy: keep pygame out of the core

        summary = run_app(
            w,
            seed=args.seed,
            max_ticks=args.ticks,
            screenshot_path=args.screenshot,
            select_slot=args.select,
        )
        print(
            f"seed={args.seed} windowed: {summary['frames']} frames, "
            f"{summary['ticks']} ticks, mean_fps={summary['mean_fps']:.1f}, "
            f"min_fps={summary['min_fps']:.1f}, alive={w.alive_count}, "
            f"selected={summary['selected']}"
        )
        return

    ticks = args.ticks if args.ticks is not None else 1000
    start = time.perf_counter()
    for _ in range(ticks):
        advance(w)
    elapsed = time.perf_counter() - start

    mean_energy = float(w.energy[w.alive].mean()) if w.alive_count else 0.0
    print(
        f"seed={args.seed} ticks={ticks} pop={args.pop} -> "
        f"{ticks / elapsed:.1f} ticks/s ({elapsed:.2f} s), "
        f"alive={w.alive_count}, mean_energy={mean_energy:.1f}"
    )


if __name__ == "__main__":
    main()
