"""Primordia CLI.

    python run.py --seed 42 --headless --ticks 1000   # no window
    python run.py --seed 42                           # pygame viewer
    python run.py --seed 42 --ticks 300 --screenshot shot.png --select 0
    python run.py --seed 42 --headless --ticks 5000 --stats s42.npz
    python run.py --seed 42 --headless --ticks 5000 --save w5k.npz
    python run.py --load w5k.npz --headless --ticks 500   # continue a run
    python run.py --headless --ticks 5000 --config overrides.json
    python run.py --headless --ticks 8000 --record out/demo --record-every 80

Windowed mode imports ``primordia.render`` lazily so the headless core never
pulls in pygame (guarded by tests/test_render_guard.py).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from primordia.bench import warm_up_jit
from primordia.config import Config, load_config
from primordia.io import load_world, save_world
from primordia.stats import StatsRecorder
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
        "--load",
        type=str,
        default=None,
        help="start from a world saved with --save (overrides --seed/--pop)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="write the final world state to this .npz (windowed or headless)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON file of Config field overrides (wins over --pop; not with --load)",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="write per-tick metrics to this .npz (headless only; see plot.py)",
    )
    parser.add_argument(
        "--screenshot", type=str, default=None, help="save the final window frame"
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="DIR",
        help="write PNG frames + demo.gif to DIR (headless only)",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=50,
        help="ticks between recorded frames (with --record)",
    )
    parser.add_argument(
        "--select",
        type=int,
        default=-1,
        help="preselect a creature by its id (the number shown in the panel)",
    )
    args = parser.parse_args()
    if args.stats is not None and not args.headless:
        parser.error("--stats requires --headless")
    if args.record is not None and not args.headless:
        parser.error("--record requires --headless")
    if args.record is not None and args.record_every < 1:
        parser.error("--record-every must be >= 1")
    if args.config is not None and args.load:
        parser.error("--config cannot be combined with --load (the save owns its Config)")

    warm_up_jit()
    if args.load:
        w = load_world(args.load)
        print(
            f"loaded {args.load}: tick={w.tick} alive={w.alive_count} "
            f"births={w.births}"
        )
    else:
        # --pop is the *initial* population; keep room to grow (reproduction
        # needs free slots) unless the user asks for a bigger starting world.
        # --config JSON fields override this base, --pop included.
        base = Config(
            max_creatures=max(args.pop, Config().max_creatures),
            initial_creatures=args.pop,
        )
        cfg = load_config(args.config, base) if args.config else base
        w = World(cfg, seed=args.seed)
    origin = f"load={args.load}" if args.load else f"seed={args.seed}"

    # --select takes the creature id (stable, shown in the panel), not the
    # slot: initial population slots come from the end of the free-list, so
    # "slot 3" would silently select nothing on a fresh world.
    select_slot = -1
    if args.select >= 0:
        hits = np.flatnonzero(w.alive & (w.creature_id == args.select))
        if hits.size == 0:
            parser.error(f"--select: creature id {args.select} is not alive")
        select_slot = int(hits[0])

    if not args.headless:
        from primordia.render import run_app  # lazy: keep pygame out of the core

        summary = run_app(
            w,
            seed=w.seed,
            max_ticks=args.ticks,
            screenshot_path=args.screenshot,
            select_slot=select_slot,
        )
        print(
            f"{origin} windowed: {summary['frames']} frames, "
            f"{summary['ticks']} ticks, mean_fps={summary['mean_fps']:.1f}, "
            f"min_fps={summary['min_fps']:.1f}, alive={w.alive_count}, "
            f"selected={summary['selected']}"
        )
        if args.save:
            save_world(w, args.save)
            print(f"world -> {args.save} (tick={w.tick})")
        return

    ticks = args.ticks if args.ticks is not None else 1000
    recorder = StatsRecorder(capacity=ticks) if args.stats else None
    frame_recorder = None
    if args.record is not None:
        from primordia.render.record import WorldRecorder  # lazy: pygame stays out

        frame_recorder = WorldRecorder(
            w, args.record, seed=w.seed, speed=args.record_every
        )
        frame_recorder.capture(w)  # tick 0
    captured = 0
    start = time.perf_counter()
    for _ in range(ticks):
        advance(w)
        if recorder is not None:
            recorder.record(w)
        if frame_recorder is not None and w.tick % args.record_every == 0:
            frame_recorder.capture(w)
            captured += 1
            if captured % 20 == 0:
                print(f"  recorded {captured} frames (tick {w.tick})", flush=True)
    elapsed = time.perf_counter() - start

    mean_energy = float(w.energy[w.alive].mean()) if w.alive_count else 0.0
    print(
        f"{origin} ticks={ticks} pop={args.pop} -> "
        f"{ticks / elapsed:.1f} ticks/s ({elapsed:.2f} s), "
        f"alive={w.alive_count}, mean_energy={mean_energy:.1f}"
    )
    if recorder is not None:
        recorder.save(args.stats, seed=w.seed)
        print(f"stats -> {args.stats} ({recorder.length} ticks)")
    if frame_recorder is not None:
        gif = frame_recorder.save_gif()
        mb = gif.stat().st_size / (1024.0 * 1024.0)
        print(
            f"frames -> {frame_recorder.frames_dir} "
            f"({len(frame_recorder.frames)}), gif -> {gif} ({mb:.2f} MB)"
        )
    if args.save:
        save_world(w, args.save)
        print(f"world -> {args.save} (tick={w.tick})")


if __name__ == "__main__":
    main()
