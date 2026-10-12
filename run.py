"""Primordia CLI.

    python run.py --seed 42 --headless --ticks 1000   # no window
    python run.py --seed 42                           # pygame viewer
    python run.py --seed 42 --ticks 300 --screenshot shot.png --select 0
    python run.py --seed 42 --headless --ticks 5000 --stats s42.npz
    python run.py --seed 42 --headless --ticks 5000 --save w5k.npz
    python run.py --load w5k.npz --headless --ticks 500   # continue a run
    python run.py --headless --ticks 5000 --config overrides.json
    python run.py --headless --ticks 8000 --record out/demo --record-every 80
    python run.py --daemon --db world.db            # run forever, state in SQLite
    python run.py --load-db world.db                # observe the last checkpoint
    python run.py --load-db world.db --headless --ticks 500 --save cont.npz

Windowed mode imports ``primordia.render`` lazily so the headless core never
pulls in pygame (guarded by tests/test_render_guard.py).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from primordia.bench import warm_up_jit
from primordia.config import Config, load_config
from primordia.database import (
    init_db,
    load_checkpoint,
    open_db,
    prune_checkpoints,
    save_checkpoint,
    timeline_row,
    insert_timeline,
)
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
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite world database (checkpoints + timeline; headless only)",
    )
    parser.add_argument(
        "--load-db",
        type=str,
        default=None,
        help="resume from the newest checkpoint in this database "
        "(replaces --load/--seed; windowed = observe that snapshot)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="run headless forever (until SIGINT), checkpointing to --db",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="ticks between world checkpoints (with --db/--daemon)",
    )
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=10,
        help="ticks between timeline gauge rows (with --db/--daemon)",
    )
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=5,
        help="checkpoints to keep after pruning (with --db/--daemon)",
    )
    args = parser.parse_args()
    if args.daemon:
        args.headless = True  # the daemon is headless by definition
    if args.stats is not None and not args.headless:
        parser.error("--stats requires --headless")
    if args.record is not None and not args.headless:
        parser.error("--record requires --headless")
    if args.record is not None and args.record_every < 1:
        parser.error("--record-every must be >= 1")
    if args.config is not None and args.load:
        parser.error("--config cannot be combined with --load (the save owns its Config)")
    if args.db is not None and not args.headless:
        parser.error("--db requires --headless (the window does not checkpoint)")
    if args.daemon and args.db is None:
        parser.error("--daemon requires --db")
    if args.daemon and args.stats is not None:
        parser.error(
            "--stats cannot be combined with --daemon "
            "(timeline gauges go to --db; full stats belong to finite runs)"
        )
    if args.load_db is not None and args.load is not None:
        parser.error("--load-db and --load are mutually exclusive")
    if args.config is not None and args.load_db is not None:
        parser.error(
            "--config cannot be combined with --load-db (the checkpoint owns its Config)"
        )
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")
    if args.metrics_every < 1:
        parser.error("--metrics-every must be >= 1")
    if args.checkpoint_keep < 1:
        parser.error("--checkpoint-keep must be >= 1")

    warm_up_jit()
    if args.load_db:
        read_conn = open_db(args.load_db)
        try:
            w = load_checkpoint(read_conn)
        finally:
            read_conn.close()
        print(
            f"loaded db {args.load_db}: tick={w.tick} alive={w.alive_count} "
            f"births={w.births}"
        )
    elif args.load:
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
    origin = (
        f"load-db={args.load_db}"
        if args.load_db
        else f"load={args.load}"
        if args.load
        else f"seed={args.seed}"
    )

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

    ticks = args.ticks
    if ticks is None:
        ticks = None if args.daemon else 1000
    recorder = (
        StatsRecorder(capacity=ticks if ticks is not None else 4096)
        if args.stats
        else None
    )
    frame_recorder = None
    if args.record is not None:
        from primordia.render.record import WorldRecorder  # lazy: pygame stays out

        frame_recorder = WorldRecorder(
            w, args.record, seed=w.seed, speed=args.record_every
        )
        frame_recorder.capture(w)  # tick 0
    db_conn = None
    timeline_batch: list[tuple] = []
    if args.db is not None:
        db_conn = open_db(args.db)
        init_db(db_conn, w)
        if args.daemon:
            print(
                f"daemon: db={args.db}, checkpoint every {args.checkpoint_every} "
                f"ticks, timeline every {args.metrics_every}, keep "
                f"{args.checkpoint_keep} (Ctrl-C to stop)",
                flush=True,
            )
    captured = 0
    start = time.perf_counter()
    n = 0
    try:
        while ticks is None or n < ticks:
            advance(w)
            n += 1
            if recorder is not None:
                recorder.record(w)
            if frame_recorder is not None and w.tick % args.record_every == 0:
                frame_recorder.capture(w)
                captured += 1
                if captured % 20 == 0:
                    print(f"  recorded {captured} frames (tick {w.tick})", flush=True)
            if db_conn is not None:
                if w.tick % args.metrics_every == 0:
                    timeline_batch.append(timeline_row(w))
                    if len(timeline_batch) >= 100:
                        insert_timeline(db_conn, timeline_batch)
                        timeline_batch.clear()
                if w.tick % args.checkpoint_every == 0:
                    insert_timeline(db_conn, timeline_batch)
                    timeline_batch.clear()
                    save_checkpoint(db_conn, w)
                    prune_checkpoints(db_conn, args.checkpoint_keep)
                    if args.daemon:
                        print(
                            f"  daemon tick={w.tick} alive={w.alive_count} "
                            f"births={w.births}",
                            flush=True,
                        )
    except KeyboardInterrupt:
        if db_conn is None:
            raise
        print(
            f"\ninterrupted at tick {w.tick} — writing final checkpoint",
            flush=True,
        )
    finally:
        if db_conn is not None:
            insert_timeline(db_conn, timeline_batch)
            save_checkpoint(db_conn, w)
            prune_checkpoints(db_conn, args.checkpoint_keep)
            db_conn.close()
            print(f"checkpoint -> {args.db} (tick={w.tick})", flush=True)
    elapsed = time.perf_counter() - start

    mean_energy = float(w.energy[w.alive].mean()) if w.alive_count else 0.0
    print(
        f"{origin} ticks={n} pop={args.pop} -> "
        f"{n / elapsed:.1f} ticks/s ({elapsed:.2f} s), "
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
