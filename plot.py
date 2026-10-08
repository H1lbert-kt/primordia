"""Plot the per-tick metrics written by ``run.py --stats``.

    python plot.py s42.npz --out figs/

Reads the .npz history (columns + seed + Config) and writes four PNGs:
``population.png``, ``energy.png``, ``deaths.png`` and ``diversity.png``.
The Agg backend is forced so the script also works over SSH without a display.
``genome_dist`` is sampled every ``sample_every`` ticks and forward-filled;
the legend says so instead of pretending it moves every tick.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from primordia.stats import StatsHistory, StatsRecorder

DPI = 120


def _title(history: StatsHistory, text: str) -> str:
    return f"{text} — seed {history.seed_label}, {len(history)} ticks"


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_history(history: StatsHistory, out_dir: Path) -> list[Path]:
    """Render the four PNGs into ``out_dir``; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tick = history["tick"]
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tick, history["alive"], color="tab:blue", label="alive")
    ax.set_xlabel("tick")
    ax.set_ylabel("population", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(tick, history["births"], color="tab:green", alpha=0.6,
             label="births/tick")
    deaths = (history["deaths_famine"] + history["deaths_age"]
              + history["deaths_predation"])
    ax2.plot(tick, deaths, color="tab:red", alpha=0.6, label="deaths/tick")
    ax2.set_ylabel("per tick", color="tab:gray")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    ax.set_title(_title(history, "population and flows"))
    paths.append(_save(fig, out_dir, "population.png"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tick, history["mean_energy"], color="tab:orange",
            label="mean energy")
    ax.set_xlabel("tick")
    ax.set_ylabel("energy")
    ax2 = ax.twinx()
    ax2.plot(tick, history["mean_age"], color="tab:purple", label="mean age")
    ax2.set_ylabel("age (ticks)", color="tab:purple")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
    ax.set_title(_title(history, "mean energy and age"))
    paths.append(_save(fig, out_dir, "energy.png"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    causes = (
        ("starvation", history["deaths_famine"], "tab:red"),
        ("old age", history["deaths_age"], "tab:gray"),
        ("predation", history["deaths_predation"], "tab:brown"),
    )
    ax.stackplot(
        tick,
        [c[1] for c in causes],
        labels=[c[0] for c in causes],
        colors=[c[2] for c in causes],
        alpha=0.85,
    )
    # symlog: the initial-generation age cliff (~400/tick) would otherwise
    # flatten the steady 1-2/tick starvation background to invisibility.
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("tick")
    ax.set_ylabel("deaths per tick (symlog)")
    ax.legend(loc="upper left")
    ax.set_title(_title(history, "causes of death"))
    paths.append(_save(fig, out_dir, "deaths.png"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for trait in ("speed", "size", "vision", "diet"):
        ax.plot(tick, history[f"{trait}_std"], label=f"{trait} std")
    # log scale: vision std (~0.9) dwarfs speed/size/diet (~0.01); zeros at
    # the untouched spawn ticks simply render as gaps.
    ax.set_yscale("log")
    ax.set_xlabel("tick")
    ax.set_ylabel("trait std (log)")
    ax2 = ax.twinx()
    ax2.plot(tick, history["genome_dist"], color="black", linewidth=1.5,
             label="genome_dist")
    ax2.set_ylabel("genome RMS distance", color="black")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower left")
    every = history.sample_every
    ax.set_title(_title(history, f"diversity (genome sampled every {every} ticks)"))
    paths.append(_save(fig, out_dir, "diversity.png"))

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", type=str, help=".npz written by run.py --stats")
    parser.add_argument("--out", type=str, default="figs",
                        help="output directory for the PNGs")
    args = parser.parse_args()

    history = StatsRecorder.load(args.stats)
    paths = plot_history(history, Path(args.out))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
