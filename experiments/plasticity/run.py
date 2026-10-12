"""Experiment (stage 21): does local Oja plasticity pay for itself?

Question: synapses now adjust during a life by a local rule with no
reward — ``dW = eta * (x*h - h^2 * W_eff)`` written only into the ``dw``
scratch (the genome is immutable inside a life: Baldwin, not Lamarck),
with an explicit decay back to the evolved blueprint, a clamp, and an
energy cost proportional to ``|dw|`` moved. Nothing rewards learning. The
experiment is the ablation the stage-21 plan pre-registers:

    off      plasticity_mode="off"                      (control)
    oja1e-4  fixed eta = 1e-4                           (sweep, plan 3)
    oja1e-3  fixed eta = 1e-3
    oja1e-2  fixed eta = 1e-2
    oja_de   fixed eta = 1e-3 + delta-E credit          (plan 5)
    gene     evolvable eta gene, spawned at 0,          (plan 4)
             pays plasticity_cost, mutation at
             plasticity_eta_std (explains the sweep band)

Pre-registered hypotheses, stated before the 3-seed 10k batch (a 2-seed
5k pilot informed plasticity_eta_std — its eta=0 purge is disclosed in
the README):

  H1 (user-declared a priori): pure Oja probably does NOT improve
     survival. The rule learns the statistics of the inputs, not what is
     good; with a price on every |dw| moved, expect oja* variants to sit
     within (or below) the control's population band, not above it.
  H2: with the energy cost, selection keeps eta near 0 unless learning
     pays — in the gene variant the eta>0 share stays a small minority
     and |dw| stays far from the clip.
  H3: the delta-E variant is the most likely to diverge from the
     control, because it is the only one with LOCAL CREDIT ASSIGNMENT —
     the learning rule itself reads the individual's payoff (energy
     change of the last tick). This is reward-like and off by default;
     declared per principle 1: energy is a state variable the creature
     already senses, never a fitness score.

Hard gate: every run of every variant finishes the requested ticks with
no extinction (min alive > 0 over the recorded trajectory), no NaN in
dw/genome, and the recorded columns present. Clones phase: two identical
genetics in two environments must diverge in dw while their genomes stay
bit-equal (the unit of "open mind, inherited blueprint").

Output: figures in ``figs/``, raw histories in ``data/`` (gitignored),
tables on stdout for the README.

    python experiments/plasticity/run.py [--ticks 10000] [--force]
    python experiments/plasticity/run.py --phase clones
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments._common import SEEDS, band, save_fig  # noqa: E402
from primordia.config import Config  # noqa: E402
from primordia.step import advance  # noqa: E402
from primordia.world import World  # noqa: E402

HERE = Path(__file__).parent

VARIANTS: dict[str, dict] = {
    "off": {"plasticity_mode": "off"},
    "oja1e-4": {"plasticity_mode": "fixed", "plasticity_rate": 1e-4},
    "oja1e-3": {"plasticity_mode": "fixed", "plasticity_rate": 1e-3},
    "oja1e-2": {"plasticity_mode": "fixed", "plasticity_rate": 1e-2},
    "oja_de": {
        "plasticity_mode": "fixed",
        "plasticity_rate": 1e-3,
        "plasticity_credit": True,
    },
    "gene": {"plasticity_mode": "gene"},  # post-gate: Config default is
    # "off"; "gene" lives in configs/proto_culture.toml (see its header)
}

COLUMNS = (
    "tick", "alive", "mean_energy", "births",
    "deaths_famine", "deaths_age", "deaths_predation",
    "dw_mean", "dw_max", "eta_mean", "eta_max", "eta_pos",
)


def record_step(w: World, t: int, cols: dict[str, np.ndarray]) -> None:
    """Sample one tick of world state (alive-only statistics)."""
    cols["tick"][t] = w.tick
    alive = np.flatnonzero(w.alive)
    cols["alive"][t] = alive.size
    cols["births"][t] = w.births
    cols["deaths_famine"][t] = w.deaths_famine
    cols["deaths_age"][t] = w.deaths_age
    cols["deaths_predation"][t] = w.deaths_predation
    if alive.size == 0:
        return
    cols["mean_energy"][t] = float(w.energy[alive].mean())
    dw = np.abs(w.dw[alive])
    cols["dw_mean"][t] = float(dw.mean())
    cols["dw_max"][t] = float(dw.max())
    eta = w.genome[alive, w.config.plasticity_gene_offset]
    cols["eta_mean"][t] = float(eta.mean())
    cols["eta_max"][t] = float(eta.max())
    cols["eta_pos"][t] = float((eta > 0.0).mean())


def run_variant(
    overrides: dict, seed: int, ticks: int, npz: Path, force: bool
) -> dict[str, np.ndarray]:
    """Run one variant/seed, recording COLUMNS every tick (npz = cache)."""
    if npz.exists() and not force:
        data = dict(np.load(npz, allow_pickle=False))
        if len(data["tick"]) == ticks:
            print(f"  seed={seed}: cached {npz.name}", flush=True)
            return data
        print(f"  seed={seed}: stale ({len(data['tick'])} != {ticks})")
    cfg = replace(Config(), **overrides)
    w = World(cfg, seed=seed)
    cols = {name: np.zeros(ticks) for name in COLUMNS}
    start = time.perf_counter()
    for t in range(ticks):
        advance(w)
        record_step(w, t, cols)
    elapsed = time.perf_counter() - start
    finite = bool(np.isfinite(cols["dw_max"]).all())
    print(
        f"  seed={seed}: {ticks / elapsed:.1f} ticks/s, final alive="
        f"{int(cols['alive'][-1])}, min={int(cols['alive'].min())}, "
        f"births={int(cols['births'][-1])}, dw_max={cols['dw_max'][-1]:.3f}"
        + ("" if finite else "  NaN!"),
        flush=True,
    )
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, config_json=np.array(str(overrides)), **cols)
    return cols


def load_hist(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def gate(histories: dict[str, list[dict[str, np.ndarray]]]) -> bool:
    """Hard gate: no extinction, no NaN, every recorded run completed."""
    ok = True
    for name, runs in histories.items():
        for i, h in enumerate(runs):
            alive = h["alive"]
            bad = []
            if alive[-1] <= 0 or alive.min() <= 0:
                bad.append(f"extinct (final {alive[-1]:.0f}, min {alive.min():.0f})")
            if not np.isfinite(h["dw_max"]).all():
                bad.append("NaN in dw")
            if bad:
                ok = False
                print(f"  GATE FAIL {name} seed={SEEDS[i]}: {', '.join(bad)}")
    print("GATE: " + ("PASS (all runs alive, no NaN)" if ok else "FAIL"))
    return ok


def report(histories: dict[str, list[dict[str, np.ndarray]]]) -> None:
    print(f"\n{'variant':9s} {'seed':>4s} {'final':>6s} {'min':>5s} "
          f"{'births':>7s} {'fame/idade/pred':>17s} {'dw_max':>7s} "
          f"{'eta_pos%':>8s} {'eta_max':>8s}")
    for name, runs in histories.items():
        for seed, h in zip(SEEDS, runs):
            print(
                f"{name:9s} {seed:4d} {int(h['alive'][-1]):6d} "
                f"{int(h['alive'].min()):5d} {int(h['births'][-1]):7d} "
                f"{int(h['deaths_famine'][-1]):5d}/{int(h['deaths_age'][-1]):5d}"
                f"/{int(h['deaths_predation'][-1]):5d} "
                f"{h['dw_max'][-1]:7.3f} {100 * h['eta_pos'][-1]:8.2f} "
                f"{h['eta_max'][-1]:8.4f}"
            )


def plot_gate(histories: dict[str, list[dict[str, np.ndarray]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, runs in histories.items():
        band(axes[0], runs, "alive", name)
    axes[0].set_xlabel("tick")
    axes[0].set_ylabel("alive")
    axes[0].set_title("survival (mean, min-max band over 3 seeds)")
    for name, runs in histories.items():
        if name == "off":
            continue
        band(axes[1], runs, "dw_max", name)
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("max |dw| (alive)")
    axes[1].set_title("engagement: learned delta, clamp = 5.0")
    axes[1].axhline(5.0, color="k", ls=":", lw=0.8, label="clip")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_fig(fig, HERE / "figs" / "gate.png")


def plot_economy(histories: dict[str, list[dict[str, np.ndarray]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, runs in histories.items():
        band(axes[0], runs, "mean_energy", name)
    axes[0].set_xlabel("tick")
    axes[0].set_ylabel("mean energy")
    axes[0].set_title("economy")
    for name, runs in histories.items():
        band(axes[1], runs, "births", name)
    axes[1].set_xlabel("tick")
    axes[1].set_ylabel("cumulative births")
    axes[1].set_title("births")
    for name, runs in histories.items():
        band(axes[2], runs, "eta_pos", name)
    axes[2].set_xlabel("tick")
    axes[2].set_ylabel("share of alive with eta > 0")
    axes[2].set_title("H2: the evolvable eta gene")
    fig.tight_layout()
    save_fig(fig, HERE / "figs" / "economy.png")


def run_gate_phase(ticks: int, force: bool) -> bool:
    histories: dict[str, list[dict[str, np.ndarray]]] = {}
    for name, overrides in VARIANTS.items():
        print(f"variant {name}: {overrides}", flush=True)
        runs = []
        for seed in SEEDS:
            npz = HERE / "data" / f"{name}_s{seed}_{ticks}.npz"
            runs.append(run_variant(overrides, seed, ticks, npz, force))
        histories[name] = runs
    report(histories)
    ok = gate(histories)
    plot_gate(histories)
    plot_economy(histories)
    print(f"figures: {(HERE / 'figs').resolve()}")
    return ok


# --- clones phase: same genetics, two environments, dw diverges ----------

CLONES_COMMON = dict(
    plasticity_mode="fixed",
    plasticity_rate=1e-3,
    plasticity_interval=10,
    mutation_rate=0.0,  # genomes must stay bit-equal: no variation channel
    reproduce_threshold=1e9,  # and no inheritance either
)


def run_clones_phase(ticks: int, force: bool) -> bool:
    npz = HERE / "data" / f"clones_{ticks}.npz"
    if npz.exists() and not force:
        data = dict(np.load(npz, allow_pickle=False))
        print(f"clones: cached {npz.name}", flush=True)
    else:
        cfg_a = replace(
            Config(), food_patch_amplitude=0.85, **CLONES_COMMON
        )
        cfg_b = replace(
            Config(), food_patch_amplitude=0.45, **CLONES_COMMON
        )
        cfg_off = replace(
            Config(),
            food_patch_amplitude=0.85,
            plasticity_mode="off",  # free control: no rate (config forbids it)
            mutation_rate=0.0,
            reproduce_threshold=1e9,
        )
        a, b, ctrl = World(cfg_a, seed=7), World(cfg_b, seed=7), World(
            cfg_off, seed=7
        )
        assert np.array_equal(a.genome, b.genome), "clone worlds start equal"
        cols = {
            "dw_diff_max": np.zeros(ticks),
            "dw_diff_mean": np.zeros(ticks),
            "genome_diff_max": np.zeros(ticks),
            "dw_ctrl": np.zeros(ticks),
            "alive_a": np.zeros(ticks),
            "alive_b": np.zeros(ticks),
        }
        for t in range(ticks):
            advance(a)
            advance(b)  # sequential: process-global Numba RNG (no draws here)
            advance(ctrl)
            d = np.abs(a.dw - b.dw)
            cols["dw_diff_max"][t] = float(d.max())
            cols["dw_diff_mean"][t] = float(d.mean())
            cols["genome_diff_max"][t] = float(np.abs(a.genome - b.genome).max())
            cols["dw_ctrl"][t] = float(np.abs(ctrl.dw).max())
            cols["alive_a"][t] = a.alive_count
            cols["alive_b"][t] = b.alive_count
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **cols)
        data = cols
        print(f"clones: {ticks} ticks", flush=True)

    ok = True
    if data["genome_diff_max"][-1] != 0.0:
        ok = False
        print("CLONES FAIL: genomes diverged (the rule leaked into the blueprint)")
    if data["dw_diff_max"][-1] <= 0.0:
        ok = False
        print("CLONES FAIL: identical environments learned identically? dw diff = 0")
    print(
        f"clones: genome_diff max = {data['genome_diff_max'].max():.6f} (must be 0), "
        f"dw_diff final max = {data['dw_diff_max'][-1]:.4f}, "
        f"ctrl dw = {data['dw_ctrl'].max():.4f} (must be 0)"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(ticks)
    ax.plot(t, data["dw_diff_max"], label="max |dw_A - dw_B| (different food)")
    ax.plot(t, data["dw_diff_mean"], label="mean |dw_A - dw_B|", alpha=0.7)
    ax.plot(t, data["dw_ctrl"], label="control (plasticity off)", ls="--")
    ax.plot(t, data["genome_diff_max"], label="max |genome_A - genome_B|", lw=3)
    ax.set_xlabel("tick")
    ax.set_ylabel("distance")
    ax.set_title("same genetics, two worlds: the delta diverges, the genome does not")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, HERE / "figs" / "clones.png")
    print(f"figure: {(HERE / 'figs' / 'clones.png').resolve()}")
    return ok


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--phase", choices=("gate", "clones", "all"), default="all"
    )
    args = parser.parse_args()
    ok = True
    if args.phase in ("gate", "all"):
        ok &= run_gate_phase(args.ticks, args.force)
    if args.phase in ("clones", "all"):
        ok &= run_clones_phase(args.ticks if args.ticks <= 2000 else 2000,
                               args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
