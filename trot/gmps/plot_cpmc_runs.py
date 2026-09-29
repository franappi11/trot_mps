"""Plot CPMC energy per block against imaginary time for runs saved with --save-walkers (mps_cpmc_gpu).

Reads each run's <tag>_walkers.npz (only the small arrays: the walkers are not loaded) or, for a run still going,
its <tag>_walkers.parts/progress.json. For every run it draws the block energy E_b at tau_b = (b + 1) * n_steps * dt
(and optionally the running energy estimate), marks the end of equilibration, draws the DMRG trial energy as a
dotted line in the run's colour, and prints the weighted mean over the sampling blocks with its blocking-analysis
error (trot.stat_utils when importable, else a plain standard error of the block energies, which ignores
autocorrelation).

    python plot_cpmc_runs.py /mnt/ceph/users/fnappi/trot_walkers/L100_U8
    python plot_cpmc_runs.py run_a_walkers.npz run_b_walkers.npz --per-site --estimate --out energy.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#d4a20f", "#d6457a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load_run(path):
    """Block energies, weights, running estimates, their imaginary times and the run's config."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)  # lazy: only the arrays read below are loaded
        run = dict(energies=z["energies"], weights=z["weights"], e_estimate=z["e_estimate"],
                   tau=z["tau_blocks"], config=json.loads(str(z["config"])), finished=True)
    else:  # a run in progress: <tag>_walkers.parts/progress.json
        progress = json.loads(path.read_text())
        blocks, config = progress["blocks"], progress["config"]
        run = dict(energies=np.array([b["energy"] for b in blocks]), weights=np.array([b["weight"] for b in blocks]),
                   e_estimate=np.array([b["e_estimate"] for b in blocks]),
                   tau=np.arange(1, len(blocks) + 1) * config["N_PROP"] * config["DT"], config=config, finished=False)
    run["path"] = path
    return run


def find_runs(paths):
    """Walker files and unfinished runs among the given files and directories (an unfinished run is skipped
    when its finished .npz exists)."""
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(p.glob("*_walkers.npz"))
            found += [q for q in sorted(p.glob("*_walkers.parts/progress.json"))
                      if not q.parent.with_name(q.parent.name.replace(".parts", ".npz")).exists()]
        else:
            found.append(p)
    return found


def sampling_stats(energies, weights, n_eql):
    """Weighted mean of the sampling blocks and its error (blocking analysis when trot is importable)."""
    e, w = energies[n_eql:], weights[n_eql:]
    if len(e) < 2:
        return float("nan"), float("nan"), "too few sampling blocks"
    try:
        from trot.stat_utils import blocking_analysis_ratio
        stats = blocking_analysis_ratio(e, w, print_q=False)
        err = float("nan") if stats["se_star"] is None else float(stats["se_star"])
        return float(stats["mu"]), err, "blocking analysis"
    except ImportError:
        mean = float(np.sum(w * e) / np.sum(w))
        return mean, float(np.std(e, ddof=1) / np.sqrt(len(e))), "plain standard error (trot not importable)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="walker .npz files, progress.json files or directories of runs")
    parser.add_argument("--per-site", action="store_true", help="plot E / L")
    parser.add_argument("--estimate", action="store_true", help="also draw the running energy estimate")
    parser.add_argument("--no-trial", action="store_true", help="do not draw the DMRG trial energies")
    parser.add_argument("--out", default="cpmc_energy_vs_tau.png", help="figure path (a .csv with the curves goes next to it)")
    parser.add_argument("--show", action="store_true", help="also open the figure window")
    args = parser.parse_args()

    runs = [load_run(p) for p in find_runs(args.paths)]
    if not runs:
        raise SystemExit("no runs found")
    runs.sort(key=lambda r: (r["config"].get("DMRG_CHI_T", 0), str(r["path"])))

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 13,
                         "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.6, "legend.frameon": False, "font.size": 10})

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    rows = ["run,trial_chi,block,tau,energy,weight,e_estimate"]
    print(f"{'run':40s} {'trial chi':>9s} {'blocks':>6s} {'E (sampling)':>16s} {'error':>10s} {'E_DMRG':>14s}  method")
    for i, r in enumerate(runs):
        cfg, color = r["config"], PALETTE[i % len(PALETTE)]
        scale = 1.0 / cfg["L"] if args.per_site else 1.0
        n_eql, chi_t = cfg["N_EQL"], cfg.get("DMRG_CHI_T")
        mean, err, method = sampling_stats(r["energies"], r["weights"], n_eql)
        name = cfg.get("tag") or r["path"].name
        state = "" if r["finished"] else " (running)"
        print(f"{name[:40]:40s} {chi_t!s:>9s} {len(r['energies']):6d} {mean:16.8f} {err:10.2e} "
              f"{cfg.get('E_DMRG', float('nan')):14.8f}  {method}{state}")
        label = (rf"trial $\chi_T$={chi_t}{state}: {mean * scale:.6f} $\pm$ {err * scale:.1e}"
                 if np.isfinite(mean) else rf"trial $\chi_T$={chi_t}{state}")
        ax.plot(r["tau"], r["energies"] * scale, "-", marker="o", ms=2.5, color=color, label=label)
        if args.estimate:
            ax.plot(r["tau"], r["e_estimate"] * scale, "--", lw=1.0, color=color, alpha=0.8)
        if not args.no_trial and "E_DMRG" in cfg:
            ax.axhline(cfg["E_DMRG"] * scale, color=color, ls=":", lw=1.2)
        for b, (t, e, w, est) in enumerate(zip(r["tau"], r["energies"], r["weights"], r["e_estimate"])):
            rows.append(f"{name},{chi_t},{b},{t},{e},{w},{est}")

    cfg0 = runs[0]["config"]
    ax.axvline(cfg0["N_EQL"] * cfg0["N_PROP"] * cfg0["DT"], color=INK2, lw=1.0, ls="--", zorder=0)
    ax.set_xlabel(r"imaginary time $\tau$")
    ax.set_ylabel("CPMC energy per block" + (" per site" if args.per_site else ""))
    ax.set_title(f"CPMC energy per block: L={cfg0['L']}, U={cfg0['U']:g}, {cfg0['N_WALKERS']} walkers, "
                 f"walker bond {cfg0.get('CHI_PROP')} per spin")
    extra = ["dashed: running estimate"] if args.estimate else []
    extra += ["dotted: DMRG trial energy"] if not args.no_trial else []
    extra += ["vertical: end of equilibration"]
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="; ".join(extra), title_fontsize=8)
    out = Path(args.out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    out.with_suffix(".csv").write_text("\n".join(rows) + "\n")
    print(f"saved {out} and {out.with_suffix('.csv')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
