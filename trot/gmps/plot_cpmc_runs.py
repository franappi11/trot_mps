"""Plot CPMC energy per block against imaginary time for runs saved with --save-walkers (run_mps_cpmc.py).

Reads each run's <tag>_walkers.npz (only the small arrays: the walkers are not loaded), for a run still going
its <tag>_walkers.parts/progress.json, or, without the walker files (e.g. on a laptop), the output directory's
blocks.jsonl (every block of every run) with results.jsonl (each finished run's settings). For every run it draws
the block energy E_b at tau_b = (b + 1) * n_steps * dt (and optionally the running energy estimate), marks the end
of equilibration, draws the DMRG trial energy as a dotted line in the run's colour, and prints the weighted mean
over the sampling blocks with its blocking-analysis error: trot.stat_utils, or a copy of trot/stat_utils.py next
to this script (it needs only numpy and jax); without either, a plain standard error of the block energies, which
ignores autocorrelation.

    python plot_cpmc_runs.py /mnt/ceph/users/fnappi/trot_walkers/L100_U8
    python plot_cpmc_runs.py run_a_walkers.npz run_b_walkers.npz --per-site --estimate --out energy.png
    python plot_cpmc_runs.py L100_U8/blocks.jsonl --estimate   # a directory holding the two .jsonl files also works
"""
import argparse
import json
from pathlib import Path

import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#d4a20f", "#d6457a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load_runs(path):
    """The runs in one file, each with its block energies, weights, running estimates, their imaginary times and
    the config: a walker .npz or progress.json holds one run, blocks.jsonl (or results.jsonl) every run of its
    directory."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return load_block_logs(path)
    if path.suffix == ".npz":
        z = np.load(path)  # lazy: only the arrays read below are loaded
        config = json.loads(str(z["config"]))
        config.update(trial_setup(config))
        run = dict(energies=z["energies"], weights=z["weights"], e_estimate=z["e_estimate"],
                   tau=z["tau_blocks"], config=config, finished=True)
    elif path.name == "progress.json":  # a run in progress: <tag>_walkers.parts/progress.json
        progress = json.loads(path.read_text())
        blocks, config = progress["blocks"], progress["config"]
        config.update(trial_setup(config))
        run = dict(energies=np.array([b["energy"] for b in blocks]), weights=np.array([b["weight"] for b in blocks]),
                   e_estimate=np.array([b["e_estimate"] for b in blocks]),
                   tau=np.arange(1, len(blocks) + 1) * config["N_PROP"] * config["DT"], config=config, finished=False)
    else:
        raise SystemExit(f"cannot read {path}: expected a _walkers.npz, progress.json, blocks.jsonl or results.jsonl")
    run["path"] = path
    return [run]


def load_block_logs(path):
    """Every finished run of an output directory from blocks.jsonl (one line per block, all runs appended to it)
    and results.jsonl beside it (one record per finished run: its Config and energies), with the walker files'
    config keys. A run without a results record (still going, or crashed) is skipped; a tag run more than once
    keeps its last run, since each run numbers its blocks from 0."""
    blocks_path = path.with_name("blocks.jsonl") if path.name == "results.jsonl" else path
    results_path = blocks_path.with_name("results.jsonl")
    if not results_path.exists():
        raise SystemExit(f"{results_path} not found: it holds the runs' settings (n_steps, dt, n_equilibration, ...)")
    results = {}
    for line in results_path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            results[record["tag"]] = record
    blocks = {}
    for line in blocks_path.read_text().splitlines():
        if line.strip():
            block = json.loads(line)
            rows = blocks.setdefault(block["tag"], [])
            if block["block"] == 0:
                rows.clear()
            rows.append(block)
    runs = []
    for tag, rows in blocks.items():
        if tag not in results:
            print(f"skipping {tag}: no record in {results_path} (still going, or crashed)")
            continue
        r = results[tag]
        config = dict(L=r.get("L", r.get("n_sites")), N_UP=r["n_up"], N_DN=r["n_down"], T=r["hopping"], U=r["interaction"],
                      N_WALKERS=r["n_walkers"], N_EQL=r["n_equilibration"], N_BLOCKS=r["n_blocks"],
                      N_PROP=r["n_steps"], DT=r["dt"], SEED=r["seed"], DMRG_CHI_T=r["trial_chi"],
                      DMRG_SWEEPS=r["dmrg_sweeps"], CHI_PROP=r["walker_channel_chi"], E_DMRG=r["dmrg_energy"],
                      E_TRIAL=r["trial_energy"], E_CPMC=r["cpmc_energy"], E_CPMC_ERR=r["cpmc_error"], tag=tag)
        if "Lx" in r:  # square lattice (mps_cpmc_2d_gpu), as in its walker files' config
            config.update(LX=r["Lx"], LY=r["Ly"])
        config.update(trial_setup(r))
        runs.append(dict(energies=np.array([b["energy"] for b in rows]), weights=np.array([b["weight"] for b in rows]),
                         e_estimate=np.array([b["e_estimate"] for b in rows]),
                         tau=(np.array([b["block"] for b in rows]) + 1) * r["n_steps"] * r["dt"],
                         config=config, finished=True, path=blocks_path))
    return runs


def rotation_angle(record):
    """The trial's spin rotation in degrees: mps_cpmc_gpu's trial_rotation, or the angle of the R matrix that the
    trot-native rotated runs stored (spin_rotation_y(beta) = [[cos beta/2, -sin beta/2], [sin beta/2, cos beta/2]])."""
    if record.get("trial_rotation") is not None:
        return float(record["trial_rotation"])
    R = record.get("rotation")
    if R is None:
        return 0.0
    return float(np.degrees(2.0 * np.arctan2(R[1][0], R[0][0])))


def trial_setup(record):
    """The config keys run_variant reads, from a results.jsonl record or a walker file's config. Runs from before
    the options existed get their meaning then: unrotated, a rotated trial projected onto the walkers' sector, and
    natural orbitals of the trial's rdm1 after the rotation."""
    beta = rotation_angle(record)
    return dict(TRIAL_ROTATION=beta,
                ROTATED_TRIAL=record.get("rotated_trial", "projected") if beta else None,
                NATURAL_RDM1=record.get("natural_rdm1", "after") if beta else None,
                PLAN_REFERENCE=record.get("plan_reference", "natural"),
                WALKER_START=record.get("walker_start", "natural"))


def run_variant(cfg):
    """How a run's trial and walkers were set up, for the legend: unrotated or rotated trial (projected onto the
    walkers' S_z sector, or used as it is without S_z labels), and where the walker plan and start come from."""
    beta = cfg.get("TRIAL_ROTATION") or 0.0
    if not beta:
        trial = "unrotated trial"
    else:
        kind = "no $S_z$ labels" if cfg.get("ROTATED_TRIAL") == "as_is" else "$S_z$-projected"
        trial = rf"rotated ${beta:g}^\circ$ trial ({kind})"
    plan, start = cfg.get("PLAN_REFERENCE", "natural"), cfg.get("WALKER_START", "natural")
    if plan == start == "natural":
        walkers = f"plan & start: {'unrotated' if cfg.get('NATURAL_RDM1') == 'before' else 'rotated'} rdm1" if beta else ""
    else:
        walkers = f"plan & start: {plan.upper()}" if plan == start else f"plan: {plan}, start: {start}"
    return ", ".join(x for x in (trial, walkers) if x)


def find_runs(paths):
    """The runs among the given files and directories; a directory is searched for walker files, unfinished runs'
    progress.json and blocks.jsonl. A run found twice (its walker file and its lines in blocks.jsonl) is kept once,
    from the first file listed: in a directory, the walker file."""
    files = []
    for p in map(Path, paths):
        if not p.exists():
            raise SystemExit(f"{p} not found")
        if p.is_dir():
            files += sorted(p.glob("*_walkers.npz")) + sorted(p.glob("*_walkers.parts/progress.json"))
            files += [p / "blocks.jsonl"] if (p / "blocks.jsonl").exists() else []
        else:
            files.append(p)
    runs = {}
    for f in files:
        for run in load_runs(f):
            runs.setdefault(run["config"].get("tag") or str(run["path"]), run)
    return list(runs.values())


def sampling_stats(energies, weights, n_eql):
    """Weighted mean of the sampling blocks and its error (blocking analysis when stat_utils is importable)."""
    e, w = energies[n_eql:], weights[n_eql:]
    if len(e) < 2:
        return float("nan"), float("nan"), "too few sampling blocks"
    try:
        from trot.stat_utils import blocking_analysis_ratio
    except ImportError:
        try:  # a copy of trot/stat_utils.py next to this script
            from stat_utils import blocking_analysis_ratio
        except ImportError:
            mean = float(np.sum(w * e) / np.sum(w))
            return mean, float(np.std(e, ddof=1) / np.sqrt(len(e))), "plain standard error (stat_utils not importable)"
    stats = blocking_analysis_ratio(e, w, print_q=False)
    err = float("nan") if stats["se_star"] is None else float(stats["se_star"])
    return float(stats["mu"]), err, "blocking analysis"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+",
                        help="walker .npz files, progress.json files, blocks.jsonl files or directories of runs")
    parser.add_argument("--per-site", action="store_true", help="plot E / L")
    parser.add_argument("--estimate", action="store_true", help="also draw the running energy estimate")
    parser.add_argument("--no-trial", action="store_true", help="do not draw the DMRG trial energies")
    parser.add_argument("--e-ref", type=float, default=None, help="reference (e.g. converged DMRG) total energy: a black solid line")
    parser.add_argument("--trial-chi", type=int, nargs="+", default=None, help="keep only runs with these trial chi")
    parser.add_argument("--out", default="cpmc_energy_vs_tau.png", help="figure path (a .csv with the curves goes next to it)")
    parser.add_argument("--show", action="store_true", help="also open the figure window")
    args = parser.parse_args()

    runs = find_runs(args.paths)
    if args.trial_chi:
        runs = [r for r in runs if r["config"].get("DMRG_CHI_T") in args.trial_chi]
    if not runs:
        raise SystemExit("no runs found")
    runs.sort(key=lambda r: (r["config"].get("DMRG_CHI_T", 0), r["config"].get("CHI_PROP") or 0,
                             r["config"].get("TRIAL_ROTATION") or 0.0, run_variant(r["config"]), r["config"]["DT"],
                             str(r["path"])))
    several_dt = len({r["config"]["DT"] for r in runs}) > 1  # then the labels say each run's time step
    several_chi_w = len({r["config"].get("CHI_PROP") for r in runs}) > 1  # ... and each run's walker bond

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
    print(f"{'run':40s} {'trial chi':>9s} {'blocks':>6s} {'E (sampling)':>16s} {'error':>10s} {'E_DMRG':>14s}  "
          f"method; setup")
    for i, r in enumerate(runs):
        cfg, color = r["config"], PALETTE[i % len(PALETTE)]
        scale = 1.0 / cfg["L"] if args.per_site else 1.0
        n_eql, chi_t = cfg["N_EQL"], cfg.get("DMRG_CHI_T")
        mean, err, method = sampling_stats(r["energies"], r["weights"], n_eql)
        name = cfg.get("tag") or r["path"].name
        state = "" if r["finished"] else " (running)"
        print(f"{name[:40]:40s} {chi_t!s:>9s} {len(r['energies']):6d} {mean:16.8f} {err:10.2e} "
              f"{cfg.get('E_DMRG', float('nan')):14.8f}  {method}{state}; {run_variant(cfg).replace('$', '')}")
        run_label = (rf"trial $\chi_T$={chi_t}" + (rf", $\chi_w$={cfg.get('CHI_PROP')}" if several_chi_w else "")
                     + (f", dt={cfg['DT']:g}" if several_dt else "") + ", " + run_variant(cfg) + state)
        label = (rf"{run_label}: {mean * scale:.6f} $\pm$ {err * scale:.1e}" if np.isfinite(mean) else run_label)
        ax.plot(r["tau"], r["energies"] * scale, "-", marker="o", ms=2.5, color=color, label=label)
        if args.estimate:
            ax.plot(r["tau"], r["e_estimate"] * scale, "--", lw=1.0, color=color, alpha=0.8)
        if not args.no_trial and "E_DMRG" in cfg:
            ax.axhline(cfg["E_DMRG"] * scale, color=color, ls=":", lw=1.2)
        for b, (t, e, w, est) in enumerate(zip(r["tau"], r["energies"], r["weights"], r["e_estimate"])):
            rows.append(f"{name},{chi_t},{b},{t},{e},{w},{est}")

    cfg0 = runs[0]["config"]
    if args.e_ref is not None:
        ax.axhline(args.e_ref * (1.0 / cfg0["L"] if args.per_site else 1.0), color=INK, lw=1.4,
                   label=f"reference energy (DMRG): {args.e_ref:.4f}")
    for tau_eql in sorted({r["config"]["N_EQL"] * r["config"]["N_PROP"] * r["config"]["DT"] for r in runs}):
        ax.axvline(tau_eql, color=INK2, lw=1.0, ls="--", zorder=0)
    ax.set_xlabel(r"imaginary time $\tau$")
    ax.set_ylabel("CPMC energy per block" + (" per site" if args.per_site else ""))
    lattice = f"{cfg0['LX']}x{cfg0['LY']}" if "LX" in cfg0 else f"L={cfg0['L']}"
    walker_bond = "" if several_chi_w else f", walker bond {cfg0.get('CHI_PROP')} per spin"
    ax.set_title(f"CPMC energy per block: {lattice}, U={cfg0['U']:g}, {cfg0['N_WALKERS']} walkers{walker_bond}")
    extra = ["dashed: running estimate"] if args.estimate else []
    extra += ["dotted: DMRG trial energy"] if not args.no_trial else []
    extra += ["vertical: end of equilibration"]
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="; ".join(extra), title_fontsize=8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".csv").write_text("\n".join(rows) + "\n")
    print(f"saved {out.with_suffix('.csv')}")
    try:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    except RuntimeError as error:
        # The python/3.12.13 module's matplotlib fails rasterizing glyphs ("FT_Render_Glyph ... raster overflow",
        # also in allocation_study.py, with or without hinting); a PDF embeds glyph outlines and never rasterizes them.
        if out.suffix.lower() == ".pdf":
            raise
        print(f"could not draw {out} ({error}); saving a PDF instead")
        out = out.with_suffix(".pdf")
        fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
