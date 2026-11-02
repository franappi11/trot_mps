"""Plot CPMC energy per block against imaginary time for runs saved with --save-walkers (run_mps_cpmc.py).

Reads each run's <tag>_walkers.npz (only the small arrays: the walkers are not loaded), for a run still going
its <tag>_walkers.parts/progress.json, or, without the walker files (e.g. on a laptop), the output directory's
blocks.jsonl (every block of every run) with results.jsonl (each finished run's settings). For every run it draws
the block energy E_b at tau_b = (b + 1) * n_steps * dt (and optionally the running energy estimate), marks the end
of equilibration, draws the DMRG trial energy as a dotted line in the run's colour, and prints the weighted mean
over the sampling blocks with its blocking-analysis error: trot.stat_utils, or a copy of trot/stat_utils.py next
to this script (it needs only numpy and jax); without either, a plain standard error of the block energies, which
ignores autocorrelation.

It also reads the runs of ex_6x6_doped_diag.py (and ex_6x6_doped.py / ex_8x8_doped.py): a run directory holding
result.npz (the start's energy, then every equilibration and sampling block, and the szsz of every block if it was
measured) with manifest.json and walkers_block0000.npz (lattice, U). For those the start's energy is drawn at tau = 0.
Equilibration checks (the walkers' start, e.g. ex_6x6_doped_diag.py --walker-start):
* --window W draws means of W blocks instead of every block;
* --szsz adds a panel with the staggered structure factor S(pi, pi) of every block's szsz;
* --discard-scan prints, for every run, the mean of blocks k..end and its blocking error for k = 0, --scan-step, ...
  (and N_EQL), of E and S(pi, pi), the MSER truncation point (the k <= half the blocks minimizing
  sum_{b >= k} (E_b - mean)^2 / (n - k)^2) and the two halves of the sampling blocks.

    python plot_cpmc_runs.py /mnt/ceph/users/fnappi/trot_walkers/L100_U8
    python plot_cpmc_runs.py run_a_walkers.npz run_b_walkers.npz --per-site --estimate --out energy.png
    python plot_cpmc_runs.py L100_U8/blocks.jsonl --estimate   # a directory holding the two .jsonl files also works
    python plot_cpmc_runs.py sq6x6_doped_diag/dmrg_chi256_w16_dense_m32* --window 10 --szsz --discard-scan
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#d4a20f", "#d6457a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load_runs(path):
    """The runs in one file, each with its block energies, weights, running estimates, their imaginary times and
    the config: a walker .npz, progress.json or result.npz holds one run, blocks.jsonl (or results.jsonl) every run of
    its directory."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return load_block_logs(path)
    if path.name == "result.npz":
        run = load_result(path)
        return [] if run is None else [run]
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


def load_result(path):
    """A run of ex_6x6_doped_diag.py (ex_6x6_doped.py, ex_8x8_doped.py): <out>/result.npz, whose block_energies and
    block_weights hold the start (the walkers' mixed energy at tau = 0) and then every equilibration and sampling
    block, and szsz_blocks (one per block) if szsz was measured; the params from <out>/manifest.json, the lattice and
    U from <out>/walkers_block0000.npz. The run is named after its directory; None (skipped) without a manifest."""
    run_dir = path.parent
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"skipping {path}: no manifest.json beside it (the run's n_eql_blocks, dt, ...)")
        return None
    manifest = json.loads(manifest_path.read_text())
    p = manifest["params"]
    z = np.load(path)
    energies, weights = np.asarray(z["block_energies"], dtype=float), np.asarray(z["block_weights"], dtype=float)
    n = len(energies) - 1
    szsz = np.asarray(z["szsz_blocks"]) if "szsz_blocks" in z.files else None
    if szsz is not None and len(szsz) != n:
        print(f"{run_dir.name}: {len(szsz)} szsz blocks for {n} energy blocks; szsz not used")
        szsz = None
    n_sites, u, nelec = (None if szsz is None else szsz.shape[-1]), float("nan"), (None, None)
    start_file = run_dir / "walkers_block0000.npz"
    if start_file.exists():
        with np.load(start_file) as s:  # lazy: the walkers are not loaded
            n_sites, u, nelec = len(s["h1"]), float(s["u"]), tuple(int(x) for x in s["nelec"])
    chi_t = re.search(r"chi(\d+)", Path(manifest["trial_file"]).stem)
    start = manifest.get("start") or dict(kind=p.get("walker_start", "natural"))
    config = dict(L=n_sites, N_UP=nelec[0], N_DN=nelec[1], U=u, N_WALKERS=p["n_walkers"], N_EQL=p["n_eql_blocks"],
                  N_BLOCKS=p["n_blocks"], N_PROP=p["n_prop_steps"], DT=p["dt"], SEED=p["seed"],
                  DMRG_CHI_T=int(chi_t.group(1)) if chi_t else None, CHI_PROP=p["walker_channel_chi"],
                  MEASURE_CHI=p.get("measure_channel_chi"), TRIAL_FILE=manifest["trial_file"],
                  TRIAL_ROTATION=float(manifest.get("rotation_degrees") or 0.0),
                  ROTATED_TRIAL=manifest.get("rotated_trial"), NATURAL_RDM1=manifest.get("natural_rdm1"),
                  PLAN_REFERENCE=p.get("plan_reference", "natural"),
                  WALKER_START=start.get("name", "walkers") if start["kind"] == "walkers" else start["kind"],
                  START_ENERGY=float(energies[0]), tag=run_dir.name)
    side = int(round(np.sqrt(n_sites))) if n_sites else 0
    if side and side * side == n_sites:
        config.update(LX=side, LY=side)
    run = dict(energies=energies[1:], weights=weights[1:], e_estimate=np.full(n, np.nan),
               tau=np.arange(1, n + 1) * p["n_prop_steps"] * p["dt"], config=config, finished=True, path=path)
    if szsz is not None:
        run["szsz"] = szsz
    return run


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
    progress.json, blocks.jsonl and result.npz. A run found twice (its walker file and its lines in blocks.jsonl) is
    kept once, from the first file listed: in a directory, the walker file."""
    files = []
    for p in map(Path, paths):
        if not p.exists():
            raise SystemExit(f"{p} not found")
        if p.is_dir():
            files += sorted(p.glob("*_walkers.npz")) + sorted(p.glob("*_walkers.parts/progress.json"))
            files += [p / "blocks.jsonl"] if (p / "blocks.jsonl").exists() else []
            files += [p / "result.npz"] if (p / "result.npz").exists() else []  # an ex_6x6_doped_diag.py run
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


def staggered_structure_factor(szsz, cfg):
    """S(pi, pi) = (1/N) sum_ij (-1)^(x_i + y_i + x_j + y_j) <S^z_i S^z_j> of every block's szsz (n, N, N), site
    y * LX + x (without LX a chain: (-1)^(i + j))."""
    n_sites = szsz.shape[-1]
    site = np.arange(n_sites)
    lx = cfg.get("LX") or n_sites
    sign = (-1.0) ** (site % lx + site // lx)
    return np.einsum("i,bij,j->b", sign, szsz, sign) / n_sites


def window_means(tau, values, weights, width):
    """Weighted means of consecutive windows of `width` blocks at their mean tau (a shorter last window is dropped)."""
    if width <= 1:
        return tau, values
    n = len(values) // width * width
    w = weights[:n].reshape(-1, width)
    return tau[:n].reshape(-1, width).mean(axis=1), np.sum(w * values[:n].reshape(-1, width), axis=1) / w.sum(axis=1)


def mser(values):
    """MSER truncation point: the k <= len(values) // 2 minimizing sum_{b >= k} (v_b - mean_k)^2 / (n - k)^2."""
    n = len(values)
    scores = [np.sum((values[k:] - values[k:].mean()) ** 2) / (n - k) ** 2 for k in range(n // 2 + 1)]
    return int(np.argmin(scores))


def discard_scan(run, step, min_keep):
    """Print how the run's means move as more leading blocks are discarded: the mean of blocks k..end with its
    blocking error for k = 0, step, 2 step, ... (and N_EQL) while at least min_keep blocks remain, for E and, if the
    run has szsz, S(pi, pi); the MSER truncation points; the two halves of the sampling blocks."""
    cfg, e, w = run["config"], run["energies"], run["weights"]
    n, n_eql, tau_block = len(e), cfg["N_EQL"], cfg["N_PROP"] * cfg["DT"]
    s = staggered_structure_factor(run["szsz"], cfg) if "szsz" in run else None
    start = f", start E {cfg['START_ENERGY']:.6f}" if "START_ENERGY" in cfg else ""
    print(f"\n== {cfg.get('tag') or run['path']}: {n} blocks of tau {tau_block:g}, N_EQL {n_eql}{start}")
    header = f"{'k':>5s} {'tau_k':>7s} {'E(k..end)':>14s} {'error':>9s}"
    print(header + (f" {'S(k..end)':>11s} {'error':>9s}" if s is not None else ""))
    for k in sorted({*range(0, n - min_keep + 1, step), *([n_eql] if n_eql <= n - 2 else [])}):
        mean, err, _ = sampling_stats(e, w, k)
        row = f"{k:5d} {k * tau_block:7.2f} {mean:14.6f} {err:9.1e}"
        if s is not None:
            s_mean, s_err, _ = sampling_stats(s, w, k)
            row += f" {s_mean:11.5f} {s_err:9.1e}"
        print(row + ("  <- N_EQL" if k == n_eql else ""))
    print(f"MSER truncation: E k* = {mser(e)}" + (f", S k* = {mser(s)}" if s is not None else ""))
    half = (n - n_eql) // 2
    if half >= 2:
        for name, values in (("E", e), ("S", s)):
            if values is None:
                continue
            a, a_err, _ = sampling_stats(values[: n_eql + half], w[: n_eql + half], n_eql)
            b, b_err, _ = sampling_stats(values, w, n_eql + half)
            z = (b - a) / np.hypot(a_err, b_err)
            print(f"sampling halves, {name}: {a:.6f}({a_err:.1e}) then {b:.6f}({b_err:.1e}): "
                  f"{b - a:+.6f} = {z:+.1f} sigma")


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
    parser.add_argument("--window", type=int, default=1, help="draw means of this many blocks (default 1: every block)")
    parser.add_argument("--szsz", action="store_true",
                        help="add a panel with S(pi, pi) of the runs whose result.npz holds szsz_blocks")
    parser.add_argument("--discard-scan", action="store_true",
                        help="print each run's means of E (and S(pi, pi)) against the leading blocks discarded")
    parser.add_argument("--scan-step", type=int, default=10, help="blocks between --discard-scan rows (default 10)")
    parser.add_argument("--min-keep", type=int, default=30, help="blocks the last --discard-scan row keeps (default 30)")
    args = parser.parse_args()

    runs = find_runs(args.paths)
    if args.trial_chi:
        runs = [r for r in runs if r["config"].get("DMRG_CHI_T") in args.trial_chi]
    if not runs:
        raise SystemExit("no runs found")
    runs.sort(key=lambda r: (r["config"].get("DMRG_CHI_T") or 0, r["config"].get("CHI_PROP") or 0,
                             r["config"].get("TRIAL_ROTATION") or 0.0, run_variant(r["config"]), r["config"]["DT"],
                             str(r["path"])))
    several_dt = len({r["config"]["DT"] for r in runs}) > 1  # then the labels say each run's time step
    several_chi_w = len({r["config"].get("CHI_PROP") for r in runs}) > 1  # ... and each run's walker bond
    several_measure = len({r["config"].get("MEASURE_CHI") for r in runs}) > 1  # ... its measurement bond
    several_seeds = len({r["config"].get("SEED") for r in runs}) > 1  # ... and its seed

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 13,
                         "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.6, "legend.frameon": False, "font.size": 10})

    if args.szsz and any("szsz" in r for r in runs):
        fig, (ax, ax_s) = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax_s = None
    rows = ["run,trial_chi,block,tau,energy,weight,e_estimate,s_pi_pi"]
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
                     + (f", measured at {cfg.get('MEASURE_CHI')}" if several_measure else "")
                     + (f", dt={cfg['DT']:g}" if several_dt else "") + ", " + run_variant(cfg)
                     + (f", seed {cfg.get('SEED')}" if several_seeds else "") + state)
        label = (rf"{run_label}: {mean * scale:.6f} $\pm$ {err * scale:.1e}" if np.isfinite(mean) else run_label)
        tau, energies = window_means(r["tau"], r["energies"], r["weights"], args.window)
        ax.plot(tau, energies * scale, "-", marker="o", ms=2.5, color=color, label=label)
        if "START_ENERGY" in cfg:  # the walkers' start, tau = 0
            ax.plot([0.0], [cfg["START_ENERGY"] * scale], marker="x", ms=6, ls="none", color=color)
        if args.estimate:
            tau, estimates = window_means(r["tau"], r["e_estimate"], r["weights"], args.window)
            ax.plot(tau, estimates * scale, "--", lw=1.0, color=color, alpha=0.8)
        if not args.no_trial and "E_DMRG" in cfg:
            ax.axhline(cfg["E_DMRG"] * scale, color=color, ls=":", lw=1.2)
        s = staggered_structure_factor(r["szsz"], cfg) if "szsz" in r else np.full(len(r["energies"]), np.nan)
        if ax_s is not None and "szsz" in r:
            ax_s.plot(*window_means(r["tau"], s, r["weights"], args.window), "-", marker="o", ms=2.5, color=color)
        for b, (t, e, w, est, s_b) in enumerate(zip(r["tau"], r["energies"], r["weights"], r["e_estimate"], s)):
            rows.append(f"{name},{chi_t},{b},{t},{e},{w},{est},{s_b}")
    if args.discard_scan:
        for r in runs:
            discard_scan(r, args.scan_step, args.min_keep)

    cfg0 = runs[0]["config"]
    if args.e_ref is not None:
        ax.axhline(args.e_ref * (1.0 / cfg0["L"] if args.per_site else 1.0), color=INK, lw=1.4,
                   label=f"reference energy (DMRG): {args.e_ref:.4f}")
    for tau_eql in sorted({r["config"]["N_EQL"] * r["config"]["N_PROP"] * r["config"]["DT"] for r in runs}):
        for a in (ax, ax_s) if ax_s is not None else (ax,):
            a.axvline(tau_eql, color=INK2, lw=1.0, ls="--", zorder=0)
    (ax if ax_s is None else ax_s).set_xlabel(r"imaginary time $\tau$")
    window = f", means of {args.window} blocks" if args.window > 1 else ""
    ax.set_ylabel("CPMC energy per block" + (" per site" if args.per_site else "") + window)
    if ax_s is not None:
        ax_s.set_ylabel(r"$S(\pi,\pi)$ of the mixed $\langle S^z_i S^z_j \rangle$" + window)
    lattice = f"{cfg0['LX']}x{cfg0['LY']}" if "LX" in cfg0 else f"L={cfg0['L']}"
    walker_bond = "" if several_chi_w else f", walker bond {cfg0.get('CHI_PROP')} per spin"
    ax.set_title(f"CPMC energy per block: {lattice}, U={cfg0['U']:g}, {cfg0['N_WALKERS']} walkers{walker_bond}")
    extra = ["dashed: running estimate"] if args.estimate else []
    extra += ["dotted: DMRG trial energy"] if not args.no_trial else []
    extra += ["x: the walkers' start"] if any("START_ENERGY" in r["config"] for r in runs) else []
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
