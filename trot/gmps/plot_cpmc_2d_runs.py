"""Plot square-lattice CPMC energy per block against imaginary time, one figure per trial bond dimension, with DMRG
references as horizontal lines.

plot_cpmc_runs.py (whose loaders and statistics this reuses) plus the ground truth: DMRG energies from
`mps_cpmc_2d.py dmrg` logs or from its kind="dmrg" records in a results.jsonl. A directory argument is also searched
for dmrg_*.log files and such records; every bond dimension whose lattice, boundaries, filling and U match the runs
is drawn, darker for larger bond, and the largest is the E_ref of the printed table. Logs of another model, or of a
DMRG run that has not finished, are skipped with the reason printed, so rerunning as runs finish adds their lines. The line is e_mps, <H> of the final MPS, which is variational (not e_davidson, which sits below it).

Each trial chi_T gets its own figure, <out>_T<chi_T>.png, holding its runs (one per walker bond chi_w, coloured the
same in every figure), the trial's energy and the references; <out>.csv holds every run's curve. The curves start at
tau = 0 with the starting walkers' mixed energy, which the block log does not record but its running estimate
determines (add_start_energy); the statistics use the logged blocks only.

    python plot_cpmc_2d_runs.py square_2d_data --out square_2d_data/sq4x4_U8_energy_vs_tau.png
    python plot_cpmc_2d_runs.py square_2d_data --dmrg square_2d_data/dmrg_4x4_open_U8_chi2000.log
    python plot_cpmc_2d_runs.py square_2d_data --reference -6.83 --reference-label "DMRG chi=2000"
"""
import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np

try:  # run as a script: plot_cpmc_runs.py sits next to it
    from plot_cpmc_runs import GRID, INK, INK2, PALETTE, SURFACE, find_runs, sampling_stats
except ImportError:
    from trot.gmps.plot_cpmc_runs import GRID, INK, INK2, PALETTE, SURFACE, find_runs, sampling_stats

REFERENCE_SHADES = ("#bdbcb6", "#8a8984", INK2, INK)  # DMRG lines, lightest for the smallest bond
MODEL = ("LX", "LY", "BOUNDARY_X", "BOUNDARY_Y", "N_UP", "N_DN", "U")  # what a reference must share with the runs
LOG_HEADER = re.compile(r"(\d+)x(\d+) lattice, boundaries x=([^,\s]+) y=([^,\s]+), \((\d+),(\d+)\), U=([-+.\deE]+)")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def add_boundaries(runs):
    """load_block_logs keeps Lx and Ly but not the boundaries: take them from each run's results.jsonl record."""
    records = {}
    for r in runs:
        results = r["path"].with_name("results.jsonl")
        if r["path"].suffix != ".jsonl" or not results.exists():
            continue
        if results not in records:
            records[results] = {rec.get("tag"): rec for rec in read_jsonl(results)}
        rec = records[results].get(r["config"].get("tag"), {})
        if "boundary_x" in rec:
            r["config"].update(BOUNDARY_X=rec["boundary_x"], BOUNDARY_Y=rec["boundary_y"])


def add_start_energy(runs):
    """r["e_start"]: the mixed energy of the starting walkers at tau = 0, where trot's running estimate starts.
    The logged estimate follows e_b = (1 - a) e_(b-1) + a E_b (the shift_ema update, a = 0.1 by default), so the
    value before block 0 is (e_0 - a E_0) / (1 - a). a is read off the later blocks, and a run whose estimate does
    not follow the rule on every pair of consecutive blocks gets no tau = 0 point."""
    for r in runs:
        E, S, cfg = r["energies"], r["e_estimate"], r["config"]
        block_length = cfg["N_PROP"] * cfg["DT"]
        consecutive = np.isclose(np.diff(r["tau"]), block_length)
        if not np.isclose(r["tau"][0], block_length) or consecutive.sum() < 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.nanmedian(((S[1:] - S[:-1]) / (E[1:] - S[:-1]))[consecutive])
        residual = np.abs(S[1:] - (1 - a) * S[:-1] - a * E[1:])[consecutive].max()
        if not residual < 1e-9 * max(1.0, np.abs(S).max()):
            print(f"no tau = 0 point for {cfg.get('tag')}: its running estimate does not follow the update rule "
                  f"(residual {residual:.1e})")
            continue
        r["e_start"] = (S[0] - a * E[0]) / (1 - a)


def reference_from_log(path):
    """(reference, "") from a `mps_cpmc_2d.py dmrg` log: the model from its lattice line, the energy from the
    record it prints last. (None, reason) when either line is missing."""
    text = path.read_text()
    final = [line for line in text.splitlines() if line.startswith("{'e_davidson'")]
    if not final:
        return None, "no final energy line (the run did not finish, or its output was cut off)"
    header = LOG_HEADER.search(text)
    if not header:
        return None, "no lattice line, so its model cannot be checked"
    record = ast.literal_eval(final[-1])
    lx, ly, bx, by, n_up, n_dn, u = header.groups()
    return dict(LX=int(lx), LY=int(ly), BOUNDARY_X=bx, BOUNDARY_Y=by, N_UP=int(n_up), N_DN=int(n_dn), U=float(u),
                energy=float(record["e_mps"]), chi=max(record["bond_dims"]), source=path.name), ""


def references_from_results(path):
    """The kind="dmrg" records of a results.jsonl (written by `mps_cpmc_2d.py dmrg result_json=...`)."""
    return [dict(LX=r["Lx"], LY=r["Ly"], BOUNDARY_X=r["boundary_x"], BOUNDARY_Y=r["boundary_y"], N_UP=r["n_up"],
                 N_DN=r["n_down"], U=float(r["interaction"]), energy=float(r["e_mps"]), chi=max(r["bond_dims"]),
                 source=f"{path.name}, dmrg record {r.get('tag') or i}")
            for i, r in enumerate(read_jsonl(path)) if r.get("kind") == "dmrg"]


def find_references(runs, paths, explicit):
    """The DMRG references matching every run's model, one per bond dimension and ordered by it, among the
    explicit files (an error if none of them matches) or else the dmrg logs and records in the directories given."""
    candidates, rejected = [], []
    if explicit:
        sources = list(map(Path, explicit))
    else:
        dirs = [p for p in map(Path, paths) if p.is_dir()]
        sources = [f for d in dirs for f in sorted(d.glob("dmrg_*.log")) + [d / "results.jsonl"] if f.exists()]
    for f in sources:
        if f.suffix == ".jsonl":
            candidates += references_from_results(f)
        else:
            ref, reason = reference_from_log(f)
            (candidates.append(ref) if ref else rejected.append(f"{f.name}: {reason}"))
    matching = []
    for ref in candidates:
        diff = sorted({f"{k} {ref[k]} vs runs' {r['config'][k]}" for r in runs for k in MODEL
                       if k in r["config"] and ref[k] != r["config"][k]})
        (rejected.append(f"{ref['source']}: model differs ({', '.join(diff)})") if diff else matching.append(ref))
    for reason in rejected:
        print(f"skipping DMRG reference {reason}")
    if explicit and not matching:
        raise SystemExit("no usable DMRG reference among " + ", ".join(explicit))
    by_chi = {}
    for ref in matching:  # a bond dimension found twice (a log and its results record) is drawn once
        by_chi.setdefault(ref["chi"], ref)
    return [by_chi[chi] for chi in sorted(by_chi)]


def draw(plt, runs, chi_t, colors, references, args, out):
    """The figure of one trial: its runs' block energies, the trial energy, the references, the equilibration end."""
    several_dt = len({r["config"]["DT"] for r in runs}) > 1  # then the labels say each run's time step
    cfg0 = runs[0]["config"]
    scale = 1.0 / cfg0["L"] if args.per_site else 1.0
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for r in runs:
        cfg, (mean, err, _) = r["config"], r["stats"]
        color = colors[cfg.get("CHI_PROP")]
        state = "" if r["finished"] else " (running)"
        run_label = rf"$\chi_w$={cfg.get('CHI_PROP')}" + (f", dt={cfg['DT']:g}" if several_dt else "") + state
        label = rf"{run_label}: {mean * scale:.6f} $\pm$ {err * scale:.1e}" if np.isfinite(mean) else run_label
        tau, energies, estimate = r["tau"], r["energies"], r["e_estimate"]
        if "e_start" in r:  # the starting walkers at tau = 0, where the running estimate also starts
            tau, energies, estimate = np.r_[0.0, tau], np.r_[r["e_start"], energies], np.r_[r["e_start"], estimate]
        ax.plot(tau, energies * scale, "-", marker="o", ms=2.5, color=color, label=label)
        if args.estimate:
            ax.plot(tau, estimate * scale, "--", lw=1.0, color=color, alpha=0.8, label=rf"{run_label}: running estimate")
    if not args.no_trial:  # <trial|H|trial> of the trial the runs used; the Davidson energy for older records
        for e_trial in sorted({r["config"].get("E_TRIAL", r["config"].get("E_DMRG")) for r in runs} - {None}):
            ax.axhline(e_trial * scale, color=INK2, ls=":", lw=1.4, label=f"trial energy: {e_trial * scale:.6f}")
    for i, ref in enumerate(references):  # ordered by bond, the largest drawn darkest
        shade = REFERENCE_SHADES[max(0, len(REFERENCE_SHADES) - len(references) + i)]
        ax.axhline(ref["energy"] * scale, color=shade, lw=1.5, zorder=3, label=f"{ref['label']}: {ref['energy'] * scale:.6f}")
    for i, tau_eql in enumerate(sorted({r["config"]["N_EQL"] * r["config"]["N_PROP"] * r["config"]["DT"] for r in runs})):
        ax.axvline(tau_eql, color=INK2, lw=1.0, ls="--", zorder=0, label="end of equilibration" if i == 0 else None)
    ax.set_xlabel(r"imaginary time $\tau$")
    ax.set_ylabel("CPMC energy per block" + (" per site" if args.per_site else ""))
    lattice = f"{cfg0['LX']}x{cfg0['LY']}" if "LX" in cfg0 else f"L={cfg0['L']}"
    boundaries = f" {cfg0['BOUNDARY_X']}/{cfg0['BOUNDARY_Y']}" if "BOUNDARY_X" in cfg0 else ""
    ax.set_title(rf"CPMC {lattice}{boundaries}, U={cfg0['U']:g}, {cfg0['N_WALKERS']} walkers, trial $\chi_T$={chi_t}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    try:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    except RuntimeError as error:  # the cluster's matplotlib glyph-raster bug, see plot_cpmc_runs.py
        if out.suffix.lower() == ".pdf":
            raise
        print(f"could not draw {out} ({error}); saving a PDF instead")
        out = out.with_suffix(".pdf")
        fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+",
                        help="walker .npz files, progress.json files, blocks.jsonl files or directories of runs")
    parser.add_argument("--dmrg", nargs="+", default=[],
                        help="DMRG reference: mps_cpmc_2d.py dmrg logs or results.jsonl files with its records "
                             "(default: search the directories given)")
    parser.add_argument("--reference", type=float, help="reference energy to draw instead, e.g. from elsewhere")
    parser.add_argument("--reference-label", default="reference", help="its legend label")
    parser.add_argument("--per-site", action="store_true", help="plot E / L")
    parser.add_argument("--estimate", action="store_true", help="also draw the running energy estimate")
    parser.add_argument("--no-trial", action="store_true", help="do not draw the trial energies")
    parser.add_argument("--out", default="cpmc_2d_energy_vs_tau.png",
                        help="figure path: each trial's figure is <stem>_T<chi_T><suffix>, the .csv with the curves "
                             "is <stem>.csv")
    parser.add_argument("--show", action="store_true", help="also open the figure windows")
    args = parser.parse_args()

    runs = find_runs(args.paths)
    if not runs:
        raise SystemExit("no runs found")
    add_boundaries(runs)
    add_start_energy(runs)
    runs.sort(key=lambda r: (r["config"].get("DMRG_CHI_T", 0), r["config"].get("CHI_PROP") or 0, r["config"]["DT"],
                             str(r["path"])))
    if args.reference is not None:
        references = [dict(energy=args.reference, label=args.reference_label)]
    else:
        references = find_references(runs, args.paths, args.dmrg)
        for ref in references:
            ref["label"] = rf"DMRG $\chi$={ref['chi']}"
            print(f"DMRG reference: {ref['source']}, chi {ref['chi']}, e_mps {ref['energy']:.8f}")
        if not references:
            print("no DMRG reference matching the runs: drawing none")

    rows = ["run,trial_chi,walker_chi,block,tau,energy,weight,e_estimate"]
    e_ref = references[-1]["energy"] if references else float("nan")  # the largest bond
    print(f"{'run':40s} {'chi_T':>5s} {'chi_w':>5s} {'blocks':>6s} {'E(tau=0)':>11s} {'E (sampling)':>14s} "
          f"{'error':>9s} {'E - E_ref':>10s} {'sigmas':>6s}  method")
    for r in runs:
        cfg = r["config"]
        chi_t, chi_w = cfg.get("DMRG_CHI_T"), cfg.get("CHI_PROP")
        r["stats"] = mean, err, method = sampling_stats(r["energies"], r["weights"], cfg["N_EQL"])
        name = cfg.get("tag") or r["path"].name
        print(f"{name[:40]:40s} {chi_t!s:>5s} {chi_w!s:>5s} {len(r['energies']):6d} {r.get('e_start', np.nan):11.6f} "
              f"{mean:14.8f} {err:9.2e} {mean - e_ref:10.5f} {(mean - e_ref) / err:6.1f}  {method}"
              + ("" if r["finished"] else " (running)"))
        if "e_start" in r:  # block -1: tau = 0, every weight 1, the running estimate's starting value
            rows.append(f"{name},{chi_t},{chi_w},-1,0.0,{r['e_start']},{cfg['N_WALKERS']},{r['e_start']}")
        blocks = np.rint(r["tau"] / (cfg["N_PROP"] * cfg["DT"])).astype(int) - 1  # a lost log line leaves a gap
        for b, t, e, w, est in zip(blocks, r["tau"], r["energies"], r["weights"], r["e_estimate"]):
            rows.append(f"{name},{chi_t},{chi_w},{b},{t},{e},{w},{est}")
    out = Path(args.out)
    out.with_suffix(".csv").write_text("\n".join(rows) + "\n")
    print(f"saved {out.with_suffix('.csv')}")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 13,
                         "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.6, "legend.frameon": False, "font.size": 10})
    walker_chis = sorted({r["config"].get("CHI_PROP") for r in runs}, key=lambda c: (c is None, c))
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(walker_chis)}  # one colour per chi_w everywhere
    for chi_t in sorted({r["config"].get("DMRG_CHI_T") for r in runs}, key=lambda c: (c is None, c)):
        group = [r for r in runs if r["config"].get("DMRG_CHI_T") == chi_t]
        draw(plt, group, chi_t, colors, references, args, out.with_name(f"{out.stem}_T{chi_t}{out.suffix}"))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
