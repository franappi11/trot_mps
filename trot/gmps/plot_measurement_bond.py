"""Figure: measure the local energy with more accurate walkers than you sample with (doped 6x6, chi_T 256).

Left: the energy error of the same saved walkers against their converged conversion (rank_exact at walker bond 64),
for the bond the energy is measured with: about +0.25 at 16, +0.06 at 32, 0 at 64. Right: the mixed energy of those
walkers at each measurement bond; walkers sampled at bond 16 and at bond 32 agree once measured at 64, so sampling at
16 is fine, and each run's own average (sampled and measured at the same bond) sits on its curve.

The numbers are the tables walker_truncation_check.py --part overlaps printed ("block <b> against rank_exact:64"),
read from its logs; the run averages from the runs' result.npz. Plain text only (no mathtext: it crashed an earlier
plot on this machine with a FreeType raster overflow). CPU, seconds, from this folder:

    sbatch -p genx --cpus-per-task=1 --mem=2G --time=00:05:00 -J plot_mbond \\
        --export=ALL,TARGET=plot_measurement_bond.py run_mps_sweep.sh
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WALKERS = Path("/mnt/ceph/users/fnappi/trot_walkers/sq6x6_doped_diag")
# (log, bond the walkers were sampled with); equilibration blocks only enter the left panel
LOGS = [(HERE / "slurm/trunc_overlaps-7256275.out", 16), (HERE / "slurm/trunc_ov_T256-7263042.out", 16),
        (HERE / "slurm/trunc_ov_w32-7263043.out", 32)]
RUNS = {16: WALKERS / "dmrg_chi256_w16/result.npz", 32: WALKERS / "dmrg_chi256_w32/result.npz"}
EQUILIBRATED = {16: 100, 32: 30}  # first sampling block of each run (N_EQL)
BONDS = (16, 32, 64)
SERIES = {16: "#2a78d6", 32: "#eb6834"}  # categorical slots 1-2 of the dataviz reference palette
SURFACE, GRID, INK, INK2 = "#fcfcfb", "#e6e5e1", "#0b0b0b", "#52514e"

HEADER = re.compile(r"^block\s+(\d+)\s+against\s+(\S+):")
ROW = re.compile(r"^\s+adaptive:(\d+)\s+(-?\d+\.\d+)\s+([+-]\d+\.\d+)\(([\d.]+e[+-]\d+)\)")


def read_tables(path):
    """{(block, measurement bond): (mixed energy, shift, shift error)} of one log's tables."""
    rows, block = {}, None
    for line in Path(path).read_text().splitlines():
        header = HEADER.match(line)
        if header:
            block = int(header.group(1))
            continue
        row = ROW.match(line)
        if row and block is not None:
            rows[block, int(row.group(1))] = tuple(float(row.group(k)) for k in (2, 3, 4))
    if not rows:
        raise SystemExit(f"no 'block ... against' tables in {path}")
    return rows


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(BONDS)))
    ax.set_xticklabels([str(b) for b in BONDS])
    ax.set_xlim(-0.35, len(BONDS) - 0.65)
    ax.set_xlabel("walker bond used to measure the energy (per spin channel)", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "plots" / "measurement_bond")
    args = parser.parse_args(argv)

    data = {16: {}, 32: {}}
    for path, sampled in LOGS:
        data[sampled].update(read_tables(path))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    style(left, "Energy error of the same walkers", "energy error vs converged conversion")
    style(right, "Mixed energy of the same walkers", "mixed energy (t = 1)")
    left.axhline(0.0, color=INK2, linewidth=0.8)

    for sampled, rows in data.items():
        color = SERIES[sampled]
        label = f"sampled with bond {sampled}"
        blocks = sorted({b for b, _ in rows})
        means, late = [], []
        for x, bond in enumerate(BONDS):
            shifts = [rows[b, bond][1] for b in blocks if (b, bond) in rows]
            errors = [rows[b, bond][2] for b in blocks if (b, bond) in rows]
            if not shifts:
                continue
            jitter = np.linspace(-0.08, 0.08, len(shifts)) if len(shifts) > 1 else [0.0]
            left.errorbar(x + np.asarray(jitter), shifts, yerr=errors, fmt="o", ms=4, color=color, alpha=0.55,
                          elinewidth=1, capsize=0, zorder=2)
            means.append((x, float(np.mean(shifts))))
            energies = [rows[b, bond][0] for b in blocks if (b, bond) in rows and b >= EQUILIBRATED[sampled]]
            if energies:
                right.plot(x + np.asarray(np.linspace(-0.08, 0.08, len(energies)) if len(energies) > 1 else [0.0]),
                           energies, "o", ms=4, color=color, alpha=0.55, zorder=2)
                late.append((x, float(np.mean(energies))))
        for ax, points in ((left, means), (right, late)):
            xs, ys = zip(*points)
            ax.plot(xs, ys, "-o", color=color, linewidth=2, ms=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=label, zorder=3, solid_capstyle="round")
        with np.load(RUNS[sampled]) as run:
            energy, error = float(run["energy"]), float(run["energy_err"])
        x = BONDS.index(sampled)
        right.errorbar([x + 0.22], [energy], yerr=[error], fmt="D", ms=7, color=color, markeredgecolor=INK,
                       markeredgewidth=1, elinewidth=1.2, capsize=0, zorder=4)
        right.annotate(f"run average {energy:.3f}", (x + 0.3, energy), color=INK2, fontsize=8, va="center")
        if sampled == 16:
            for x, y in means:
                left.annotate(f"{y:+.2f}", (x, y), xytext=(10, 4), textcoords="offset points", color=INK, fontsize=9)

    left.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    right.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.suptitle("Doped 6x6 Hubbard, U = 8, trial bond 256: the energy needs bond 64, the sampling does not",
                 color=INK, fontsize=11, x=0.01, ha="left")
    # Small dots: one saved block each (400 walkers; left error bars are paired standard errors); large dots: block
    # means; diamonds: each run's own average. No bbox_inches="tight" and no wrapped text: both draw the text with the
    # Agg rasterizer, whose FreeType fails here ("FT_Render_Glyph ... raster overflow", job 7279425, also in
    # plot_cpmc_runs.py), even for a PDF. The PDF embeds glyph outlines and is written first; the PNG is optional.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"saved {args.out.with_suffix('.pdf')}", flush=True)
    try:
        fig.savefig(args.out.with_suffix(".png"), dpi=200, facecolor=SURFACE)
        print(f"saved {args.out.with_suffix('.png')}", flush=True)
    except RuntimeError as error:
        print(f"no PNG ({error}); use the PDF", flush=True)
    for sampled, rows in data.items():
        for bond in BONDS:
            values = [rows[k] for k in sorted(rows) if k[1] == bond]
            if values:
                print(f"sampled {sampled}, measured {bond}: mean shift {np.mean([v[1] for v in values]):+.4f} over "
                      f"{len(values)} blocks", flush=True)


if __name__ == "__main__":
    main()
