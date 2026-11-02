"""Plots of the error_analysis_8x8.py results, read from its slurm log (numpy and matplotlib only, to run offline).

    python plot_error_analysis.py slurm/error_analysis_8x8_ref1024-7291195.out [--out DIR] [--title TEXT] [--show]

Writes three figures to --out (default error_analysis_plots/ next to this script):
  entropy_by_cut.png          the walker entanglement entropy at every cut: weighted mean, median and 10-90% band
  infidelity_new.png          infidelity against the exact walker (the walker's own plan: 1 - (1 - plan loss)(1 - discarded
                              weight)) versus the walker bond: median, worst, and every walker
  infidelity_old_vs_new.png   the walkers scored both ways, old (overlap with the reference walker) against new, and the
                              ratio old / new; dotted: how far the old way's reference is from the exact walker
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
FLOOR = 1e-16  # values at or below this (exact zeros, rounding noise) are drawn at this height on the log axes


def floats(text):
    return np.array([float(x) for x in re.findall(FLOAT, text)])


def array(text, name):
    """The numpy array printed after `name` at the start of a line (None if the log has no such line)."""
    match = re.search(rf"^{re.escape(name)}\s+array\(\[(.*?)\]", text, re.M | re.S)
    return None if match is None else floats(match.group(1))


def comparison(text):
    """{walker: (reference chi, reference distance from exact, {chi_w: (old, new)})} of the old-versus-new block."""
    rows, walker = {}, None
    for line in text.splitlines():
        match = re.match(rf"walker (\d+): the chi (\d+) reference itself is ({FLOAT}) from", line)
        if match:
            walker = int(match[1])
            rows[walker] = (int(match[2]), float(match[3]), {})
            continue
        match = re.match(rf"\s+chi_w\s+(\d+): old ({FLOAT})\s+new ({FLOAT})", line)
        if match and walker is not None:
            rows[walker][2][int(match[1])] = (float(match[2]), float(match[3]))
    return rows


def clip(values):
    return np.maximum(np.asarray(values, float), FLOOR)


def bond_axis(ax, chis):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(chis)
    ax.set_xticklabels([str(c) for c in chis])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(alpha=0.25)


def plot_entropy(text, title, out):
    cuts, mean = array(text, "cuts"), array(text, "cut_mean")
    q10, q50, q90 = (array(text, f"cut_q{p}") for p in (10, 50, 90))
    center = re.search(r"^center_cut (\d+)", text, re.M)
    if any(x is None for x in (cuts, mean, q10, q50, q90)) or center is None:
        print("no entropy arrays in the log: entropy_by_cut.png skipped")
        return
    cuts = cuts.astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(cuts, q10, q90, color="C1", alpha=0.20, label="10--90% walkers")
    ax.plot(cuts, q50, ":", color="C1", label="median walker")
    ax.plot(cuts, mean, "o-", color="C1", label="weighted mean")
    ax.axvline(int(center[1]), color="k", ls="--", lw=1, label="center cut")
    ax.set_xlabel("cut after site")
    ax.set_ylabel("entropy S (nats)")
    ax.set_xticks(cuts[::4])
    ax.set_title(f"{title} walker entanglement", fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    fig.savefig(out / "entropy_by_cut.png", dpi=150, bbox_inches="tight")


def plot_new(text, title, out, rows):
    chis, median, worst = array(text, "chi_w"), array(text, "median infidelity"), array(text, "worst infidelity")
    if any(x is None for x in (chis, median, worst)):
        print("no infidelity arrays in the log: infidelity_new.png skipped")
        return
    chis = chis.astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, chi in enumerate(chis):  # every scored walker at this bond
        match = re.search(rf"^infidelity per walker, chi_w {chi}\s+array\(\[(.*?)\]", text, re.M | re.S)
        if match:
            y = clip(floats(match.group(1)))
            ax.plot(np.full(len(y), chi), y, ".", color="0.6", ms=5, label="each walker" if k == 0 else None)
    ax.plot(chis, clip(median), "o-", color="C0", label="median walker")
    ax.plot(chis, clip(worst), "o-", color="C3", label="worst walker")
    for chi in sorted({row[0] for row in rows.values()}):
        ax.axvline(chi, color="k", ls=":", lw=1, label="the old way's reference bond")
    bond_axis(ax, chis)
    ax.set_xlabel("chi_w")
    ax.set_ylabel("infidelity against the exact walker")
    ax.set_title(f"{title} walker truncation error (own plan)", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.savefig(out / "infidelity_new.png", dpi=150, bbox_inches="tight")


def plot_old_vs_new(title, out, rows):
    if not rows:
        print("no old-versus-new block in the log: infidelity_old_vs_new.png skipped")
        return
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.6]))
    all_chis = sorted({chi for _, _, by_chi in rows.values() for chi in by_chi})
    for n, (walker, (ref_chi, ref_error, by_chi)) in enumerate(rows.items()):
        chis = sorted(by_chi)
        old, new = (np.array([by_chi[chi][j] for chi in chis]) for j in range(2))
        color = f"C{n}"
        ax.plot(chis, clip(old), "o-", color=color, label=f"walker {walker}: old (overlap with the chi {ref_chi} walker)")
        ax.plot(chis, clip(new), "s--", color=color, label=f"walker {walker}: new (own plan, against the exact walker)")
        ax.axhline(max(ref_error, FLOOR), color=color, ls=":", lw=1)
        bx.plot(chis, np.where(new > 0, old / np.where(new > 0, new, 1.0), np.nan), "o-", color=color)
    ax.plot([], [], ":", color="k", lw=1, label="the reference's own distance from the exact walker")
    bond_axis(ax, all_chis)
    ax.set_ylabel("infidelity")
    ax.set_title(f"{title} walker truncation error, old and new way", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)
    bx.axhline(1.0, color="k", ls="--", lw=1)
    bx.set_ylim(bottom=0)
    bx.set_xlabel("chi_w")
    bx.set_ylabel("old / new")
    bx.grid(alpha=0.25)
    fig.savefig(out / "infidelity_old_vs_new.png", dpi=150, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("log", type=Path, help="the slurm log of error_analysis_8x8.py")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "error_analysis_plots")
    p.add_argument("--title", default="8×8 Hubbard U = 8")
    p.add_argument("--show", action="store_true", help="also open the figures")
    args = p.parse_args()

    text = args.log.read_text()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = comparison(text)
    plot_entropy(text, args.title, args.out)
    plot_new(text, args.title, args.out, rows)
    plot_old_vs_new(args.title, args.out, rows)
    print(f"figures in {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
