"""Local energies of the saved walkers of a uhf_trial_cpmc_gpu.py run, against trot's energy window.

trot's block energy (trot.prop.blocks.block) replaces every walker's local energy that lies more than
sqrt(2/dt) from the running energy estimate by the estimate itself. With a weak trial the local energies
spread wide and skew low, so the replacement could raise the block energies (the RHF-trial run at L = 100,
U = 8 is 0.27 above DMRG). This takes the walkers of the run's last saved state (after the last block's
reconfiguration, so all weights are equal) and gives how many lie outside the window, and the mean with
trot's replacement, with the window edges as caps instead, and with no window.

It is one snapshot of the population: the fraction outside the window is the main number; the shifts of the
mean carry roughly the scatter of one block, and copies made by the reconfiguration are not independent.
The numbers go to <out>/<tag>_local_energy_check.json before the histogram <tag>_local_energy_hist.pdf.

    # CPU only: a genx job (never the login node), from trot/gmps, with the run's arguments
    sbatch -p genx -J uhf_eloc --time=00:10:00 --cpus-per-task=2 --mem=4G \
        --export=ALL,TARGET=uhf_local_energy_check.py run_mps_sweep.sh --trial rhf
"""
import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from trot.core.ops import k_energy
from trot.core.system import System
from trot.gmps.uhf_cpmc import rhf_scf, uhf_scf
from trot.ham.chol import HamChol
from trot.meas.uhf import make_uhf_meas_ops
from trot.trial.uhf import UhfTrial, overlap_u

HERE = Path(__file__).resolve().parent
# as uhf_trial_cpmc_gpu.py: (L, n_up, n_dn, t, U) -> DMRG reference
REFERENCES = {(100, 50, 50, 1.0, 8.0): -32.545774923969844}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
INK2, GRID, SURFACE = "#52514e", "#e6e5e1", "#fcfcfb"

# the run's arguments, as in uhf_trial_cpmc_gpu.py, to rebuild its tag and trial
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--L", type=int, default=100)
parser.add_argument("--n-up", type=int, default=None)
parser.add_argument("--n-dn", type=int, default=None)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--U", type=float, default=8.0)
parser.add_argument("--walkers", type=int, default=400)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--prop", default="fast", choices=["fast", "slow"])
parser.add_argument("--trial", default="uhf", choices=["uhf", "rhf"])
parser.add_argument("--walker-start", default="trial", choices=["trial", "uhf"])
parser.add_argument("--out", default=str(HERE / "uhf_cpmc_data"), help="the run's --out folder")
parser.add_argument("--state", default=None, help="a state .npz (default: the run's last saved state in --out)")
args = parser.parse_args()

L, T, U, DT = args.L, args.t, args.U, args.dt
N_UP = L // 2 if args.n_up is None else args.n_up
N_DN = L // 2 if args.n_dn is None else args.n_dn
WALKER_START = "trial" if args.trial == "uhf" else args.walker_start
TAG = (f"L{L}_N{N_UP}-{N_DN}_U{U:g}_nw{args.walkers}_dt{DT:g}_n{args.steps}_s{args.seed}"
       + ("" if args.trial == "uhf" else f"_{args.trial}trial") + ("_wuhf" if WALKER_START == "uhf" else "")
       + ("_fast" if args.prop == "fast" else ""))
DATA_DIR = Path(args.out).resolve()
if args.state:
    STATE = Path(args.state).resolve()
else:
    found = sorted(DATA_DIR.glob(f"{TAG}_state_*.npz"))
    if len(found) != 1:
        raise SystemExit(f"expected one {TAG}_state_*.npz in {DATA_DIR}, found {[f.name for f in found]}")
    STATE = found[0]
E_REF = REFERENCES.get((L, N_UP, N_DN, T, U))
THRESH = float(np.sqrt(2.0 / DT))           # trot.prop.blocks.block: jnp.sqrt(2.0 / params.dt)
print(f"jax {jax.__version__} on {jax.default_backend()}; run {TAG}; walkers from {STATE}", flush=True)

# ---- the run's trial and energy kernel (uhf_trial_cpmc_gpu.run)
h1 = np.zeros((L, L))
i = np.arange(L - 1)
h1[i, i + 1] = h1[i + 1, i] = -T
if args.trial == "rhf":
    C, e_trial, _, _ = rhf_scf(h1, U, N_UP)
    Ta, Tb = C, C
else:
    Ta, Tb, e_trial, _, _ = uhf_scf(h1, U, N_UP, N_DN)
onsite = np.zeros((L, L, L))
onsite[np.arange(L), np.arange(L), np.arange(L)] = np.sqrt(U)
ham_meas = HamChol(h0=jnp.zeros(()), h1=jnp.asarray(h1), chol=jnp.asarray(onsite))
system = System(norb=L, nelec=(N_UP, N_DN), walker_kind="unrestricted")
trial = UhfTrial(mo_coeff_a=jnp.asarray(Ta), mo_coeff_b=jnp.asarray(Tb))
meas_ops = make_uhf_meas_ops(system)
meas_ctx = meas_ops.build_meas_ctx(ham_meas, trial)
e_kernel = meas_ops.require_kernel(k_energy)

# ---- the saved walkers (trot.prop.blocks.dump_prop_state_npz)
z = np.load(STATE)
walkers = (jnp.asarray(z["walkers_0"]), jnp.asarray(z["walkers_1"]))
w = np.asarray(z["weights"], float)
e_ref = float(z["e_estimate"])              # the estimate the next block's window is centred on
ov = np.asarray(jax.vmap(overlap_u, in_axes=(0, None))(walkers, trial))
# the run stored |<T|W>| up to the signs of the trial's orbitals, which a fresh eigh may flip
ov_check = float(np.max(np.abs(np.abs(ov) - np.abs(z["overlaps"]))) / np.max(np.abs(z["overlaps"])))
# 50 walkers at a time: the exchange term holds (L, N, N) per walker and spin (2 MB at L = 100), so all 400
# at once would need a few GB on the CPU
energy = jax.jit(jax.vmap(e_kernel, in_axes=(0, None, None, None)))
e = np.concatenate([np.asarray(jnp.real(energy((walkers[0][s:s + 50], walkers[1][s:s + 50]), ham_meas, meas_ctx,
                                               trial))) for s in range(0, walkers[0].shape[0], 50)])

# ---- trot's rule against capping and no window
ok = np.isfinite(e)
outside = ok & (np.abs(e - e_ref) > THRESH)
below, above = int(np.sum(ok & (e < e_ref - THRESH))), int(np.sum(ok & (e > e_ref + THRESH)))
w_ok = np.where(ok, w, 0.0)
mean_trot = float(np.sum(w_ok * np.where(outside, e_ref, np.where(ok, e, 0.0))) / np.sum(w_ok))
mean_cap = float(np.sum(w_ok * np.clip(np.where(ok, e, e_ref), e_ref - THRESH, e_ref + THRESH)) / np.sum(w_ok))
mean_raw = float(np.sum(w_ok * np.where(ok, e, 0.0)) / np.sum(w_ok))
std = float(np.sqrt(np.sum(w_ok * (np.where(ok, e, mean_raw) - mean_raw) ** 2) / np.sum(w_ok)))
skew = float(np.sum(w_ok * (np.where(ok, e, mean_raw) - mean_raw) ** 3) / np.sum(w_ok) / std ** 3)
n, n_ok = len(e), int(ok.sum())
n_distinct = len(np.unique(np.round(np.abs(ov) / np.max(np.abs(ov)), 12)))   # copies from the reconfiguration
quant = dict(zip(["min", "q01", "q05", "median", "q95", "q99", "max"],
                 np.quantile(e[ok], [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).tolist()))

print(f"{args.trial.upper()} trial E = {e_trial:.6f}; overlaps recomputed vs saved (abs): relative deviation {ov_check:.1e}")
print(f"{n} walkers ({n_distinct} distinct after the reconfiguration), {n - n_ok} with a non-finite local energy")
print(f"energy estimate (window centre) {e_ref:.6f}, window ±{THRESH:.3f}: [{e_ref - THRESH:.3f}, {e_ref + THRESH:.3f}]")
print(f"local energies: mean {mean_raw:.6f}, std {std:.4f}, skewness {skew:+.3f}; "
      + ", ".join(f"{k} {v:.3f}" for k, v in quant.items()))
print(f"outside the window: {int(outside.sum())} of {n_ok} ({outside.sum() / n_ok:.2%}): {below} below, {above} above")
print(f"mean with trot's replacement {mean_trot:.6f}, with capping at the edges {mean_cap:.6f}, with no window "
      f"{mean_raw:.6f}; trot - no window {mean_trot - mean_raw:+.6f} (walker scatter std/sqrt(n) {std / np.sqrt(n_ok):.4f},"
      f" an underestimate with copies)")
if E_REF is not None:
    print(f"DMRG reference {E_REF:.6f}")

out = dict(tag=TAG, state=str(STATE), trial=args.trial, e_trial=e_trial, dt=DT, window=THRESH, e_estimate=e_ref,
           n_walkers=n, n_distinct=n_distinct, n_nonfinite=n - n_ok, n_outside=int(outside.sum()), n_below=below,
           n_above=above, mean_trot=mean_trot, mean_capped=mean_cap, mean_no_window=mean_raw, std=std, skewness=skew,
           quantiles=quant, overlap_check=ov_check, e_ref=E_REF)
json_path = DATA_DIR / f"{TAG}_local_energy_check.json"
tmp = json_path.with_suffix(".tmp.json")
tmp.write_text(json.dumps(out, indent=2) + "\n")
os.replace(tmp, json_path)
print(f"saved {json_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.axvspan(e_ref - THRESH, e_ref + THRESH, color=GRID, alpha=0.6, lw=0, zorder=0,
               label=f"trot's window, estimate ± {THRESH:.0f}")
    ax.hist(e[ok], bins=60, color=PALETTE[0], alpha=0.85, label=f"local energies, {n_ok} walkers")
    ax.axvline(e_ref, color=INK2, ls=":", lw=1.5, label=f"energy estimate: {e_ref:.3f}")
    ax.axvline(mean_raw, color=PALETTE[1], ls="--", lw=1.5, label=f"mean, no window: {mean_raw:.3f}")
    if E_REF is not None:
        ax.axvline(E_REF, color=PALETTE[2], ls="--", lw=1.5, label=f"DMRG: {E_REF:.3f}")
    ax.set_xlabel("local energy")
    ax.set_ylabel("walkers")
    ax.set_title(f"{args.trial.upper()}-trial CPMC, L = {L}, U = {U:g}, Δτ = {DT:g}: last saved walkers")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    pdf = DATA_DIR / f"{TAG}_local_energy_hist.pdf"   # a PDF: the module's matplotlib cannot rasterize text
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {pdf}")
except Exception:
    traceback.print_exc()
    raise SystemExit(f"the histogram failed; the numbers are in {json_path}")
