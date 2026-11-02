"""Spin-spin correlations with MPS-CPMC: half-filled 4x4 Hubbard (script version of szsz_correlation_4x4.ipynb).

observable_names=("szsz",) makes every block also measure the mixed estimator <T|S^z_i S^z_j|phi> / <T|phi>,
weighted over the walkers like the energy: an (L, L) matrix per block. The trial's own <S^z_i S^z_j> (an MPS
expectation) gives the extrapolated estimator 2*mixed - trial, and a larger DMRG (REF_CHI) gives the reference.
Open 4x4, 8 up + 8 dn, U = 4, t = 1. Three parts, so the GPU is only held for the CPMC:
  --part cpmc       GPU job: trial DMRG + CPMC; writes cpmc.npz
  --part reference  CPU job: DMRG chi = --ref-chi correlations; writes reference.npz
  --part plot       login-free, light: reads both npz in --out, writes the figures (run it as a small CPU job)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from trot import config

config.configure_once()  # precision; before importing jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from trot.core.system import System
from trot.gmps.dmrg import make_dmrg_trial
from trot.gmps.driver import run_qmc_mps
from trot.gmps.engine import szsz_dense
from trot.ham.hubbard import HamHubbard, square_hopping_matrix
from trot.prop.types import QmcParamsMps

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--part", choices=("cpmc", "reference", "plot"), required=True)
ap.add_argument("--U", type=float, default=8.0)
ap.add_argument("--walkers", type=int, default=32)
ap.add_argument("--chi-w", type=int, default=16, help="walker channel chi")
ap.add_argument("--trial-chi", type=int, default=256, help="trial MPS chi")
ap.add_argument("--ref-chi", type=int, default=1024, help="DMRG reference chi for the plots")
ap.add_argument("--eql-blocks", type=int, default=20)
ap.add_argument("--blocks", type=int, default=200)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--out", type=Path, default=HERE / "plots" / "szsz_correlation_4x4")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

Lx = Ly = 4
U = args.U
N = Lx * Ly
h1 = square_hopping_matrix(Lx, Ly, 1.0)  # open boundaries, site = x * Ly + y
n_elec = (11, 11)
sys = System(N, n_elec, "unrestricted")  #doped case
ham = HamHubbard(jnp.asarray(h1), U)
params = QmcParamsMps(
    n_walkers=args.walkers,
    n_eql_blocks=args.eql_blocks,
    n_blocks=args.blocks,
    dt=0.01,
    n_prop_steps=50, 
    weight_floor=1e-8,
    seed=args.seed,
    trial_chi=args.trial_chi,
    walker_channel_chi=args.chi_w,
)
REF_CHI = args.ref_chi

r0 = 1 * Ly + 1  # reference site (x, y) = (1, 1)
xy = np.array([(s // Ly, s % Ly) for s in range(N)])
dist = np.linalg.norm(xy - xy[r0], axis=1)
if args.part == "reference":
    ref = make_dmrg_trial(ham, sys, chi=REF_CHI, n_sweeps=params.dmrg_sweeps)
    print(f"E_DMRG(chi={REF_CHI}) = {ref.variational_energy:.4f}")
    C_ref = np.asarray(szsz_dense(ref.trial.tensors, ref.trial.tensors))
    np.savez(args.out / "reference.npz", C_ref=C_ref, E_dmrg_ref=float(ref.variational_energy), ref_chi=REF_CHI, U=U)
    print(f"saved {args.out / 'reference.npz'}")

elif args.part == "cpmc":
    trial = make_dmrg_trial(ham, sys, chi=params.trial_chi, n_sweeps=params.dmrg_sweeps).trial
    result = run_qmc_mps(sys=sys, params=params, ham_data=ham, trial_data=trial, observable_names=("szsz",))
    print(f"E = {float(result.mean_energy):.4f} +/- {float(result.stderr_energy):.4f}", flush=True)
    C_trial = np.asarray(szsz_dense(trial.tensors, trial.tensors))
    C_mix = np.asarray(result.observable_means["szsz"])
    C_err = np.asarray(result.observable_stderrs["szsz"])
    blocks = np.asarray(result.block_observables["szsz"])  # (blocks, L, L)
    print(f"sum rule  sum_j <S0 Sj> = {C_mix[r0].sum():.2e} (0 for S_z = 0);  <S0 S0> = {C_mix[r0, r0]:.4f}")
    np.savez(args.out / "cpmc.npz", C_trial=C_trial, C_mix=C_mix, C_err=C_err, blocks=blocks,
             E=float(result.mean_energy), E_err=float(result.stderr_energy), r0=r0, U=U, n_walkers=params.n_walkers,
             chi_w=params.walker_channel_chi, trial_chi=params.trial_chi, dt=params.dt,
             n_prop_steps=params.n_prop_steps, n_eql_blocks=params.n_eql_blocks, seed=params.seed)
    print(f"saved {args.out / 'cpmc.npz'}")

else:
    cp = np.load(args.out / "cpmc.npz")
    rf = np.load(args.out / "reference.npz")
    C_trial, C_mix, C_err, blocks = cp["C_trial"], cp["C_mix"], cp["C_err"], cp["blocks"]
    C_ref, REF_CHI = rf["C_ref"], int(rf["ref_chi"])
    C_ext = 2 * C_mix - C_trial
    dt, n_prop_steps, n_eql_blocks = float(cp["dt"]), int(cp["n_prop_steps"]), int(cp["n_eql_blocks"])
    print(f"E_CPMC = {float(cp['E']):.4f} +/- {float(cp['E_err']):.4f};  E_DMRG(chi={REF_CHI}) = "
          f"{float(rf['E_dmrg_ref']):.4f}")

    # Correlations vs distance from r0
    order = np.argsort(dist, kind="stable")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="0.7", lw=0.8)
    series = [
        (C_trial[r0], None, "o", "Trial  <T|S0 Sj|T>"),
        (C_mix[r0], C_err[r0], "s", "CPMC mixed"),
        (C_ext[r0], 2 * C_err[r0], "D", "CPMC extrapolated (2 mixed - trial)"),
        (C_ref[r0], None, "x", f"DMRG chi={REF_CHI}"),
    ]
    for k, (values, err, marker, label) in enumerate(series):
        shift = 0.04 * (k - 1.5)
        ax.errorbar(dist[order] + shift, values[order], yerr=None if err is None else err[order], fmt=marker, capsize=3,
                    label=label)
    ax.set_xlabel("distance |r_j - r_0|")
    ax.set_ylabel("<S^z_0 S^z_j>")
    ax.set_title(f"Spin correlations from site (1, 1), 4×4 Hubbard, U = {float(cp['U']):g}, half filling", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig(args.out / "szsz_vs_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Real-space maps around r0
    maps = [("Trial", C_trial), ("CPMC (mixed)", C_mix), ("CPMC extrapolated", C_ext), (f"DMRG chi={REF_CHI}", C_ref)]
    off = np.arange(N) != r0  # the on-site value <Sz^2> would set the colour scale
    vmax = max(np.abs(m[r0, off]).max() for _, m in maps)
    fig, axes = plt.subplots(1, len(maps), figsize=(14, 3.8), constrained_layout=True)
    for ax, (name, m) in zip(axes, maps):
        row = m[r0].reshape(Lx, Ly).copy()
        row[r0 // Ly, r0 % Ly] = np.nan  # the on-site value dominates the scale
        im = ax.imshow(row.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot(r0 // Ly, r0 % Ly, "k*", ms=12)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("x")
        ax.set_xticks(range(Lx)), ax.set_yticks(range(Ly))
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.85, label="<S^z_0 S^z_j>")
    fig.suptitle("Antiferromagnetic correlations around the site (1, 1) (star)", fontsize=14)
    fig.savefig(args.out / "szsz_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Block values of the mixed estimator
    tau = (np.arange(len(blocks)) + 1) * n_prop_steps * dt
    partners = {"nearest neighbour (2, 1)": 2 * Ly + 1, "next-nearest (2, 2)": 2 * Ly + 2, "third (3, 1)": 3 * Ly + 1}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, j in partners.items():
        line, = ax.plot(tau, blocks[:, r0, j], label=label)
        ax.axhline(C_mix[r0, j], color=line.get_color(), ls="--", lw=1)
    ax.axvline(n_eql_blocks * n_prop_steps * dt, color="k", ls=":", lw=1, label="end of equilibration")
    ax.set_xlabel("τ")
    ax.set_ylabel("<S^z_0 S^z_j>  (block mixed estimator)")
    ax.set_title("Block values of the mixed estimator (dashed: sampling mean)", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.savefig(args.out / "szsz_blocks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved data and figures in {args.out}")
