"""<S^z_i S^z_j> of a saved doped 6x6 DMRG trial after the production spin rotation, and its spin content.

The CPMC uses the DMRG trial rotated by 90 degrees about y and used as it is (ex_6x6_doped_diag.load_trial:
make_mps_trial(rotate_spin(tensors, spin_rotation_y(90)))). For a state with S_z = 0, R^dag S^z_i R = -+S^x_i, so the
rotated trial's <S^z_i S^z_j> is the unrotated one's <S^x_i S^x_j>, equal to its <S^z_i S^z_j> only for a singlet,
which a DMRG state with fixed (N_up, N_dn) at a finite bond need not be. On the saved trial (default: the block2
chi 5000 of job 7317904) this measures:

  * C_rot, the <S^z_i S^z_j> of R(90)|psi>: rotate_spin site by site, then trot's szsz_dense, as
    dmrg_trial_6x6_block2.py measured the unrotated C (its _szsz.npz, --szsz);
  * the spin weights w_S from f(x) = <psi|R(beta)|psi> = sum_S w_S P_S(x), x = cos(beta), at K Gauss-Legendre nodes
    (exact for every S <= 2K - 1 - S_max; S_max = 16 here, so K = 17 covers all, trial_spin_check.py's method);
    <S^2> = sum_S S(S+1) w_S, checked against 2 sum_ij C_rot (= 2 <(S^x_tot)^2> = <S^2> for S_z = 0), and the weight
    of R(90)|psi> in the walkers' S_z = 0 sector, sum_S w_S P_S(cos beta)^2 (at 90 degrees odd S drop out).

It prints C_rot against C and the staggered structure factors S^zz(pi) (from C) and S^xx(pi) (from C_rot), and saves
C_rot, C and the spin data to --out (default: next to the trial, <trial>_szsz_rot90.npz). CPU only, one genx job; the
trial is 23 GB on the host, the szsz_dense stack ~40 GB more:

    sbatch -p genx -C "icelake&rocky9" --time=01:30:00 --cpus-per-task=32 --mem=120G -J szsz_rot \\
        --export=ALL,TARGET=szsz_rotated_6x6.py run_mps_sweep.sh
"""
from trot.config import configure_once

configure_once()  # float64

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

from trot.gmps.engine import szsz_dense
from trot.gmps.prepare_htrial_saved import read_saved_trial
from trot.trial.mps import rotate_spin, spin_rotation_unitary, spin_rotation_y

STATE = Path("/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00")
TRIAL = STATE / "dmrg_chi5000_from2048_down_block2.npz"


def rotated_overlap(tensors, beta_deg):
    """<psi|R(beta)|psi> of real tensors (Dl, 4, Dr), R applied on the ket's sites as rotate_spin applies it."""
    M = spin_rotation_unitary(spin_rotation_y(beta_deg))
    E = np.ones((1, 1))
    for A in tensors:
        Y = M @ np.tensordot(E, A, axes=([1], [0]))  # (a, p, x): the rotated ket's site under the environment
        E = np.tensordot(A, Y, axes=([0, 1], [0, 1]))  # (r, x)
    return float(E[0, 0])


def staggered(C, side):
    """S(pi) = sum_ij (-1)^(x_i + y_i + x_j + y_j) C_ij / L, sites y * side + x."""
    s = np.array([(-1.0) ** (i // side + i % side) for i in range(len(C))])
    return float(s @ C @ s / len(C))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", type=Path, default=TRIAL)
    parser.add_argument("--szsz", type=Path, default=None,
                        help="the unrotated <S^z_i S^z_j> (default: <trial>_szsz.npz of dmrg_trial_6x6_block2.py)")
    parser.add_argument("--rotation", type=float, default=90.0, help="degrees about y, as the production trial")
    parser.add_argument("--nodes", type=int, default=17, help="Gauss-Legendre nodes for the spin weights")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    szsz_file = args.szsz or args.trial.with_name(args.trial.stem + "_szsz.npz")
    out = args.out or args.trial.with_name(args.trial.stem + f"_szsz_rot{args.rotation:g}.npz")

    start = time.perf_counter()
    h1, u, nelec, tensors, _, energy, _ = read_saved_trial(args.trial)
    side, n = int(round(np.sqrt(len(h1)))), len(h1)
    with np.load(szsz_file) as z:
        C = np.asarray(z["C"])
    print(f"{args.trial}: {n} sites, nelec {nelec}, bond {max(A.shape[2] for A in tensors)}, E = {energy:.10f}; "
          f"loaded in {time.perf_counter() - start:.0f} s; unrotated C from {szsz_file}", flush=True)

    # spin weights on the unrotated tensors
    start = time.perf_counter()
    s_max = min(sum(nelec), 2 * n - sum(nelec)) // 2
    x, c = leggauss(args.nodes)
    norm = rotated_overlap(tensors, 0.0)
    f = np.array([rotated_overlap(tensors, np.degrees(np.arccos(xk))) for xk in x]) / norm
    S = np.arange(s_max + 1)
    weights = (2 * S + 1) / 2 * (legvander(x, s_max).T @ (c * f))
    exact = min(s_max, 2 * args.nodes - 1 - s_max)
    d00 = legvander(np.array([np.cos(np.radians(args.rotation))]), s_max)[0]  # <S,0|R|S,0> = P_S(cos beta)
    s2 = float(np.sum(S * (S + 1) * weights))
    in_sector = float(np.sum(weights * d00 ** 2))
    print(f"spin weights from {args.nodes + 1} rotated overlaps in {time.perf_counter() - start:.0f} s (exact for "
          f"S <= {exact}): norm {norm:.12f}, sum w_S - 1 = {weights.sum() - 1:.1e}, <S^2> = {s2:.6f}", flush=True)
    print("  " + ", ".join(f"w_{s} = {w:.6f}" for s, w in enumerate(weights[:8])), flush=True)
    print(f"  weight of R({args.rotation:g})|psi> in the walkers' S_z = 0 sector: sum_S w_S P_S(cos beta)^2 = "
          f"{in_sector:.6f}", flush=True)

    # the production rotation, then szsz_dense as for the unrotated C
    start = time.perf_counter()
    R = spin_rotation_y(args.rotation)
    for i in range(n):  # site by site: the host holds one copy of the trial
        tensors[i] = jnp.asarray(rotate_spin([tensors[i]], R)[0])
    C_rot = np.asarray(szsz_dense(tensors, tensors))
    del tensors
    print(f"rotated and measured in {time.perf_counter() - start:.0f} s", flush=True)

    s2_szsz = 2.0 * float(C_rot.sum())
    diff = C_rot - C
    off = ~np.eye(n, dtype=bool)
    nn = (np.abs(h1) > 0) & off
    print(f"<S^2> = 2 sum_ij C_rot = {s2_szsz:.6f} (weights: {s2:.6f}, difference {s2_szsz - s2:.1e}); "
          f"sum_ij C = {C.sum():.1e}", flush=True)
    print(f"C_rot - C: max |.| {np.abs(diff).max():.4e}, rms off-diagonal {np.sqrt(np.mean(diff[off] ** 2)):.4e}, "
          f"diagonal max |.| {np.abs(np.diag(diff)).max():.2e}; nearest-neighbour mean C {C[nn].mean():.6f}, "
          f"C_rot {C_rot[nn].mean():.6f}", flush=True)
    print(f"staggered structure factor: S^zz(pi) = {staggered(C, side):.6f} (unrotated), S^xx(pi) = "
          f"{staggered(C_rot, side):.6f} (rotated)", flush=True)

    np.savez(out, C_rot=C_rot, C=C, rotation_degrees=args.rotation, spin_weights=weights,
             spin_weights_exact_up_to=exact, s2_weights=s2, s2_szsz=s2_szsz, weight_sz0=in_sector, nodes_x=x, nodes_f=f, norm=norm, energy=energy,
             h1=h1, u=u, nelec=np.asarray(nelec), trial_file=str(args.trial), szsz_file=str(szsz_file))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
