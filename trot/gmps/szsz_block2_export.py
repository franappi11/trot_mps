"""Export an MPS-CPMC block's <S^z_i S^z_j> mixed estimator for the block2 cross-check.

    ~/.trot/bin/python trot/gmps/szsz_block2_export.py --case sz   --out sz.npz
    ~/.trot/bin/python trot/gmps/szsz_block2_export.py --case nosz --out nosz.npz
    ~/.trot/bin/python trot/gmps/szsz_block2_check.py sz.npz       # block2, no trot/JAX needed

Case "sz": the DMRG trial with (N_up, N_dn) bond labels. Case "nosz": the same trial rotated by 90 degrees about y and
used as it is (particle-number labels, no definite S_z). The walkers always have definite (N_up, N_dn).

An open Hubbard chain, a truncated pyblock3 DMRG trial and an exact walker plan. The walkers diffuse for a few
blocks; then the n_prop_steps of one more block are run by hand (the block's own half-step scan), so that the
exact walkers the block measures are known. The block's szsz (trot.prop.mps_cpmc.block), the batched kernel and the
per-walker observable must agree here; the .npz holds the walkers, the weights, the trial's determinant
expansion and trot's values, which the block2 script reproduces independently.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from trot import config

config.configure_once()

from trot.core.system import System  # noqa: E402
from trot.gmps import engine  # noqa: E402
from trot.gmps.dmrg import make_dmrg_trial  # noqa: E402
from trot.gmps.driver import prepare_mps_cpmc  # noqa: E402
from trot.ham.hubbard import HamHubbard, hopping_matrix  # noqa: E402
from trot.meas.mps import szsz_fns  # noqa: E402
from trot.prop import mps_cpmc  # noqa: E402
from trot.prop.types import QmcParamsMps  # noqa: E402
from trot.trial.mps import make_mps_trial, rotate_spin, spin_rotation_y  # noqa: E402


def full_vector(tensors):
    """Dense interleaved 4^L vector of an MPS (site 0 most significant, local index n_up + 2 n_dn)."""
    v = np.ones((1, 1))
    for A in tensors:
        A = np.asarray(A)
        v = np.einsum("xa,apb->xpb", v, A).reshape(-1, A.shape[2])
    return v[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("sz", "nosz"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--nelec", type=int, nargs=2, default=(4, 4))
    parser.add_argument("--chi", type=int, default=16, help="DMRG trial bond dimension (truncated: nontrivial)")
    parser.add_argument("--walkers", type=int, default=6)
    parser.add_argument("--warm-blocks", type=int, default=3)
    parser.add_argument("--prop-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--kernel", choices=("blocked", "dense"), default="blocked")
    args = parser.parse_args()

    L, nelec = args.L, tuple(args.nelec)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=args.U), System(norb=L, nelec=nelec, walker_kind="unrestricted")
    params = QmcParamsMps(
        dt=0.05, n_walkers=args.walkers, n_prop_steps=args.prop_steps, n_eql_blocks=5, n_blocks=10, seed=args.seed,
        auto_n_chunks=False, n_chunks=1, orbital_plan="maximal", walker_channel_chi=None, energy_kernel=args.kernel,
    )
    dmrg = make_dmrg_trial(ham, sys_, chi=args.chi, n_sweeps=10, seed=0)
    trial = dmrg.trial
    if args.case == "nosz":
        rotated = rotate_spin([np.asarray(A) for A in trial.tensors], spin_rotation_y(90.0))
        trial = make_mps_trial(rotated, nelec=nelec)
    expected_width = 2 if args.case == "sz" else 1
    assert trial.label_width == expected_width, (args.case, trial.label_width)
    print(f"[export] case={args.case} L={L} U={args.U} nelec={nelec} trial bonds max {max(trial.bond_dims)} "
          f"labels {'(N_up, N_dn)' if trial.label_width == 2 else 'N'}")

    run = prepare_mps_cpmc(sys=sys_, params=params, ham_data=ham, trial=trial, verbose=False)
    ops, ctx, prop_ctx = run.ops, run.meas_ctx, run.prop_ctx
    names = ("szsz",)
    kwargs = dict(sys=sys_, params=params, ham_data=ham, trial_data=trial, trial_ops=ops.trial_ops,
                  meas_ops=ops.meas_ops, meas_ctx=ctx, prop_ops=ops.prop_ops, prop_ctx=prop_ctx)
    block = jax.jit(lambda s: mps_cpmc.block(s, observable_names=names, **kwargs))
    state = run.state
    for _ in range(args.warm_blocks):
        state, _ = block(state)

    # the block to measure: its scan of half steps by hand, to know the walkers it measures
    data = ctx.data(prop_ctx)
    half = engine.make_half_step(ctx.kernels, params, 1)
    stepped = state
    for i in range(2 * params.n_prop_steps):
        stepped = half(stepped, i, data)
    qu, _ = ctx.kernels.batch_qr(stepped.walkers[0])
    qd, _ = ctx.kernels.batch_qr(stepped.walkers[1])
    weights = np.asarray(stepped.weights)

    _, obs = block(state)
    block_szsz = np.asarray(obs.observables["szsz"])
    batch = np.asarray(szsz_fns(ctx).batch(stepped.walkers[0], stepped.walkers[1], data))
    one = np.array([np.asarray(ops.meas_ops.observables["szsz"]((qu[i], qd[i]), ham, ctx, trial))
                    for i in range(params.n_walkers)])
    overlaps = np.asarray(jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))((qu, qd), trial))
    by_hand = np.tensordot(weights, batch, axes=1) / weights.sum()
    print(f"[export] block vs by-hand weighted mean: {np.abs(block_szsz - by_hand).max():.2e}; "
          f"batched vs per-walker: {np.abs(batch - one).max():.2e}")
    assert np.abs(block_szsz - by_hand).max() < 1e-12 and np.abs(batch - one).max() < 1e-12
    M = 0.5 * (nelec[0] - nelec[1])
    print(f"[export] sum rule: sum_ij C_ij = {block_szsz.sum():.12f} (M^2 = {M * M})")

    v = full_vector(trial.tensors)
    nz = np.flatnonzero(v)
    codes = np.array([[(k // 4 ** (L - 1 - s)) % 4 for s in range(L)] for k in nz], dtype=np.uint8)
    np.savez(
        args.out, case=args.case, L=L, U=args.U, nelec=np.array(nelec), h1=h1,
        trial_codes=codes, trial_values=v[nz], trial_energy=float(ctx.trial_energy),
        walkers_up=np.asarray(qu), walkers_dn=np.asarray(qd), weights=weights, trot_overlaps=overlaps,
        trot_szsz_walkers=batch, trot_szsz_block=block_szsz,
    )
    print(f"[export] wrote {args.out}: {len(nz)} trial determinants, {params.n_walkers} walkers")


if __name__ == "__main__":
    main()
