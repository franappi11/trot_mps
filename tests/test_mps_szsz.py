"""Tests for the mixed <S^z_i S^z_j> estimator of MPS-CPMC (trot.meas.mps szsz, trot.gmps.engine.szsz_contract).

References never come from the MPS code under test: exact Fock-space enumeration of the trial's walker-sector
amplitudes (tests/helpers/hubbard_fock.py), trot's GHF density_corr observable for determinant trials, and the
sum rule sum_ij <S^z_i S^z_j> = M^2 (walkers are S^z eigenstates with M = (N_up - N_dn) / 2, whatever the trial).
Trials: the exact ground state and an unrestricted determinant (bond labels (N_up, N_dn)), and the spin-rotated
determinant used as it is (particle-number labels N, no definite S_z).
"""

from trot import config

config.configure_once()

import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers import hubbard_fock as hf
from tests.test_mps_cpmc import (
    U,
    _determinant,
    _diffused_state,
    _exact_trial,
    _mps_side,
    _noisy_trial,
    _params,
    _sd_side,
    _spin_rotation,
    _system,
)
from tests.test_mps_meas import (  # noqa: F401  (fixtures and the case builder are shared)
    ENERGY_CASES,
    KERNELS,
    NELEC,
    ed_2x3,
    rotated_sd_chain,
    sd_chain,
    case,
)
import jax
import scipy.linalg

from trot.driver import run_qmc
from trot.gmps.driver import make_mps_cpmc_ops, run_qmc_mps
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.meas.ghf import density_corr_kernel_uw
from trot.meas.mps import szsz_fns
from trot.prop import blocks, mps_cpmc
from trot.trial.ghf import GhfTrial
from trot.trial.mps import make_mps_trial, mps_trial_from_sd, rotate_spin


def _szsz_by_enumeration(amplitudes, Wa, Wb):
    """<T|S^z_i S^z_j|SD> / <T|SD> from the sector amplitudes: S^z_i S^z_j is diagonal, (n_a - n_b)_i (n_a - n_b)_j / 4."""
    L = Wa.shape[0]
    _, occ_a = hf.sector_basis(L, Wa.shape[1])
    _, occ_b = hf.sector_basis(L, Wb.shape[1])
    weights = amplitudes * hf.sd_amplitudes(Wa, Wb)
    sz = 0.5 * (occ_a[:, None, :] - occ_b[None, :, :])  # (n_a, n_b, L)
    return np.einsum("ab,abi,abj->ij", weights, sz, sz) / weights.sum()


def _mps_szsz(case, plan_kind, kernel):
    op = case.ops[plan_kind, kernel].observables["szsz"]
    ctx = case.ctx[plan_kind, kernel]
    return np.array([np.asarray(op(w, case.ham, ctx, case.trial)) for w in case.walkers])


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_szsz_matches_enumeration(case, kernel):
    """With exact walkers the observable is <T|S^z_i S^z_j|W>/<T|W> of the enumerated sector amplitudes, for the
    (N_up, N_dn)-labelled trials and the N-labelled rotated one."""
    want = np.array(
        [_szsz_by_enumeration(case.amplitudes, np.asarray(wa), np.asarray(wb)) for wa, wb in case.walkers]
    )
    got = _mps_szsz(case, "exact", kernel)
    assert got.shape == want.shape == (len(case.walkers), case.trial.norb, case.trial.norb)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-11)
    assert np.abs(want - want.mean(axis=0)).max() > 1e-3  # the walkers differ: the check is not trivial


@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_truncated_walkers_blocked_equals_dense(case):
    """With chi=2 adaptive walker channels the blocked and dense kernels agree, and truncation is visible."""
    blocked = _mps_szsz(case, "truncated", "blocked")
    np.testing.assert_allclose(blocked, _mps_szsz(case, "truncated", "dense"), rtol=0, atol=1e-11)
    if case.E0 is None:
        assert np.abs(blocked - _mps_szsz(case, "exact", "blocked")).max() > 1e-6


@pytest.mark.parametrize("plan_kind", ["exact", "truncated"])
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_szsz_is_symmetric_and_obeys_the_sum_rule(case, kernel, plan_kind):
    """C is symmetric and sum_ij C_ij = M^2 for every walker (an S^z eigenstate), truncated or not."""
    C = _mps_szsz(case, plan_kind, kernel)
    np.testing.assert_allclose(C, np.swapaxes(C, 1, 2), rtol=0, atol=1e-13)
    M = 0.5 * (NELEC[0] - NELEC[1])
    np.testing.assert_allclose(C.sum(axis=(1, 2)), M**2, rtol=0, atol=1e-11)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", ["sd_chain", "rotated_sd_chain"], indirect=True)
def test_szsz_matches_ghf_density_correlations(case, kernel):
    """For (spin-rotated) determinant trials: S^z S^z = (uu + dd - ud - ud^T) / 4 of trot's GHF density_corr."""
    ghf = GhfTrial(mo_coeff=jnp.asarray(case.C))
    want = []
    for w in case.walkers:
        uu, ud, dd = np.asarray(density_corr_kernel_uw(w, case.ham, None, ghf))
        want.append(0.25 * (uu + dd - ud - ud.T))
    np.testing.assert_allclose(_mps_szsz(case, "exact", kernel), want, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------------------------
# The block: observable_names=("szsz",) in trot.prop.mps_cpmc.block, blocks.block and run_qmc(_mps)
# ---------------------------------------------------------------------------------------------


def _block_kwargs(ham, sys_, params, trial, ops):
    return dict(
        sys=sys_,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=ops.trial_ops,
        meas_ops=ops.meas_ops,
        meas_ctx=ops.meas_ops.build_meas_ctx(ham, trial),
        prop_ops=ops.prop_ops,
        prop_ctx=ops.prop_ops.build_prop_ctx(ham, None, params),
    )


@pytest.mark.parametrize("walker_channel_chi, energy_kernel", [(None, "blocked"), (2, "blocked"), (None, "dense")])
def test_mps_block_szsz_matches_trots_block(walker_channel_chi, energy_kernel):
    """The MPS block's szsz (batched, after det-R rescaling) equals blocks.block's (vmapped observable kernel) and is
    the weighted mean of the per-walker values; the energy and the walkers are unchanged by asking for it."""
    L, nelec = 6, (3, 2)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(orbital_plan="adaptive", walker_channel_chi=walker_channel_chi, energy_kernel=energy_kernel)
    Ca, Cb = _determinant("uhf", h1, nelec)
    trial = mps_trial_from_sd(Ca, Cb)
    ops = make_mps_cpmc_ops(ham, trial, sys_, params)
    state = _diffused_state(trial, ops, h1, Ca, Cb, params.n_walkers, seed=4)
    rng = np.random.default_rng(4)
    walkers = tuple(
        jnp.asarray(np.stack([hf.nonorthonormal(np.asarray(w), rng) for w in spin])) for spin in state.walkers
    )
    state = state._replace(
        walkers=walkers,
        weights=jnp.asarray(rng.uniform(0.5, 1.5, params.n_walkers)),
        overlaps=jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))(walkers, trial),
    )
    kwargs = _block_kwargs(ham, sys_, params, trial, ops)
    got, got_obs = jax.jit(lambda s: mps_cpmc.block(s, observable_names=("szsz",), **kwargs))(state)
    want, want_obs = jax.jit(lambda s: blocks.block(s, observable_names=("szsz",), **kwargs))(state)
    plain, plain_obs = jax.jit(lambda s: mps_cpmc.block(s, **kwargs))(state)

    assert set(got_obs.observables) == {"szsz"} and plain_obs.observables == {}
    C = np.asarray(got_obs.observables["szsz"])
    assert C.shape == (L, L)
    np.testing.assert_allclose(C, np.asarray(want_obs.observables["szsz"]), rtol=0, atol=1e-9)
    assert np.abs(C - np.diag(np.diag(C))).max() > 1e-3
    for key in ("energy", "weight"):
        assert float(got_obs.scalars[key]) == float(plain_obs.scalars[key])
    for x, y in zip(got.walkers, plain.walkers):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    # the batched kernel (the block's) equals the per-walker observable (blocks.block's) on the same walkers
    ctx = kwargs["meas_ctx"]
    batched = np.asarray(szsz_fns(ctx).batch(walkers[0], walkers[1], ctx.data()))
    per_walker = np.array(
        [
            np.asarray(ops.meas_ops.observables["szsz"]((walkers[0][i], walkers[1][i]), ham, ctx, trial))
            for i in range(params.n_walkers)
        ]
    )
    np.testing.assert_allclose(batched, per_walker, rtol=0, atol=1e-11)


def test_mps_block_rejects_unknown_observables():
    L, nelec = 4, (2, 2)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params()
    trial = mps_trial_from_sd(*_determinant("uhf", h1, nelec))
    ops = make_mps_cpmc_ops(ham, trial, sys_, params)
    state = _diffused_state(trial, ops, h1, *_determinant("uhf", h1, nelec), params.n_walkers, seed=1)
    with pytest.raises(ValueError, match="rdm1"):
        mps_cpmc.block(state, observable_names=("rdm1",), **_block_kwargs(ham, sys_, params, trial, ops))


@pytest.mark.parametrize("trial_kind", ["uhf", "rotated"])
def test_szsz_blocks_equal_ghf_density_corr_blocks(trial_kind):
    """Plain-SD trot CPMC measuring density_corr and the SD-as-MPS run measuring szsz (the MPS block) agree block
    for block: SzSz = (uu + dd - ud - ud^T) / 4. For "rotated", with the spin-rotated determinant used as it is
    (particle-number labels) against the rotated GhfTrial."""
    L, nelec = 6, (3, 2)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(n_walkers=8, n_prop_steps=3, n_eql_blocks=5, n_blocks=10, dt=0.05, weight_floor=1e-3)
    Ca, Cb = _determinant("uhf", h1, nelec)
    if trial_kind == "uhf":
        C_ghf, mps_trial = scipy.linalg.block_diag(Ca, Cb), None
    else:
        R = _spin_rotation("rotation")
        C_ghf = np.kron(R, np.eye(L)) @ scipy.linalg.block_diag(Ca, Cb)
        tensors = [np.asarray(t) for t in sd_to_gmps(Ca, Cb, mode="maximal").tensors]
        mps_trial = make_mps_trial(rotate_spin(tensors, R), nelec=nelec)
    sd = _sd_side(ham, sys_, params, C_ghf)
    mps_trial = mps_trial or mps_trial_from_sd(Ca, Cb, rdm1=sd[1].get_rdm1(sd[0]))
    if trial_kind == "rotated":
        mps_trial = make_mps_trial(mps_trial.tensors, nelec=nelec, rdm1=sd[1].get_rdm1(sd[0]))
    mps = _mps_side(ham, sys_, params, mps_trial)

    def run(side, block_fn, names):
        trial, trial_ops, meas_ops, prop_ops, prop_ctx = side
        return run_qmc(sys=sys_, params=params, ham_data=ham, trial_data=trial, trial_ops=trial_ops,
                       meas_ops=meas_ops, prop_ops=prop_ops, prop_ctx=prop_ctx, block_fn=block_fn,
                       observable_names=names)

    sd_run = run(sd, blocks.block, ("density_corr",))
    mps_run = run(mps, mps_cpmc.block, ("szsz",))
    uu, ud, dd = np.moveaxis(np.asarray(sd_run.block_observables["density_corr"]), 1, 0)
    want = 0.25 * (uu + dd - ud - np.swapaxes(ud, 1, 2))
    got = np.asarray(mps_run.block_observables["szsz"])
    assert got.shape == want.shape and got.shape[1:] == (L, L)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)
    assert np.ptp(want, axis=0).max() > 1e-3  # a stochastic run
    uu, ud, dd = np.asarray(sd_run.observable_means["density_corr"])
    np.testing.assert_allclose(
        np.asarray(mps_run.observable_means["szsz"]), 0.25 * (uu + dd - ud - ud.T), rtol=0, atol=1e-8
    )


def test_run_qmc_mps_returns_szsz_means_and_errors():
    """run_qmc_mps(..., observable_names=("szsz",)) gives block values, a mean and a stderr like the energy; the
    mean obeys the sum rule and the values do not disturb the energies."""
    pytest.importorskip("pyblock3")
    h1 = hopping_matrix(4, 1.0)
    nelec = (2, 2)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(4, nelec)
    e0, trial = _exact_trial(h1, nelec)
    noisy = _noisy_trial(trial, nelec, seed=4)
    params = _params(n_walkers=6, n_prop_steps=3, n_eql_blocks=5, n_blocks=20, seed=3)
    with_obs = run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=noisy, observable_names=("szsz",))
    without = run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=noisy)
    np.testing.assert_array_equal(np.asarray(with_obs.block_energies), np.asarray(without.block_energies))
    blocks_szsz = np.asarray(with_obs.block_observables["szsz"])
    mean, err = np.asarray(with_obs.observable_means["szsz"]), np.asarray(with_obs.observable_stderrs["szsz"])
    assert blocks_szsz.shape[1:] == mean.shape == err.shape == (4, 4)
    np.testing.assert_allclose(mean, mean.T, rtol=0, atol=1e-12)
    assert abs(mean.sum()) < 1e-10  # (2, 2): M = 0
    assert np.all(np.isfinite(err)) or np.all(np.isnan(err))
