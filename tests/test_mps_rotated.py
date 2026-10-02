"""The rotated MPS trial used as it is (RotatedMpsTrial: no definite S_z, bonds unchanged).

Walkers are S_z eigenstates and H conserves S_z, so overlaps must be sqrt(w) times those of the
trial projected onto the walkers' sector (w its weight there) and local energies must be the same.
One step and whole run_qmc runs then agree with the projection (approach 1), the S_z-conserving
MPO (approach 2) and the rotated GHF trial.
"""

from trot import config

config.configure_once()

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from tests.helpers import hubbard_fock as hf
from tests.test_mps_cpmc import (
    _assert_identical_runs,
    _assert_identical_states,
    _blocks_twin,
    _compare_states,
    _initial_state,
    _mps_side,
    _params,
    _run,
    _sd_side,
    _system,
    _three_steps,
)
from trot.core.ops import k_energy
from trot.gmps.driver import make_mps_cpmc_ops, make_rotated_mps_cpmc_ops
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.trial.ghf import GhfTrial, get_rdm1_block_diag, make_ghf_trial_ops
from trot.trial.mps import make_mps_trial, natural_orbitals, one_rdm, rotate_spin
from trot.trial.mps_rotation import (
    RotatedMpsTrial,
    make_rotated_mps_trial,
    rotate_spin_mpo,
)

U = 4.0
L = 6
_C, _S = np.cos(0.7), np.sin(0.7)
ROTATIONS = {
    "rotation": np.array([[_C, -_S], [_S, _C]]),
    "reflection": np.array([[_C, _S], [_S, -_C]]),  # det = -1
}


# ---------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------


def _model(walkers):
    h1 = hopping_matrix(L, 1.0)
    return h1, HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, walkers)


def _rotated_ghf(R, nelec):
    """The staggered UHF determinant SD(Ca, Cb) of nelec and its rotated GhfTrial."""
    Ca, Cb = hf.staggered_determinant(hopping_matrix(L, 1.0), *nelec)
    C_ghf = np.kron(R, np.eye(L)) @ cast(np.ndarray, scipy.linalg.block_diag(Ca, Cb))
    return Ca, Cb, GhfTrial(mo_coeff=jnp.asarray(C_ghf))


def _three_trials(R, Ca, Cb, walkers, rdm1):
    """One rotated trial three ways: projected (1), MPO (2) and used as it is (3)."""
    gmps = sd_to_gmps(Ca, Cb, mode="maximal")
    projected = make_mps_trial(rotate_spin(gmps.tensors, R), nelec=walkers, rdm1=rdm1)
    tensors, charges, _ = rotate_spin_mpo(gmps.tensors, gmps.charges, R, nelec=walkers)
    mpo = make_mps_trial(tensors, charges, nelec=walkers, rdm1=rdm1)
    rotated = make_rotated_mps_trial(gmps.tensors, R, nelec=walkers, rdm1=rdm1)
    return projected, mpo, rotated


def _rotated_side(ham, sys_, params, trial):
    ops = make_rotated_mps_cpmc_ops(ham, trial, sys_, params)
    prop_ctx = ops.prop_ops.build_prop_ctx(ham, ops.trial_ops.get_rdm1(trial), params)
    return trial, ops.trial_ops, ops.meas_ops, ops.prop_ops, prop_ctx


def _walkers(h1, rdm1, walkers, n=8, seed=3, nonorthonormal=True):
    """Determinants diffused from the natural orbitals of rdm1 (and made non-orthonormal)."""
    Ra = natural_orbitals(np.asarray(rdm1[0]), walkers[0])[0]
    Rb = natural_orbitals(np.asarray(rdm1[1]), walkers[1])[0]
    out = hf.random_field_walkers(h1, U, 0.1, Ra, Rb, n=n, steps=20, seed=seed)
    if nonorthonormal:
        rng = np.random.default_rng(seed)
        out = [(hf.nonorthonormal(a, rng), hf.nonorthonormal(b, rng)) for a, b in out]
    return out


def _stack(walkers):
    return tuple(jnp.asarray(np.stack([w[s] for w in walkers])) for s in range(2))


def _overlaps(ops, walkers, trial):
    return np.asarray(jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))(_stack(walkers), trial))


def _energies(meas_ops, walkers, ham, meas_ctx, trial):
    kernel = jax.vmap(meas_ops.kernels[k_energy], in_axes=(0, None, None, None))
    return np.asarray(kernel(_stack(walkers), ham, meas_ctx, trial))


# ---------------------------------------------------------------------------------------------
# The trial
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROTATIONS)
def test_rotated_trial_keeps_the_rotated_tensors_and_the_ghf_walker_start(name):
    R = ROTATIONS[name]
    Ca, Cb, ghf = _rotated_ghf(R, (3, 2))
    gmps = sd_to_gmps(Ca, Cb, mode="maximal")
    rotated = rotate_spin(gmps.tensors, R)
    trial = make_rotated_mps_trial(gmps.tensors, R, nelec=(3, 2))
    assert trial.nelec == (3, 2) and trial.norb == L
    assert trial.bond_dims == tuple(len(q) for q in gmps.charges)
    vector = hf.mps_full_vector(rotated)
    np.testing.assert_allclose(
        hf.mps_full_vector(trial.tensors), vector / np.linalg.norm(vector), rtol=0, atol=1e-13
    )
    pre_rotated = make_rotated_mps_trial(rotated, nelec=(3, 2))
    for a, b in zip(pre_rotated.tensors, trial.tensors):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_allclose(np.asarray(trial.rdm1), np.stack(one_rdm(rotated)), atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(trial.rdm1), np.asarray(get_rdm1_block_diag(ghf)), rtol=0, atol=1e-12
    )
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    assert len(leaves) == L + 1
    same = jax.jit(lambda t: t)(trial)
    assert isinstance(same, RotatedMpsTrial) and same.nelec == (3, 2)
    assert jax.tree_util.tree_structure(same) == treedef


def test_rotated_trial_validates_its_input():
    Ca, Cb, _ = _rotated_ghf(ROTATIONS["rotation"], (3, 2))
    tensors = [np.asarray(t) for t in sd_to_gmps(Ca, Cb, mode="maximal").tensors]
    with pytest.raises(TypeError, match="real"):
        make_rotated_mps_trial([t.astype(complex) for t in tensors], nelec=(3, 2))
    with pytest.raises(ValueError, match="4"):
        make_rotated_mps_trial([t[:, :3] for t in tensors], nelec=(3, 2))
    with pytest.raises(ValueError, match="bond"):
        make_rotated_mps_trial([tensors[0][:, :, :1]] + tensors[1:], nelec=(3, 2))
    with pytest.raises(ValueError, match="sector"):
        make_rotated_mps_trial(tensors, nelec=(7, -2))
    with pytest.raises(ValueError, match="shape"):
        make_rotated_mps_trial(tensors, nelec=(3, 2), rdm1=np.zeros((2, L, L + 1)))
    with pytest.raises(ValueError, match="orthogonal"):
        make_rotated_mps_trial(tensors, np.ones((2, 2)), nelec=(3, 2))
    trial = make_rotated_mps_trial(tensors, nelec=(3, 2))
    _, ham, sys_ = _model((3, 3))
    with pytest.raises(ValueError, match="does not match the system"):
        make_rotated_mps_cpmc_ops(ham, trial, sys_, _params())
    projected = cast(Any, make_mps_trial(tensors, nelec=(3, 2)))  # the wrong trial type on purpose
    with pytest.raises(TypeError, match="RotatedMpsTrial"):
        make_rotated_mps_cpmc_ops(ham, projected, sys_, _params())


# ---------------------------------------------------------------------------------------------
# Overlap and local energy
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROTATIONS)
@pytest.mark.parametrize("nelec, walkers", [((3, 2), (3, 2)), ((3, 3), (3, 3)), ((3, 2), (2, 3))])
def test_overlaps_are_sqrt_w_times_the_projected_trials(nelec, walkers, name):
    R = ROTATIONS[name]
    h1, ham, sys_ = _model(walkers)
    Ca, Cb, ghf = _rotated_ghf(R, nelec)
    rdm1 = get_rdm1_block_diag(ghf)
    projected, mpo, rotated = _three_trials(R, Ca, Cb, walkers, rdm1)
    params = _params()
    ws = _walkers(h1, rdm1, walkers)
    ov1 = _overlaps(make_mps_cpmc_ops(ham, projected, sys_, params), ws, projected)
    ov2 = _overlaps(make_mps_cpmc_ops(ham, mpo, sys_, params), ws, mpo)
    ov3 = _overlaps(make_rotated_mps_cpmc_ops(ham, rotated, sys_, params), ws, rotated)
    root_w = np.sqrt(projected.sector_weight)
    assert 1e-3 < projected.sector_weight < 0.9
    np.testing.assert_allclose(ov3, root_w * ov1, rtol=1e-11, atol=0)
    np.testing.assert_allclose(ov3, root_w * ov2, rtol=1e-11, atol=0)
    amplitudes = hf.mps_sector_amplitudes(rotated.tensors, *walkers)
    oracle = [hf.overlap_with_sd(amplitudes, Wa, Wb) for Wa, Wb in ws]
    np.testing.assert_allclose(ov3, oracle, rtol=1e-10, atol=0)
    ghf_ops = make_ghf_trial_ops(sys_)
    ratio = ov3 / np.array([float(ghf_ops.overlap(tuple(map(jnp.asarray, w)), ghf)) for w in ws])
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)


def test_truncated_walkers_give_the_same_overlaps_up_to_sqrt_w():
    """chi=2 walkers: both trials contract the same truncated walker MPS."""
    R = ROTATIONS["rotation"]
    h1, ham, sys_ = _model((3, 2))
    Ca, Cb, ghf = _rotated_ghf(R, (3, 2))
    rdm1 = get_rdm1_block_diag(ghf)
    projected, _, rotated = _three_trials(R, Ca, Cb, (3, 2), rdm1)
    params = _params(walker_channel_chi=2)
    ws = _walkers(h1, rdm1, (3, 2))
    ov1 = _overlaps(make_mps_cpmc_ops(ham, projected, sys_, params), ws, projected)
    ov3 = _overlaps(make_rotated_mps_cpmc_ops(ham, rotated, sys_, params), ws, rotated)
    np.testing.assert_allclose(ov3, np.sqrt(projected.sector_weight) * ov1, rtol=1e-11, atol=0)
    exact = _overlaps(make_rotated_mps_cpmc_ops(ham, rotated, sys_, _params()), ws, rotated)
    assert np.max(np.abs(ov3 / exact - 1)) > 1e-6, "chi=2 must actually truncate"


@pytest.mark.parametrize("name", ROTATIONS)
@pytest.mark.parametrize("nelec, walkers", [((3, 2), (3, 2)), ((3, 2), (2, 3))])
def test_local_energies_equal_the_projected_trials_and_the_ghf(nelec, walkers, name):
    R = ROTATIONS[name]
    h1, ham, sys_ = _model(walkers)
    Ca, Cb, ghf = _rotated_ghf(R, nelec)
    rdm1 = get_rdm1_block_diag(ghf)
    projected, _, rotated = _three_trials(R, Ca, Cb, walkers, rdm1)
    ws = _walkers(h1, rdm1, walkers)
    ops3 = make_rotated_mps_cpmc_ops(ham, rotated, sys_, _params())
    e3 = _energies(ops3.meas_ops, ws, ham, ops3.meas_ops.build_meas_ctx(ham, rotated), rotated)
    for kernel in ("blocked", "dense"):
        ops1 = make_mps_cpmc_ops(ham, projected, sys_, _params(energy_kernel=kernel))
        ctx1 = ops1.meas_ops.build_meas_ctx(ham, projected)
        np.testing.assert_allclose(
            e3, _energies(ops1.meas_ops, ws, ham, ctx1, projected), rtol=0, atol=1e-11
        )
    ghf_meas = make_ghf_meas_ops_hubbard(sys_)
    e_ghf = _energies(ghf_meas, ws, ham, ghf_meas.build_meas_ctx(ham, ghf), ghf)
    np.testing.assert_allclose(e3, e_ghf, rtol=0, atol=1e-11)


@pytest.mark.parametrize("walkers", [(3, 2), (2, 3)])
def test_a_rotated_exact_doublet_has_zero_variance_in_both_sectors(walkers):
    """U(R)|S=1/2, S_z=1/2> = R00 |S_z=1/2> + R10 |S_z=-1/2>: an eigenstate in both sectors."""
    h1, ham, sys_ = _model(walkers)
    e0, psi = hf.ground_state(h1, U, 3, 2)
    tensors, _ = hf.exact_mps_from_sector_state(psi, L, 3, 2)
    trial = make_rotated_mps_trial(tensors, ROTATIONS["rotation"], nelec=walkers)
    ops = make_rotated_mps_cpmc_ops(ham, trial, sys_, _params())
    ws = _walkers(h1, np.asarray(trial.rdm1), walkers, n=6)
    energies = _energies(ops.meas_ops, ws, ham, ops.meas_ops.build_meas_ctx(ham, trial), trial)
    np.testing.assert_allclose(energies, e0, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "propagator, floor", [("fast", 0.0), ("fast", 0.5), ("slow", 0.0), ("slow", 0.25)]
)
def test_steps_match_the_projected_trial(propagator, floor):
    """Three steps from identical walkers: same walkers, weights, nodes and RNG; overlaps x sqrt(w)."""
    R = ROTATIONS["rotation"]
    h1, ham, sys_ = _model((3, 2))
    Ca, Cb, ghf = _rotated_ghf(R, (3, 2))
    rdm1 = get_rdm1_block_diag(ghf)
    projected, _, rotated = _three_trials(R, Ca, Cb, (3, 2), rdm1)
    params = _params(dt=0.1, weight_floor=floor, propagator=propagator)
    states = []
    for trial, ops in (
        (projected, make_mps_cpmc_ops(ham, projected, sys_, params)),
        (rotated, make_rotated_mps_cpmc_ops(ham, rotated, sys_, params)),
    ):
        meas_ctx = ops.meas_ops.build_meas_ctx(ham, trial)
        start = _initial_state(ham, trial, ops, params, meas_ctx, sys_)
        end = _three_steps(
            ops.prop_ops,
            start,
            params=params,
            ham=ham,
            trial=trial,
            trial_ops=ops.trial_ops,
            meas_ops=ops.meas_ops,
            meas_ctx=meas_ctx,
            prop_ctx=ops.prop_ops.build_prop_ctx(ham, ops.trial_ops.get_rdm1(trial), params),
        )
        states.append((start, end))
    (start1, end1), (start3, end3) = states
    for a, b in zip(start1.walkers, start3.walkers):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    root_w = np.sqrt(projected.sector_weight)
    end1_scaled = end1._replace(overlaps=root_w * end1.overlaps)
    _compare_states(end3, end1_scaled, walkers_atol=1e-12, rtol=1e-10)


# ---------------------------------------------------------------------------------------------
# Capstone: the three approaches and the rotated GHF trial
# ---------------------------------------------------------------------------------------------

# (trial nelec, walkers' nelec, spin rotation, weight_floor, dt, propagator on every side)
CAPSTONE_THREE = [
    pytest.param((3, 2), (3, 2), "rotation", 1e-3, 0.05, "fast", id="three-rotation-32-plain"),
    pytest.param((3, 3), (3, 3), "reflection", 0.5, 0.1, "fast", id="three-reflection-33-floor"),
    pytest.param((3, 2), (3, 2), "rotation", 0.25, 0.1, "slow", id="three-rotation-32-floor-slow"),
    pytest.param((3, 2), (2, 3), "rotation", 1e-3, 0.05, "fast", id="three-rotation-32-to-23"),
]


@pytest.mark.parametrize("nelec, walkers, rotation, floor, dt, propagator", CAPSTONE_THREE)
def test_three_rotation_approaches_and_the_rotated_ghf_run_identically(
    nelec, walkers, rotation, floor, dt, propagator
):
    """Projection, MPO and the rotated trial used as it is all reproduce the rotated GhfTrial."""
    if propagator == "fast":
        assert floor <= 0.5 and dt * U <= 0.4 + 1e-12  # no site with both proposals floored
    else:
        assert floor <= 0.25  # cpmc_slow at floor 0.5 follows one deterministic path
    h1, ham, sys_ = _model(walkers)
    params = _params(dt=dt, weight_floor=floor, propagator=propagator)
    R = ROTATIONS[rotation]
    Ca, Cb, ghf = _rotated_ghf(R, nelec)
    sd = _sd_side(ham, sys_, params, np.asarray(ghf.mo_coeff), propagator)
    projected, mpo, rotated = _three_trials(R, Ca, Cb, walkers, get_rdm1_block_diag(sd[0]))
    assert rotated.bond_dims == tuple(len(q) for q in sd_to_gmps(Ca, Cb, mode="maximal").charges)
    sides = {
        "projected": _mps_side(ham, sys_, params, projected),
        "mpo": _mps_side(ham, sys_, params, mpo),
        "rotated": _rotated_side(ham, sys_, params, rotated),
    }

    reference = _run(sys_, params, ham, sd)
    for side in sides.values():
        _assert_identical_runs(reference, _run(sys_, params, ham, side))
    if floor > 0.1:
        twin = _blocks_twin(sys_, params, ham, sd)
        for side in sides.values():
            _assert_identical_states(
                twin, _blocks_twin(sys_, params, ham, side), nodes_positive=True
            )
