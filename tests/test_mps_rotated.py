"""The spin-rotated MPS trial used as it is: make_mps_trial(rotate_spin(T, R), nelec=...), an MpsTrial with
particle-number labels (no definite S_z, bonds unchanged, sector_weight None).

Walkers are S_z eigenstates and H conserves S_z, so overlaps must be sqrt(w) times those of the
trial projected onto the walkers' sector (w its weight there) and local energies must be the same.
One step and whole run_qmc runs then agree with the projected trial (rotate_mps_trial, the
S_z-conserving MPO), with the projection by exact enumeration and with the rotated GHF trial.
"""

from trot import config

config.configure_once()

from typing import cast

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
from trot.gmps.driver import make_mps_cpmc_ops, run_qmc_mps
from trot.gmps.utils import number_labels, sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.meas.mps import apply_mpo, hubbard_mpo_from_h1, trial_times_h
from trot.trial.ghf import GhfTrial, get_rdm1_block_diag, make_ghf_trial_ops
from trot.trial.mps import (
    MpsTrial,
    compress_mps_qn,
    label_array,
    make_mps_trial,
    natural_orbitals,
    one_rdm,
    physical_charge,
    rotate_spin,
)
from trot.trial.mps_rotation import rotate_mps_trial

U = 4.0
L = 6
_C, _S = np.cos(0.7), np.sin(0.7)
ROTATIONS = {
    "rotation": np.array([[_C, -_S], [_S, _C]]),
    "reflection": np.array([[_C, _S], [_S, -_C]]),  # det = -1
}
N_PHYSICAL = physical_charge(1)[:, 0]  # n_up + n_dn of |0>, |up>, |dn>, |up dn>


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
    """One rotated trial three ways: projected onto the walkers' sector by exact enumeration (1) and by the
    S_z-conserving MPO (2, rotate_mps_trial), and used as it is (3, particle-number labels)."""
    gmps = sd_to_gmps(Ca, Cb, mode="maximal")
    as_is = make_mps_trial(rotate_spin(gmps.tensors, R), nelec=walkers, rdm1=rdm1)
    unrotated = make_mps_trial(gmps.tensors, gmps.charges, nelec=(Ca.shape[1], Cb.shape[1]))
    projected = rotate_mps_trial(unrotated, R, nelec=walkers, rdm1=rdm1)
    amplitudes = hf.mps_sector_amplitudes(as_is.tensors, *walkers)
    tensors, charges = hf.exact_mps_from_sector_state(amplitudes, L, *walkers)
    enumerated = make_mps_trial(tensors, charges, nelec=walkers, rdm1=rdm1)
    return enumerated, projected, as_is


def _root_weight(projected) -> float:
    """sqrt(w), w = <P U T|P U T>/<T|T> the walkers' sector weight of the rotated trial (rotate_mps_trial)."""
    w = projected.sector_weight
    assert w is not None and 1e-3 < w < 0.9, "the rotation must move weight out of the walkers' sector"
    return float(np.sqrt(w))


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


def _rotated_case(R, nelec, walkers):
    """Model, the projected and as-is rotated trials of one rotated UHF determinant, and walkers."""
    h1, ham, sys_ = _model(walkers)
    Ca, Cb, ghf = _rotated_ghf(R, nelec)
    rdm1 = get_rdm1_block_diag(ghf)
    _, projected, as_is = _three_trials(R, Ca, Cb, walkers, rdm1)
    return h1, ham, sys_, projected, as_is, _walkers(h1, rdm1, walkers)


def _energy_by_enumeration(h1, amplitudes, nelec):
    """<psi|H|psi>/<psi|psi> of alpha-block sector amplitudes, with the dense sector Hamiltonian."""
    psi = np.asarray(amplitudes).ravel()
    return float(psi @ hf.hubbard_sector_hamiltonian(h1, U, *nelec) @ psi / (psi @ psi))


# ---------------------------------------------------------------------------------------------
# The trial
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROTATIONS)
def test_rotated_trial_keeps_the_rotated_tensors_and_the_ghf_walker_start(name):
    R = ROTATIONS[name]
    Ca, Cb, ghf = _rotated_ghf(R, (3, 2))
    gmps = sd_to_gmps(Ca, Cb, mode="maximal")
    rotated = rotate_spin(gmps.tensors, R)
    trial = make_mps_trial(rotated, nelec=(3, 2))
    assert trial.nelec == (3, 2) and trial.norb == L
    assert trial.label_width == 1 and trial.sector_weight is None
    assert trial.bond_dims == tuple(len(q) for q in gmps.charges)
    vector = hf.mps_full_vector(rotated)
    np.testing.assert_allclose(
        hf.mps_full_vector(trial.tensors), vector / np.linalg.norm(vector), rtol=0, atol=1e-13
    )
    np.testing.assert_allclose(np.asarray(trial.rdm1), np.stack(one_rdm(rotated)), atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(trial.rdm1), np.asarray(get_rdm1_block_diag(ghf)), rtol=0, atol=1e-12
    )
    # the projected trial starts its walkers from the same (unprojected) rotated 1-RDM
    projected = rotate_mps_trial(make_mps_trial(gmps.tensors, gmps.charges, nelec=(3, 2)), R)
    assert projected.label_width == 2 and projected.nelec == (3, 2)
    np.testing.assert_allclose(np.asarray(projected.rdm1), np.asarray(trial.rdm1), rtol=0, atol=1e-12)
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    assert len(leaves) == L + 1
    same = jax.jit(lambda t: t)(trial)
    assert isinstance(same, MpsTrial) and same.nelec == (3, 2) and same.sector_weight is None
    assert same.charges == trial.charges
    assert jax.tree_util.tree_structure(same) == treedef


def test_rotated_trial_validates_its_input():
    R = ROTATIONS["rotation"]
    Ca, Cb, _ = _rotated_ghf(R, (3, 2))
    gmps = sd_to_gmps(Ca, Cb, mode="maximal")
    tensors = rotate_spin(gmps.tensors, R)
    with pytest.raises(TypeError, match="real"):
        make_mps_trial([t.astype(complex) for t in tensors], nelec=(3, 2))
    with pytest.raises(ValueError, match="4"):
        make_mps_trial([t[:, :3] for t in tensors], nelec=(3, 2))
    with pytest.raises(ValueError, match="bond"):
        make_mps_trial([tensors[0][:, :, :1]] + tensors[1:], nelec=(3, 2))
    with pytest.raises(ValueError, match="sector"):
        make_mps_trial(tensors, nelec=(7, -2))
    with pytest.raises(ValueError, match="shape"):
        make_mps_trial(tensors, nelec=(3, 2), rdm1=np.zeros((2, L, L + 1)))
    with pytest.raises(ValueError, match="N = 5"):
        make_mps_trial(tensors, nelec=(3, 3))
    with pytest.raises(ValueError, match="orthogonal"):
        rotate_spin(gmps.tensors, np.ones((2, 2)))
    unrotated = make_mps_trial(gmps.tensors, gmps.charges, nelec=(3, 2))
    with pytest.raises(ValueError, match="orthogonal"):
        rotate_mps_trial(unrotated, np.ones((2, 2)))
    with pytest.raises(ValueError, match="keeps N = 5"):
        rotate_mps_trial(unrotated, R, nelec=(3, 3))
    trial = make_mps_trial(tensors, nelec=(3, 2))
    with pytest.raises(ValueError, match="bond labels"):
        rotate_mps_trial(trial, R)  # N labels only: no (N_up, N_dn) sector to rotate from
    _, ham, sys_ = _model((3, 3))
    with pytest.raises(ValueError, match="does not match the system"):
        make_mps_cpmc_ops(ham, trial, sys_, _params())


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
    enumerated, projected, as_is = _three_trials(R, Ca, Cb, walkers, rdm1)
    params = _params()
    ws = _walkers(h1, rdm1, walkers)
    ov1 = _overlaps(make_mps_cpmc_ops(ham, enumerated, sys_, params), ws, enumerated)
    ov2 = _overlaps(make_mps_cpmc_ops(ham, projected, sys_, params), ws, projected)
    ov3 = _overlaps(make_mps_cpmc_ops(ham, as_is, sys_, params), ws, as_is)
    amplitudes = hf.mps_sector_amplitudes(as_is.tensors, *walkers)
    root_w = _root_weight(projected)
    assert root_w**2 == pytest.approx(np.sum(amplitudes**2), rel=1e-10)  # w by enumeration
    np.testing.assert_allclose(ov2, ov1, rtol=1e-11, atol=0)
    np.testing.assert_allclose(ov3, root_w * ov2, rtol=1e-11, atol=0)
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
    _, projected, as_is = _three_trials(R, Ca, Cb, (3, 2), rdm1)
    params = _params(walker_channel_chi=2)
    ws = _walkers(h1, rdm1, (3, 2))
    ov2 = _overlaps(make_mps_cpmc_ops(ham, projected, sys_, params), ws, projected)
    ov3 = _overlaps(make_mps_cpmc_ops(ham, as_is, sys_, params), ws, as_is)
    np.testing.assert_allclose(ov3, _root_weight(projected) * ov2, rtol=1e-11, atol=0)
    exact = _overlaps(make_mps_cpmc_ops(ham, as_is, sys_, _params()), ws, as_is)
    assert np.max(np.abs(ov3 / exact - 1)) > 1e-6, "chi=2 must actually truncate"


@pytest.mark.parametrize("name", ROTATIONS)
@pytest.mark.parametrize("nelec, walkers", [((3, 2), (3, 2)), ((3, 2), (2, 3))])
def test_local_energies_equal_the_projected_trials_and_the_ghf(nelec, walkers, name):
    R = ROTATIONS[name]
    h1, ham, sys_ = _model(walkers)
    Ca, Cb, ghf = _rotated_ghf(R, nelec)
    rdm1 = get_rdm1_block_diag(ghf)
    _, projected, as_is = _three_trials(R, Ca, Cb, walkers, rdm1)
    ws = _walkers(h1, rdm1, walkers)
    ghf_meas = make_ghf_meas_ops_hubbard(sys_)
    e_ghf = _energies(ghf_meas, ws, ham, ghf_meas.build_meas_ctx(ham, ghf), ghf)
    for kernel in ("blocked", "dense"):
        for trial in (projected, as_is):
            ops = make_mps_cpmc_ops(ham, trial, sys_, _params(energy_kernel=kernel))
            ctx = ops.meas_ops.build_meas_ctx(ham, trial)
            np.testing.assert_allclose(
                _energies(ops.meas_ops, ws, ham, ctx, trial), e_ghf, rtol=0, atol=1e-11
            )


@pytest.mark.parametrize("walkers", [(3, 2), (2, 3)])
def test_a_rotated_exact_doublet_has_zero_variance_in_both_sectors(walkers):
    """U(R)|S=1/2, S_z=1/2> = R00 |S_z=1/2> + R10 |S_z=-1/2>: an eigenstate in both sectors."""
    h1, ham, sys_ = _model(walkers)
    e0, psi = hf.ground_state(h1, U, 3, 2)
    tensors, _ = hf.exact_mps_from_sector_state(psi, L, 3, 2)
    trial = make_mps_trial(rotate_spin(tensors, ROTATIONS["rotation"]), nelec=walkers)
    assert trial.label_width == 1
    ops = make_mps_cpmc_ops(ham, trial, sys_, _params())
    ctx = ops.meas_ops.build_meas_ctx(ham, trial)
    assert ctx.trial_energy == pytest.approx(e0, abs=1e-10)
    ws = _walkers(h1, np.asarray(trial.rdm1), walkers, n=6)
    energies = _energies(ops.meas_ops, ws, ham, ctx, trial)
    np.testing.assert_allclose(energies, e0, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------------------------

STEP_SETTINGS = {
    "exact": {},
    "truncated-native": dict(orbital_plan="adaptive", walker_channel_chi=2, walker_qr="native"),
    "truncated-cholesky": dict(orbital_plan="adaptive", walker_channel_chi=2, walker_qr="cholesky"),
}


@pytest.mark.parametrize("setting", STEP_SETTINGS)
@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_steps_match_the_projected_trial(floor, setting):
    """Three steps from identical walkers: same walkers, weights, shift, nodes and RNG; overlaps x sqrt(w)."""
    _, ham, sys_, projected, as_is, _ = _rotated_case(ROTATIONS["rotation"], (3, 2), (3, 2))
    params = _params(dt=0.1, weight_floor=floor, **STEP_SETTINGS[setting])
    states = []
    for trial in (projected, as_is):
        ops = make_mps_cpmc_ops(ham, trial, sys_, params)
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
    (start2, end2), (start3, end3) = states
    for a, b in zip(start2.walkers, start3.walkers):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    end2_scaled = end2._replace(overlaps=_root_weight(projected) * end2.overlaps)
    _compare_states(end3, end2_scaled, walkers_atol=1e-12, rtol=1e-10)


def test_run_qmc_mps_takes_a_rotated_trial():
    """run_qmc_mps end to end (the MPS block, truncated walkers) with the trial used as it is and with the
    projected trial: the same blocks."""
    _, ham, sys_, projected, as_is, _ = _rotated_case(ROTATIONS["rotation"], (3, 2), (3, 2))
    params = _params(orbital_plan="adaptive", walker_channel_chi=2, walker_qr="native", n_eql_blocks=2, n_blocks=4)
    runs = [run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=t) for t in (projected, as_is)]
    for key in ("block_energies", "block_weights"):
        np.testing.assert_allclose(
            np.asarray(getattr(runs[1], key)), np.asarray(getattr(runs[0], key)), rtol=1e-9
        )


# ---------------------------------------------------------------------------------------------
# Capstone: the three approaches and the rotated GHF trial
# ---------------------------------------------------------------------------------------------

# (trial nelec, walkers' nelec, spin rotation, weight_floor, dt, trot propagator of the GHF side)
CAPSTONE_THREE = [
    pytest.param((3, 2), (3, 2), "rotation", 1e-3, 0.05, "fast", id="three-rotation-32-plain"),
    pytest.param((3, 3), (3, 3), "reflection", 0.5, 0.1, "fast", id="three-reflection-33-floor"),
    pytest.param((3, 2), (3, 2), "rotation", 0.0, 0.1, "slow", id="three-rotation-32-slow-vs-fast"),
    pytest.param((3, 2), (2, 3), "rotation", 1e-3, 0.05, "fast", id="three-rotation-32-to-23"),
]


@pytest.mark.parametrize("nelec, walkers, rotation, floor, dt, sd_propagator", CAPSTONE_THREE)
def test_three_rotation_approaches_and_the_rotated_ghf_run_identically(
    nelec, walkers, rotation, floor, dt, sd_propagator
):
    """Projection by enumeration, the MPO and the rotated trial used as it is all reproduce the rotated GhfTrial."""
    if sd_propagator == "fast":
        assert floor <= 0.5 and dt * U <= 0.4 + 1e-12  # no site with both proposals floored
    else:
        assert floor == 0.0  # cpmc_slow floors half the ratio: its rule and the MPS step's coincide only at 0
    _, ham, sys_ = _model(walkers)
    params = _params(dt=dt, weight_floor=floor)
    R = ROTATIONS[rotation]
    Ca, Cb, ghf = _rotated_ghf(R, nelec)
    sd = _sd_side(ham, sys_, params, np.asarray(ghf.mo_coeff), sd_propagator)
    enumerated, projected, as_is = _three_trials(R, Ca, Cb, walkers, get_rdm1_block_diag(sd[0]))
    assert as_is.bond_dims == tuple(len(q) for q in sd_to_gmps(Ca, Cb, mode="maximal").charges)
    sides = {
        "enumerated": _mps_side(ham, sys_, params, enumerated),
        "projected": _mps_side(ham, sys_, params, projected),
        "as-is": _mps_side(ham, sys_, params, as_is),
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


# ---------------------------------------------------------------------------------------------
# Particle-number labels: the engine blocks the trial used as it is on N
# ---------------------------------------------------------------------------------------------


def _respects_number_labels(tensors, labels) -> bool:
    """Every nonzero entry A[i, p, j] out of a reached index i has label_j = label_i + n_up(p) + n_dn(p)."""
    for s, A in enumerate(tensors):
        i, p, j = np.nonzero(np.asarray(A))
        reached = labels[s][i, 0] >= 0
        i, p, j = i[reached], p[reached], j[reached]
        if not np.array_equal(labels[s][i, 0] + N_PHYSICAL[p], labels[s + 1][j, 0]):
            return False
    return True


def _break_number_conservation(tensors):
    """Copy of an N-labelled MPS with one entry that joins two particle numbers at a bond index."""
    broken = [np.array(A) for A in tensors]
    site = L // 2
    labels = number_labels(broken)
    assert labels is not None
    i, p, j = next(e for e in np.argwhere(broken[site] != 0) if labels[site][e[0], 0] >= 0)
    broken[site][i, 0 if N_PHYSICAL[p] else 1, j] = 0.1
    return broken


@pytest.mark.parametrize("name", ROTATIONS)
def test_number_labels_survive_the_rotation(name):
    """A spin rotation keeps N site by site: the trial used as it is carries the input's N labels."""
    R = ROTATIONS[name]
    Ca, Cb, _ = _rotated_ghf(R, (3, 2))
    gmps = sd_to_gmps(Ca, Cb, mode="maximal")
    trial = make_mps_trial(rotate_spin(gmps.tensors, R), nelec=(3, 2))
    labels = trial.charge_arrays()
    assert trial.label_width == 1 and int(labels[-1][0, 0]) == 5
    assert all(q.shape == (D, 1) for q, D in zip(labels, trial.bond_dims))
    assert _respects_number_labels(trial.tensors, labels)
    for got, pairs in zip(labels, gmps.charges):
        reached = got[:, 0] >= 0  # an index without incoming nonzeros gets -1
        np.testing.assert_array_equal(got[reached], label_array(pairs).sum(axis=1, keepdims=True)[reached])
    unrotated = number_labels(gmps.tensors)
    assert unrotated is not None
    for a, b in zip(labels, unrotated):
        np.testing.assert_array_equal(a, b)
    assert number_labels(_break_number_conservation(trial.tensors)) is None


def test_trial_times_h_and_compress_mps_qn_keep_number_labels():
    """H|T> of the rotated trial with N labels (width 1) is H|T> with exact labels, and the sector-by-sector
    compression keeps the state and the labels."""
    R = ROTATIONS["reflection"]
    h1, _, _ = _model((3, 2))
    Ca, Cb, _ = _rotated_ghf(R, (3, 2))
    trial = make_mps_trial(rotate_spin(sd_to_gmps(Ca, Cb, mode="maximal").tensors, R), nelec=(3, 2))
    tensors = [np.asarray(A) for A in trial.tensors]
    labels = trial.charge_arrays()
    assert trial.label_width == 1
    assert all(np.all(q >= 0) for q in labels), "precondition: every trial index is reached"
    W = hubbard_mpo_from_h1(h1, U)
    want = hf.mps_full_vector(apply_mpo(W, tensors))
    exact, exact_labels = trial_times_h(W, tensors, labels)
    small, small_labels = compress_mps_qn(exact, exact_labels)
    for mps, labels in ((exact, exact_labels), (small, small_labels)):
        assert all(q.shape[1] == 1 for q in labels)
        assert _respects_number_labels(mps, labels)
        np.testing.assert_allclose(hf.mps_full_vector(mps), want, rtol=0, atol=1e-12)
    assert max(A.shape[0] for A in small) <= max(A.shape[0] for A in exact)


@pytest.mark.parametrize("name", ROTATIONS)
@pytest.mark.parametrize("nelec, walkers", [((3, 2), (3, 2)), ((3, 3), (3, 3)), ((3, 2), (2, 3))])
@pytest.mark.parametrize("energy_kernel", ["blocked", "dense"])
@pytest.mark.parametrize("walker_channel_chi", [None, 2])
def test_overlaps_and_energies_match_the_projected_trial_on_every_setting(
    name, nelec, walkers, energy_kernel, walker_channel_chi
):
    """Adaptive plan, exact and truncated walkers, both energy kernels: the trial used as it is (N-blocked, walker
    channels apart) gives sqrt(w) times the projected trial's overlaps and the same local energies. Its <T|H|T> runs
    over every sector, so it is the unrotated determinant's (H is spin-rotation invariant); the projected trial's
    is the walkers' sector's."""
    R = ROTATIONS[name]
    h1, ham, sys_, projected, as_is, ws = _rotated_case(R, nelec, walkers)
    params = _params(orbital_plan="adaptive", walker_channel_chi=walker_channel_chi, energy_kernel=energy_kernel,
                     walker_qr="cholesky")
    results = []
    for trial in (projected, as_is):
        ops = make_mps_cpmc_ops(ham, trial, sys_, params)
        ctx = ops.meas_ops.build_meas_ctx(ham, trial)
        results.append((_overlaps(ops, ws, trial), _energies(ops.meas_ops, ws, ham, ctx, trial), ctx))
    (ov2, e2, ctx2), (ov3, e3, ctx3) = results
    np.testing.assert_allclose(ov3, _root_weight(projected) * ov2, rtol=1e-10)
    np.testing.assert_allclose(e3, e2, rtol=0, atol=1e-10)
    Ca, Cb, _ = _rotated_ghf(R, nelec)
    e_sd = _energy_by_enumeration(h1, hf.sd_amplitudes(Ca, Cb), nelec)
    e_sector = _energy_by_enumeration(h1, hf.mps_sector_amplitudes(as_is.tensors, *walkers), walkers)
    assert ctx3.trial_energy == pytest.approx(e_sd, abs=1e-10)
    assert ctx2.trial_energy == pytest.approx(e_sector, abs=1e-10)


def test_unreached_trial_indices_are_dropped_exactly():
    """A bond index that no nonzero entry reaches gets the label -1, pairs with no walker label, and changes
    neither the overlaps nor the local energies."""
    R = ROTATIONS["rotation"]
    h1, ham, sys_ = _model((3, 2))
    Ca, Cb, ghf = _rotated_ghf(R, (3, 2))
    rdm1 = get_rdm1_block_diag(ghf)
    tensors = rotate_spin(sd_to_gmps(Ca, Cb, mode="maximal").tensors, R)
    c = L // 2
    padded = list(tensors)
    padded[c - 1] = np.concatenate([tensors[c - 1], np.zeros(tensors[c - 1].shape[:2] + (1,))], axis=2)
    rows = np.random.default_rng(0).standard_normal((1,) + tensors[c].shape[1:])
    padded[c] = np.concatenate([tensors[c], rows], axis=0)
    ws = _walkers(h1, rdm1, (3, 2))
    params = _params(orbital_plan="adaptive", walker_qr="native")
    results = []
    for t in (tensors, padded):
        trial = make_mps_trial(t, nelec=(3, 2), rdm1=rdm1)
        ops = make_mps_cpmc_ops(ham, trial, sys_, params)
        ctx = ops.meas_ops.build_meas_ctx(ham, trial)
        results.append((trial, _overlaps(ops, ws, trial), _energies(ops.meas_ops, ws, ham, ctx, trial)))
    (trial, overlaps, energies), (padded_trial, padded_overlaps, padded_energies) = results
    assert padded_trial.bond_dims[c] == trial.bond_dims[c] + 1
    assert int(padded_trial.charge_arrays()[c][-1, 0]) == -1
    np.testing.assert_allclose(padded_overlaps, overlaps, rtol=1e-13, atol=0)
    np.testing.assert_allclose(padded_energies, energies, rtol=0, atol=1e-11)


def test_a_trial_that_mixes_particle_numbers_is_rejected():
    """The engine blocks on particle number at least: an MPS without N labels has nothing to block on."""
    R = ROTATIONS["rotation"]
    Ca, Cb, _ = _rotated_ghf(R, (3, 2))
    broken = _break_number_conservation(rotate_spin(sd_to_gmps(Ca, Cb, mode="maximal").tensors, R))
    assert number_labels(broken) is None
    with pytest.raises(ValueError, match="mixes particle numbers"):
        make_mps_trial(broken, nelec=(3, 2))
