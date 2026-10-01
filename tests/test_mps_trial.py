"""Tests for trot/trial/mps.py: the MpsTrial pytree, trials from determinants, dense MPS and DMRG,
the exact (N_up, N_dn) sector projection, spin rotations, walker plans and the blocked overlap.

Every reference comes from tests/helpers/hubbard_fock.py (exact Fock-space enumeration), from
trot's GHF trial or from pyblock3, never from the MPS code under test. Systems: an L=6 open chain
(t=1, U=4) with nelec (3, 3) and (3, 2), and a 2x3 lattice periodic in x (doubled rungs) with two
on-site terms, U=4, (3, 2).
"""

from trot import config

config.configure_once()

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from tests.helpers import hubbard_fock as hf
from trot.core.system import System
from trot.gmps import utils as gmps_utils
from trot.gmps.utils import combine_channels, sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix, square_hopping_matrix
from trot.meas.mps import hubbard_mpo_from_h1, trial_times_h
from trot.prop.types import QmcParamsMps
from trot.trial.ghf import GhfTrial, get_rdm1_block_diag, overlap_u
from trot.trial.mps import (
    PHYSICAL_CHARGE,
    MpsTrial,
    as_mps_trial,
    compress_mps_qn,
    contraction_layout,
    convert_walker,
    extract_fixed_blocks,
    gather_blocks,
    make_mps_trial,
    make_mps_trial_ops,
    make_walker_plan,
    make_walker_plan_from_reference,
    mps_overlap,
    mps_trial_from_pyblock3,
    mps_trial_from_sd,
    project_to_sector,
    rotate_spin,
    spin_rotation_unitary,
)
from trot.walkers import vmap_chunked

REPO = Path(__file__).resolve().parents[1]
U = 4.0
L = 6
NELEC = (3, 2)
E_CHAIN = {(3, 3): -3.092565319505, (3, 2): -3.984358962762}  # exact, L=6 open chain, U=4
E_L4 = -1.953145308685  # exact, L=4 open chain, (2, 2), U=4
THETA = 0.7
ROTATION = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])
REFLECTION = np.array([[np.cos(THETA), np.sin(THETA)], [np.sin(THETA), -np.cos(THETA)]])
SPIN_MAPS = {"rotation": ROTATION, "reflection": REFLECTION}
TRIALS = ("ed", "sd", "rotated_sd")
PLAN_SETTINGS = {
    "exact": dict(orbital_plan="maximal", walker_channel_chi=None),
    "truncated": dict(orbital_plan="adaptive", walker_channel_chi=2),
}


# ---------------------------------------------------------------------------------------------
# Builders and checks
# ---------------------------------------------------------------------------------------------


def _lattice_h1():
    h1 = square_hopping_matrix(2, 3, 1.0, "periodic", "open")
    h1[0, 0] = 0.3
    h1[4, 4] = -0.2
    return h1


def _sd_tensors(Ca, Cb):
    """The exact d=4 MPS of SD(Ca, Cb) (maximal plan, no truncation) as NumPy arrays."""
    return [np.asarray(t) for t in sd_to_gmps(Ca, Cb, mode="maximal").tensors]


def _ghf_orbitals(R, Ca, Cb):
    """kron(R, I_L) @ block_diag(Ca, Cb): spin orbitals of the spin-rotated determinant."""
    return np.kron(R, np.eye(len(Ca))) @ scipy.linalg.block_diag(Ca, Cb)


def _jnp_walker(wa, wb):
    return jnp.asarray(wa), jnp.asarray(wb)


def _walkers(h1, n, seed, all_nonorthonormal=False):
    """CPMC-like determinants (unguided random HS fields); odd ones (or all) non-orthonormal."""
    rng = np.random.default_rng(seed)
    start = hf.staggered_determinant(h1, *NELEC, field=0.3)
    out = []
    for i, (wa, wb) in enumerate(
        hf.random_field_walkers(h1, U, 0.1, *start, n=n, steps=10, seed=seed)
    ):
        if all_nonorthonormal or i % 2:  # the same determinants up to a scalar
            wa, wb = hf.nonorthonormal(wa, rng), hf.nonorthonormal(wb, rng)
        out.append((wa, wb))
    return out


def _sd_overlaps(tensors, walkers):
    """<T|SD(Wa, Wb)> of a d=4 MPS by enumeration of its (N_up, N_dn) = NELEC amplitudes."""
    amplitudes = hf.mps_sector_amplitudes(tensors, *NELEC)
    return np.array([hf.overlap_with_sd(amplitudes, wa, wb) for wa, wb in walkers])


def _scaled(got, want):
    """c * want with the one free global constant c = got / want at the largest |want| entry."""
    got, want = np.asarray(got), np.asarray(want)
    k = np.unravel_index(np.argmax(np.abs(want)), want.shape)
    return got[k] / want[k] * want


def _fit_error(got, want):
    """max |got - c want| / max |c want|."""
    scaled = _scaled(got, want)
    return np.abs(np.asarray(got) - scaled).max() / np.abs(scaled).max()


def _assert_close_up_to_constant(got, want, tol):
    """got == c * want for one global constant c (atol tol * max |c want|)."""
    scaled = _scaled(got, want)
    np.testing.assert_allclose(got, scaled, rtol=0, atol=tol * np.abs(scaled).max())


def _assert_labels_respected(tensors, charges, nelec):
    """(N_up, N_dn) bond labels from (0, 0) to nelec that every nonzero entry respects."""
    Q = [np.asarray(q, int).reshape(-1, 2) for q in charges]
    assert len(Q) == len(tensors) + 1
    np.testing.assert_array_equal(Q[0], [[0, 0]])
    np.testing.assert_array_equal(Q[-1], [list(nelec)])
    for s, A in enumerate(tensors):
        A = np.asarray(A)
        assert A.shape[0] == len(Q[s]) and A.shape[2] == len(Q[s + 1])
        for p, delta in enumerate(PHYSICAL_CHARGE):
            left, right = np.nonzero(A[:, p, :])
            np.testing.assert_array_equal(Q[s + 1][right], Q[s][left] + delta)


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chain():
    """L=6 open chain: h1 and the staggered determinants for (3, 3) and (3, 2)."""
    h1 = hopping_matrix(L, 1.0)
    for nelec, energy in E_CHAIN.items():  # the oracle's Hamiltonian is right
        assert abs(hf.ground_state(h1, U, *nelec)[0] - energy) < 1e-10
    assert abs(hf.ground_state(hopping_matrix(4, 1.0), U, 2, 2)[0] - E_L4) < 1e-10
    return SimpleNamespace(h1=h1, sd={ne: hf.staggered_determinant(h1, *ne) for ne in E_CHAIN})


@pytest.fixture(scope="module")
def lattice():
    """2x3 lattice, (3, 2): ED ground state, three trials with oracles, walkers, walker plans."""
    h1 = _lattice_h1()
    E0, psi = hf.ground_state(h1, U, *NELEC)
    Ca, Cb = hf.staggered_determinant(h1, *NELEC)
    trials = {
        "ed": make_mps_trial(*hf.exact_mps_from_sector_state(psi, L, *NELEC), nelec=NELEC),
        "sd": mps_trial_from_sd(Ca, Cb),
        "rotated_sd": make_mps_trial(rotate_spin(_sd_tensors(Ca, Cb), ROTATION), nelec=NELEC),
    }
    oracles = {  # independent alpha-block amplitudes, each proportional to its trial's
        "ed": psi,
        "sd": hf.sd_amplitudes(Ca, Cb),
        "rotated_sd": hf.ghf_amplitudes(_ghf_orbitals(ROTATION, Ca, Cb), *NELEC),
    }
    ham = HamHubbard(h1=jnp.asarray(h1), u=U)
    sys_ = System(norb=L, nelec=NELEC, walker_kind="unrestricted")
    plans = {  # natural-orbital reference, as in a run
        (name, kind): make_walker_plan(ham, trial, sys_, QmcParamsMps(seed=0, **settings))
        for name, trial in trials.items()
        for kind, settings in PLAN_SETTINGS.items()
    }
    return SimpleNamespace(
        h1=h1,
        E0=E0,
        psi=psi,
        sd=(Ca, Cb),
        trials=trials,
        oracles=oracles,
        walkers=_walkers(h1, 8, seed=1),
        plans=plans,
    )


@pytest.fixture(scope="module")
def dmrg_2x3(lattice):
    """2x3 lattice: pyblock3 DMRG at chi=64, which is exact for six sites."""
    pytest.importorskip("pyblock3")
    from trot.gmps.dmrg import make_dmrg_trial

    ham = HamHubbard(h1=jnp.asarray(lattice.h1), u=U)
    sys_ = System(norb=L, nelec=NELEC, walker_kind="unrestricted")
    return make_dmrg_trial(ham, sys_, chi=64, n_sweeps=8, seed=0)


# ---------------------------------------------------------------------------------------------
# 1: the pytree
# ---------------------------------------------------------------------------------------------


def test_mps_trial_is_a_pytree_with_static_labels(lattice):
    """MpsTrial flattens to L tensors + rdm1, labels and sector weight are its hashable aux."""
    trial, other = lattice.trials["sd"], lattice.trials["ed"]
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    assert len(leaves) == L + 1
    assert all(a is b for a, b in zip(leaves, (*trial.tensors, trial.rdm1)))
    _, aux = trial.tree_flatten()
    assert aux == (trial.charges, trial.sector_weight)
    hash(aux)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, MpsTrial)
    assert rebuilt.charges == trial.charges and rebuilt.sector_weight == trial.sector_weight
    assert all(a is b for a, b in zip(jax.tree_util.tree_leaves(rebuilt), leaves))

    def first_sum(t):
        assert t.charges == trial.charges and t.nelec == NELEC  # aux stays Python data in jit
        return t.tensors[0].sum()

    np.testing.assert_allclose(
        jax.jit(first_sum)(trial), np.asarray(trial.tensors[0]).sum(), rtol=0, atol=1e-14
    )
    passed = jax.jit(lambda t: t)(trial)
    assert isinstance(passed, MpsTrial) and passed.charges == trial.charges

    assert jax.tree_util.tree_structure(trial) == treedef
    assert jax.tree_util.tree_structure(rebuilt) == treedef
    assert jax.tree_util.tree_structure(mps_trial_from_sd(*lattice.sd)) == treedef
    assert other.charges != trial.charges
    assert jax.tree_util.tree_structure(other) != treedef


# ---------------------------------------------------------------------------------------------
# 2-5: trials from determinants, sector projection and spin rotations
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nelec", [(3, 3), (3, 2)], ids=["3up3dn", "3up2dn"])
def test_trial_from_sd_is_exact(chain, nelec):
    """mps_trial_from_sd is SD(Ca, Cb): amplitudes, unit norm, projector rdm1, labels, weight 1."""
    nup, ndn = nelec
    Ca, Cb = chain.sd[nelec]
    trial = mps_trial_from_sd(Ca, Cb)
    want = hf.sd_amplitudes(Ca, Cb)
    _assert_close_up_to_constant(hf.mps_sector_amplitudes(trial.tensors, nup, ndn), want, 1e-12)
    vector = hf.mps_full_vector(trial.tensors)
    assert abs(vector @ vector - 1.0) < 1e-13

    Qa, Qb = np.linalg.qr(Ca)[0], np.linalg.qr(Cb)[0]
    projectors = np.stack([Qa @ Qa.T, Qb @ Qb.T])
    np.testing.assert_allclose(trial.rdm1, projectors, rtol=0, atol=1e-13)
    # the projectors are the determinant's 1-RDM by enumeration
    np.testing.assert_allclose(hf.dense_rdm1(want, L, nup, ndn), projectors, rtol=0, atol=1e-13)

    assert trial.charges[0] == ((0, 0),) and trial.charges[-1] == ((nup, ndn),)
    assert trial.nelec == nelec and trial.norb == L
    np.testing.assert_array_equal(PHYSICAL_CHARGE, [[p % 2, p // 2] for p in range(4)])
    _assert_labels_respected(trial.tensors, trial.charges, nelec)
    assert abs(trial.sector_weight - 1.0) < 1e-12


def test_project_to_sector_is_the_exact_projection(chain, lattice):
    """project_to_sector is P|T> with SZ labels, keeps valid SZ input and rejects empty sectors."""
    Ca, Cb = chain.sd[NELEC]
    gmps = sd_to_gmps(Ca, Cb, mode="maximal")
    rotated = rotate_spin([np.asarray(t) for t in gmps.tensors], ROTATION)
    rng = np.random.default_rng(7)
    bonds = [1, 4, 8, 8, 4, 1]
    unlabelled = [rng.standard_normal((bonds[i], 4, bonds[i + 1])) for i in range(5)]

    # (a) the spin-rotated determinant onto (3, 2); (b) an unlabelled random MPS onto (2, 2)
    results = {}
    for name, tensors, nelec in (("a", rotated, NELEC), ("b", unlabelled, (2, 2))):
        projected, charges, norm2 = project_to_sector(tensors, nelec)
        want = hf.sector_from_full(hf.mps_full_vector(tensors), len(tensors), *nelec)
        got = hf.mps_sector_amplitudes(projected, *nelec)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())
        np.testing.assert_allclose(norm2, np.sum(want**2), rtol=1e-12)
        _assert_labels_respected(projected, charges, nelec)
        results[name] = projected, charges, norm2
    # labels that the tensors violate (the unrotated determinant's) are ignored, not an error
    stale = project_to_sector(rotated, NELEC, gmps.charges)
    assert all(np.array_equal(x, y) for x, y in zip(stale[0], results["a"][0]))
    assert all(np.array_equal(x, y) for x, y in zip(stale[1], results["a"][1]))

    # (c) valid SZ labels that end at nelec: returned unchanged (sd_to_gmps output, and the
    # projection of (a), so projecting twice is projecting once). Not hf.exact_mps_from_sector_state:
    # its last SVD remainder breaks its labels by ~1e-15, which correctly triggers the projection.
    for tensors, charges in ((gmps.tensors, gmps.charges), results["a"][:2]):
        out, out_charges, norm2 = project_to_sector(tensors, NELEC, charges)
        assert len(out) == len(tensors) and len(out_charges) == len(charges)
        assert all(np.array_equal(x, np.asarray(y)) for x, y in zip(out, tensors))
        assert all(np.array_equal(x, np.asarray(y)) for x, y in zip(out_charges, charges))
        vector = hf.mps_full_vector(tensors)
        np.testing.assert_allclose(norm2, vector @ vector, rtol=1e-12)

    # (d) the spin swap maps the (3, 2) determinant into (2, 3): no (3, 2) weight is left
    swapped = rotate_spin([np.asarray(t) for t in gmps.tensors], [[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="no weight"):
        make_mps_trial(swapped, nelec=NELEC)
    assert abs(make_mps_trial(swapped, nelec=(2, 3)).sector_weight - 1.0) < 1e-12


@pytest.mark.parametrize("spin_map", list(SPIN_MAPS))
def test_rotate_spin_is_the_ghf_determinant(chain, spin_map):
    """rotate_spin(SD, R) is the determinant kron(R, I) @ blockdiag(Ca, Cb): R, not R^T, det R."""
    R = SPIN_MAPS[spin_map]
    Ca, Cb = chain.sd[NELEC]
    tensors = _sd_tensors(Ca, Cb)
    C = _ghf_orbitals(R, Ca, Cb)
    want = hf.ghf_full_vector(C)
    _assert_close_up_to_constant(hf.mps_full_vector(rotate_spin(tensors, R)), want, 1e-12)
    M = spin_rotation_unitary(R)
    np.testing.assert_allclose(
        M, cast(np.ndarray, scipy.linalg.block_diag(1.0, R, np.linalg.det(R))), rtol=0, atol=1e-15
    )
    if spin_map == "rotation":  # R^T is the inverse rotation: a different state
        assert _fit_error(hf.mps_full_vector(rotate_spin(tensors, R.T)), want) > 1e-3
        return

    # a 4x4 site map without det(R) = -1 on |up dn> is not the fermionic spin reflection
    no_det = M.copy()
    no_det[3, 3] = 1.0
    rng = np.random.default_rng(4)
    walkers = [(rng.standard_normal((L, 3)), rng.standard_normal((L, 2))) for _ in range(6)]
    ghf = GhfTrial(mo_coeff=jnp.asarray(C))
    reference = np.array([float(overlap_u(_jnp_walker(wa, wb), ghf)) for wa, wb in walkers])
    no_det_tensors = [np.einsum("ps,asb->apb", no_det, A) for A in tensors]
    correct = _sd_overlaps(rotate_spin(tensors, R), walkers) / reference
    wrong = _sd_overlaps(no_det_tensors, walkers) / reference
    # both states are normalised, so the correct ratio is one constant of modulus 1
    np.testing.assert_allclose(np.abs(correct), 1.0, rtol=0, atol=1e-10)
    assert np.ptp(correct) < 1e-10
    assert np.ptp(wrong) > 1


@pytest.mark.parametrize("spin_map", list(SPIN_MAPS))
def test_rotated_trial_rdm1_is_the_ghf_block_diagonal(chain, spin_map):
    """The rotated trial's rdm1 is get_rdm1_block_diag of the rotated GhfTrial (unprojected)."""
    R = SPIN_MAPS[spin_map]
    Ca, Cb = chain.sd[NELEC]
    trial = make_mps_trial(rotate_spin(_sd_tensors(Ca, Cb), R), nelec=NELEC)
    want = get_rdm1_block_diag(GhfTrial(mo_coeff=jnp.asarray(_ghf_orbitals(R, Ca, Cb))))
    np.testing.assert_allclose(trial.rdm1, want, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# 6-9: walker plans, conversion and the overlap
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_plan", ["maximal", "rank_exact"])
def test_exact_plans_convert_walkers_exactly(chain, orbital_plan):
    """Exact plan, no truncation: prefactor * MPS(W) is SD(W) for non-orthonormal walkers W."""
    Ca, Cb = chain.sd[NELEC]
    plan = make_walker_plan_from_reference(
        Ca, Cb, orbital_plan=orbital_plan, walker_channel_chi=None
    )
    for wa, wb in _walkers(chain.h1, 8, seed=6, all_nonorthonormal=True):
        alpha, qa, beta, qb, prefactor = convert_walker(_jnp_walker(wa, wb), plan)
        tensors, charges = combine_channels(alpha, qa, beta, qb)
        got = float(prefactor) * hf.mps_sector_amplitudes(tensors, *NELEC)
        want = hf.sd_amplitudes(wa, wb)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-11 * np.abs(want).max())
        assert all(np.array_equal(x, y) for x, y in zip(charges, plan.walker_charges))
        _assert_labels_respected(tensors, charges, NELEC)


@pytest.mark.parametrize("name", TRIALS)
def test_overlap_matches_enumeration(lattice, name):
    """The jitted overlap is <T|SD(W)> by enumeration; vmap and vmap_chunked give the same."""
    trial = lattice.trials[name]
    overlap = make_mps_trial_ops(lattice.plans[name, "exact"]).overlap
    walkers = [_jnp_walker(wa, wb) for wa, wb in lattice.walkers]
    first = overlap(walkers[0], trial)
    assert first.dtype == jnp.float64 and first.shape == ()
    got = np.array([float(overlap(w, trial)) for w in walkers])
    own = hf.mps_sector_amplitudes(trial.tensors, *NELEC)
    for amplitudes in (own, lattice.oracles[name]):
        want = np.array([hf.overlap_with_sd(amplitudes, wa, wb) for wa, wb in lattice.walkers])
        _assert_close_up_to_constant(got, want, 1e-10)

    batch = tuple(jnp.stack(x) for x in zip(*walkers))
    vmapped = jax.vmap(overlap, in_axes=((0, 0), None))(batch, trial)
    chunked = vmap_chunked(overlap, 2, in_axes=(0, None))(batch, trial)
    np.testing.assert_allclose(vmapped, got, rtol=1e-14, atol=0)
    np.testing.assert_allclose(chunked, got, rtol=1e-14, atol=0)


@pytest.mark.parametrize("name", TRIALS)
def test_truncated_overlap_equals_dense_contraction(lattice, name):
    """chi=2 adaptive plan: the blocked overlap is prefactor * <T|MPS(W)> contracted densely."""
    trial = lattice.trials[name]
    plan = lattice.plans[name, "truncated"]
    overlap = make_mps_trial_ops(plan).overlap
    exact = make_mps_trial_ops(lattice.plans[name, "exact"]).overlap
    trial_vector = hf.mps_full_vector(trial.tensors)
    blocked, dense, untruncated = [], [], []
    for wa, wb in lattice.walkers:
        walker = _jnp_walker(wa, wb)
        alpha, qa, beta, qb, prefactor = convert_walker(walker, plan)
        tensors, _ = combine_channels(alpha, qa, beta, qb)
        dense.append(float(prefactor) * (hf.mps_full_vector(tensors) @ trial_vector))
        blocked.append(float(overlap(walker, trial)))
        untruncated.append(float(exact(walker, trial)))
    np.testing.assert_allclose(blocked, dense, rtol=1e-11, atol=0)
    # sanity: the truncation is real
    assert np.abs(np.array(blocked) / np.array(untruncated) - 1.0).max() > 1e-6


@pytest.mark.parametrize("kind", list(PLAN_SETTINGS))
def test_separable_gather_equals_extract_fixed_blocks(lattice, kind):
    """gather_blocks (eager and in-graph) is bitwise the host extract_fixed_blocks padding."""
    trial = lattice.trials["ed"]
    layout = contraction_layout(lattice.plans["ed", kind], trial.charges)
    want = extract_fixed_blocks([np.asarray(A) for A in trial.tensors], layout.contraction)
    eager = gather_blocks(trial.tensors, layout.gather)
    jitted = jax.jit(lambda tensors: gather_blocks(tensors, layout.gather))(trial.tensors)
    assert len(want) == len(eager) == len(jitted) == L
    for w, e, j in zip(want, eager, jitted):
        assert np.array_equal(e, w) and np.array_equal(j, w)


# ---------------------------------------------------------------------------------------------
# 10-12: validation, duck typing and the compress copy
# ---------------------------------------------------------------------------------------------


def test_validation_errors(chain, lattice):
    """Bad shapes, nelec mismatch, positional nelec, complex data and wrong trials all raise."""
    rng = np.random.default_rng(10)
    bad_shapes = {
        "4, D_right": [(1, 3, 2), (2, 3, 1)],  # physical dimension 3
        "disagree on the bond dimension": [(1, 4, 2), (3, 4, 1)],
        "boundary bonds": [(2, 4, 2), (2, 4, 1)],
    }
    for message, shapes in bad_shapes.items():
        with pytest.raises(ValueError, match=message):
            make_mps_trial([rng.standard_normal(s) for s in shapes], nelec=(1, 1))

    Ca, Cb = chain.sd[NELEC]
    gmps = sd_to_gmps(Ca, Cb, mode="maximal")
    tensors = [np.asarray(t) for t in gmps.tensors]
    trial = make_mps_trial(tensors, gmps.charges, nelec=NELEC)
    with pytest.raises(ValueError, match="trial has nelec"):
        as_mps_trial(trial, nelec=(2, 2))
    with pytest.raises(TypeError, match="keyword-only argument: 'nelec'"):
        make_mps_trial(tensors, NELEC)  # pyright: ignore[reportCallIssue]  # nelec is keyword-only
    with pytest.raises(TypeError, match="nelec by keyword"):
        make_mps_trial(tensors, NELEC, nelec=NELEC)  # an nelec-like pair passed as charges
    with pytest.raises(TypeError, match="complex tensors"):
        make_mps_trial([t.astype(complex) for t in tensors], nelec=NELEC)
    with pytest.raises(TypeError, match="rdm1 must be real"):
        make_mps_trial(
            tensors, gmps.charges, nelec=NELEC, rdm1=np.asarray(trial.rdm1).astype(complex)
        )

    plan = lattice.plans["sd", "exact"]
    overlap = make_mps_trial_ops(plan).overlap
    walker = _jnp_walker(*lattice.walkers[0])
    complex_walker = (walker[0].astype(jnp.complex128), walker[1])
    with pytest.raises(TypeError, match="real floating-point"):
        convert_walker(complex_walker, plan)
    with pytest.raises(TypeError, match="real floating-point"):
        overlap(complex_walker, lattice.trials["sd"])
    other = mps_trial_from_sd(*hf.staggered_determinant(lattice.h1, 3, 3))
    with pytest.raises(ValueError, match="does not match the walker plan"):
        overlap(walker, other)
    ghf = GhfTrial(mo_coeff=jnp.asarray(scipy.linalg.block_diag(*lattice.sd)))
    with pytest.raises(TypeError, match="must be an MpsTrial"):
        mps_overlap(walker, ghf, plan)  # pyright: ignore[reportArgumentType]


def test_as_mps_trial_accepts_gmps_from_a_second_module_copy(chain):
    """as_mps_trial duck-types Gmps: one from a bare second copy of gmps/utils.py is accepted."""
    spec = importlib.util.spec_from_file_location(
        "trot_gmps_utils_second_copy", gmps_utils.__file__
    )
    assert spec is not None
    second = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(second)
    Ca, Cb = chain.sd[NELEC]
    foreign = second.sd_to_gmps(Ca, Cb, mode="maximal")
    assert type(foreign) is not gmps_utils.Gmps and not isinstance(foreign, gmps_utils.Gmps)
    got = as_mps_trial(foreign)
    want = as_mps_trial(gmps_utils.sd_to_gmps(Ca, Cb, mode="maximal"))
    assert isinstance(got, MpsTrial) and got.nelec == NELEC
    assert got.charges == want.charges and len(got.tensors) == L
    assert all(np.array_equal(a, b) for a, b in zip(got.tensors, want.tensors))


def test_compress_mps_qn_is_bitwise_the_gpu_module_copy(lattice):
    """trot.trial.mps.compress_mps_qn is bitwise trot.gmps.mps_cpmc_gpu.compress_mps_qn."""
    gpu = pytest.importorskip("trot.gmps.mps_cpmc_gpu")
    trial = lattice.trials["rotated_sd"]
    W = hubbard_mpo_from_h1(lattice.h1, U)
    tensors, charges = trial_times_h(W, trial.tensors, trial.charge_arrays())
    rng = np.random.default_rng(12)
    # same labels and sparsity, generic singular values
    scrambled = [np.where(A != 0, rng.standard_normal(A.shape), 0.0) for A in tensors]
    for inputs in (tensors, scrambled):
        ours, our_charges = compress_mps_qn(inputs, charges)
        theirs, their_charges = gpu.compress_mps_qn(inputs, charges)
        assert len(ours) == len(theirs) == L and len(our_charges) == len(their_charges) == L + 1
        assert all(np.array_equal(a, b) for a, b in zip(ours, theirs))
        assert all(np.array_equal(a, b) for a, b in zip(our_charges, their_charges))


# ---------------------------------------------------------------------------------------------
# 13-14: pyblock3 optional / pyblock3 DMRG trials
# ---------------------------------------------------------------------------------------------


PYBLOCK3_FREE_SCRIPT = textwrap.dedent("""
    import importlib.abc
    import sys


    class BlockPyblock3(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "pyblock3" or name.startswith("pyblock3."):
                raise ImportError(f"pyblock3 is blocked in this test: {name}")
            return None


    sys.meta_path.insert(0, BlockPyblock3())
    try:
        import pyblock3
    except ImportError:
        pass
    else:
        raise SystemExit("the finder did not block pyblock3")

    from trot import config

    config.configure_once()

    import numpy as np

    import trot.gmps.driver
    import trot.gmps.utils
    import trot.ham.hubbard
    import trot.meas.mps
    import trot.prop.mps_cpmc
    import trot.trial.mps

    rng = np.random.default_rng(0)
    Ca = np.linalg.qr(rng.standard_normal((3, 2)))[0]
    Cb = np.linalg.qr(rng.standard_normal((3, 1)))[0]
    trial = trot.trial.mps.mps_trial_from_sd(Ca, Cb)
    assert trial.nelec == (2, 1), trial.nelec
    loaded = [m for m in sys.modules if m == "pyblock3" or m.startswith("pyblock3.")]
    assert "pyblock3" not in sys.modules and not loaded, loaded
    print("PYBLOCK3_FREE_OK")
    """)


def test_mps_stack_imports_and_builds_trials_without_pyblock3():
    """With pyblock3 unimportable the MPS-CPMC modules import and mps_trial_from_sd works."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    env.setdefault("JAX_PLATFORMS", "cpu")
    result = subprocess.run(
        [sys.executable, "-c", PYBLOCK3_FREE_SCRIPT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "PYBLOCK3_FREE_OK" in result.stdout


def test_dmrg_trial_densifies_identically_and_is_exact(lattice, dmrg_2x3):
    """All pyblock3 paths give one trial; chi=64 DMRG on 6 sites is the ED state (E, rdm1)."""
    dense = gmps_utils.densify(dmrg_2x3.mps)
    want = make_mps_trial(dense.tensors, dense.charges, nelec=NELEC)
    for trial in (mps_trial_from_pyblock3(dmrg_2x3.mps), dmrg_2x3.trial):
        assert trial.charges == want.charges and trial.sector_weight == want.sector_weight
        assert len(trial.tensors) == L
        assert all(np.array_equal(a, b) for a, b in zip(trial.tensors, want.tensors))
        assert np.array_equal(trial.rdm1, want.rdm1)
    assert abs(dmrg_2x3.variational_energy - lattice.E0) < 1e-9
    np.testing.assert_allclose(
        dmrg_2x3.trial.rdm1, hf.dense_rdm1(lattice.psi, L, *NELEC), rtol=0, atol=1e-7
    )
