"""Tests for trot/trial/mps.py: the MpsTrial pytree, trials from determinants, dense MPS and DMRG, the bond labels
make_mps_trial reads ((N_up, N_dn) pairs or particle numbers N; it never projects), spin rotations, walker plans,
the engine's walker conversion and the overlap of make_mps_trial_ops.

Every reference comes from tests/helpers/hubbard_fock.py (exact Fock-space enumeration), from the NumPy host oracle
of the conversion (trot.gmps.engine.channel_mps_host), from trot's GHF trial or from pyblock3, never from the MPS code
under test. Systems: an L=6 open chain (t=1, U=4) with nelec (3, 3) and (3, 2), and a 2x3 lattice periodic in x
(doubled rungs) with two on-site terms, U=4, (3, 2). The lattice trials are the ED ground state and a determinant
with (N_up, N_dn) labels, and the spin-rotated determinant used as it is (N labels) and projected onto (3, 2)
(rotate_mps_trial).
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
from trot.gmps import engine
from trot.gmps import utils as gmps_utils
from trot.gmps.utils import combine_channels, combined_charges, sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix, square_hopping_matrix
from trot.meas.mps import hubbard_mpo_from_h1, trial_times_h
from trot.prop.types import QmcParamsMps
from trot.trial.ghf import GhfTrial, get_rdm1_block_diag, overlap_u
from trot.trial.mps import (
    PHYSICAL_CHARGE,
    MpsTrial,
    _hashable_charges,
    as_mps_trial,
    check_trial,
    check_walker,
    compress_mps_qn,
    make_mps_trial,
    make_mps_trial_ops,
    make_walker_plan,
    make_walker_plan_from_reference,
    mps_overlap,
    mps_trial_from_pyblock3,
    mps_trial_from_sd,
    rotate_spin,
    spin_rotation_unitary,
)
from trot.trial.mps_rotation import rotate_mps_trial
from trot.walkers import _qr as qr_with_det
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
SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])  # up <-> down
SPIN_MAPS = {"rotation": ROTATION, "reflection": REFLECTION}
TRIALS = ("ed", "sd", "rotated_sd", "projected_sd")
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


def _make_trials(psi, Ca, Cb):
    """The lattice trials: the ED ground state and SD(Ca, Cb) with (N_up, N_dn) labels, and SD rotated by ROTATION
    used as it is (N labels, sector_weight None) and projected onto NELEC (rotate_mps_trial, sector_weight w)."""
    sd = mps_trial_from_sd(Ca, Cb)
    return {
        "ed": make_mps_trial(*hf.exact_mps_from_sector_state(psi, L, *NELEC), nelec=NELEC),
        "sd": sd,
        "rotated_sd": make_mps_trial(rotate_spin(_sd_tensors(Ca, Cb), ROTATION), nelec=NELEC),
        "projected_sd": rotate_mps_trial(sd, ROTATION),
    }


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


def _engine_walker_mps(walker, convert, labels):
    """An SD walker converted by a plan's engine: trot's QR, the compiled circuit (convert), channels interleaved.
    Returns the d=4 tensors and the prefactor det R_a det R_b g_a g_b."""
    (qa, det_a), (qb, det_b) = (qr_with_det(jnp.asarray(C)) for C in walker)
    alpha, beta, (gauge_a, gauge_b) = convert(qa, qb)
    tensors, _ = combine_channels(alpha, labels[0], beta, labels[1])
    return [np.asarray(t) for t in tensors], float(det_a * det_b * gauge_a * gauge_b)


def _host_walker_mps(walker, plan):
    """The same with the NumPy host oracle of the conversion (engine.channel_mps_host) on the plan's frozen circuit."""
    parts, prefactor = [], 1.0
    for C, orbital_plan, bond_plan in zip(walker, plan.orbital_plans, plan.bond_plans):
        Q, det_r = qr_with_det(jnp.asarray(C))
        tensors, charges, gauge = engine.channel_mps_host(np.asarray(Q), orbital_plan, bond_plan)
        parts += [tensors, charges]
        prefactor *= float(det_r) * gauge
    tensors, _ = combine_channels(*parts)
    return [np.asarray(t) for t in tensors], prefactor


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


def _assert_close(got, want, tol):
    np.testing.assert_allclose(got, want, rtol=0, atol=tol * np.abs(want).max())


def _assert_labels_respected(tensors, charges, nelec):
    """Bond labels from 0 to nelec that every nonzero entry respects: (N_up, N_dn) pairs, or particle numbers N
    (width 1; an index that no nonzero entry reaches is labelled -1, and the entries leaving it are not checked)."""
    Q = [np.asarray(q, int).reshape(len(q), -1) for q in charges]
    width = Q[0].shape[1]
    physical = PHYSICAL_CHARGE if width == 2 else PHYSICAL_CHARGE.sum(axis=1, keepdims=True)
    assert len(Q) == len(tensors) + 1
    np.testing.assert_array_equal(Q[0], [[0] * width])
    np.testing.assert_array_equal(Q[-1], [list(nelec) if width == 2 else [sum(nelec)]])
    for s, A in enumerate(tensors):
        A = np.asarray(A)
        assert A.shape[0] == len(Q[s]) and A.shape[2] == len(Q[s + 1])
        for p, delta in enumerate(physical):
            left, right = np.nonzero(A[:, p, :])
            reached = np.all(Q[s][left] >= 0, axis=1)
            np.testing.assert_array_equal(Q[s + 1][right[reached]], Q[s][left[reached]] + delta)


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
    """2x3 lattice, (3, 2): ED ground state, four trials with oracles, walkers, walker plans."""
    h1 = _lattice_h1()
    E0, psi = hf.ground_state(h1, U, *NELEC)
    Ca, Cb = hf.staggered_determinant(h1, *NELEC)
    trials = _make_trials(psi, Ca, Cb)
    rotated_ghf = hf.ghf_amplitudes(_ghf_orbitals(ROTATION, Ca, Cb), *NELEC)
    oracles = {  # independent alpha-block amplitudes, each proportional to its trial's
        "ed": psi,
        "sd": hf.sd_amplitudes(Ca, Cb),
        "rotated_sd": rotated_ghf,
        "projected_sd": rotated_ghf,
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


@pytest.mark.parametrize("name", TRIALS)
def test_mps_trial_is_a_pytree_with_static_labels(lattice, name):
    """MpsTrial flattens to L tensors + rdm1; charges, nelec and sector_weight are its hashable aux, kept by the
    round trip and static inside jit. Rebuilt from scratch it has the same tree structure, other trials not."""
    trial = lattice.trials[name]
    leaves, treedef = jax.tree_util.tree_flatten(trial)
    assert len(leaves) == L + 1
    assert all(a is b for a, b in zip(leaves, (*trial.tensors, trial.rdm1)))
    _, aux = trial.tree_flatten()
    assert aux == (trial.charges, trial.nelec, trial.sector_weight) and trial.nelec == NELEC
    hash(aux)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, MpsTrial)
    assert (rebuilt.charges, rebuilt.nelec, rebuilt.sector_weight) == aux
    assert rebuilt.label_width == trial.label_width and rebuilt.bond_dims == trial.bond_dims
    assert all(a is b for a, b in zip(jax.tree_util.tree_leaves(rebuilt), leaves))

    def first_sum(t):
        assert (t.charges, t.nelec, t.sector_weight) == aux  # aux stays Python data in jit
        return t.tensors[0].sum()

    np.testing.assert_allclose(
        jax.jit(first_sum)(trial), np.asarray(trial.tensors[0]).sum(), rtol=0, atol=1e-14
    )
    passed = jax.jit(lambda t: t)(trial)
    assert isinstance(passed, MpsTrial)
    assert (passed.charges, passed.nelec, passed.sector_weight) == aux

    assert jax.tree_util.tree_structure(trial) == treedef
    assert jax.tree_util.tree_structure(rebuilt) == treedef
    again = _make_trials(lattice.psi, *lattice.sd)[name]
    assert jax.tree_util.tree_structure(again) == treedef
    for other, other_trial in lattice.trials.items():
        if other != name:
            assert jax.tree_util.tree_structure(other_trial) != treedef


# ---------------------------------------------------------------------------------------------
# 2-6: trials from determinants, bond labels and spin rotations
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
    assert trial.nelec == nelec and trial.norb == L and trial.label_width == 2
    np.testing.assert_array_equal(PHYSICAL_CHARGE, [[p % 2, p // 2] for p in range(4)])
    _assert_labels_respected(trial.tensors, trial.charges, nelec)
    assert abs(trial.sector_weight - 1.0) < 1e-12


def test_make_mps_trial_reads_the_labels_off_the_tensors(lattice):
    """Without valid charges make_mps_trial reads the labels off the nonzero entries and projects nothing:
    (N_up, N_dn) labels for a sector state (sector_weight 1), particle-number labels for the spin-rotated
    determinant (used as it is, sector_weight None). Labels the tensors violate are ignored, not an error."""
    # (a) the exact MPS of the ED ground state, with and without its labels
    tensors, charges = hf.exact_mps_from_sector_state(lattice.psi, L, *NELEC)
    labelled = make_mps_trial(tensors, charges, nelec=NELEC)
    read = make_mps_trial(tensors, nelec=NELEC)
    assert read.charges == labelled.charges == _hashable_charges(charges)
    assert read.charges == _hashable_charges(gmps_utils.sz_labels(tensors))
    assert read.label_width == 2 and read.sector_weight == 1.0 and read.nelec == NELEC
    assert all(np.array_equal(a, b) for a, b in zip(read.tensors, labelled.tensors))
    _assert_labels_respected(read.tensors, read.charges, NELEC)

    # (b) the spin-rotated determinant has no definite (N_up, N_dn): particle-number labels, bonds unchanged
    gmps = sd_to_gmps(*lattice.sd, mode="maximal")
    rotated = rotate_spin([np.asarray(t) for t in gmps.tensors], ROTATION)
    assert gmps_utils.sz_labels(rotated) is None
    as_is = make_mps_trial(rotated, nelec=NELEC)
    stale = make_mps_trial(rotated, gmps.charges, nelec=NELEC)  # the unrotated labels: violated, ignored
    assert as_is.label_width == 1 and as_is.sector_weight is None and as_is.nelec == NELEC
    assert stale.charges == as_is.charges == _hashable_charges(gmps_utils.number_labels(rotated))
    assert stale.sector_weight is None
    assert all(np.array_equal(a, b) for a, b in zip(stale.tensors, as_is.tensors))
    assert as_is.bond_dims == tuple(len(q) for q in gmps.charges)
    for q, q_sz in zip(as_is.charge_arrays(), gmps.charges):  # reached indices keep N = N_up + N_dn
        reached = q[:, 0] >= 0
        np.testing.assert_array_equal(q[reached, 0], np.asarray(q_sz).sum(axis=1)[reached])
    _assert_labels_respected(as_is.tensors, as_is.charges, NELEC)
    vector = hf.mps_full_vector(rotated)
    _assert_close(hf.mps_full_vector(as_is.tensors), vector / np.linalg.norm(vector), 1e-13)


def test_make_mps_trial_rejects_trials_the_walkers_cannot_see(lattice):
    """(N_up, N_dn) labels that end at another sector, particle-number labels that end at another N, and an MPS
    that mixes particle numbers (no labels to block on) are errors; the mirrored sector is accepted."""
    tensors, _ = hf.exact_mps_from_sector_state(lattice.psi, L, *NELEC)
    swapped = rotate_spin(tensors, SWAP)  # the (3, 2) ground state moved whole into (2, 3)
    with pytest.raises(ValueError, match=r"\(N_up, N_dn\) = \(2, 3\) and the walkers \(3, 2\)"):
        make_mps_trial(swapped, nelec=NELEC)
    mirrored = make_mps_trial(swapped, nelec=(2, 3))
    assert mirrored.label_width == 2 and mirrored.sector_weight == 1.0 and mirrored.nelec == (2, 3)
    _assert_labels_respected(mirrored.tensors, mirrored.charges, (2, 3))
    vector = hf.mps_full_vector(swapped)
    np.testing.assert_allclose(np.sum(hf.sector_from_full(vector, L, 2, 3) ** 2), vector @ vector, rtol=1e-12)

    rotated = rotate_spin(_sd_tensors(*lattice.sd), ROTATION)
    with pytest.raises(ValueError, match="the trial has N = 5 and the walkers 6"):
        make_mps_trial(rotated, nelec=(3, 3))

    rng = np.random.default_rng(7)
    bonds = [1, 4, 8, 8, 4, 1]
    mixed = [rng.standard_normal((bonds[i], 4, bonds[i + 1])) for i in range(5)]
    with pytest.raises(ValueError, match="mixes particle numbers"):
        make_mps_trial(mixed, nelec=(2, 2))


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
    """The rotated trial's rdm1, as it is and projected (rotate_mps_trial), is get_rdm1_block_diag of the rotated
    GhfTrial (the unprojected state's), so both start the walkers where the GHF trial does."""
    R = SPIN_MAPS[spin_map]
    Ca, Cb = chain.sd[NELEC]
    want = get_rdm1_block_diag(GhfTrial(mo_coeff=jnp.asarray(_ghf_orbitals(R, Ca, Cb))))
    as_is = make_mps_trial(rotate_spin(_sd_tensors(Ca, Cb), R), nelec=NELEC)
    projected = rotate_mps_trial(mps_trial_from_sd(Ca, Cb), R)
    for trial in (as_is, projected):
        np.testing.assert_allclose(trial.rdm1, want, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# 7-12: walker plans, the engine's conversion and the overlap
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("orbital_plan", ["maximal", "rank_exact"])
def test_exact_plans_convert_walkers_exactly(chain, orbital_plan):
    """Exact plan, no truncation: the engine's conversion (engine.converter_for) of the QR'd walker times
    det R_a det R_b g_a g_b is SD(W) by enumeration, for non-orthonormal walkers W; its labels are the plan's."""
    Ca, Cb = chain.sd[NELEC]
    plan = make_walker_plan_from_reference(
        Ca, Cb, orbital_plan=orbital_plan, walker_channel_chi=None
    )
    converter = engine.converter_for(plan)
    convert = jax.jit(converter.convert)
    labels = combined_charges(*converter.charges)
    assert len(labels) == len(plan.walker_charges) == L + 1
    assert all(np.array_equal(x, y) for x, y in zip(labels, plan.walker_charges))
    for wa, wb in _walkers(chain.h1, 8, seed=6, all_nonorthonormal=True):
        tensors, prefactor = _engine_walker_mps((wa, wb), convert, converter.charges)
        got = prefactor * hf.mps_sector_amplitudes(tensors, *NELEC)
        _assert_close(got, hf.sd_amplitudes(wa, wb), 1e-11)
        _assert_labels_respected(tensors, labels, NELEC)


@pytest.mark.parametrize("name", TRIALS)
def test_overlap_matches_enumeration(lattice, name):
    """The jitted overlap (exact walker plan) is <T|SD(W)> by enumeration, the trial's NELEC amplitudes against
    det(Wa[A]) det(Wb[B]), for (N_up, N_dn) and N labels alike, and proportional to the independent oracle's;
    vmap and vmap_chunked give the same."""
    trial = lattice.trials[name]
    overlap = make_mps_trial_ops(lattice.plans[name, "exact"]).overlap
    walkers = [_jnp_walker(wa, wb) for wa, wb in lattice.walkers]
    first = overlap(walkers[0], trial)
    assert first.dtype == jnp.float64 and first.shape == ()
    got = np.array([float(overlap(w, trial)) for w in walkers])
    _assert_close(got, _sd_overlaps(trial.tensors, lattice.walkers), 1e-10)
    oracle = np.array([hf.overlap_with_sd(lattice.oracles[name], wa, wb) for wa, wb in lattice.walkers])
    _assert_close_up_to_constant(got, oracle, 1e-10)

    batch = tuple(jnp.stack(x) for x in zip(*walkers))
    vmapped = jax.vmap(overlap, in_axes=((0, 0), None))(batch, trial)
    chunked = vmap_chunked(overlap, 2, in_axes=(0, None))(batch, trial)
    np.testing.assert_allclose(vmapped, got, rtol=1e-14, atol=0)
    np.testing.assert_allclose(chunked, got, rtol=1e-14, atol=0)


@pytest.mark.parametrize("name", TRIALS)
def test_truncated_overlap_equals_the_host_conversion(lattice, name):
    """chi=2 adaptive plan: the overlap is prefactor * <T|MPS(W)>, with MPS(W) converted by the NumPy host oracle
    (engine.channel_mps_host) on the plan's frozen circuit and contracted densely."""
    trial = lattice.trials[name]
    plan = lattice.plans[name, "truncated"]
    overlap = make_mps_trial_ops(plan).overlap
    exact = make_mps_trial_ops(lattice.plans[name, "exact"]).overlap
    trial_vector = hf.mps_full_vector(trial.tensors)
    blocked, dense, untruncated = [], [], []
    for wa, wb in lattice.walkers:
        walker = _jnp_walker(wa, wb)
        tensors, prefactor = _host_walker_mps((wa, wb), plan)
        dense.append(prefactor * (hf.mps_full_vector(tensors) @ trial_vector))
        blocked.append(float(overlap(walker, trial)))
        untruncated.append(float(exact(walker, trial)))
    np.testing.assert_allclose(blocked, dense, rtol=1e-10, atol=0)
    # sanity: the truncation is real
    assert np.abs(np.array(blocked) / np.array(untruncated) - 1.0).max() > 1e-6


@pytest.mark.parametrize("kind", list(PLAN_SETTINGS))
def test_rotated_trial_overlaps_are_sqrt_w_times_the_projected(lattice, kind):
    """Walkers are S_z eigenstates: on one walker plan the rotated trial used as it is (N labels) has overlaps
    sqrt(w) times those of the projected trial (rotate_mps_trial, sector_weight w), exact or truncated walkers;
    w is the as-is trial's NELEC weight by enumeration."""
    as_is, projected = lattice.trials["rotated_sd"], lattice.trials["projected_sd"]
    weight = projected.sector_weight
    np.testing.assert_allclose(
        weight, np.sum(hf.mps_sector_amplitudes(as_is.tensors, *NELEC) ** 2), rtol=1e-12
    )
    assert 1e-3 < weight < 0.9, "the rotation must move weight between sectors"
    overlap = make_mps_trial_ops(lattice.plans["projected_sd", kind]).overlap
    walkers = [_jnp_walker(wa, wb) for wa, wb in lattice.walkers]
    got = np.array([float(overlap(w, as_is)) for w in walkers])
    want = np.sqrt(weight) * np.array([float(overlap(w, projected)) for w in walkers])
    _assert_close(got, want, 1e-10)


@pytest.mark.parametrize("name", ["ed", "rotated_sd"])
@pytest.mark.parametrize("kind", list(PLAN_SETTINGS))
def test_trial_blocks_gathered_in_graph_equal_the_host_padding(lattice, kind, name):
    """engine.fixed_blocks of the trial tensors (eager and in-graph, as mps_overlap gathers them) is bitwise the host
    padding (xp=np) of the plan's layout, for (N_up, N_dn) and N labels."""
    trial = lattice.trials[name]
    layout = engine.layout_for(lattice.plans[name, kind], trial.charges)
    want = engine.fixed_blocks([np.asarray(A) for A in trial.tensors], layout, xp=np)
    eager = engine.fixed_blocks(trial.tensors, layout)
    jitted = jax.jit(lambda tensors: engine.fixed_blocks(tensors, layout))(trial.tensors)
    assert len(want) == len(eager) == len(jitted) == L
    for w, e, j in zip(want, eager, jitted):
        assert np.array_equal(np.asarray(e), w) and np.array_equal(np.asarray(j), w)


def test_walker_plan_carries_the_engine_settings(lattice, monkeypatch):
    """make_walker_plan takes sector_buckets and walker_qr from QmcParamsMps (defaults (8, 16), "auto") for every
    trial, and the plan's engine (engine.kernels_for, cached on the plan) is built with them; the buckets change the
    padding only, so the overlaps are those of the default plan."""
    ham = HamHubbard(h1=jnp.asarray(lattice.h1), u=U)
    sys_ = System(norb=L, nelec=NELEC, walker_kind="unrestricted")
    for trial in lattice.trials.values():
        default = make_walker_plan(ham, trial, sys_, QmcParamsMps(seed=0))
        assert default.sector_buckets == (8, 16) and default.walker_qr == "auto"
        assert default.nelec == NELEC and default.norb == L
    trial = lattice.trials["rotated_sd"]
    params = QmcParamsMps(seed=0, sector_buckets=(2, 4), walker_qr="native", **PLAN_SETTINGS["exact"])

    seen = {}
    make_converter, make_kernels = engine.make_converter, engine.make_kernels

    def converter_spy(*args, **kwargs):
        seen["buckets"] = kwargs["buckets"]
        return make_converter(*args, **kwargs)

    def kernels_spy(*args, **kwargs):
        seen["walker_qr"] = args[4]
        return make_kernels(*args, **kwargs)

    monkeypatch.setattr(engine, "make_converter", converter_spy)
    monkeypatch.setattr(engine, "make_kernels", kernels_spy)
    plan = make_walker_plan(ham, trial, sys_, params)  # compiles the conversion circuit with the buckets
    assert plan.sector_buckets == (2, 4) and plan.walker_qr == "native"
    assert seen == {"buckets": (2, 4)}
    kernels = engine.kernels_for(plan, trial.charges, energy=None)
    assert seen == {"buckets": (2, 4), "walker_qr": "native"}
    assert engine.kernels_for(plan, trial.charges, energy=None) is kernels

    overlap = make_mps_trial_ops(plan).overlap
    reference = make_mps_trial_ops(lattice.plans["rotated_sd", "exact"]).overlap
    walkers = [_jnp_walker(wa, wb) for wa, wb in lattice.walkers]
    want = np.array([float(reference(w, trial)) for w in walkers])
    _assert_close(np.array([float(overlap(w, trial)) for w in walkers]), want, 1e-11)


# ---------------------------------------------------------------------------------------------
# 13-15: validation, duck typing and the per-sector compression
# ---------------------------------------------------------------------------------------------


def test_validation_errors(chain, lattice):
    """Bad shapes and sectors, nelec mismatch, positional nelec, complex data, zero norm, bad rdm1, wrong trials
    and complex walkers all raise."""
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
    rdm1 = np.asarray(trial.rdm1)
    with pytest.raises(ValueError, match="trial has nelec"):
        as_mps_trial(trial, nelec=(2, 2))
    with pytest.raises(ValueError, match="is not a sector"):
        make_mps_trial(tensors, nelec=(L + 1, 0))
    with pytest.raises(TypeError, match="keyword-only argument: 'nelec'"):
        make_mps_trial(tensors, NELEC)  # pyright: ignore[reportCallIssue]  # nelec is keyword-only
    with pytest.raises(TypeError, match="nelec by keyword"):
        make_mps_trial(tensors, NELEC, nelec=NELEC)  # an nelec-like pair passed as charges
    with pytest.raises(TypeError, match="complex tensors"):
        make_mps_trial([t.astype(complex) for t in tensors], nelec=NELEC)
    with pytest.raises(TypeError, match="rdm1 must be real"):
        make_mps_trial(tensors, gmps.charges, nelec=NELEC, rdm1=rdm1.astype(complex))
    with pytest.raises(ValueError, match="rdm1 must have shape"):
        make_mps_trial(tensors, gmps.charges, nelec=NELEC, rdm1=rdm1[0])
    with pytest.raises(ValueError, match="zero norm"):
        make_mps_trial([np.zeros_like(t) for t in tensors], nelec=NELEC, rdm1=rdm1)

    ham = HamHubbard(h1=jnp.asarray(lattice.h1), u=U)
    sys_ = System(norb=L, nelec=NELEC, walker_kind="unrestricted")
    plan = lattice.plans["sd", "exact"]
    overlap = make_mps_trial_ops(plan).overlap
    walker = _jnp_walker(*lattice.walkers[0])
    complex_walker = (walker[0].astype(jnp.complex128), walker[1])
    # mps_overlap no longer validates (the checks were commented out for speed); the checks it used are
    # check_walker / check_trial, which the measurement kernels and the step still call
    with pytest.raises(TypeError, match="real floating-point"):
        check_walker(complex_walker)
    check_walker(walker)
    other = mps_trial_from_sd(*hf.staggered_determinant(lattice.h1, 3, 3))
    with pytest.raises(ValueError, match="does not match the walker plan"):
        check_trial(other, plan)
    with pytest.raises(ValueError, match="does not match the system"):
        make_walker_plan(ham, other, sys_, QmcParamsMps(seed=0))
    ghf = GhfTrial(mo_coeff=jnp.asarray(scipy.linalg.block_diag(*lattice.sd)))
    with pytest.raises(TypeError, match="must be an MpsTrial"):
        check_trial(ghf, plan)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="must be an MpsTrial"):
        make_walker_plan(ham, ghf, sys_, QmcParamsMps(seed=0))  # pyright: ignore[reportArgumentType]


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


@pytest.mark.parametrize("name", ["projected_sd", "rotated_sd"])
def test_compress_mps_qn_keeps_the_state_and_the_labels(lattice, name):
    """compress_mps_qn factors each charge sector on its own: H|trial> (trial_times_h, (N_up, N_dn) or N labels)
    and a generic MPS with the same labels and sparsity come back as the same state by enumeration, with labels
    the tensors respect and smaller bonds."""
    trial = lattice.trials[name]
    W = hubbard_mpo_from_h1(lattice.h1, U)
    tensors, charges = trial_times_h(W, [np.asarray(A) for A in trial.tensors], trial.charge_arrays())
    rng = np.random.default_rng(12)
    scrambled = [np.where(A != 0, rng.standard_normal(A.shape), 0.0) for A in tensors]
    for inputs in (tensors, scrambled):
        out, out_charges = compress_mps_qn(inputs, charges)
        assert len(out) == L and len(out_charges) == L + 1
        assert np.shape(out_charges[0])[1] == trial.label_width
        _assert_close(hf.mps_full_vector(out), hf.mps_full_vector(inputs), 1e-11)
        _assert_labels_respected(out, out_charges, NELEC)
        assert sum(map(len, out_charges)) < sum(map(len, charges))


# ---------------------------------------------------------------------------------------------
# 16-17: pyblock3 optional / pyblock3 DMRG trials
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
    import trot.gmps.engine
    import trot.gmps.trials
    import trot.gmps.utils
    import trot.ham.hubbard
    import trot.meas.mps
    import trot.prop.mps_cpmc
    import trot.trial.mps
    import trot.trial.mps_rotation

    rng = np.random.default_rng(0)
    Ca = np.linalg.qr(rng.standard_normal((3, 2)))[0]
    Cb = np.linalg.qr(rng.standard_normal((3, 1)))[0]
    trial = trot.trial.mps.mps_trial_from_sd(Ca, Cb)
    assert trial.nelec == (2, 1) and trial.label_width == 2, trial.nelec
    rotated = trot.trial.mps_rotation.rotate_mps_trial(trial, trot.trial.mps.spin_rotation_y(90.0))
    assert rotated.label_width == 2 and 0.0 < rotated.sector_weight <= 1.0, rotated.sector_weight
    loaded = [m for m in sys.modules if m == "pyblock3" or m.startswith("pyblock3.")]
    assert "pyblock3" not in sys.modules and not loaded, loaded
    print("PYBLOCK3_FREE_OK")
    """)


def test_mps_stack_imports_and_builds_trials_without_pyblock3():
    """With pyblock3 unimportable the MPS-CPMC modules import and SD trials are built and rotated."""
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
    assert want.label_width == 2 and want.sector_weight == 1.0
    for trial in (mps_trial_from_pyblock3(dmrg_2x3.mps), dmrg_2x3.trial):
        assert trial.charges == want.charges and trial.sector_weight == want.sector_weight
        assert trial.nelec == NELEC and len(trial.tensors) == L
        assert all(np.array_equal(a, b) for a, b in zip(trial.tensors, want.tensors))
        assert np.array_equal(trial.rdm1, want.rdm1)
    assert abs(dmrg_2x3.variational_energy - lattice.E0) < 1e-9
    np.testing.assert_allclose(
        dmrg_2x3.trial.rdm1, hf.dense_rdm1(lattice.psi, L, *NELEC), rtol=0, atol=1e-7
    )
