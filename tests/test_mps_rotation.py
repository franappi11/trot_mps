"""Spin rotations of MPS trials as an S_z-conserving MPO (trot.trial.mps_rotation).

The MPO must equal the projected product of site rotations. Applied to a labelled MPS, it must give
the state of rotate_spin + project_to_sector, the rotated GHF determinant, and the exact
transformation of spin multiplets. The capstone runs CPMC with it, identical to the rotated GHF trial.
"""

from trot import config

config.configure_once()

from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from tests.helpers import hubbard_fock as hf
from tests.test_mps_cpmc import (
    _assert_identical_runs,
    _assert_identical_states,
    _blocks_twin,
    _mps_side,
    _params,
    _run,
    _sd_side,
    _system,
)
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.trial import mps as trial_mps
from trot.trial.ghf import GhfTrial, get_rdm1_block_diag
from trot.trial.mps import (
    PHYSICAL_CHARGE,
    _hashable_charges,
    _valid_sz_labels,
    make_mps_trial,
    mps_trial_from_sd,
    one_rdm,
    project_to_sector,
    rotate_spin,
    spin_rotation_unitary,
)
from trot.trial.mps_rotation import (
    apply_rotation_mpo,
    rotate_mps_trial,
    rotate_spin_mpo,
    rotated_rdm1,
    spin_rotation_mpo,
    spin_rotation_pieces,
)

U = 4.0
L = 6
_C, _S = np.cos(0.7), np.sin(0.7)
ROTATIONS = {
    "rotation": np.array([[_C, -_S], [_S, _C]]),
    "reflection": np.array([[_C, _S], [_S, -_C]]),  # det = -1
    "swap": np.array([[0.0, 1.0], [1.0, 0.0]]),  # up <-> down, det = -1
    "identity": np.eye(2),
}


# ---------------------------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------------------------


def _site_charges(n_sites):
    """(N_up, N_dn) of every index of the interleaved 4^L vector, site 0 most significant."""
    digits = (np.arange(4**n_sites)[:, None] // 4 ** np.arange(n_sites - 1, -1, -1)) % 4
    return PHYSICAL_CHARGE[digits].sum(axis=1)


def _dense_mpo(W):
    """(out, in) matrix of an MPO with boundary bonds of dimension 1, site 0 most significant."""
    op = np.ones((1, 1, 1))  # (out, in, bond)
    for w in W:
        op = np.einsum("xyk,kpqm->xpyqm", op, w)
        x, p, y, q, m = op.shape
        op = op.reshape(x * p, y * q, m)
    return op[:, :, 0]


def _rotated_vector(vector, R, target):
    """P_target (M x ... x M) vector, with M applied on each site of the dense 4^L vector."""
    M = spin_rotation_unitary(R)
    v = vector.reshape((4,) * L)
    for s in range(L):
        v = np.moveaxis(np.tensordot(M, v, axes=(1, s)), 0, s)
    v = v.reshape(-1).copy()
    v[np.any(_site_charges(L) != np.asarray(target), axis=1)] = 0.0
    return v


def _uhf(nelec):
    return hf.staggered_determinant(hopping_matrix(L, 1.0), *nelec)


def _input_mps(kind, nelec):
    """A labelled MPS: the staggered UHF determinant, the ED ground state, or a random sector state."""
    if kind == "uhf":
        gmps = sd_to_gmps(*_uhf(nelec), mode="maximal")
        return [np.asarray(t) for t in gmps.tensors], tuple(np.asarray(q) for q in gmps.charges)
    if kind == "ground":
        _, psi = hf.ground_state(hopping_matrix(L, 1.0), U, *nelec)
    else:
        shape = (len(hf.sector_basis(L, nelec[0])[0]), len(hf.sector_basis(L, nelec[1])[0]))
        psi = np.random.default_rng(7).normal(size=shape)
        psi /= np.linalg.norm(psi)
    return hf.exact_mps_from_sector_state(psi, L, *nelec)


def _assert_close_vectors(got, want, tol=1e-12):
    np.testing.assert_allclose(got, want, rtol=0, atol=tol * np.abs(want).max())


# ---------------------------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROTATIONS)
def test_pieces_sum_to_the_site_rotation_and_move_sz_by_one(name):
    R = ROTATIONS[name]
    pieces = spin_rotation_pieces(R)
    assert sorted(pieces) == [-1, 0, 1]
    assert np.array_equal(pieces[0] + pieces[1] + pieces[-1], spin_rotation_unitary(R))
    assert np.array_equal(np.diag(pieces[0]), [1.0, R[0, 0], R[1, 1], np.linalg.det(R)])
    assert np.array_equal(pieces[0], np.diag(np.diag(pieces[0])))
    assert pieces[1][1, 2] == R[0, 1] and np.count_nonzero(pieces[1]) <= 1
    assert pieces[-1][2, 1] == R[1, 0] and np.count_nonzero(pieces[-1]) <= 1
    for d, piece in pieces.items():
        for p, q in np.argwhere(piece != 0):
            assert tuple(PHYSICAL_CHARGE[p] - PHYSICAL_CHARGE[q]) == (d, -d)


@pytest.mark.parametrize("n_sites", [1, 2, 3, 4])
@pytest.mark.parametrize("name", ROTATIONS)
def test_mpo_equals_the_projected_product_of_site_rotations(name, n_sites):
    """Every shift: the contracted MPO is (M x ... x M) restricted to dN_up = shift, dN = 0."""
    R = ROTATIONS[name]
    product = np.ones((1, 1))
    for _ in range(n_sites):
        product = np.kron(product, spin_rotation_unitary(R))
    q = _site_charges(n_sites)
    same_n = q.sum(axis=1)[:, None] == q.sum(axis=1)[None, :]
    for shift in range(-n_sites, n_sites + 1):
        W, ks = spin_rotation_mpo(R, n_sites, shift)
        mask = same_n & (q[:, None, 0] - q[None, :, 0] == shift)
        np.testing.assert_allclose(_dense_mpo(W), np.where(mask, product, 0.0), rtol=0, atol=1e-15)


@pytest.mark.parametrize("shift", [-3, -1, 0, 2, 6])
def test_mpo_bonds_count_the_down_to_up_moves(shift):
    W, ks = spin_rotation_mpo(ROTATIONS["rotation"], L, shift)
    assert len(W) == L and len(ks) == L + 1
    assert list(ks[0]) == [0] and list(ks[L]) == [shift]
    for c, k in enumerate(ks):
        assert np.array_equal(k, np.arange(max(-c, shift - (L - c)), min(c, shift + (L - c)) + 1))
        if shift == 0:
            assert len(k) == 2 * min(c, L - c) + 1
    for s, w in enumerate(W):
        assert w.shape == (len(ks[s]), 4, 4, len(ks[s + 1]))
        step = ks[s + 1][None, :] - ks[s][:, None]
        jumps = np.moveaxis(w, 3, 1)[np.abs(step) > 1]  # (k, k', out, in) blocks with |k' - k| > 1
        assert not np.any(jumps), "a site moves at most one electron"
    with pytest.raises(ValueError, match="shift"):
        spin_rotation_mpo(ROTATIONS["rotation"], 3, 4)


# ---------------------------------------------------------------------------------------------
# Applied to labelled MPS
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rotation", "reflection"])
@pytest.mark.parametrize(
    "kind, nelec", [("uhf", (3, 3)), ("uhf", (3, 2)), ("ground", (3, 2)), ("random", (3, 2))]
)
def test_raw_application_is_labelled_and_lives_in_the_target_sector(kind, nelec, name):
    R = ROTATIONS[name]
    tensors, charges = _input_mps(kind, nelec)
    vector = hf.mps_full_vector(tensors)
    flipped = (nelec[0] + 1, nelec[1] - 1) if nelec == (3, 3) else (nelec[1], nelec[0])
    for target in (nelec, flipped):
        W, ks = spin_rotation_mpo(R, L, target[0] - nelec[0])
        raw, labels = apply_rotation_mpo(W, ks, tensors, charges)
        assert _valid_sz_labels(raw, labels, target)
        assert [t.shape[0] for t in raw] == [len(k) * len(q) for k, q in zip(ks, charges)][:-1]
        got = hf.mps_full_vector(raw)
        _assert_close_vectors(got, _rotated_vector(vector, R, target))
        outside = np.any(_site_charges(L) != np.asarray(target), axis=1)
        assert np.all(got[outside] == 0.0)


@pytest.mark.parametrize("name", ["rotation", "reflection"])
@pytest.mark.parametrize(
    "nelec, target", [((3, 3), (3, 3)), ((3, 3), (4, 2)), ((3, 2), (3, 2)), ((3, 2), (2, 3))]
)
def test_mpo_path_gives_the_state_of_rotate_then_project(nelec, target, name):
    R = ROTATIONS[name]
    tensors, charges = _input_mps("uhf", nelec)
    got, got_charges, got_norm2 = rotate_spin_mpo(tensors, charges, R, nelec=target)
    want, want_charges, want_norm2 = project_to_sector(rotate_spin(tensors, R), target)
    assert _valid_sz_labels(got, got_charges, target)
    _assert_close_vectors(hf.mps_full_vector(got), hf.mps_full_vector(want))
    np.testing.assert_allclose(got_norm2, want_norm2, rtol=1e-12)
    assert 1e-3 < got_norm2 < 0.9, "the rotation must move weight between sectors"
    assert [len(q) for q in got_charges] == [len(q) for q in want_charges]


@pytest.mark.parametrize("name", ["rotation", "reflection"])
@pytest.mark.parametrize("nelec, target", [((3, 3), (3, 3)), ((3, 2), (3, 2)), ((3, 2), (2, 3))])
def test_mpo_rotation_matches_the_rotated_ghf_determinant(nelec, target, name):
    R = ROTATIONS[name]
    Ca, Cb = _uhf(nelec)
    tensors, charges = _input_mps("uhf", nelec)
    rotated, _, _ = rotate_spin_mpo(tensors, charges, R, nelec=target)
    got = hf.mps_sector_amplitudes(rotated, *target)
    C_ghf = np.kron(R, np.eye(L)) @ cast(np.ndarray, scipy.linalg.block_diag(Ca, Cb))
    want = hf.ghf_amplitudes(C_ghf, *target)
    scale = np.vdot(want, got) / np.vdot(want, want)
    np.testing.assert_allclose(got, scale * want, rtol=0, atol=1e-12 * np.abs(got).max())


@pytest.mark.parametrize("name, sign", [("rotation", 1.0), ("reflection", -1.0)])
def test_rotations_keep_a_singlet_and_reflections_flip_its_sign(name, sign):
    """U(R) = (det R)^(N/2) on S = 0 (here N = 6), independent of any GHF code."""
    tensors, charges = _input_mps("ground", (3, 3))
    rotated, _, norm2 = rotate_spin_mpo(tensors, charges, ROTATIONS[name])
    np.testing.assert_allclose(norm2, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        hf.mps_full_vector(rotated), sign * hf.mps_full_vector(tensors), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("name", ["rotation", "reflection"])
def test_a_doublet_transforms_with_r(name):
    """S = 1/2 ground state of (3, 2): <S_z = 1/2|U(R)|S_z = 1/2> = R00, and R10 into S_z = -1/2."""
    R = ROTATIONS[name]
    tensors, charges = _input_mps("ground", (3, 2))
    same, _, norm2 = rotate_spin_mpo(tensors, charges, R)
    np.testing.assert_allclose(norm2, R[0, 0] ** 2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        hf.mps_full_vector(same), R[0, 0] * hf.mps_full_vector(tensors), rtol=0, atol=1e-12
    )
    _, _, norm2_flipped = rotate_spin_mpo(tensors, charges, R, nelec=(2, 3))
    np.testing.assert_allclose(norm2_flipped, R[1, 0] ** 2, rtol=0, atol=1e-12)


def test_the_swap_moves_the_whole_state_to_the_mirrored_sector():
    R = ROTATIONS["swap"]
    tensors, charges = _input_mps("uhf", (3, 2))
    with pytest.raises(ValueError, match="no weight"):
        rotate_spin_mpo(tensors, charges, R)
    with pytest.raises(ValueError, match="no weight"):
        project_to_sector(rotate_spin(tensors, R), (3, 2))  # the projection path agrees
    rotated, charges_out, norm2 = rotate_spin_mpo(tensors, charges, R, nelec=(2, 3))
    vector = hf.mps_full_vector(tensors)
    np.testing.assert_allclose(norm2, vector @ vector, rtol=1e-12)
    assert _valid_sz_labels(rotated, charges_out, (2, 3))
    _assert_close_vectors(hf.mps_full_vector(rotated), _rotated_vector(vector, R, (2, 3)))


def test_rotate_spin_mpo_validates_its_input():
    R = ROTATIONS["rotation"]
    tensors, charges = _input_mps("uhf", (3, 2))
    with pytest.raises(ValueError, match="bond labels"):
        rotate_spin_mpo(tensors, None, R)
    with pytest.raises(ValueError, match="bond labels"):
        rotate_spin_mpo(rotate_spin(tensors, R), charges, R)  # rotated tensors break the labels
    with pytest.raises(ValueError, match="keeps N"):
        rotate_spin_mpo(tensors, charges, R, nelec=(3, 3))
    with pytest.raises(ValueError, match="not a sector"):
        rotate_spin_mpo(tensors, charges, R, nelec=(7, -2))
    with pytest.raises(ValueError, match="orthogonal"):
        rotate_spin_mpo(tensors, charges, np.array([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(TypeError, match="real"):
        rotate_spin_mpo([t.astype(complex) for t in tensors], charges, R)


# ---------------------------------------------------------------------------------------------
# The trial
# ---------------------------------------------------------------------------------------------


def test_make_mps_trial_uses_the_mpo_output_without_projecting(monkeypatch):
    tensors, charges = _input_mps("uhf", (3, 2))
    rotated, labels, norm2 = rotate_spin_mpo(tensors, charges, ROTATIONS["rotation"])

    def no_compression(*args, **kwargs):
        raise AssertionError("make_mps_trial projected the MPO output again")

    monkeypatch.setattr(trial_mps, "compress_mps_qn", no_compression)
    trial = make_mps_trial(rotated, labels, nelec=(3, 2))
    assert np.array_equal(np.asarray(trial.tensors[0]), rotated[0] / np.sqrt(norm2))
    for a, b in zip(trial.tensors[1:], rotated[1:]):
        assert np.array_equal(np.asarray(a), b)
    assert trial.charges == _hashable_charges(labels)
    assert trial.sector_weight == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", ["rotation", "reflection"])
@pytest.mark.parametrize("nelec", [(3, 3), (3, 2)])
def test_rotated_rdm1_is_the_ghf_walker_start(nelec, name):
    R = ROTATIONS[name]
    Ca, Cb = _uhf(nelec)
    tensors, _ = _input_mps("uhf", nelec)
    got = rotated_rdm1(np.stack(one_rdm(tensors)), R)
    C_ghf = np.kron(R, np.eye(L)) @ cast(np.ndarray, scipy.linalg.block_diag(Ca, Cb))
    ghf = np.asarray(get_rdm1_block_diag(GhfTrial(mo_coeff=jnp.asarray(C_ghf))))
    np.testing.assert_allclose(got, ghf, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got, np.stack(one_rdm(rotate_spin(tensors, R))), rtol=0, atol=1e-12)


@pytest.mark.parametrize("target", [None, (2, 3)])
def test_rotate_mps_trial_equals_the_projected_trial(target):
    R = ROTATIONS["rotation"]
    trial = mps_trial_from_sd(*_uhf((3, 2)))
    got = rotate_mps_trial(trial, R, nelec=target)
    want = make_mps_trial(rotate_spin(trial.tensors, R), nelec=target or (3, 2))
    assert got.nelec == want.nelec == (target or (3, 2))
    vector = hf.mps_full_vector(got.tensors)
    _assert_close_vectors(vector, hf.mps_full_vector(want.tensors))
    np.testing.assert_allclose(vector @ vector, 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(got.rdm1), np.asarray(want.rdm1), rtol=0, atol=1e-12)
    assert got.sector_weight == pytest.approx(want.sector_weight, rel=1e-12)
    assert got.sector_weight < 0.9
    assert got.bond_dims == want.bond_dims


# ---------------------------------------------------------------------------------------------
# Capstone
# ---------------------------------------------------------------------------------------------

# (trial nelec, walkers' nelec, spin rotation, weight_floor, dt, propagator on both sides)
CAPSTONE_MPO = [
    pytest.param((3, 2), (3, 2), "rotation", 1e-3, 0.05, "fast", id="mpo-rotation-32-plain"),
    pytest.param((3, 3), (3, 3), "reflection", 0.5, 0.1, "fast", id="mpo-reflection-33-floor"),
    pytest.param((3, 2), (2, 3), "rotation", 1e-3, 0.05, "fast", id="mpo-rotation-32-to-23"),
    pytest.param((3, 3), (3, 3), "rotation", 1e-3, 0.05, "fast", id="mpo-rotation-33-plain"),
    pytest.param((3, 2), (3, 2), "reflection", 0.5, 0.1, "fast", id="mpo-reflection-32-floor"),
]


@pytest.mark.parametrize("nelec, walkers, rotation, floor, dt, propagator", CAPSTONE_MPO)
def test_rotated_ghf_and_mpo_rotated_mps_runs_are_identical(
    nelec, walkers, rotation, floor, dt, propagator
):
    """A spin-rotated GhfTrial and the MPO-rotated SD-as-MPS trial agree block for block."""
    assert floor <= 0.5 and dt * U <= 0.4 + 1e-12  # no site with both proposals floored
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, walkers)
    params = _params(dt=dt, weight_floor=floor, propagator=propagator)
    R = ROTATIONS[rotation]
    Ca, Cb = _uhf(nelec)
    C_ghf = np.kron(R, np.eye(L)) @ cast(np.ndarray, scipy.linalg.block_diag(Ca, Cb))
    sd = _sd_side(ham, sys_, params, C_ghf, propagator)
    tensors, charges = _input_mps("uhf", nelec)
    rotated, labels, norm2 = rotate_spin_mpo(tensors, charges, R, nelec=walkers)
    assert 1e-3 < norm2 < 0.9, "the rotation must move weight between sectors"
    trial = make_mps_trial(rotated, labels, nelec=walkers, rdm1=get_rdm1_block_diag(sd[0]))
    mps = _mps_side(ham, sys_, params, trial)

    _assert_identical_runs(_run(sys_, params, ham, sd), _run(sys_, params, ham, mps))
    if floor > 0.1:
        _assert_identical_states(
            _blocks_twin(sys_, params, ham, sd),
            _blocks_twin(sys_, params, ham, mps),
            nodes_positive=True,
        )
