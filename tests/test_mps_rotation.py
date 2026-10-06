"""Spin rotations of MPS trials as an S_z-conserving MPO (trot.trial.mps_rotation).

The MPO must equal the projected product of site rotations. Applied to a labelled MPS, it must give P_target of
rotate_spin's state by exact enumeration (tests/helpers/hubbard_fock.py), the rotated GHF determinant, and the exact
transformation of spin multiplets. rotate_mps_trial is the normalised projection with sector weight w, and
make_mps_trial uses the rotated MPS as it is (particle-number labels), with sector amplitudes sqrt(w) times the
projected ones. The capstone runs trot's CPMC with both, identical to the rotated GHF trial.
"""

from trot import config

config.configure_once()

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from tests.helpers import hubbard_fock as hf
from trot.core.system import System
from trot.driver import make_run_blocks, run_qmc
from trot.gmps.driver import make_mps_cpmc_ops
from trot.gmps.utils import number_labels, sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.prop import blocks, cpmc
from trot.prop.types import QmcParamsMps
from trot.trial import mps as trial_mps
from trot.trial.ghf import GhfTrial, get_rdm1_block_diag, make_ghf_trial_ops
from trot.trial.mps import (
    PHYSICAL_CHARGE,
    _hashable_charges,
    _valid_sz_labels,
    make_mps_trial,
    mps_trial_from_sd,
    one_rdm,
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


def _ghf_orbitals(R, nelec):
    """kron(R, I_L) @ block_diag(Ca, Cb) of the staggered UHF determinant: the rotated GHF determinant."""
    return np.kron(R, np.eye(L)) @ cast(np.ndarray, scipy.linalg.block_diag(*_uhf(nelec)))


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
def test_mpo_path_is_the_projection_of_the_rotated_state(nelec, target, name):
    """rotate_spin_mpo is P_target rotate_spin(T, R): sector amplitudes and norm by enumeration of the rotated
    state, nothing outside the target, exact labels, and no bond above the projected state's Schmidt rank."""
    R = ROTATIONS[name]
    tensors, charges = _input_mps("uhf", nelec)
    got, got_charges, got_norm2 = rotate_spin_mpo(tensors, charges, R, nelec=target)
    want = hf.sector_from_full(hf.mps_full_vector(rotate_spin(tensors, R)), L, *target)
    _assert_close_vectors(hf.mps_sector_amplitudes(got, *target), want)
    np.testing.assert_allclose(got_norm2, np.sum(want**2), rtol=1e-12)
    assert 1e-3 < got_norm2 < 0.9, "the rotation must move weight between sectors"
    vector = hf.mps_full_vector(got)
    assert np.all(vector[np.any(_site_charges(L) != np.asarray(target), axis=1)] == 0.0)
    assert _valid_sz_labels(got, got_charges, target)
    projected = hf.full_from_sector(want, L, *target)
    for c in range(1, L):
        schmidt = np.linalg.svd(projected.reshape(4**c, -1), compute_uv=False)
        assert len(got_charges[c]) <= np.sum(schmidt > 1e-14 * schmidt[0]), f"bond {c} is not compressed"


@pytest.mark.parametrize("name", ["rotation", "reflection"])
@pytest.mark.parametrize("nelec, target", [((3, 3), (3, 3)), ((3, 2), (3, 2)), ((3, 2), (2, 3))])
def test_mpo_rotation_matches_the_rotated_ghf_determinant(nelec, target, name):
    R = ROTATIONS[name]
    tensors, charges = _input_mps("uhf", nelec)
    rotated, _, _ = rotate_spin_mpo(tensors, charges, R, nelec=target)
    got = hf.mps_sector_amplitudes(rotated, *target)
    want = hf.ghf_amplitudes(_ghf_orbitals(R, nelec), *target)
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
    swapped = hf.mps_full_vector(rotate_spin(tensors, R))
    assert not np.any(hf.sector_from_full(swapped, L, 3, 2)), "enumeration agrees: no (3, 2) amplitude"
    rotated, charges_out, norm2 = rotate_spin_mpo(tensors, charges, R, nelec=(2, 3))
    vector = hf.mps_full_vector(tensors)
    np.testing.assert_allclose(norm2, vector @ vector, rtol=1e-12)
    assert _valid_sz_labels(rotated, charges_out, (2, 3))
    _assert_close_vectors(hf.mps_full_vector(rotated), _rotated_vector(vector, R, (2, 3)))
    _assert_close_vectors(hf.mps_full_vector(rotated), swapped)


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
    as_is = make_mps_trial(rotate_spin(tensors, R), nelec=(3, 2))  # particle-number labels only
    with pytest.raises(ValueError, match="bond labels"):
        rotate_mps_trial(as_is, R)


# ---------------------------------------------------------------------------------------------
# The trial
# ---------------------------------------------------------------------------------------------


def test_make_mps_trial_uses_the_mpo_output_without_projecting(monkeypatch):
    """Valid (N_up, N_dn) labels are used as given: nothing is compressed or read off the tensors again."""
    tensors, charges = _input_mps("uhf", (3, 2))
    rotated, labels, norm2 = rotate_spin_mpo(tensors, charges, ROTATIONS["rotation"])

    def forbidden(*args, **kwargs):
        raise AssertionError("make_mps_trial reworked the MPO output")

    for name in ("compress_mps_qn", "sz_labels", "number_labels"):
        monkeypatch.setattr(trial_mps, name, forbidden)
    trial = make_mps_trial(rotated, labels, nelec=(3, 2))
    assert np.array_equal(np.asarray(trial.tensors[0]), rotated[0] / np.sqrt(norm2))
    for a, b in zip(trial.tensors[1:], rotated[1:]):
        assert np.array_equal(np.asarray(a), b)
    assert trial.charges == _hashable_charges(labels) and trial.label_width == 2
    assert trial.sector_weight == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", ["rotation", "reflection"])
@pytest.mark.parametrize("nelec", [(3, 3), (3, 2)])
def test_rotated_rdm1_is_the_ghf_walker_start(nelec, name):
    R = ROTATIONS[name]
    tensors, _ = _input_mps("uhf", nelec)
    got = rotated_rdm1(np.stack(one_rdm(tensors)), R)
    ghf = np.asarray(get_rdm1_block_diag(GhfTrial(mo_coeff=jnp.asarray(_ghf_orbitals(R, nelec)))))
    np.testing.assert_allclose(got, ghf, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got, np.stack(one_rdm(rotate_spin(tensors, R))), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["rotation", "reflection"])
@pytest.mark.parametrize("target", [None, (2, 3)])
def test_rotated_trial_as_it_is_and_projected(target, name):
    """make_mps_trial(rotate_spin(T, R)) is the rotated state as it is: particle-number labels (number_labels),
    sector_weight None, the input's bonds. rotate_mps_trial is its normalised projection onto the target sector
    with sector_weight w = <P U T|P U T>/<T|T>, and there the as-is amplitudes are sqrt(w) times the projected
    ones. Both start the walkers from the same rdm1. References: enumeration of rotate_spin's state."""
    R = ROTATIONS[name]
    sector = target or (3, 2)
    trial = mps_trial_from_sd(*_uhf((3, 2)))
    rotated = rotate_spin(trial.tensors, R)
    full = hf.mps_full_vector(rotated)
    np.testing.assert_allclose(full @ full, 1.0, rtol=1e-12)  # a normalised trial, an orthogonal M
    want = hf.sector_from_full(full, L, *sector)
    weight = float(np.sum(want**2))
    assert 1e-3 < weight < 0.9, "the rotation must move weight between sectors"

    as_is = make_mps_trial(rotated, nelec=sector)
    assert as_is.nelec == sector and as_is.label_width == 1 and as_is.sector_weight is None
    assert as_is.charges == _hashable_charges(number_labels(rotated))
    assert as_is.bond_dims == trial.bond_dims
    _assert_close_vectors(hf.mps_full_vector(as_is.tensors), full)

    projected = rotate_mps_trial(trial, R, nelec=target)
    tensors = [np.asarray(A) for A in projected.tensors]
    assert projected.nelec == sector and projected.label_width == 2
    assert _valid_sz_labels(tensors, projected.charge_arrays(), sector)
    np.testing.assert_allclose(projected.sector_weight, weight, rtol=1e-12)
    vector = hf.mps_full_vector(tensors)
    np.testing.assert_allclose(vector @ vector, 1.0, rtol=1e-12)
    amplitudes = hf.mps_sector_amplitudes(tensors, *sector)
    _assert_close_vectors(amplitudes, want / np.sqrt(weight))
    _assert_close_vectors(
        hf.mps_sector_amplitudes(as_is.tensors, *sector), np.sqrt(projected.sector_weight) * amplitudes
    )
    np.testing.assert_allclose(np.asarray(projected.rdm1), np.asarray(as_is.rdm1), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# Capstone
# ---------------------------------------------------------------------------------------------


def _params(**overrides: Any) -> QmcParamsMps:
    values: dict[str, Any] = dict(
        dt=0.05,
        n_walkers=10,
        n_prop_steps=5,
        n_eql_blocks=5,
        n_blocks=10,
        seed=11,
        auto_n_chunks=False,
        n_chunks=1,
        orbital_plan="maximal",
        walker_channel_chi=None,
    )
    values.update(overrides)
    return QmcParamsMps(**values)


def _ghf_side(ham, sys_, params, ghf):
    """The plain trot CPMC template: GhfTrial, GHF trial and measurement ops, trot's CPMC propagator."""
    trial_ops = make_ghf_trial_ops(sys_)
    prop_ops = cpmc.make_prop_ops(ham, "unrestricted", trial_ops)
    prop_ctx = prop_ops.build_prop_ctx(ham, None, params)  # pyright: ignore[reportArgumentType]
    return ghf, trial_ops, make_ghf_meas_ops_hubbard(sys_), prop_ops, prop_ctx


def _mps_side(ham, sys_, params, trial):
    ops = make_mps_cpmc_ops(ham, trial, sys_, params)
    prop_ctx = ops.prop_ops.build_prop_ctx(ham, ops.trial_ops.get_rdm1(trial), params)
    return trial, ops.trial_ops, ops.meas_ops, ops.prop_ops, prop_ctx


def _run(sys_, params, ham, side):
    trial, trial_ops, meas_ops, prop_ops, prop_ctx = side
    return run_qmc(
        sys=sys_,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        prop_ctx=prop_ctx,
        block_fn=blocks.block,
    )


def _two_blocks(sys_, params, ham, side):
    """The initial state and the state and block scalars after two jitted blocks."""
    trial, trial_ops, meas_ops, prop_ops, prop_ctx = side
    meas_ctx = meas_ops.build_meas_ctx(ham, trial)
    state = prop_ops.init_prop_state(
        sys=sys_,
        ham_data=ham,
        trial_ops=trial_ops,
        trial_data=trial,
        meas_ops=meas_ops,
        params=params,
        meas_ctx=meas_ctx,
    )
    run_blocks = make_run_blocks(
        block_fn=blocks.block,
        sys=sys_,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )
    final, scalars, _ = run_blocks(
        state, ham_data=ham, trial_data=trial, meas_ctx=meas_ctx, prop_ctx=prop_ctx, n_blocks=2
    )
    return state, final, scalars


# (trial nelec, walkers' nelec, spin rotation, weight_floor, dt, trial form)
CAPSTONE = [
    pytest.param((3, 2), (3, 2), "rotation", 1e-3, 0.05, "projected", id="mpo-rotation-32-plain"),
    pytest.param((3, 3), (3, 3), "reflection", 0.5, 0.1, "projected", id="mpo-reflection-33-floor"),
    pytest.param((3, 2), (2, 3), "rotation", 1e-3, 0.05, "projected", id="mpo-rotation-32-to-23"),
    pytest.param((3, 3), (3, 3), "rotation", 1e-3, 0.05, "projected", id="mpo-rotation-33-plain"),
    pytest.param((3, 2), (3, 2), "reflection", 0.5, 0.1, "projected", id="mpo-reflection-32-floor"),
    pytest.param((3, 2), (2, 3), "rotation", 1e-3, 0.05, "as_is", id="as-is-rotation-32-to-23"),
    pytest.param((3, 3), (3, 3), "reflection", 0.5, 0.1, "as_is", id="as-is-reflection-33-floor"),
]


@pytest.mark.parametrize("nelec, walkers, rotation, floor, dt, form", CAPSTONE)
def test_rotated_ghf_and_rotated_mps_runs_are_identical(nelec, walkers, rotation, floor, dt, form):
    """A spin-rotated GhfTrial and the rotated SD-as-MPS trial, projected (rotate_mps_trial) or used as it is
    (make_mps_trial of rotate_spin), make the same CPMC decisions: block energies and weights agree to rounding,
    and with a high floor (the constraint acting) so do walkers, weights, node counts and RNG after two blocks."""
    assert floor <= 0.5 and dt * U <= 0.4 + 1e-12  # no site with both proposals floored
    ham = HamHubbard(h1=jnp.asarray(hopping_matrix(L, 1.0)), u=U)
    sys_ = System(norb=L, nelec=tuple(walkers), walker_kind="unrestricted")
    params = _params(dt=dt, weight_floor=floor)
    R = ROTATIONS[rotation]
    ghf = GhfTrial(mo_coeff=jnp.asarray(_ghf_orbitals(R, nelec)))
    rdm1 = get_rdm1_block_diag(ghf)  # the same walker start on both sides
    sd_trial = mps_trial_from_sd(*_uhf(nelec))
    if form == "projected":
        trial = rotate_mps_trial(sd_trial, R, nelec=walkers, rdm1=rdm1)
        assert 1e-3 < trial.sector_weight < 0.9, "the rotation must move weight between sectors"
    else:
        trial = make_mps_trial(rotate_spin(sd_trial.tensors, R), nelec=walkers, rdm1=rdm1)
        assert trial.label_width == 1 and trial.sector_weight is None
    sd, mps = _ghf_side(ham, sys_, params, ghf), _mps_side(ham, sys_, params, trial)

    run_sd, run_mps = _run(sys_, params, ham, sd), _run(sys_, params, ham, mps)
    e_sd, e_mps = np.asarray(run_sd.block_energies), np.asarray(run_mps.block_energies)
    w_sd, w_mps = np.asarray(run_sd.block_weights), np.asarray(run_mps.block_weights)
    assert e_sd.shape == e_mps.shape and w_sd.shape == w_mps.shape
    np.testing.assert_allclose(e_mps, e_sd, rtol=1e-9, atol=0)
    np.testing.assert_allclose(w_mps, w_sd, rtol=1e-9, atol=0)
    assert np.ptp(e_sd[1:]) > 1e-3, "the comparison needs a genuinely stochastic run"
    np.testing.assert_allclose(float(run_mps.mean_energy), float(run_sd.mean_energy), rtol=1e-9, atol=0)
    if np.isfinite(run_sd.stderr_energy):
        np.testing.assert_allclose(
            float(run_mps.stderr_energy), float(run_sd.stderr_energy), rtol=1e-9, atol=1e-12
        )
    else:
        assert not np.isfinite(run_mps.stderr_energy)
    if floor <= 0.1:
        return

    (sd_start, sd_final, sd_scalars) = _two_blocks(sys_, params, ham, sd)
    (mps_start, mps_final, mps_scalars) = _two_blocks(sys_, params, ham, mps)
    for a, b in zip(mps_start.walkers, sd_start.walkers):
        assert np.array_equal(np.asarray(a), np.asarray(b)), "starting walkers must be identical"
    for a, b in zip(mps_final.walkers, sd_final.walkers):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
    w_sd, w_mps = np.asarray(sd_final.weights), np.asarray(mps_final.weights)
    assert np.all(np.isfinite(w_mps)) and np.all(np.isfinite(w_sd))
    assert np.array_equal(w_sd == 0.0, w_mps == 0.0), "a walker died on one side only"
    np.testing.assert_allclose(w_mps, w_sd, rtol=1e-10, atol=0)
    ratio = np.asarray(mps_final.overlaps) / np.asarray(sd_final.overlaps)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)
    assert int(mps_final.node_encounters) == int(sd_final.node_encounters) > 0, "the constraint never acted"
    assert np.array_equal(np.asarray(mps_final.rng_key), np.asarray(sd_final.rng_key))
    for key in ("energy", "weight"):
        assert np.all(np.isfinite(np.asarray(mps_scalars[key])))
        np.testing.assert_allclose(
            np.asarray(mps_scalars[key]), np.asarray(sd_scalars[key]), rtol=1e-10, atol=0
        )
