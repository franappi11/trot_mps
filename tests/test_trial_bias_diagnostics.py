"""Physics checks and independent oracles for the optional trial-bias experiment.

    python -m pytest -q tests/test_trial_bias_diagnostics.py

No pyblock3, FCI artifacts, full 4x4 Hilbert space, or stochastic energy ranking
is needed. Spin operators are applied independently in the small Fock space;
Slater amplitudes come from determinant minors rather than the conversion code.
"""

from trot import config

config.configure_once(single_precision=False, matmul_precision="highest")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre

from tests.helpers import hubbard_fock as hf
from trot.gmps import engine
from trot.gmps import trial_bias_diagnostics as diag
from trot.gmps import trials
from trot.gmps.utils import make_orbital_plan, plan_bonds
from trot.ham.hubbard import square_hopping_matrix
from trot.trial.mps import rotate_spin, spin_rotation_y


def _sum_on_sites(v, operator, nsites):
    a = v.reshape((4,) * nsites)
    result = np.zeros_like(a)
    for i in range(nsites):
        result += np.moveaxis(np.tensordot(operator, a, axes=(1, i)), 0, i)
    return result.ravel()


def _s2_vector(v, nsites):
    # Local basis |0>, |up>, |down>, |up down>; spin bilinears have no JW string.
    up = np.zeros((4, 4))
    up[1, 2] = 1.
    z = np.diag([0., .5, -.5, 0.])
    return (_sum_on_sites(_sum_on_sites(v, up, nsites), up.T, nsites)
            + _sum_on_sites(_sum_on_sites(v, z, nsites), z, nsites)
            + _sum_on_sites(v, z, nsites))


def _sector_spin_matrix(nsites, nelec):
    shape = tuple(len(hf.sector_basis(nsites, n)[0]) for n in nelec)
    dim = int(np.prod(shape))
    matrix = np.zeros((dim, dim))
    for k in range(dim):
        column = np.zeros(dim)
        column[k] = 1.
        v = hf.full_from_sector(column.reshape(shape), nsites, *nelec)
        matrix[:, k] = hf.sector_from_full(_s2_vector(v, nsites), nsites, *nelec).ravel()
    return matrix


def _projected_amplitudes(tensors, nelec, angle):
    return hf.mps_sector_amplitudes(rotate_spin(tensors, spin_rotation_y(angle)), *nelec)


@pytest.fixture(scope="module")
def doped():
    nsites, nelec, u = 6, (2, 2), 8.
    h1 = np.asarray(square_hopping_matrix(2, 3, 1.))
    psi = np.random.default_rng(37).normal(size=(15, 15))
    psi /= np.linalg.norm(psi)
    tensors, charges = hf.exact_mps_from_sector_state(psi, nsites, *nelec)
    h = hf.hubbard_sector_hamiltonian(h1, u, *nelec)
    s2 = _sector_spin_matrix(nsites, nelec)
    return nsites, nelec, u, h1, psi, tensors, charges, h, s2


def test_global_rotation_preserves_energy_and_spin_but_moves_moment():
    states = [1, 2, 1, 2]
    tensors = [np.eye(4)[p].reshape(1, 4, 1) for p in states]
    rotated = rotate_spin(tensors, spin_rotation_y(90.))
    before, after = map(hf.mps_full_vector, (tensors, rotated))
    h1 = np.asarray(square_hopping_matrix(2, 2, 1.))
    for v in (before, after):
        np.testing.assert_allclose(v @ _s2_vector(v, 4), 2., atol=1e-12)
        np.testing.assert_allclose(v @ hf.apply_hubbard_full(h1, 8., v), 0., atol=1e-12)
    z = diag.local_elements(tensors, tensors, diag.SZ)
    np.testing.assert_allclose(diag.local_elements(rotated, rotated, diag.SX), z, atol=1e-12)
    np.testing.assert_allclose(diag.local_elements(rotated, rotated, diag.SZ), 0., atol=1e-12)
    np.testing.assert_allclose(diag.spin_squared(rotated), 2., atol=1e-12)


def test_sz_zero_after_rotation_retains_quintet_contamination():
    tensors = [np.eye(4)[p].reshape(1, 4, 1) for p in [1, 2, 1, 2]]
    psi = _projected_amplitudes(tensors, (2, 2), 90.)
    np.testing.assert_allclose(np.vdot(psi, psi), 3 / 8, atol=1e-12)
    v = hf.full_from_sector(psi / np.linalg.norm(psi), 4, 2, 2)
    np.testing.assert_allclose(v @ _s2_vector(v, 4), 2 / 3, atol=1e-12)
    sector_tensors, _ = hf.exact_mps_from_sector_state(psi / np.linalg.norm(psi), 4, 2, 2)
    for op in (diag.SZ, diag.SX):
        np.testing.assert_allclose(diag.local_elements(sector_tensors, sector_tensors, op), 0., atol=1e-12)


def test_singlet_quadrature_matches_independent_s2_projector(doped):
    n, nelec, _, _, psi, tensors, _, h, s2 = doped
    eig, vectors = np.linalg.eigh(s2)
    singlets = vectors[:, abs(eig) < 1e-10]
    expected = singlets @ (singlets.T @ psi.ravel())
    angles, weights = diag.singlet_quadrature(n, nelec)
    got = sum(w * _projected_amplitudes(tensors, nelec, angle).ravel()
              for angle, w in zip(angles, weights, strict=True))
    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=0)
    got /= np.linalg.norm(got)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(got @ s2 @ got, 0., atol=1e-10)
    np.testing.assert_allclose(got @ h @ got, expected @ h @ expected, atol=1e-10)


def test_four_angles_eliminate_every_higher_spin_for_fourteen_electrons():
    angles, weights = diag.singlet_quadrature(16, (7, 7))
    assert len(angles) == 4
    x = np.cos(np.deg2rad(angles))
    # Independent analytic representation matrix elements <S,0|Ry|S,0>.
    for spin in range(8):
        np.testing.assert_allclose(weights @ Legendre.basis(spin)(x), float(spin == 0), atol=1e-14)


def test_spin_weights_effective_profiles_and_energy_moments_match_enumeration(doped):
    n, nelec, u, h1, psi, tensors, charges, h, s2 = doped
    hp, hq, energy, variance = diag.prepare_htrial(tensors, charges, h1, u)
    hv = h @ psi.ravel()
    np.testing.assert_allclose(energy, psi.ravel() @ hv, atol=1e-10)
    np.testing.assert_allclose(variance, hv @ hv - energy**2, atol=1e-10)
    result = diag.spin_diagnostics(tensors, charges, nelec, (hp, hq))
    eig, vectors = np.linalg.eigh(s2)
    coefficients = vectors.T @ psi.ravel()
    want = [np.sum(coefficients[np.isclose(eig, s * (s + 1))]**2) for s in range(3)]
    np.testing.assert_allclose(result["original"]["spin_weights"], want, atol=1e-10)
    np.testing.assert_allclose(result["original"]["s2"], psi.ravel() @ s2 @ psi.ravel(), atol=1e-10)
    projected = _projected_amplitudes(tensors, nelec, 90.)
    norm = float(np.vdot(projected, projected))
    projected /= np.sqrt(norm)
    effective = result["effective"]
    np.testing.assert_allclose(effective["sector_weight"], norm, atol=1e-10)
    np.testing.assert_allclose(effective["energy"], projected.ravel() @ h @ projected.ravel(), atol=1e-10)
    np.testing.assert_allclose(effective["s2"], projected.ravel() @ s2 @ projected.ravel(), atol=1e-10)
    rdm = hf.dense_rdm1(projected, n, *nelec)
    np.testing.assert_allclose(effective["density"], np.diag(rdm[0] + rdm[1]), atol=1e-10)
    np.testing.assert_allclose(np.diag(rdm[0] - rdm[1]), 0., atol=1e-10)
    singlet = vectors[:, np.isclose(eig, 0.)] @ coefficients[np.isclose(eig, 0.)]
    np.testing.assert_allclose(result["singlet_projected_energy"], singlet @ h @ singlet / (singlet @ singlet), atol=1e-10)
    # Independently check block rotation of H|T>, including fermionic ordering.
    hr, qr = diag.number_blocks(hp, hq, rotation=spin_rotation_y(90.))
    dense_hr = hf.mps_full_vector(trials.from_blocks(hr, qr))
    full_rotated = hf.mps_full_vector(rotate_spin(tensors, spin_rotation_y(90.)))
    np.testing.assert_allclose(dense_hr, hf.apply_hubbard_full(h1, u, full_rotated), atol=1e-10)


def test_adaptive_reference_accuracy_does_not_imply_accuracy_for_other_walkers():
    reference = np.eye(8)[:, [0, 2, 4, 6]]
    walker = np.linalg.qr(np.random.default_rng(19).normal(size=(8, 4)))[0]
    adaptive = make_orbital_plan(reference, "adaptive")
    for c, should_be_exact in ((reference, True), (walker, False)):
        tensors, _, gauge = engine.channel_mps_host(c, adaptive)
        error = diag.channel_error(tensors, gauge, diag.determinant_amplitudes(c))
        if should_be_exact:
            assert error["relative_l2"] < 1e-10
        else:
            assert error["relative_l2"] > 1e-4
    exact = make_orbital_plan(reference, "rank_exact")
    tensors, _, gauge = engine.channel_mps_host(walker, exact)
    np.testing.assert_allclose(diag.channel_amplitudes(tensors, gauge), diag.determinant_amplitudes(walker), atol=1e-12)


@pytest.mark.parametrize("device", [False, True], ids=["host", "compiled"])
def test_exact_conversion_and_bond_truncation_are_separate(device):
    rng = np.random.default_rng(81)
    reference = np.linalg.qr(rng.normal(size=(8, 4)))[0]
    walker = np.linalg.qr(rng.normal(size=(8, 4)))[0]
    plan = make_orbital_plan(reference, "rank_exact")
    for cap in (None, 2):
        bond = None if cap is None else plan_bonds(reference, plan, cap)
        if device:
            converter = jax.jit(engine.make_converter(plan, plan, bond, bond).convert)
            alpha, _, gauges = converter(jnp.asarray(walker), jnp.asarray(walker))
            alpha, gauge = list(map(np.asarray, alpha)), float(gauges[0])
        else:
            alpha, _, gauge = engine.channel_mps_host(walker, plan, bond)
        error = diag.channel_error(alpha, gauge, diag.determinant_amplitudes(walker))
        if cap is None:
            assert error["relative_l2"] < 1e-10
        else:
            assert error["relative_l2"] > 1e-4


def test_overlap_and_local_energy_diagnostics_match_fock_space(doped):
    n, nelec, u, h1, _, tensors, charges, h, _ = doped
    hp, hq, _, _ = diag.prepare_htrial(tensors, charges, h1, u)
    rng = np.random.default_rng(52)
    reference = tuple(np.linalg.qr(rng.normal(size=(n, nn)))[0] for nn in nelec)
    walkers = [tuple(np.linalg.qr(rng.normal(size=(n, nn)))[0] for nn in nelec) for _ in range(2)]
    report = diag.compare_conversions(tensors, charges, (hp, hq), reference, walkers,
                                     cap=2, say=lambda *a, **k: None)
    amplitudes = _projected_amplitudes(tensors, nelec, 90.)
    for pair, row in zip(walkers, report["rank_exact_uncapped"]["walkers"], strict=True):
        sd = hf.sd_amplitudes(*pair).ravel()
        expected_overlap = amplitudes.ravel() @ sd
        numerator = amplitudes.ravel() @ h @ sd
        np.testing.assert_allclose(row["overlap"], expected_overlap, atol=1e-10)
        np.testing.assert_allclose(row["numerator"], numerator, atol=1e-10)
        np.testing.assert_allclose(row["local_energy"], numerator / expected_overlap, atol=1e-9)
    near = diag.compare_conversions(tensors, charges, (hp, hq), reference, walkers,
                                   cap=2, near_zero=2., say=lambda *a, **k: None)
    for result in near.values():
        for row in result["walkers"]:
            assert row["near_zero"]
            assert row["local_energy"] is None and row["sign_disagreement"] is None
            assert np.isfinite(row["overlap"]) and np.isfinite(row["numerator"])


@pytest.mark.parametrize("shape", [(2, 2), (2, 4), (4, 4)])
def test_both_hole_guesses_preserve_particle_numbers(shape):
    expected = np.prod(shape) // 2 - 1
    for current in (False, True):
        states = diag.product_guess(shape, current)
        assert np.count_nonzero(states == 0) == 2
        assert np.count_nonzero(states == 1) == expected
        assert np.count_nonzero(states == 2) == expected


def test_probe_walkers_are_reproducible_and_snapshots_preserve_gauge(tmp_path):
    h1 = np.asarray(square_hopping_matrix(2, 2, 1.))
    first = diag.make_probe_walkers(h1, 8., (1, 1), count=2, steps=3)
    second = diag.make_probe_walkers(h1, 8., (1, 1), count=2, steps=3)
    np.testing.assert_array_equal(first, second)
    path = tmp_path / "walkers.npz"
    np.savez(path, up=np.stack([p[0] for p in first]), dn=np.stack([p[1] for p in first]))
    np.testing.assert_array_equal(diag.load_probe_walkers([path], 4, (1, 1), 2), first)
    with pytest.raises(ValueError, match="only 2"):
        diag.load_probe_walkers([path], 4, (1, 1), 3)
    np.savez(path, up=np.stack([p[0] * 2 for p in first]), dn=np.stack([p[1] for p in first]))
    with pytest.raises(ValueError, match="orthonormal"):
        diag.load_probe_walkers([path], 4, (1, 1), 1)
