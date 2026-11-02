"""Independent Fock-space checks of notebook contractions and trial moments."""

from itertools import product

from trot.config import configure_once

configure_once(quiet=True)

import numpy as np

from trot.gmps import engine, utils
from trot.gmps.notebook_walker_tools import cas_trial_moments, determinant_mps_transition


def test_subset_overlap_and_all_densities_match_fock_with_permuted_bonds():
    rng = np.random.default_rng(321)
    L, N = 8, 3
    D = np.linalg.qr(rng.normal(size=(L, N)))[0]
    Q = np.linalg.qr(rng.normal(size=(L, N)))[0]
    plan = utils.make_orbital_plan(D, "rank_exact")
    bond = utils.plan_bonds(D, plan, 4)
    tensors, charges, gauge = engine.channel_mps_host(Q, plan, bond)
    # Dense conversion's bond labels need not be sorted. The contraction must
    # use labels as data rather than assume contiguous fixed charge sectors.
    for i in range(1, L):
        order = rng.permutation(len(charges[i]))
        tensors[i - 1] = tensors[i - 1][:, :, order]
        tensors[i] = tensors[i][order]
        charges[i] = charges[i][order]
    gauge *= -0.7
    v = np.ones((1, 1))
    for A in tensors:
        v = np.einsum("xa,apb->xpb", v, A).reshape(-1, A.shape[-1])
    v = gauge * v[:, 0]
    configs = np.array(list(product((0, 1), repeat=L)))
    bra = np.zeros(2**L)
    for i, occ in enumerate(configs):
        if occ.sum() == N:
            bra[i] = np.linalg.det(D[np.flatnonzero(occ)])
    overlap, density = determinant_mps_transition(D, tensors, charges, gauge)
    np.testing.assert_allclose(overlap, bra @ v, atol=2e-13)
    np.testing.assert_allclose(density, ((bra * v)[:, None] * configs).sum(axis=0), atol=2e-13)
    overlap_only, omitted_density = determinant_mps_transition(
        D, tensors, charges, gauge, with_density=False
    )
    np.testing.assert_allclose(overlap_only, bra @ v, atol=2e-13)
    assert omitted_density is None


def test_subset_contraction_retains_density_when_overlap_is_zero():
    D = np.array([[1.0], [1.0]]) / np.sqrt(2)
    Q = np.array([[1.0], [-1.0]]) / np.sqrt(2)
    plan = utils.make_orbital_plan(Q, "rank_exact")
    tensors, charges, gauge = engine.channel_mps_host(Q, plan)
    overlap, density = determinant_mps_transition(D, tensors, charges, gauge)
    np.testing.assert_allclose(overlap, 0, atol=2e-14)
    np.testing.assert_allclose(density, [0.5, -0.5], atol=2e-14)


def test_full_cas_moments_match_hubbard_fock_including_external_excitations():
    from tests.helpers.hubbard_fock import hubbard_sector_hamiltonian, sd_amplitudes

    h1 = np.array([[0, -1, -1, 0], [-1, 0, 0, -1],
                   [-1, 0, 0, -1], [0, -1, -1, 0]], dtype=float)
    _, orbitals = np.linalg.eigh(h1)
    core, up = orbitals[:, :1], orbitals[:, 1:3]
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    down = up @ rotation
    coefficients = np.array([np.sqrt(0.3), -np.sqrt(0.7)])
    psi = sum(c * sd_amplitudes(np.column_stack((core, up[:, r])),
                                np.column_stack((core, down[:, r])))
              for r, c in enumerate(coefficients)).ravel()
    H = hubbard_sector_hamiltonian(h1, 4.0, 2, 2)
    mean = psi @ H @ psi
    residual = H @ psi - mean * psi
    expected = (mean, residual @ residual)
    actual = cas_trial_moments(h1, 4.0, core, up, down, coefficients)
    np.testing.assert_allclose(actual, expected, atol=5e-13)
    assert actual[1] > 0
