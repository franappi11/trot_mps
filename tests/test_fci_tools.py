"""trot.gmps.fci_tools on small lattices, against the exact Fock-space oracles of tests/helpers/hubbard_fock.py
(alpha-block sector amplitudes in lexicographic order) and pyscf's own operators.

The conventions checked: the alpha-block amplitudes are pyscf's CI vectors up to one sign (same energy as the
oracle's Hamiltonian), MPS amplitudes convert with the interleave sign (any MPS, also a spin-rotated one), lattice
permutations keep the Hubbard energy (the reordering signs), Schmidt values equal those of the dense vector, and the
correlation functions equal their definitions.
"""

import numpy as np
import pytest

from tests.helpers import hubbard_fock as hf
from trot.gmps import fci_tools
from trot.ham.hubbard import hopping_matrix, square_hopping_matrix

pyscf = pytest.importorskip("pyscf")
from pyscf import fci  # noqa: E402

U = 4.0
CASES = {  # name: (h1, nelec, (Lx, Ly))
    "sq2x3": (square_hopping_matrix(2, 3, 1.0), (2, 2), (2, 3)),
    "sq3x3": (square_hopping_matrix(3, 3, 1.0), (3, 2), (3, 3)),  # odd L, N_up != N_dn, C4v
    "L5": (hopping_matrix(5, 1.0), (2, 1), (5, 1)),
}


def to_pyscf(psi, norb, nelec):
    """Lexicographic alpha-block amplitudes (hubbard_fock.sector_basis order) in pyscf's string order."""
    out = np.zeros_like(psi)
    rows, cols = (hf.sector_basis(norb, n)[1] @ (1 << np.arange(norb)) for n in nelec)
    out[np.ix_(fci_tools.strings(norb, nelec[0])[2][rows], fci_tools.strings(norb, nelec[1])[2][cols])] = psi
    return out


def random_state(norb, nelec, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(len(hf.sector_basis(norb, nelec[0])[0]), len(hf.sector_basis(norb, nelec[1])[0])))
    return psi / np.linalg.norm(psi)


def eri_of(norb, u):
    eri = np.zeros((norb,) * 4)
    eri[np.arange(norb), np.arange(norb), np.arange(norb), np.arange(norb)] = u
    return eri


@pytest.mark.parametrize("name", CASES)
def test_alpha_block_amplitudes_are_pyscf_vectors(name):
    """Same energy from pyscf (dense two-electron contraction), from hubbard_energy and from the oracle's sector
    Hamiltonian: the alpha-block convention is pyscf's up to one sign."""
    h1, nelec, _ = CASES[name]
    norb = len(h1)
    psi = random_state(norb, nelec)
    H = hf.hubbard_sector_hamiltonian(h1, U, *nelec)
    exact = float(psi.ravel() @ H @ psi.ravel())
    ci = to_pyscf(psi, norb, nelec)
    assert np.isclose(fci.direct_spin1.energy(h1, eri_of(norb, U), ci, norb, nelec), exact, atol=1e-10)
    assert np.isclose(fci_tools.hubbard_energy(ci, h1, U, nelec), exact, atol=1e-10)


@pytest.mark.parametrize("name", CASES)
def test_mps_to_fci_matches_the_dense_amplitudes(name):
    """An exact MPS of a random sector state gives back the state; a spin-rotated MPS (particle-number labels, all
    sectors mixed) gives the oracle's amplitudes of its (N_up, N_dn) part."""
    from trot.trial.mps import rotate_spin, spin_rotation_y

    h1, nelec, _ = CASES[name]
    norb = len(h1)
    psi = random_state(norb, nelec, seed=1)
    tensors, _ = hf.exact_mps_from_sector_state(psi, norb, *nelec)
    np.testing.assert_allclose(fci_tools.mps_to_fci(tensors, nelec), to_pyscf(psi, norb, nelec), atol=1e-12)
    rotated = rotate_spin(tensors, spin_rotation_y(90.0))
    expected = to_pyscf(hf.mps_sector_amplitudes(rotated, *nelec), norb, nelec)
    np.testing.assert_allclose(fci_tools.mps_to_fci(rotated, nelec), expected, atol=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_lattice_permutations_keep_the_hubbard_energy_and_match_transform_ci(name):
    """g|psi> has the energy of |psi> for every symmetry operation (wrong reordering signs would flip hoppings), and
    equals pyscf's transform_ci with the permutation matrix (for g or its inverse, up to sign)."""
    h1, nelec, (Lx, Ly) = CASES[name]
    norb = len(h1)
    ci = to_pyscf(random_state(norb, nelec, seed=2), norb, nelec)
    energy = fci_tools.hubbard_energy(ci, h1, U, nelec)
    ops, classes, table = fci_tools.square_point_group(Lx, Ly)
    for g, perm in ops.items():
        moved = fci_tools.permute_ci(ci, norb, nelec, perm)
        assert np.isclose(fci_tools.hubbard_energy(moved, h1, U, nelec), energy, atol=1e-10), g
        u = np.eye(norb)[:, perm]
        transformed = fci.addons.transform_ci(ci, nelec, u)
        inverse = fci_tools.permute_ci(ci, norb, nelec, np.argsort(perm))
        errors = [np.abs(transformed - s * c).max() for s in (1, -1) for c in (moved, inverse)]
        assert min(errors) < 1e-12, g
    # the irrep weights of a random state sum to one
    _, weights = fci_tools.point_group_weights(ci, norb, nelec, (ops, classes, table))
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-12)


def test_ground_state_irrep_is_pure():
    """A non-degenerate ground state lies in one irrep: one weight 1, characters +-1."""
    h1, nelec, (Lx, Ly) = CASES["sq2x3"]
    norb = len(h1)
    _, psi = hf.ground_state(h1, U, *nelec)
    ci = to_pyscf(psi, norb, nelec)
    characters, weights = fci_tools.point_group_weights(ci, norb, nelec, fci_tools.square_point_group(Lx, Ly))
    assert max(weights.values()) > 1 - 1e-10
    assert all(np.isclose(abs(c), 1.0, atol=1e-10) for c in characters.values())


@pytest.mark.parametrize("name", CASES)
def test_schmidt_values_match_the_dense_svd(name):
    h1, nelec, _ = CASES[name]
    norb = len(h1)
    psi = random_state(norb, nelec, seed=3)
    vector = hf.full_from_sector(psi, norb, *nelec)
    ci = to_pyscf(psi, norb, nelec)
    for cut in range(1, norb):
        dense = np.linalg.svd(vector.reshape(4**cut, -1), compute_uv=False)
        values = np.sort(np.concatenate(list(fci_tools.schmidt_values(ci, norb, nelec, cut).values())))[::-1]
        dense = dense[: len(values)]
        np.testing.assert_allclose(values[values > 1e-12], dense[dense > 1e-12], atol=1e-12)
    summary = fci_tools.schmidt_summary(fci_tools.schmidt_values(ci, norb, nelec, norb // 2), chis=(1, 4, 10**6))
    assert np.isclose(summary["discarded"][10**6], 0.0, atol=1e-12)
    assert np.isclose(sum(summary["sectors"].values()), 1.0)


@pytest.mark.parametrize("name", CASES)
def test_correlations_match_their_definitions(name):
    """n_i, <n_i n_j>, <S^z_i S^z_j>, double occupancy and <h_i h_j> summed over the sector's configurations."""
    h1, nelec, _ = CASES[name]
    norb = len(h1)
    psi = random_state(norb, nelec, seed=4)
    _, occ_a = hf.sector_basis(norb, nelec[0])
    _, occ_b = hf.sector_basis(norb, nelec[1])
    w = psi**2
    n_up = np.einsum("ab,ai->i", w, occ_a)
    n_dn = np.einsum("ab,bi->i", w, occ_b)
    n = occ_a[:, None, :] + occ_b[None, :, :]
    sz = 0.5 * (occ_a[:, None, :] - occ_b[None, :, :])
    holes = (1 - occ_a[:, None, :]) * (1 - occ_b[None, :, :])
    ci = to_pyscf(psi, norb, nelec)
    diagonal = fci_tools.diagonal_correlations(ci, norb, nelec)
    np.testing.assert_allclose(diagonal["n_up"], n_up, atol=1e-12)
    np.testing.assert_allclose(diagonal["n_dn"], n_dn, atol=1e-12)
    np.testing.assert_allclose(diagonal["nn"], np.einsum("ab,abi,abj->ij", w, n, n), atol=1e-12)
    np.testing.assert_allclose(diagonal["szsz"], np.einsum("ab,abi,abj->ij", w, sz, sz), atol=1e-12)
    np.testing.assert_allclose(diagonal["double"], np.einsum("ab,ai,bi->i", w, occ_a, occ_b), atol=1e-12)
    h, hh = fci_tools.hole_correlations(ci, norb, nelec)
    np.testing.assert_allclose(hh, np.einsum("ab,abi,abj->ij", w, holes, holes), atol=1e-12)
    np.testing.assert_allclose(h, np.einsum("ab,abi->i", w, holes), atol=1e-12)


def test_pair_correlations_match_pyscf_operators():
    """<D+_u D_v> from the 2-RDM equals <D_u psi|D_v psi> with D_{ij} = (c_{j dn} c_{i up} - c_{j up} c_{i dn})/sqrt(2)
    built from pyscf's annihilation operators."""
    h1, nelec, (Lx, Ly) = CASES["sq2x3"]
    norb = len(h1)
    ci = to_pyscf(random_state(norb, nelec, seed=5), norb, nelec)
    bonds = fci_tools.lattice_bonds(Lx, Ly)
    na, nb = nelec

    def pair_annihilated(i, j):
        first = fci.addons.des_b(fci.addons.des_a(ci, norb, (na, nb), i), norb, (na - 1, nb), j)
        second = fci.addons.des_a(fci.addons.des_b(ci, norb, (na, nb), i), norb, (na, nb - 1), j)
        return (first - second) / np.sqrt(2.0)

    states = [pair_annihilated(i, j) for i, j, _ in bonds]
    expected = np.array([[np.vdot(a, b) for b in states] for a in states])
    np.testing.assert_allclose(fci_tools.pair_correlations(ci, norb, nelec, bonds), expected, atol=1e-12)
    d_wave = fci_tools.d_wave_correlations(expected, bonds, norb)
    np.testing.assert_allclose(d_wave, d_wave.T, atol=1e-12)


def test_sign_agreement_and_distances():
    rng = np.random.default_rng(6)
    ref = rng.normal(size=(20, 20))
    assert fci_tools.sign_agreement(ref, ref) == 1.0
    assert fci_tools.sign_agreement(ref, -2.0 * ref) == 1.0
    flipped = ref.copy()
    flipped[0, 0] *= -1
    assert np.isclose(fci_tools.sign_agreement(ref, flipped), 1.0 - ref[0, 0] ** 2 / np.sum(ref**2))
    curve = fci_tools.distance_average(np.ones((9, 9)), 3, 3, min_distance=1.0)
    assert min(curve) == 1.0 and all(v == 1.0 for v in curve.values())
