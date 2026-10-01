"""Test-only oracles for MPS-CPMC: exact Fock-space Hubbard objects for small lattices.

Independent of the production MPS code. Conventions match trot.trial.mps:

- local basis |0>, |up>, |dn>, |up dn> with index n_up + 2*n_dn, |up dn> = c+_up c+_dn |0>;
- fermion modes ordered site by site with spin up first (mode 2*i + spin), the "interleaved" order
  of the MPS; a dense 4^L vector has site 0 as its most significant index.

Sector amplitudes are matrices psi[a, b] over the up configurations a and down configurations b
(sector_basis order) in the "alpha-block" convention natural for determinants: configuration
(A, B) is prod_{i in A} c+_{i up} prod_{j in B} c+_{j dn} |0> with increasing sites, so an
unrestricted determinant has amplitude det(Ca[A]) det(Cb[B]). Interleaved (MPS) amplitudes
differ by (-1)^K with K = #{(i, j): i in A, j in B, j < i} (reorder_sign).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


def sector_basis(L, n):
    """(rows, occ): the C(L, n) occupied-site tuples in lexicographic order and their 0/1 rows."""
    if n == 0:
        rows = np.zeros((1, 0), dtype=int)
    else:
        rows = np.array(list(combinations(range(L), n)), dtype=int)
    occ = np.zeros((len(rows), L), dtype=int)
    occ[np.arange(len(rows))[:, None], rows] = 1
    return rows, occ


def reorder_sign(occ_a, occ_b):
    """(-1)^K between alpha-block and interleaved amplitudes, shape (n_a, n_b)."""
    L = occ_a.shape[1]
    lower = np.tril(np.ones((L, L), dtype=int), -1)
    return 1 - 2 * ((occ_a @ lower @ occ_b.T) & 1)


def full_index(occ_a, occ_b):
    """Index into the 4^L interleaved vector of each (alpha row, beta row) pair, shape (n_a, n_b)."""
    L = occ_a.shape[1]
    weights = 4 ** np.arange(L - 1, -1, -1)
    return (occ_a @ weights)[:, None] + 2 * (occ_b @ weights)[None, :]


def mps_full_vector(tensors):
    """Dense 4^L vector of an MPS (D_left, 4, D_right) with boundary bonds of dimension 1."""
    v = np.ones((1, 1))
    for A in tensors:
        A = np.asarray(A)
        v = np.einsum("xa,apb->xpb", v, A).reshape(-1, A.shape[2])
    return v[:, 0]


def sector_from_full(vector, L, nup, ndn):
    """Alpha-block sector amplitudes of an interleaved 4^L vector."""
    _, occ_a = sector_basis(L, nup)
    _, occ_b = sector_basis(L, ndn)
    return reorder_sign(occ_a, occ_b) * np.asarray(vector)[full_index(occ_a, occ_b)]


def full_from_sector(psi, L, nup, ndn):
    """Interleaved 4^L vector of alpha-block sector amplitudes (zero outside the sector)."""
    _, occ_a = sector_basis(L, nup)
    _, occ_b = sector_basis(L, ndn)
    vector = np.zeros(4**L)
    vector[full_index(occ_a, occ_b)] = reorder_sign(occ_a, occ_b) * psi
    return vector


def mps_sector_amplitudes(tensors, nup, ndn):
    return sector_from_full(mps_full_vector(tensors), len(tensors), nup, ndn)


def sd_amplitudes(Ca, Cb):
    """det(Ca[A]) det(Cb[B]) for every (A, B): the alpha-block amplitudes of SD(Ca, Cb)."""
    L = Ca.shape[0]
    rows_a, _ = sector_basis(L, Ca.shape[1])
    rows_b, _ = sector_basis(L, Cb.shape[1])
    da = np.array([np.linalg.det(Ca[r]) for r in rows_a])
    db = np.array([np.linalg.det(Cb[r]) for r in rows_b])
    return np.outer(da, db)


def ghf_amplitudes(C, nup, ndn):
    """Alpha-block (nup, ndn) amplitudes of the GHF determinant C (2L x N, rows 0..L-1 spin up)."""
    L = C.shape[0] // 2
    rows_a, _ = sector_basis(L, nup)
    rows_b, _ = sector_basis(L, ndn)
    amp = np.zeros((len(rows_a), len(rows_b)))
    for a, ra in enumerate(rows_a):
        for b, rb in enumerate(rows_b):
            amp[a, b] = np.linalg.det(C[np.concatenate([ra, rb + L])])
    return amp


def ghf_full_vector(C):
    """Interleaved 4^L vector of the GHF determinant C (all sectors with N = C.shape[1])."""
    L, N = C.shape[0] // 2, C.shape[1]
    vector = np.zeros(4**L)
    for nup in range(max(0, N - L), min(N, L) + 1):
        vector += full_from_sector(ghf_amplitudes(C, nup, N - nup), L, nup, N - nup)
    return vector


def _species_operators(L, n, h1):
    """One-species hopping matrix T[c', c] = sum_ij h_ij <c'|c+_i c_j|c> and the c+_i c_j blocks."""
    rows, occ = sector_basis(L, n)
    index = {tuple(r): k for k, r in enumerate(rows)}
    ops = np.zeros((L, L, len(rows), len(rows)))
    for k, o in enumerate(occ):
        for j in range(L):
            if not o[j]:
                continue
            for i in range(L):
                if i != j and o[i]:
                    continue
                if i == j:
                    ops[i, i, k, k] += 1.0
                    continue
                lo, hi = min(i, j), max(i, j)
                sign = (-1) ** int(o[lo + 1 : hi].sum())
                new = o.copy()
                new[j], new[i] = 0, 1
                ops[i, j, index[tuple(np.flatnonzero(new))], k] += sign
    T = np.einsum("ij,ijxy->xy", np.asarray(h1, dtype=float), ops)
    return T, ops


def hubbard_sector_hamiltonian(h1, u, nup, ndn):
    """Dense Hubbard Hamiltonian in the (nup, ndn) sector, index a * n_b + b (alpha-block)."""
    h1 = np.asarray(h1, dtype=float)
    L = len(h1)
    Ta, _ = _species_operators(L, nup, h1)
    Tb, _ = _species_operators(L, ndn, h1)
    _, occ_a = sector_basis(L, nup)
    _, occ_b = sector_basis(L, ndn)
    double = (occ_a @ occ_b.T).ravel()
    return np.kron(Ta, np.eye(len(occ_b))) + np.kron(np.eye(len(occ_a)), Tb) + u * np.diag(double)


def ground_state(h1, u, nup, ndn):
    """(E0, psi) of the sector Hamiltonian; psi is (n_a, n_b), normalised, sign fixed (max > 0)."""
    H = hubbard_sector_hamiltonian(h1, u, nup, ndn)
    if len(H) <= 2000:
        energies, vectors = np.linalg.eigh(H)
        e0, v = energies[0], vectors[:, 0]
    else:
        energies, vectors = scipy.sparse.linalg.eigsh(scipy.sparse.csr_matrix(H), k=1, which="SA")
        e0, v = energies[0], vectors[:, 0]
    v = v * np.sign(v[np.argmax(np.abs(v))])
    L = len(h1)
    n_b = len(sector_basis(L, ndn)[0])
    return float(e0), v.reshape(-1, n_b)


def hubbard_full_matrix(h1, u):
    """Sparse Hubbard Hamiltonian on the full 4^L interleaved Fock space (all sectors)."""
    h1 = np.asarray(h1, dtype=float)
    L = len(h1)
    n_modes = 2 * L
    dim = 4**L
    rows, cols, vals = [], [], []

    def occupation(state):
        # local digit of site s is the base-4 digit with site 0 most significant
        digits = [(state // 4 ** (L - 1 - s)) % 4 for s in range(L)]
        modes = np.zeros(n_modes, dtype=int)
        for s, p in enumerate(digits):
            modes[2 * s] = p % 2
            modes[2 * s + 1] = p // 2
        return modes

    def encode(modes):
        return int(sum((modes[2 * s] + 2 * modes[2 * s + 1]) * 4 ** (L - 1 - s) for s in range(L)))

    for state in range(dim):
        m = occupation(state)
        diag = u * sum(m[2 * s] * m[2 * s + 1] for s in range(L))
        diag += sum(h1[s, s] * (m[2 * s] + m[2 * s + 1]) for s in range(L))
        if diag:
            rows.append(state)
            cols.append(state)
            vals.append(diag)
        for spin in (0, 1):
            for i in range(L):
                for j in range(L):
                    if i == j or not h1[i, j]:
                        continue
                    mi, mj = 2 * i + spin, 2 * j + spin
                    if not m[mj] or m[mi]:
                        continue
                    lo, hi = min(mi, mj), max(mi, mj)
                    sign = (-1) ** int(m[lo + 1 : hi].sum())
                    new = m.copy()
                    new[mj], new[mi] = 0, 1
                    rows.append(encode(new))
                    cols.append(state)
                    vals.append(sign * h1[i, j])
    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def apply_hubbard_full(h1, u, vector):
    return hubbard_full_matrix(h1, u) @ np.asarray(vector)


def exact_mps_from_sector_state(psi, L, nup, ndn, tolerance=1.0e-14):
    """Exact charge-labelled d=4 MPS of alpha-block sector amplitudes psi (n_a x n_b).

    The interleave sign is applied first, then a left-to-right SVD sweep factors every bond
    charge sector separately, so each bond index carries an exact (N_up, N_dn) label.
    Returns (tensors, charges).
    """
    vector = full_from_sector(psi, L, nup, ndn)
    physical = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    M = vector.reshape(1, -1)
    left_labels = np.zeros((1, 2), dtype=int)
    tensors: list[np.ndarray] = []
    charges: list[np.ndarray] = [left_labels]
    for s in range(L - 1):
        D = M.shape[0]
        R = M.reshape(D * 4, -1)
        row_labels = (left_labels[:, None, :] + physical[None]).reshape(-1, 2)
        scale = np.linalg.norm(R)
        columns_u, labels, remainder = [], [], []
        for label in sorted(set(map(tuple, row_labels.tolist()))):
            rows = np.flatnonzero(np.all(row_labels == label, axis=1))
            u, sv, vh = np.linalg.svd(R[rows], full_matrices=False)
            keep = sv > tolerance * scale
            for k in np.flatnonzero(keep):
                column = np.zeros(D * 4)
                column[rows] = u[:, k]
                columns_u.append(column)
                labels.append(label)
                remainder.append(sv[k] * vh[k])
        U = np.stack(columns_u, axis=1)
        tensors.append(U.reshape(D, 4, -1))
        left_labels = np.asarray(labels, dtype=int)
        charges.append(left_labels)
        M = np.stack(remainder)
    last = M.reshape(M.shape[0], 4, 1).copy()
    # entries that would break the final label are rounding noise of the SVD: zero them exactly
    allowed = np.all(left_labels[:, None, :] + physical[None] == np.array([nup, ndn]), axis=-1)
    last[~allowed] = 0.0
    tensors.append(last)
    charges.append(np.array([[nup, ndn]]))
    return tensors, tuple(charges)


def dense_rdm1(psi, L, nup, ndn):
    """(2, L, L) spin-resolved <c+_i,s c_j,s> of normalised alpha-block sector amplitudes."""
    psi = psi / np.linalg.norm(psi)
    _, ops_a = _species_operators(L, nup, np.zeros((L, L)))
    _, ops_b = _species_operators(L, ndn, np.zeros((L, L)))
    gamma_a = np.einsum("ab,ijac,cb->ij", psi, ops_a, psi)
    gamma_b = np.einsum("ab,ijbc,ac->ij", psi, ops_b, psi)
    return np.stack([gamma_a, gamma_b])


def overlap_with_sd(amplitudes, Wa, Wb):
    """<trial|SD(Wa, Wb)> for real alpha-block trial amplitudes."""
    return float(np.sum(amplitudes * sd_amplitudes(Wa, Wb)))


def local_energy(amplitudes, H, Wa, Wb):
    """<trial|H|SD> / <trial|SD> with the sector Hamiltonian H (hubbard_sector_hamiltonian)."""
    w = sd_amplitudes(Wa, Wb).ravel()
    t = amplitudes.ravel()
    return float(t @ (H @ w)) / float(t @ w)


def staggered_determinant(h1, nup, ndn, field=0.7):
    """Spin-dependent mean-field-like orbitals: lowest eigenvectors of h1 -+ field*(-1)^i."""
    h1 = np.asarray(h1, dtype=float)
    stag = np.diag((-1.0) ** np.arange(len(h1)))
    Ca = np.linalg.eigh(h1 + field * stag)[1][:, :nup]
    Cb = np.linalg.eigh(h1 - field * stag)[1][:, :ndn]
    return Ca, Cb


def random_field_walkers(h1, u, dt, Ca, Cb, n, steps, seed):
    """Determinants propagated by the CPMC propagator with unguided random HS fields, with QR."""
    rng = np.random.default_rng(seed)
    half = scipy.linalg.expm(-0.5 * dt * np.asarray(h1, dtype=float))
    gamma = np.arccosh(np.exp(0.5 * dt * u))
    L = len(h1)
    out = []
    for _ in range(n):
        ca, cb = Ca.copy(), Cb.copy()
        for _ in range(steps):
            field = rng.integers(0, 2, L) * 2 - 1
            ca = np.linalg.qr(half @ (np.exp(gamma * field)[:, None] * (half @ ca)))[0]
            cb = np.linalg.qr(half @ (np.exp(-gamma * field)[:, None] * (half @ cb)))[0]
        out.append((ca, cb))
    return out


def nonorthonormal(C, rng, scale=0.3):
    """C @ (I + scale * randn): same determinant up to a known scalar, not orthonormal."""
    n = C.shape[1]
    return C @ (np.eye(n) + scale * rng.standard_normal((n, n)))
