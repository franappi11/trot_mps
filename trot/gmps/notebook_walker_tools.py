"""Analysis-only contractions for the walker comparison notebooks.

Contract a real determinant bra with a number-labelled d=2 MPS, without an
approximation to the determinant or enumeration of the spatial Fock space.
The auxiliary states are subsets of the determinant's occupied orbitals:
cost is exponential in particle number (2**N), rather than site number.
This is practical for the notebooks' N=7 and N=16, not a production QMC kernel.
Also evaluate full-Hubbard moments of the notebooks' kinetic-eigenstate CAS
trial using analytic density transitions, including excitations outside its CAS.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np


def cas_trial_moments(h1, interaction, core, active_up, active_down, coefficients):
    """Full-Hubbard mean and variance for a real, kinetic-eigenstate CAS trial.

    The normalized spin-Schmidt form is sum_r coefficients[r]
    |core, active_up[:,r]> |core, active_down[:,r]>. Each orbital set is
    orthonormal, and all paired determinants must have the same kinetic
    energy. This restricted helper includes interaction-induced transitions
    outside the CAS; it does not use the projected CAS Hamiltonian's variance.
    """
    h1, core = np.asarray(h1), np.asarray(core)
    active_up, active_down = np.asarray(active_up), np.asarray(active_down)
    coefficients = np.asarray(coefficients)
    if any(np.iscomplexobj(x) for x in (h1, core, active_up, active_down, coefficients)):
        raise TypeError("CAS moment helper supports real orbitals and coefficients only")
    np.testing.assert_allclose(h1, h1.T, atol=1e-12)
    np.testing.assert_allclose(coefficients @ coefficients, 1, atol=1e-12)
    energies = []
    for active in (active_up, active_down):
        orbitals = np.column_stack((core, active))
        np.testing.assert_allclose(orbitals.T @ orbitals, np.eye(orbitals.shape[1]), atol=1e-12)
        np.testing.assert_allclose(h1 @ core, core @ (core.T @ h1 @ core), atol=1e-12)
        levels = np.einsum("ir,ij,jr->r", active, h1, active)
        np.testing.assert_allclose(h1 @ active, active * levels, atol=1e-12)
        energies.append(levels)
    paired = energies[0] + energies[1]
    np.testing.assert_allclose(paired, np.full_like(paired, paired[0]), atol=1e-12)
    kinetic = 2 * np.trace(core.T @ h1 @ core) + paired[0]
    projector = core @ core.T
    rho = np.diag(projector)

    def density_transitions(active):
        count = active.shape[1]
        one = np.zeros((count, count, len(h1)))
        two = np.zeros((count, count, len(h1), len(h1)))
        for r in range(count):
            for s in range(count):
                ar, ass = active[:, r], active[:, s]
                diagonal = ar * ass
                one[r, s] = (r == s) * rho + diagonal
                two[r, s] = (
                    (r == s) * (np.outer(rho, rho) - projector * projector.T)
                    + np.outer(rho, diagonal) + np.outer(diagonal, rho)
                    - projector * np.outer(ass, ar)
                    - projector.T * np.outer(ar, ass)
                )
                np.fill_diagonal(two[r, s], one[r, s])
        return one, two

    a1, a2 = density_transitions(active_up)
    b1, b2 = density_transitions(active_down)
    mean_double = np.einsum("r,s,rsi,rsi", coefficients, coefficients, a1, b1)
    second_double = np.einsum("r,s,rsij,rsij", coefficients, coefficients, a2, b2)
    variance = interaction**2 * (second_double - mean_double**2)
    if variance < -1e-10:
        raise ArithmeticError("Negative full-Hubbard variance")
    return float(kinetic + interaction * mean_double), float(max(0, variance))


@lru_cache(maxsize=4)
def _subset_maps(n):
    masks = tuple(np.array([sum(1 << j for j in c) for c in combinations(range(n), q)],
                           dtype=np.int64) for q in range(n + 1))
    parity = np.array([k.bit_count() % 2 for k in range(1 << n)], dtype=np.int8)
    maps = []
    for q in range(n):
        positions = np.full(1 << n, -1, dtype=np.int32)
        positions[masks[q + 1]] = np.arange(len(masks[q + 1]))
        entries = []
        for j in range(n):
            src = np.flatnonzero((masks[q] & (1 << j)) == 0)
            dst = positions[masks[q][src] | (1 << j)]
            sign = 1 - 2 * parity[masks[q][src] >> (j + 1)]
            entries.append((src, dst, sign))
        maps.append(tuple(entries))
    return tuple(map(len, masks)), tuple(maps)


def determinant_mps_transition(orbitals, tensors, charges, gauge=1.0, *, with_density=True):
    """Return (<D|MPS>, [<D|n_i|MPS>]) including the ket's gauge.

    D has real orthonormal occupied orbitals (L,N); the real MPS tensors have
    shapes (Dl,2,Dr), with charge arrays on all L+1 bonds. Labels are cumulative
    particle counts from the left. Zero-weight padding is allowed. No Schmidt
    truncation or numerical charge projection is performed by this contraction.
    Set with_density=False to return (overlap, None) using only the backward pass.
    """
    D = np.asarray(orbitals)
    if np.iscomplexobj(D) or any(np.iscomplexobj(A) for A in tensors):
        raise TypeError("The notebook contraction supports real states only")
    L, N = D.shape
    if len(tensors) != L or len(charges) != L + 1:
        raise ValueError("Expected one tensor per site and one label array per bond")
    sizes, maps = _subset_maps(N)
    groups = []
    for labels in charges:
        labels = np.asarray(labels).ravel()
        rounded = np.rint(labels).astype(int)
        if not np.allclose(labels, rounded, atol=1e-8, rtol=0):
            raise ValueError("Bond charges must be integers")
        groups.append({q: np.flatnonzero(rounded == q) for q in np.unique(rounded) if 0 <= q <= N})
    blocks = []
    for i, A in enumerate(tensors):
        A = np.asarray(A)
        site = {}
        for q, left in groups[i].items():
            for p in (0, 1):
                right = groups[i + 1].get(q + p)
                if right is not None:
                    M = A[:, p, :][np.ix_(left, right)]
                    if np.any(M):
                        site[q, p] = M
        blocks.append(site)

    def raise_subset(V, q, row):
        out = np.zeros((sizes[q + 1], V.shape[1]))
        for j, (src, dst, sign) in enumerate(maps[q]):
            if row[j] != 0:
                out[dst] += (row[j] * sign[:, None]) * V[src]
        return out

    def lower_subset(V, q, row):
        out = np.zeros((sizes[q], V.shape[1]))
        for j, (src, dst, sign) in enumerate(maps[q]):
            if row[j] != 0:
                out[src] += (row[j] * sign[:, None]) * V[dst]
        return out

    right = [None] * (L + 1)
    right[L] = {N: np.ones((1, 1))}
    for i in range(L - 1, -1, -1):
        env = {}
        for q, left in groups[i].items():
            if N - q > L - i:
                continue
            value = np.zeros((sizes[q], len(left)))
            if (q, 0) in blocks[i] and q in right[i + 1]:
                value += right[i + 1][q] @ blocks[i][q, 0].T
            if q < N and (q, 1) in blocks[i] and q + 1 in right[i + 1]:
                value += lower_subset(right[i + 1][q + 1], q, D[i]) @ blocks[i][q, 1].T
            if np.any(value):
                env[q] = value
        right[i] = env

    overlap = float(right[0].get(0, np.zeros((1, 1)))[0, 0])
    if not with_density:
        return float(gauge) * overlap, None
    density = np.zeros(L)
    forward = {0: np.ones((1, 1))}
    for i in range(L):
        env = {}
        for q, value in forward.items():
            if (q, 0) in blocks[i]:
                part = value @ blocks[i][q, 0]
                env[q] = env.get(q, 0) + part
            if q < N and (q, 1) in blocks[i]:
                part = raise_subset(value @ blocks[i][q, 1], q, D[i])
                env[q + 1] = env.get(q + 1, 0) + part
                if q + 1 in right[i + 1]:
                    density[i] += np.sum(part * right[i + 1][q + 1])
        forward = env
    forward_overlap = float(forward.get(N, np.zeros((1, 1)))[0, 0])
    np.testing.assert_allclose(forward_overlap, overlap, atol=2e-12, rtol=2e-8)
    np.testing.assert_allclose(density.sum(), N * overlap, atol=2e-11, rtol=2e-8)
    return float(gauge) * overlap, float(gauge) * density
