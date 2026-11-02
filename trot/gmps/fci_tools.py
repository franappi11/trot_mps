"""Exact (FCI) states of the Hubbard model in pyscf's CI format, and the tools that compare MPS trials with them.

A CI matrix ci[a, b] holds the amplitude of the determinant whose up electrons fill pyscf string a and whose down
electrons fill string b (pyscf.fci.cistring.make_strings order: bit k is site k). The amplitudes are those of
prod_{i in A} c+_{i up} prod_{j in B} c+_{j dn} |0> with increasing sites (the alpha-block convention of
tests/helpers/hubbard_fock.py), which is pyscf's own convention up to one sign for the whole (N_up, N_dn) sector, so
pyscf's energies and density matrices of these matrices are exact, and overlaps with pyscf vectors are exact up to
that sign. MPS amplitudes (trot.trial.mps: local index n_up + 2 n_dn, modes interleaved site by site, up first)
differ by (-1)^K, K = #{(i in A, j in B): j < i}.

Sites are those of h1; on an Lx x Ly lattice site x * Ly + y (trot.ham.hubbard.square_hopping_matrix).

* mps_to_fci: the (N_up, N_dn) amplitudes of any real MPS (also a spin-rotated one with particle-number labels),
  from the two halves of the chain contracted separately and multiplied sector by sector.
* hubbard_energy: <H> of a CI matrix (pyscf's one-body contraction and the diagonal U term).
* square_point_group, permute_ci, point_group_weights: the symmetry operations of the open lattice (C4v for a
  square, C2v for Lx != Ly, a reflection for a chain) as site permutations, their action on a CI matrix, and the
  weight of each irrep.
* schmidt_values, schmidt_summary: Schmidt coefficients across a cut of the site order, sector by sector, and the
  entropy, sector weights and the smallest discarded weight possible at each bond dimension.
* diagonal_correlations, hole_correlations, pair_correlations, d_wave_correlations, distance_average: density,
  spin, hole and singlet-pair correlations.
* sign_agreement: the weight of the configurations on which two states agree in sign.
"""

from __future__ import annotations

from math import comb

import numpy as np


def _popcount(x):
    x = np.asarray(x, dtype=np.int64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    count = np.zeros(x.shape, dtype=np.int64)
    for k in range(int(x.max(initial=0)).bit_length()):
        count += (x >> k) & 1
    return count


def strings(norb, nelec):
    """pyscf's strings of nelec electrons on norb sites: (bit strings, 0/1 occupations (n_strings, norb), and the
    address of every bit string as an array of 2^norb entries, -1 for bit strings with another electron count)."""
    from pyscf.fci import cistring

    if norb > 26:
        raise ValueError("the address table has 2^norb entries: norb <= 26")
    bits = np.asarray(cistring.make_strings(range(norb), int(nelec)), dtype=np.int64)
    occupations = ((bits[:, None] >> np.arange(norb)) & 1).astype(np.int8)
    address = np.full(1 << norb, -1, dtype=np.int64)
    address[bits] = np.arange(len(bits))
    return bits, occupations, address


def _block_amplitudes(tensors, sites, at_left_end):
    """Amplitudes of every configuration of the consecutive sites, with the bond to the rest of the chain left open:
    (amplitudes (4^len(sites), D), up bits, down bits). at_left_end: the sites start the chain (bond on their right),
    else they end it (bond on their left)."""
    amplitudes = np.ones((1, 1))
    up = np.zeros(1, dtype=np.int64)
    dn = np.zeros(1, dtype=np.int64)
    for k in (sites if at_left_end else sites[::-1]):
        A = np.asarray(tensors[k], dtype=float)
        if at_left_end:
            parts = [amplitudes @ A[:, p, :] for p in range(4)]
        else:
            parts = [amplitudes @ A[:, p, :].T for p in range(4)]
        amplitudes = np.concatenate(parts, axis=0)
        up = np.concatenate([up | ((p & 1) << k) for p in range(4)])
        dn = np.concatenate([dn | ((p >> 1) << k) for p in range(4)])
    return amplitudes, up, dn


def _crossings(up, dn, sites):
    """K = #{(i in A, j in B): j < i} inside a block of sites, for every configuration (up, dn bits)."""
    count = np.zeros(len(up), dtype=np.int64)
    for k in sites:
        count += ((up >> k) & 1) * _popcount(dn & ((1 << k) - 1))
    return count


def mps_to_fci(tensors, nelec):
    """The (N_up, N_dn) amplitudes of a real d=4 MPS (L sites, boundary bonds 1) as a pyscf CI matrix.

    Not normalised: a state outside one sector (e.g. a spin-rotated trial) keeps only this sector's part. The
    left half (sites 0..L/2-1) and the right half are contracted for all their 4^(L/2) configurations; each sector
    (n_up_left, n_dn_left) is then one matrix product. The interleaved-to-alpha-block sign
    (-1)^(K_left + K_right + |A_right| |B_left|) is applied per configuration.
    """
    n = len(tensors)
    na, nb = (int(x) for x in nelec)
    m = n // 2
    left, up_l, dn_l = _block_amplitudes(tensors, list(range(m)), True)
    right, up_r, dn_r = _block_amplitudes(tensors, list(range(m, n)), False)
    sign_l = 1 - 2 * (_crossings(up_l, dn_l, range(m)) & 1)
    sign_r = 1 - 2 * (_crossings(up_r, dn_r, range(m, n)) & 1)
    na_l, nb_l = _popcount(up_l), _popcount(dn_l)
    na_r, nb_r = _popcount(up_r), _popcount(dn_r)
    address_a, address_b = strings(n, na)[2], strings(n, nb)[2]
    ci = np.zeros((comb(n, na), comb(n, nb)))
    for a_left in range(max(0, na - (n - m)), min(m, na) + 1):
        for b_left in range(max(0, nb - (n - m)), min(m, nb) + 1):
            rows = np.flatnonzero((na_l == a_left) & (nb_l == b_left))
            cols = np.flatnonzero((na_r == na - a_left) & (nb_r == nb - b_left))
            if not len(rows) or not len(cols):
                continue
            block = left[rows] @ right[cols].T
            block *= sign_l[rows][:, None] * sign_r[cols][None, :] * (-1) ** ((na - a_left) * b_left)
            ci[address_a[up_l[rows][:, None] | up_r[cols][None, :]],
               address_b[dn_l[rows][:, None] | dn_r[cols][None, :]]] = block
    return ci


def hubbard_energy(ci, h1, u, nelec):
    """<ci|H|ci>/<ci|ci> of the Hubbard model: pyscf's one-body contraction plus the diagonal U term (cheaper than
    pyscf's general two-electron contraction, which treats U as a dense (pq|rs) tensor)."""
    from pyscf.fci import direct_spin1

    na, nb = (int(x) for x in nelec)
    h1 = np.asarray(h1, dtype=float)
    norb = len(h1)
    ci = np.ascontiguousarray(ci, dtype=float)
    one_body = float(np.vdot(ci, direct_spin1.contract_1e(h1, ci, norb, (na, nb))))
    double = strings(norb, na)[1].astype(float) @ strings(norb, nb)[1].T.astype(float)
    return (one_body + float(u) * float(np.sum(ci * ci * double))) / float(np.vdot(ci, ci))


def square_point_group(Lx, Ly):
    """The symmetry operations of the open Lx x Ly lattice (site x * Ly + y) as site permutations perm (site i goes
    to perm[i]), with their classes and the character table: C4v for Lx == Ly (classes E, C2, C4, sv = the mirrors
    x -> -x and y -> -y, sd = the diagonal mirrors; B1 is even under sv, the d_{x^2-y^2} irrep), C2v for Lx != Ly,
    and identity and reflection for a chain. Returns (ops, classes, table)."""
    x, y = np.divmod(np.arange(Lx * Ly), Ly)

    def site(X, Y):
        return X * Ly + Y

    if Lx == 1 or Ly == 1:
        return ({"E": site(x, y), "s": site(Lx - 1 - x, Ly - 1 - y)}, {"E": "E", "s": "s"},
                {"A": {"E": 1, "s": 1}, "B": {"E": 1, "s": -1}})
    ops = {"E": site(x, y), "C2": site(Lx - 1 - x, Ly - 1 - y), "sx": site(Lx - 1 - x, y), "sy": site(x, Ly - 1 - y)}
    if Lx != Ly:
        classes = {name: name for name in ops}
        table = {"A1": {"E": 1, "C2": 1, "sx": 1, "sy": 1}, "A2": {"E": 1, "C2": 1, "sx": -1, "sy": -1},
                 "B1": {"E": 1, "C2": -1, "sx": 1, "sy": -1}, "B2": {"E": 1, "C2": -1, "sx": -1, "sy": 1}}
        return ops, classes, table
    L = Lx
    ops.update({"C4": site(y, L - 1 - x), "C4^3": site(L - 1 - y, x), "sd": site(y, x),
                "sd'": site(L - 1 - y, L - 1 - x)})
    classes = {"E": "E", "C2": "C2", "C4": "C4", "C4^3": "C4", "sx": "sv", "sy": "sv", "sd": "sd", "sd'": "sd"}
    table = {"A1": {"E": 1, "C2": 1, "C4": 1, "sv": 1, "sd": 1}, "A2": {"E": 1, "C2": 1, "C4": 1, "sv": -1, "sd": -1},
             "B1": {"E": 1, "C2": 1, "C4": -1, "sv": 1, "sd": -1}, "B2": {"E": 1, "C2": 1, "C4": -1, "sv": -1, "sd": 1},
             "E": {"E": 2, "C2": -2, "C4": 0, "sv": 0, "sd": 0}}
    return ops, classes, table


def permute_ci(ci, norb, nelec, perm):
    """g|ci> for the site permutation g: c+_{i s} -> c+_{perm[i] s} for both spins, with the sign of reordering
    each spin string."""
    perm = np.asarray(perm, dtype=np.int64)
    targets, signs = [], []
    for n_s in (int(x) for x in nelec):
        bits, occupations, address = strings(norb, n_s)
        if n_s == 0:
            targets.append(address[bits])
            signs.append(np.ones(len(bits), dtype=np.int64))
            continue
        occupied = np.argsort(-occupations, axis=1, kind="stable")[:, :n_s]  # occupied sites, ascending
        mapped = perm[occupied]
        targets.append(address[(np.int64(1) << mapped).sum(axis=1)])
        upper_u, upper_v = np.triu_indices(n_s, 1)
        signs.append(1 - 2 * ((mapped[:, upper_u] > mapped[:, upper_v]).sum(axis=1) & 1))
    out = np.empty_like(ci)
    out[np.ix_(targets[0], targets[1])] = ci * signs[0][:, None] * signs[1][None, :]
    return out


def point_group_weights(ci, norb, nelec, group):
    """(characters <ci|g ci>/<ci|ci> of every operation, weight of every irrep d/|G| sum_g chi(g) <ci|g ci>/<ci|ci>)
    for group = square_point_group(...)."""
    ops, classes, table = group
    norm = float(np.vdot(ci, ci))
    characters = {name: float(np.vdot(ci, permute_ci(ci, norb, nelec, perm))) / norm for name, perm in ops.items()}
    weights = {irrep: row["E"] / len(ops) * sum(row[classes[name]] * characters[name] for name in ops)
               for irrep, row in table.items()}
    return characters, weights


def schmidt_values(ci, norb, nelec, cut):
    """{(n_up_left, n_dn_left): Schmidt coefficients} of the CI state across the cut between sites cut - 1 and cut
    (unnormalised state: the coefficients are scaled with it). The interleave signs are a sign per row and per
    column of each sector block (and one per block), so the singular values of the gathered blocks are the
    Schmidt coefficients."""
    from pyscf.fci import cistring

    na, nb = (int(x) for x in nelec)
    address_a, address_b = strings(norb, na)[2], strings(norb, nb)[2]

    def parts(n_total, n_left):
        left = np.asarray(cistring.make_strings(range(cut), n_left), dtype=np.int64)
        right = np.asarray(cistring.make_strings(range(norb - cut), n_total - n_left), dtype=np.int64) << cut
        return left, right

    values = {}
    for a_left in range(max(0, na - (norb - cut)), min(cut, na) + 1):
        left_a, right_a = parts(na, a_left)
        rows_a = address_a[left_a[:, None] | right_a[None, :]]
        for b_left in range(max(0, nb - (norb - cut)), min(cut, nb) + 1):
            left_b, right_b = parts(nb, b_left)
            rows_b = address_b[left_b[:, None] | right_b[None, :]]
            block = ci[rows_a[:, None, :, None], rows_b[None, :, None, :]]
            block = block.reshape(len(left_a) * len(left_b), len(right_a) * len(right_b))
            values[(a_left, b_left)] = np.linalg.svd(block, compute_uv=False)
    return values


def schmidt_summary(values, chis=(64, 128, 192, 256, 384, 512)):
    """Entropy, rank, weight per left sector (keyed "N_left,2Sz_left") and, for each bond chi, the smallest
    discarded weight a chi-state truncation of this cut can have (the best chi Schmidt states of all sectors)."""
    squares = {key: s ** 2 for key, s in values.items()}
    total = sum(float(w.sum()) for w in squares.values())
    p = np.sort(np.concatenate(list(squares.values())))[::-1] / total
    nonzero = p[p > 0]
    return dict(entropy=float(-np.sum(nonzero * np.log(nonzero))), rank=int(np.sum(p > 1e-14)),
                discarded={int(chi): float(max(0.0, 1.0 - p[:chi].sum())) for chi in chis},
                sectors={f"{a + b},{a - b}": float(w.sum() / total) for (a, b), w in squares.items()})


def diagonal_correlations(ci, norb, nelec):
    """Correlations diagonal in the occupation basis: n_up, n_dn (L), <n_i n_j>, <S^z_i S^z_j> (L x L) and the double
    occupancy <n_i up n_i dn> (L)."""
    na, nb = (int(x) for x in nelec)
    weights = ci * ci
    weights /= weights.sum()
    occ_a = strings(norb, na)[1].astype(float)
    occ_b = strings(norb, nb)[1].astype(float)
    p_a, p_b = weights.sum(axis=1), weights.sum(axis=0)
    aa = occ_a.T @ (p_a[:, None] * occ_a)
    bb = occ_b.T @ (p_b[:, None] * occ_b)
    ab = occ_a.T @ (weights @ occ_b)
    return dict(n_up=occ_a.T @ p_a, n_dn=occ_b.T @ p_b, nn=aa + bb + ab + ab.T, szsz=0.25 * (aa + bb - ab - ab.T),
                double=np.diag(ab).copy())


def hole_correlations(ci, norb, nelec):
    """(<h_i>, <h_i h_j>) for the empty-site projector h_i = (1 - n_i up)(1 - n_i dn)."""
    na, nb = (int(x) for x in nelec)
    weights = ci * ci
    weights /= weights.sum()
    empty_a = 1.0 - strings(norb, na)[1]
    empty_b = 1.0 - strings(norb, nb)[1]
    pairs_a = (empty_a[:, :, None] * empty_a[:, None, :]).reshape(len(empty_a), -1)
    pairs_b = (empty_b[:, :, None] * empty_b[:, None, :]).reshape(len(empty_b), -1)
    hh = np.einsum("ak,ak->k", pairs_a, weights @ pairs_b).reshape(norb, norb)
    return np.diag(hh).copy(), hh


def lattice_bonds(Lx, Ly):
    """Nearest-neighbour bonds (i, j, f) of the open lattice (site x * Ly + y), f = +1 along x and -1 along y (the
    d_{x^2-y^2} form factor)."""
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                bonds.append((i, i + Ly, 1.0))
            if y + 1 < Ly:
                bonds.append((i, i + 1, -1.0))
    return bonds


def pair_correlations(ci, norb, nelec, bonds):
    """P[u, v] = <D+_u D_v> for the singlet pair operators D+_{ij} = (c+_{i up} c+_{j dn} - c+_{i dn} c+_{j up})/sqrt(2)
    on bonds[u] = (i, j, ...), from pyscf's 2-RDM, dm2ab[p, q, r, s] = <p+_up r+_dn s_dn q_up>."""
    from pyscf.fci import direct_spin1

    state = ci / np.sqrt(np.vdot(ci, ci))
    (_, _), (_, dm2ab, _) = direct_spin1.make_rdm12s(state, norb, tuple(int(x) for x in nelec))
    out = np.empty((len(bonds), len(bonds)))
    for u, (i, i2, *_) in enumerate(bonds):
        for v, (j, j2, *_) in enumerate(bonds):
            out[u, v] = 0.5 * (dm2ab[i, j, i2, j2] + dm2ab[i, j2, i2, j] + dm2ab[i2, j, i, j2] + dm2ab[i2, j2, i, j])
    return out


def d_wave_correlations(pair, bonds, norb):
    """P_d[i, j] = <D_d+(i) D_d(j)> with D_d(i) = sum of f_b D_b over the bonds b at site i (pair_correlations)."""
    form = np.zeros((norb, len(bonds)))
    for b, (i, j, f) in enumerate(bonds):
        form[i, b] = form[j, b] = f
    return form @ pair @ form.T


def distance_average(matrix, Lx, Ly, min_distance=0.0):
    """{r: mean of matrix[i, j] over the site pairs at distance r >= min_distance} on the Lx x Ly lattice."""
    x, y = np.divmod(np.arange(Lx * Ly), Ly)
    distance = np.round(np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]), 8)
    return {float(r): float(np.asarray(matrix)[distance == r].mean()) for r in np.unique(distance)
            if r >= min_distance}


def sign_agreement(reference, ci):
    """sum of reference^2 over the configurations where ci has the sign of reference (global sign aligned first),
    over sum of reference^2: 1 for the same sign structure."""
    s = 1.0 if np.vdot(reference, ci) >= 0 else -1.0
    weights = reference * reference
    counted = weights > 1e-24 * weights.max()
    agree = counted & (np.sign(ci) * s == np.sign(reference))
    return float(weights[agree].sum() / weights[counted].sum())
