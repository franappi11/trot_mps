"""Reference implementation of the Gaussian-MPS (Fishman-White) machinery.

A Slater determinant becomes a charge-labelled MPS one spin channel at a time:
an orbital plan fixes a circuit of nearest-neighbour Givens gates, the gates act
on an occupation state with the orthogonality centre moved onto each gate, and
every split is done charge block by charge block, optionally truncated to fixed
ranks. The two channels are then interleaved into the d=4 basis of a pyblock3
DMRG trial, which densify turns into plain arrays. mps_cpmc_new and the scripts
built on it import these functions from here.

Conventions: real orbitals; local basis |0>, |up>, |dn>, |up dn> indexed
n_up + 2*n_dn; fermion operators ordered site by site with up before down; bond
charges counted from the left.

High-level entry points, at the end of the file: channel_to_gmps, sd_to_gmps, densify.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import scipy.linalg

# Precision follows trot: float32 unless trot.config.configure_once() (default) enables float64.

"""
Orbital plan is the fixed block sizes and occupation number for all walkers GMPS conversions, computed from a reference state
"""
class OrbitalPlan(NamedTuple):
    occupation: np.ndarray
    block_sizes: np.ndarray
    references: tuple[np.ndarray, ...]
    exact_for_all_walkers: bool


class BondPlan(NamedTuple):
    kept_per_sector: tuple[tuple[int, ...], ...]
    charges: tuple[np.ndarray, ...]
    max_bond: int
    reference_discarded_weight: float

"""
Sector plan is the fixed shape of the charges that allow more efficient block-diagonal matrices operations.
"""
class SectorPlan(NamedTuple):
    sectors: tuple[tuple[np.ndarray, np.ndarray, int], ...]
    middle_charges: np.ndarray
    row_map: np.ndarray
    column_map: np.ndarray


def _rotate_mode_to_front(orbitals, start, vector):
    """Apply adjacent row rotations and return their (site, angle) description."""
    gates = []
    v = vector.copy()
    for j in range(len(v) - 1, 0, -1):
        theta = np.arctan2(v[j], v[j - 1])
        c, s = np.cos(theta), np.sin(theta)
        v[j - 1], v[j] = c * v[j - 1] + s * v[j], 0.0
        p = start + j - 1
        left, right = orbitals[p].copy(), orbitals[p + 1].copy()
        orbitals[p], orbitals[p + 1] = c * left + s * right, -s * left + c * right
        gates.append((p, theta))
    return gates


def make_orbital_plan(C: np.ndarray, mode="rank_exact", eps=1.0e-10) -> OrbitalPlan:
    """Plan the Gaussian-to-MPS gates on a reference determinant.

    rank_exact uses blocks of size min(N, L-N)+1.  Rank counting guarantees an
    exact empty/occupied mode in every such block for any LxN orthonormal walker.
    adaptive stops when the reference has an approximately pure local mode.
    maximal uses the full remaining block. Here we just want to compute the shapes 
    of the blocks and compute the occupied modes
    """

    C = np.asarray(C, float)
    np.testing.assert_allclose(C.T @ C, np.eye(C.shape[1]), atol=1e-10, rtol=0)
    U = C.copy()
    L, particles = C.shape
    holes = L - particles
    occ = np.empty(L, dtype=int)
    block_sizes, references = [], []

    if mode == "rank_exact":
        isolate_occupied = particles > holes
        n_isolated = particles if isolate_occupied else holes
        fixed_B = min(particles, holes) + 1
    elif mode in ("adaptive", "maximal"):
        isolate_occupied = False  # selected independently at every step
        n_isolated = L - 1
        fixed_B = None
    else:
        raise ValueError("orbital_plan must be 'rank_exact', 'adaptive', or 'maximal'")

    for k in range(L - 1):
        P = U @ U.T
        if mode == "rank_exact" and k >= n_isolated:
            B = 1
        elif mode == "rank_exact":
            B = fixed_B
        elif mode == "maximal":
            B = L - k
        else:
            for B in range(2, L - k + 1):
                values = np.linalg.eigvalsh(P[k:k + B, k:k + B])
                if min(values[0], 1.0 - values[-1]) < eps:
                    break

        values, vectors = np.linalg.eigh(P[k:k + B, k:k + B])
        if mode == "rank_exact":
            occupied = isolate_occupied if k < n_isolated else not isolate_occupied
        else:
            occupied = values[0] > 1.0 - values[-1]
        v = vectors[:, -1 if occupied else 0]
        occ[k] = int(occupied)
        block_sizes.append(B)
        references.append(v.copy())
        _rotate_mode_to_front(U, k, v)

    occ[-1] = int(round(float(U[-1] @ U[-1])))
    if occ.sum() != particles:
        raise AssertionError("particle number lost while planning orbital rotations")
    exact = mode in ("rank_exact", "maximal")
    return OrbitalPlan(occ, np.asarray(block_sizes), tuple(references), exact)


def channel_angles(C, plan: OrbitalPlan, xp=jnp):
    """Recompute numerical rotation angles for the current orthonormal walker following 
    Fishman White paper."""
    rows = [xp.asarray(C)[i] for i in range(C.shape[0])]
    angles = []
    for k, (B, reference) in enumerate(zip(plan.block_sizes, plan.references)):
        B = int(B)
        if B == 1:
            continue
        block = xp.stack(rows[k:k + B])
        if plan.exact_for_all_walkers and not plan.occupation[k]:
            # B > number of occupied orbitals, so the last complete-QR vector
            # lies exactly in null(block.T): it is the required empty mode.
            vectors, _ = xp.linalg.qr(block, mode="complete")
            v = vectors[:, -1]
        else:
            _, vectors = xp.linalg.eigh(block @ block.T)
            v = vectors[:, -1 if plan.occupation[k] else 0]
        ref = xp.asarray(reference)
        v = xp.where(v @ ref < 0, -v, v)  # unlike sign(dot), never zeroes v
        v = [v[j] for j in range(B)]
        for j in range(B - 1, 0, -1):
            theta = xp.arctan2(v[j], v[j - 1])
            c, s = xp.cos(theta), xp.sin(theta)
            v[j - 1] = c * v[j - 1] + s * v[j]
            p = k + j - 1
            rows[p], rows[p + 1] = c * rows[p] + s * rows[p + 1], -s * rows[p] + c * rows[p + 1]
            angles.append((p, theta))
    return angles, rows


def gate_pair(A, B, theta, xp=jnp):
    """Contract adjacent d=2 tensors and apply the number-conserving Givens gate."""
    c, s = xp.cos(theta), xp.sin(theta)
    t00 = A[:, 0, :] @ B[:, 0, :]
    t01 = A[:, 0, :] @ B[:, 1, :]
    t10 = A[:, 1, :] @ B[:, 0, :]
    t11 = A[:, 1, :] @ B[:, 1, :]
    return xp.stack((xp.stack((t00, c*t01 + s*t10), axis=1),
                     xp.stack((-s*t01 + c*t10, t11), axis=1)), axis=1)


def _key(labels):
    return tuple(map(int, labels))


@lru_cache(maxsize=None)
def _block_plan(row_key, column_key, kept_key=None) -> SectorPlan:
    """Static charge blocks and maps of a matrix with charge-labelled rows/columns."""
    row_charge, column_charge = np.asarray(row_key), np.asarray(column_key)
    shared = sorted(set(row_charge) & set(column_charge))

    sectors, middle, row_order, column_order = [], [], [], []
    for s, charge in enumerate(shared):
        rows = np.flatnonzero(row_charge == charge)
        columns = np.flatnonzero(column_charge == charge)
        full_rank = min(len(rows), len(columns))
        rank = full_rank if kept_key is None else int(kept_key[s])
        if rank:
            sectors.append((rows, columns, rank))
            middle.extend([charge] * rank)
            row_order.append(rows)
            column_order.append(columns)

    row_order = np.concatenate(row_order)
    column_order = np.concatenate(column_order)
    row_map = np.full(len(row_charge), len(row_order), dtype=int)
    column_map = np.full(len(column_charge), len(column_order), dtype=int)
    row_map[row_order] = np.arange(len(row_order))
    column_map[column_order] = np.arange(len(column_order))
    return SectorPlan(tuple(sectors), np.asarray(middle, int), row_map, column_map)


def sector_plan(ql, qr, kept=None):
    """Blocks of a two-site tensor reshaped to (left bond, n_p) x (n_p+1, right bond)."""
    row_charge = (np.asarray(ql)[:, None] + np.arange(2)).ravel()
    column_charge = (np.asarray(qr)[None, :] - np.arange(2)[:, None]).ravel()
    kept_key = None if kept is None else _key(kept)
    return _block_plan(_key(row_charge), _key(column_charge), kept_key)


def _assemble(left_blocks, right_blocks, plan, xp=jnp):
    """Scatter per-sector factors back to the full row and column order."""
    block_diag = scipy.linalg.block_diag if xp is np else jax.scipy.linalg.block_diag
    left = block_diag(*left_blocks)
    right = block_diag(*right_blocks)
    left = xp.concatenate((left, xp.zeros((1, left.shape[1]), left.dtype)), axis=0)[plan.row_map]
    right = xp.concatenate((right, xp.zeros((right.shape[0], 1), right.dtype)), axis=1)[:, plan.column_map]
    return left, right


def _shift_centre(tensors, charges, site, step, xp=jnp):
    """Move the orthogonality centre from `site` to `site + step` (step = +1 or -1).

    A charge-blocked QR (step +1) or LQ (step -1) leaves `site` an isometry and
    pushes the remainder onto the neighbour. Exact; a bond sector shrinks only
    where its dimension exceeds the rank its charges allow, which depends on the
    labels alone, so shapes stay static.
    """
    A = tensors[site]
    Dl, _, Dr = A.shape
    ql, qr = charges[site], charges[site + 1]
    if step > 0:
        M = A.reshape(2 * Dl, Dr)
        rows, columns = (ql[:, None] + np.arange(2)).ravel(), qr
    else:  # LQ as the QR of the transpose
        M = A.reshape(Dl, 2 * Dr).T
        rows, columns = (qr[None, :] - np.arange(2)[:, None]).ravel(), ql
    plan = _block_plan(_key(rows), _key(columns))
    left, right = [], []
    for r, c, rank in plan.sectors:
        q, rr = _factor_block(M[np.ix_(r, c)], rank, False, xp)
        left.append(q)
        right.append(rr)
    Q, R = _assemble(left, right, plan, xp)
    if step > 0:
        tensors[site] = Q.reshape(Dl, 2, -1)
        tensors[site + 1] = xp.tensordot(R, tensors[site + 1], axes=1)
        charges[site + 1] = plan.middle_charges
    else:
        tensors[site] = Q.T.reshape(-1, 2, Dr)
        tensors[site - 1] = xp.tensordot(tensors[site - 1], R.T, axes=1)
        charges[site] = plan.middle_charges


def _move_centre(tensors, charges, centre, target, xp=jnp):
    """Walk the orthogonality centre to `target`. centre=None marks the initial
    product state, in which every tensor is already an isometry."""
    if centre is not None:
        while centre > target:
            _shift_centre(tensors, charges, centre, -1, xp)
            centre -= 1
        while centre < target:
            _shift_centre(tensors, charges, centre, +1, xp)
            centre += 1
    return target


def _vector_qr(block, xp=jnp):
    """Reduced QR of a block with a single row or column, in closed form."""
    rows = block.shape[0]
    if rows == 1:
        return xp.ones((1, 1), block.dtype), block
    norm = xp.sqrt(xp.sum(block * block))
    safe = xp.where(norm > 0, norm, 1.0)
    q = xp.where(norm > 0, block / safe, xp.eye(rows, 1, dtype=block.dtype))
    return q, norm.reshape(1, 1)


def _factor_block(block, rank, truncate, xp=jnp):
    """Left isometry and remainder of one charge block, keeping `rank` vectors.

    A single row or column is done in closed form, an untruncated block by QR.
    A truncated block needs only its top-`rank` left singular subspace, taken from
    one eigh of its smaller Gram matrix (M M^T = U S^2 U^T, or M^T M = V S^2 V^T).
    That replaces the QR + eigh(R R^T) of arXiv:2212.09782 with a single LAPACK
    call: same subspace and the same squared-singular-value precision.
    """
    rows, cols = block.shape
    if min(rows, cols) == 1:
        return _vector_qr(block, xp)
    if not truncate:
        q, r = xp.linalg.qr(block, mode="reduced")
        return q[:, :rank], r[:rank]
    if rows <= cols:
        _, U = xp.linalg.eigh(block @ block.T)
        Uk = U[:, ::-1][:, :rank]
        return Uk, Uk.T @ block
    w, V = xp.linalg.eigh(block.T @ block)
    Vk = V[:, ::-1][:, :rank]
    sk = xp.sqrt(xp.maximum(w[::-1][:rank], 0.0))
    return (block @ Vk) / xp.where(sk > 0, sk, 1.0), sk[:, None] * Vk.T


def split_pair(T, ql, qr, kept=None):
    """Charge-preserving split into a left isometry and the new centre; with kept,
    truncate each sector to fixed rank. The truncation keeps the largest Schmidt
    values only because the orthogonality centre has been moved onto the pair."""
    Dl, _, _, Dr = T.shape
    M = T.reshape(2 * Dl, 2 * Dr)
    plan = sector_plan(ql, qr, kept)
    left, right = [], []
    for rows, columns, rank in plan.sectors:
        truncate = rank < min(len(rows), len(columns))
        q, r = _factor_block(M[np.ix_(rows, columns)], rank, truncate)
        left.append(q)
        right.append(r)
    A, B = _assemble(left, right, plan)
    return A.reshape(Dl, 2, -1), B.reshape(-1, 2, Dr), plan.middle_charges


def channel_mps(C, plan: OrbitalPlan, bond_plan: BondPlan | None = None):
    """Convert one orthonormal spin channel to a charge-labelled d=2 MPS.

    The orthogonality centre is moved onto every gate before its split, so a
    truncating split discards the smallest Schmidt values of the current state.
    """

    one_hot = (jnp.array([[[1.0], [0.0]]]), jnp.array([[[0.0], [1.0]]]))
    tensors = [one_hot[int(o)] for o in plan.occupation]
    charges = [np.zeros(1, int)]
    for o in plan.occupation:
        charges.append(charges[-1] + int(o))

    angles, rotated_rows = channel_angles(C, plan)
    centre = None
    for gate_index, (site, theta) in enumerate(reversed(angles)):
        centre = _move_centre(tensors, charges, centre, site)
        kept = None if bond_plan is None else bond_plan.kept_per_sector[gate_index]
        pair = gate_pair(tensors[site], tensors[site + 1], theta)
        tensors[site], tensors[site + 1], charges[site + 1] = split_pair(
            pair, charges[site], charges[site + 2], kept)
        centre = site + 1

    occupied_rows = np.flatnonzero(plan.occupation)
    gauge = jnp.linalg.det(jnp.stack([rotated_rows[i] for i in occupied_rows]))
    return tensors, charges, gauge


def combine_channels(alpha, qa, beta, qb):
    """Interleave alpha/beta channels, including the fermionic reordering sign."""
    tensors = []
    for site, (Aa, Ab) in enumerate(zip(alpha, beta)):
        Dal, _, Dar = Aa.shape
        Dbl, _, Dbr = Ab.shape
        physical = []
        for nb in (0, 1):
            for na in (0, 1):
                sign = (-1.0) ** (na * qb[site])
                physical.append(jnp.einsum("ar,b,bs->abrs", Aa[:, na, :], sign, Ab[:, nb, :]))
        tensors.append(jnp.stack(physical, axis=2).reshape(Dal*Dbl, 4, Dar*Dbr))
    return tensors, combined_charges(qa, qb)


def combined_charges(qa, qb):
    """Bond labels in the alpha-major flattened product basis.
    As an example:bond 4: alpha charges = [0, 1, 1, 1, 1, 2]
        beta  charges = [0, 1, 1, 1, 2, 2]
        combined      = [(0,0), (0,1), (0,1), (0,1), (0,2), (0,2),(1,0),...]   

    """
    charges = [np.zeros((1, 2), int)]
    for a, b in zip(qa[1:], qb[1:]):
        charges.append(np.stack((np.repeat(a, len(b)), np.tile(b, len(a))), axis=1))
    return tuple(charges)


def plan_bonds(C, orbital_plan, chi_max=None, cutoff=0.0) -> BondPlan:
    """Freeze per-sector retained ranks using a NumPy dry run on the reference.

    Follows channel_mps's centre moves exactly, so the singular values seen here
    are Schmidt values and reference_discarded_weight is the norm actually lost.
    """
    #Create product state and its charges
    tensors = [np.eye(2)[int(o)].reshape(1, 2, 1) for o in orbital_plan.occupation]
    charges = [np.zeros(1, int)]
    for o in orbital_plan.occupation:
        charges.append(charges[-1] + int(o))
    #Getting the angles
    angles, _ = channel_angles(np.asarray(C), orbital_plan, xp=np)

    kept_all, discarded, centre = [], 0.0, None
    for site, theta in reversed(angles):
        centre = _move_centre(tensors, charges, centre, site, xp=np)
        pair = gate_pair(tensors[site], tensors[site + 1], theta, xp=np)
        Dl, _, _, Dr = pair.shape
        matrix = pair.reshape(2*Dl, 2*Dr)
        full_plan = sector_plan(charges[site], charges[site + 2])
        decompositions, singular_values = [], []
        for rows, columns, _ in full_plan.sectors:
            u, s, vh = np.linalg.svd(matrix[np.ix_(rows, columns)], full_matrices=False)
            decompositions.append((u, s, vh))
            singular_values.append(s)

        flat = np.concatenate(singular_values)
        order = np.argsort(-flat, kind="stable")  # exact ties: earlier sector first, reproducibly
        count = len(flat) if chi_max is None else min(int(chi_max), len(flat))
        selected = order[:count]
        if cutoff:
            selected = selected[flat[selected] > cutoff * flat[order[0]]]
        keep_mask = np.zeros(len(flat), bool)
        keep_mask[selected] = True
        discarded += float(flat[~keep_mask] @ flat[~keep_mask])

        kept, offset = [], 0
        for s in singular_values:
            kept.append(int(keep_mask[offset:offset + len(s)].sum()))
            offset += len(s)
        kept = tuple(kept)
        kept_all.append(kept)

        truncated = sector_plan(charges[site], charges[site + 2], kept)
        left, right = [], []
        for (u, s, vh), rank in zip(decompositions, kept):
            if rank:
                left.append(u[:, :rank])
                right.append(s[:rank, None] * vh[:rank])
        left, right = _assemble(left, right, truncated, xp=np)
        tensors[site], tensors[site + 1] = left.reshape(Dl, 2, -1), right.reshape(-1, 2, Dr)
        charges[site + 1] = truncated.middle_charges
        centre = site + 1

    return BondPlan(tuple(kept_all), tuple(charges), max(map(len, charges)), discarded)


def contract_real(left_mps, right_mps):
    """Real MPS contraction.  Complex support would require conjugating the left MPS."""
    env = jnp.ones((1, 1))
    for left, right in zip(left_mps, right_mps):
        env = jnp.einsum("ab,apr,bps->rs", env, left, right, optimize=True)
    return env.reshape(())


def spin_occupations(charge):
    return ((int(charge.n) + int(charge.twos)) // 2,
            (int(charge.n) - int(charge.twos)) // 2)


def flat_blocks(mps, site):
    from pyblock3.algebra.symmetry import SZ  # lazy: only densify needs pyblock3

    tensor = mps[site]
    for k in range(tensor.n_blocks):
        labels = tuple(SZ.from_flat(int(x)) for x in tensor.q_labels[k])
        shape = tuple(map(int, tensor.shapes[k]))
        data = np.asarray(tensor.data[tensor.idxs[k]:tensor.idxs[k + 1]]).reshape(shape)
        yield labels, shape, data


def densify_with_charges(mps, L):
    """Convert a pyblock3 MPS to dense tensors and matching (N_alpha,N_beta) labels."""

    key = lambda q: (int(q.n), int(q.twos))
    left, right = [], []
    for site in range(L):
        lo, ro = {}, {}
        for (ql, _, qr), shape, _ in flat_blocks(mps, site):
            lo[key(ql)], ro[key(qr)] = shape[0], shape[2]
        left.append(lo)
        right.append(ro)
    for site in range(L - 1):
        if left[site + 1] != right[site]:
            raise AssertionError(f"bond {site + 1} differs between neighboring tensors")

    bond_sectors = [left[i] for i in range(L)] + [right[-1]]
    offsets, bond_charges = [], []
    for sectors in bond_sectors:
        offset, labels, start = {}, [], 0
        for q, multiplicity in sorted(sectors.items()):
            offset[q] = (start, multiplicity)
            spin_charge = ((q[0] + q[1]) // 2, (q[0] - q[1]) // 2)
            labels.extend([spin_charge] * multiplicity)
            start += multiplicity
        offsets.append((offset, start))
        bond_charges.append(np.asarray(labels, int))

    dense = []
    for site in range(L):
        A = np.zeros((offsets[site][1], 4, offsets[site + 1][1]))
        for (ql, qp, qr), shape, data in flat_blocks(mps, site):
            if shape[1] != 1:
                raise AssertionError("expected one-dimensional physical charge blocks")
            na, nb = spin_occupations(qp)
            ol, dl = offsets[site][0][key(ql)]
            or_, dr = offsets[site + 1][0][key(qr)]
            A[ol:ol + dl, na + 2*nb, or_:or_ + dr] = data[:, 0, :]
        dense.append(A)
    return dense, tuple(bond_charges)


class DenseMps(NamedTuple):
    tensors: list       # NumPy site tensors (D_left, 4, D_right), local index n_up + 2*n_dn
    charges: tuple      # L+1 bond labels, one (N_up, N_dn) row per bond index


class Gmps(NamedTuple):
    tensors: list       # site tensors (D_left, d, D_right): d=2 for one channel, d=4 for both
    charges: tuple      # L+1 bond labels: N per channel, (N_up, N_dn) rows when combined
    prefactor: float    # |SD> = prefactor * |MPS>, exactly when untruncated
    discarded: float    # 1 - <MPS|MPS>: the squared norm truncation removed (0 when exact)


def channel_to_gmps(C, chi=None, cutoff=0.0, mode="adaptive", eps=1.0e-10) -> Gmps:
    """One spin channel, an L x N matrix of real orbitals (any full-rank columns),
    as a d=2 charge-labelled MPS by Fishman-White.

    The gate circuit and the kept count of every charge block are planned on C
    itself, so chi (max bond) and cutoff (relative singular value) truncate gate by
    gate at the Schmidt values of C. Planning runs in NumPy, so this is not
    jit-able; to convert many walkers against one reference, freeze the plans and
    call channel_mps directly.
    """
    if np.iscomplexobj(C):
        raise ValueError("the conversion is real-valued; complex orbitals are not supported")
    C = np.asarray(C, float)
    L, N = C.shape
    if N == 0:  # vacuum: channel_mps has no occupied rows to take its gauge from
        empty = jnp.array([1.0, 0.0]).reshape(1, 2, 1)
        return Gmps([empty] * L, (np.zeros(1, int),) * (L + 1), 1.0, 0.0)

    Q, R = np.linalg.qr(C)
    plan = make_orbital_plan(Q, mode, eps)
    bonds = None if chi is None and not cutoff else plan_bonds(Q, plan, chi, cutoff)
    tensors, charges, gauge = channel_mps(jnp.asarray(Q), plan, bonds)
    discarded = 0.0 if bonds is None else bonds.reference_discarded_weight
    return Gmps(tensors, tuple(charges), float(np.linalg.det(R) * gauge), discarded)


def sd_to_gmps(Ca, Cb, chi=None, cutoff=0.0, mode="adaptive", eps=1.0e-10) -> Gmps:
    """Slater determinant SD(Ca, Cb) as one d=4 MPS in the DMRG trial's basis.

    Each spin channel is converted by channel_to_gmps with the same options (chi is
    the per-channel bond, so the combined bond is at most chi**2) and the two are
    interleaved with the fermionic reordering sign.
    """
    alpha = channel_to_gmps(Ca, chi, cutoff, mode, eps)
    beta = channel_to_gmps(Cb, chi, cutoff, mode, eps)
    tensors, charges = combine_channels(alpha.tensors, alpha.charges, beta.tensors, beta.charges)
    kept = (1.0 - alpha.discarded) * (1.0 - beta.discarded)
    return Gmps(tensors, charges, alpha.prefactor * beta.prefactor, 1.0 - kept)


def densify(mps) -> DenseMps:
    """A pyblock3 SZ MPS (flat or not, e.g. a DMRG trial) as dense site tensors.

    Each bond's charge sectors are laid out contiguously in sorted order and the
    labels say which sector every index belongs to, so the result contracts
    directly with sd_to_gmps output (contract_real) or feeds the charge-blocked
    contraction (mps_cpmc_new.make_contraction_plan).
    """
    if mps.const:
        raise ValueError("MPS.const is an added constant term that dense tensors cannot carry")
    tensors, charges = densify_with_charges(mps.to_flat(), mps.n_sites)
    return DenseMps(tensors, charges)
