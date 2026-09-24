"""CPMC with a DMRG trial and locally cached Hubbard-Stratonovich updates.

The walkers remain Slater determinants.  Each spin channel is converted to a
charge-labelled MPS only when an overlap is needed.  Required two-spin charge
blocks are formed directly, without materialising the much larger combined
d=4 walker MPS.  During the diagonal HS sweep, one conversion plus cached
left/right environments serves both field proposals at all sites.

Walker truncation happens gate by gate with the orthogonality centre moved onto
each gate first, so every split discards the smallest Schmidt values. Each
truncated charge block costs one eigh of its Gram matrix; single-row or
single-column blocks are factored in closed form.

The implementation is real-valued and assumes the spatial local basis
|0>, |alpha>, |beta>, |alpha beta>, indexed by n_alpha + 2*n_beta.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import scipy.linalg

jax.config.update("jax_enable_x64", True)

from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.symmetry import SZ
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian

from trot import walkers as wk
from trot.core.ops import MeasOps, k_energy
from trot.core.system import System
from trot.driver import make_run_blocks
from trot.stat_utils import blocking_analysis_ratio, reject_outliers
from trot.ham.hubbard import HamHubbard
from trot.prop import blocks
from trot.prop.cpmc import init_prop_state
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx, make_hubbard_cpmc_ops
from trot.prop.types import PropOps, PropState, QmcParams
from trot.trial.auto import make_auto_trial_ops
from trot.trial.uhf import UhfTrial, get_rdm1 as uhf_get_rdm1
from trot.walkers import _qr as qr_with_det


@dataclass(frozen=True)
class Config:
    L: int = 16
    n_up: int = 8
    n_down: int = 8
    hopping: float = 1.0
    interaction: float = 4.0
    trial_chi: int = 64
    dmrg_sweeps: int = 14
    dmrg_seed: int = 0
    # rank_exact is exact for every full-rank walker before bond truncation.
    # adaptive is cheaper but is only guaranteed for the reference determinant.
    # maximal is exact but usually creates more gates than rank_exact.
    orbital_plan: str = "adaptive"  # rank_exact, adaptive, maximal
    occupation_tolerance: float = 1.0e-10
    walker_channel_chi: int | None = 4
    walker_cutoff: float = 0.0
    n_walkers: int = 32
    n_blocks: int = 40
    n_equilibration: int = 15
    n_steps: int = 20
    dt: float = 0.01
    weight_floor: float = 1.0e-8
    seed: int = 1234
    n_chunks: int = 1
    result_json: str = ""
    block_log: str = ""  # if set, append every block's scalars here as it finishes
    tag: str = ""


CFG = Config()
PHYSICAL_CHARGE = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


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


class SectorPlan(NamedTuple):
    sectors: tuple[tuple[np.ndarray, np.ndarray, int], ...]
    middle_charges: np.ndarray
    row_map: np.ndarray
    column_map: np.ndarray


def hopping_matrix(n: int, hopping: float) -> np.ndarray:
    h1 = np.zeros((n, n))
    i = np.arange(n - 1)
    h1[i, i + 1] = h1[i + 1, i] = -hopping
    return h1


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
        order = np.argsort(-flat)
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


def hubbard_mpo(L, hopping, interaction):
    """Open-chain Hubbard MPO in the spatial local basis, virtual dimension six."""
    eye = np.eye(4)
    create_a = np.zeros((4, 4)); create_a[1, 0] = create_a[3, 2] = 1.0
    create_b = np.zeros((4, 4)); create_b[2, 0] = 1.0; create_b[3, 1] = -1.0
    annihilate_a, annihilate_b = create_a.T, create_b.T
    parity_a = np.diag([1.0, -1.0, 1.0, -1.0])
    parity_b = np.diag([1.0, 1.0, -1.0, -1.0])
    double = np.diag([0.0, 0.0, 0.0, 1.0])

    W = np.zeros((L, 6, 4, 4, 6))
    for i in range(L):
        W[i, 0, :, :, 0] = eye
        W[i, 5, :, :, 5] = eye
        W[i, 0, :, :, 5] = interaction * double
        if i < L - 1:
            W[i, 0, :, :, 1] = create_a @ parity_b
            W[i, 0, :, :, 2] = annihilate_a @ parity_b
            W[i, 0, :, :, 3] = parity_a @ create_b
            W[i, 0, :, :, 4] = parity_a @ annihilate_b
        if i:
            W[i, 1, :, :, 5] = -hopping * annihilate_a
            W[i, 2, :, :, 5] = -hopping * create_a
            W[i, 3, :, :, 5] = -hopping * annihilate_b
            W[i, 4, :, :, 5] = -hopping * create_b
    return W


def apply_mpo(W, tensors):
    out = []
    for i, (operator, A) in enumerate(zip(W, tensors)):
        if i == 0:
            operator = operator[:1]
        if i == len(tensors) - 1:
            operator = operator[:, :, :, 5:6]
        T = np.einsum("apqb,cqd->acpbd", operator, np.asarray(A))
        dl, cl, d, dr, cr = T.shape
        out.append(T.reshape(dl*cl, d, dr*cr))
    return out


def compress_mps(tensors, relative_tolerance=1.0e-13):
    """Compress a fixed MPS by a host QR/SVD sweep."""
    tensors = [np.array(A, copy=True) for A in tensors]
    for i in range(len(tensors) - 1):
        Dl, d, Dr = tensors[i].shape
        q, r = np.linalg.qr(tensors[i].reshape(Dl*d, Dr))
        tensors[i] = q.reshape(Dl, d, -1)
        tensors[i + 1] = np.tensordot(r, tensors[i + 1], axes=1)
    for i in range(len(tensors) - 1, 0, -1):
        Dl, d, Dr = tensors[i].shape
        u, s, vh = np.linalg.svd(tensors[i].reshape(Dl, d*Dr), full_matrices=False)
        rank = max(1, int(np.sum(s > relative_tolerance * max(s[0], 1e-300))))
        tensors[i] = vh[:rank].reshape(rank, d, Dr)
        tensors[i - 1] = np.tensordot(tensors[i - 1], u[:, :rank] * s[:rank], axes=1)
    return tensors


def build_dmrg_hamiltonian(cfg: Config):
    h1 = hopping_matrix(cfg.L, cfg.hopping)
    g2 = np.zeros((cfg.L,) * 4)
    i = np.arange(cfg.L)
    g2[i, i, i, i] = cfg.interaction
    fcidump = FCIDUMP(pg="c1", n_sites=cfg.L, n_elec=cfg.n_up + cfg.n_down,
                      twos=cfg.n_up - cfg.n_down, ipg=0, h1e=h1, g2e=g2)
    return Hamiltonian(fcidump, flat=True)


def run_dmrg(hamiltonian, cfg: Config):
    np.random.seed(cfg.dmrg_seed)
    mpo, _ = hamiltonian.build_qc_mpo().compress(cutoff=1.0e-12)
    mps = hamiltonian.build_mps(cfg.trial_chi)
    result = MPE(mps, mpo, mps).dmrg(
        bdims=[cfg.trial_chi] * cfg.dmrg_sweeps,
        noises=[1.0e-5] * min(6, cfg.dmrg_sweeps) + [0.0],
        dav_thrds=[1.0e-10], iprint=-1, n_sweeps=cfg.dmrg_sweeps)
    return mps, float(result.energies[-1])


def spin_occupations(charge):
    return ((int(charge.n) + int(charge.twos)) // 2,
            (int(charge.n) - int(charge.twos)) // 2)


def flat_blocks(mps, site):
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
        left.append(lo); right.append(ro)
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


def _charge_index(labels):
    grouped = {}
    for i, charge in enumerate(map(tuple, np.asarray(labels).tolist())):
        grouped.setdefault(charge, []).append(i)
    return {charge: np.asarray(indices, int) for charge, indices in grouped.items()}


def make_contraction_plan(walker_charges, trial_charges):
    """Prepare dense padded charge blocks for repeated walker/trial contractions."""
    n = len(walker_charges) - 1
    walker_index = [_charge_index(q) for q in walker_charges]
    trial_index = [_charge_index(q) for q in trial_charges]
    shared, walker_pad, trial_pad = [], [], []
    for bond in range(n + 1):
        charges = sorted(set(walker_index[bond]) & set(trial_index[bond]))
        shared.append(charges)
        walker_pad.append(max(map(lambda q: len(walker_index[bond][q]), charges), default=0))
        trial_pad.append(max(map(lambda q: len(trial_index[bond][q]), charges), default=0))

    sites = []
    for site in range(n):
        incoming = {q: i for i, q in enumerate(shared[site])}
        outgoing = {q: i for i, q in enumerate(shared[site + 1])}
        src, dst, rows, columns, masks, physical = [], [], [], [], [], []
        for charge in shared[site]:
            left_indices = walker_index[site][charge]
            for p, delta in enumerate(PHYSICAL_CHARGE):
                next_charge = tuple(np.asarray(charge) + delta)
                if next_charge not in outgoing:
                    continue
                right_indices = walker_index[site + 1][next_charge]
                shape = (walker_pad[site], walker_pad[site + 1])
                r = np.zeros(shape, int); c = np.zeros(shape, int); mask = np.zeros(shape)
                r[:len(left_indices), :len(right_indices)] = left_indices[:, None]
                c[:len(left_indices), :len(right_indices)] = right_indices[None, :]
                mask[:len(left_indices), :len(right_indices)] = 1.0
                src.append(incoming[charge]); dst.append(outgoing[next_charge])
                rows.append(r); columns.append(c); masks.append(mask); physical.append(p)
        sites.append(dict(src=np.asarray(src, int), dst=np.asarray(dst, int),
                          rows=np.stack(rows), columns=np.stack(columns),
                          mask=np.stack(masks), physical=np.asarray(physical, int),
                          n_out=len(shared[site + 1])))
    return dict(sites=tuple(sites), shared=shared, walker_pad=walker_pad, trial_pad=trial_pad,
                walker_index=walker_index, trial_index=trial_index, n=n)


def extract_fixed_blocks(tensors, plan):
    blocks = []
    for site, layout in enumerate(plan["sites"]):
        out = np.zeros((len(layout["src"]), plan["trial_pad"][site], plan["trial_pad"][site + 1]))
        for t, (qin, qout, p) in enumerate(zip(layout["src"], layout["dst"], layout["physical"])):
            charge_in = plan["shared"][site][qin]
            charge_out = plan["shared"][site + 1][qout]
            rows = plan["trial_index"][site][charge_in]
            columns = plan["trial_index"][site + 1][charge_out]
            out[t, :len(rows), :len(columns)] = np.asarray(tensors[site])[np.ix_(rows, [p], columns)][:, 0]
        blocks.append(jnp.asarray(out))
    return tuple(blocks)


def make_channel_block_maps(plan, alpha_charges, beta_charges):
    """Map allowed combined-spin blocks back to the two channel tensors."""
    maps = []
    for site, layout in enumerate(plan["sites"]):
        left_beta = len(beta_charges[site])
        right_beta = len(beta_charges[site + 1])
        physical = layout["physical"]
        beta_left = layout["rows"] % left_beta
        maps.append(dict(
            alpha_left=layout["rows"] // left_beta,
            beta_left=beta_left,
            alpha_right=layout["columns"] // right_beta,
            beta_right=layout["columns"] % right_beta,
            n_alpha=(physical % 2)[:, None, None],
            n_beta=(physical // 2)[:, None, None],
            sign=layout["mask"] * (-1.0) ** (
                (physical % 2)[:, None, None] * beta_charges[site][beta_left]),
        ))
    return tuple(maps)


def extract_channel_blocks(alpha_mps, beta_mps, channel_maps):
    """Build only the charge blocks needed by the overlap, never the dense d=4 MPS."""
    blocks = []
    for Aa, Ab, m in zip(alpha_mps, beta_mps, channel_maps):
        a = Aa[m["alpha_left"], m["n_alpha"], m["alpha_right"]]
        b = Ab[m["beta_left"], m["n_beta"], m["beta_right"]]
        blocks.append(m["sign"] * a * b)
    return tuple(blocks)


def blocked_contract_from_blocks(walker_blocks, trial_blocks, plan):
    env = jnp.ones((1, plan["walker_pad"][0], plan["trial_pad"][0]))
    for wb, tb, layout in zip(walker_blocks, trial_blocks, plan["sites"]):
        incoming = env[layout["src"]]
        temp = jnp.einsum("tij,tik->tjk", wb, incoming)
        contributions = jnp.einsum("tjk,tkl->tjl", temp, tb)
        env = jax.ops.segment_sum(contributions, layout["dst"], num_segments=layout["n_out"])
    return env.reshape(())


def contraction_report(plan):
    exact = dense = padded = 0
    for bond in range(plan["n"] + 1):
        for charge in plan["shared"][bond]:
            exact += len(plan["walker_index"][bond][charge]) * len(plan["trial_index"][bond][charge])
        dense += sum(map(len, plan["walker_index"][bond].values())) * sum(map(len, plan["trial_index"][bond].values()))
        padded += len(plan["shared"][bond]) * plan["walker_pad"][bond] * plan["trial_pad"][bond]
    return dict(dense=dense, exact_blocks=exact, padded_blocks=padded,
                transitions=sum(len(site["src"]) for site in plan["sites"]))


def right_environments(walker_blocks, trial_blocks, plan):
    right = [jnp.ones((1, plan["walker_pad"][-1], plan["trial_pad"][-1]))]
    for site in range(plan["n"] - 1, -1, -1):
        layout = plan["sites"][site]
        wb, tb = walker_blocks[site], trial_blocks[site]
        following = right[-1][layout["dst"]]
        temp = jnp.einsum("tij,tjk->tik", wb, following)
        contributions = jnp.einsum("tik,tlk->til", temp, tb)
        right.append(jax.ops.segment_sum(contributions, layout["src"],
                                         num_segments=len(plan["shared"][site])))
    return tuple(reversed(right))


def constrain_ratio(ratio, weight_floor):
    """trot's cpmc_step rule: zero every overlap ratio at or below the floor.

    This is also the constrained-path condition (a sign change gives ratio <= 0),
    so it cannot be dropped; weight_floor=0 keeps only the constraint.
    """
    return jnp.where(ratio <= weight_floor, 0.0, ratio)


def make_fast_sweep(convert_channels, channel_maps, trial_blocks, plan):
    """Build the one-conversion HS sweep for a fixed trial and static layouts."""
    def sweep(ca, cb, randoms, hs, weight_floor):
        alpha, beta, prefactor = convert_channels(ca, cb)
        walker_blocks = extract_channel_blocks(alpha, beta, channel_maps)
        right = right_environments(walker_blocks, trial_blocks, plan)
        overlap_in = prefactor * right[0][0, 0, 0]

        left = jnp.ones((1, plan["walker_pad"][0], plan["trial_pad"][0]))
        overlap, log_weight = overlap_in, jnp.zeros(())
        nodes = jnp.zeros((), jnp.int64)
        diagonal = jnp.stack((jnp.ones(2), hs[:, 0], hs[:, 1], hs[:, 0]*hs[:, 1]), axis=1)

        for site, (wb, tb, layout) in enumerate(zip(walker_blocks, trial_blocks, plan["sites"])):
            temp = jnp.einsum("tij,tik->tjk", wb, left[layout["src"]])
            local = jnp.einsum("tjk,tkl->tjl", temp, tb)
            by_transition = jnp.einsum("tjl,tjl->t", local, right[site + 1][layout["dst"]])
            marginal = jax.ops.segment_sum(by_transition, layout["physical"], num_segments=4)

            proposed = prefactor * (diagonal @ marginal)
            ratios = constrain_ratio(proposed / overlap, weight_floor)
            nodes += jnp.sum(ratios <= 0.0, dtype=jnp.int64)
            probabilities = 0.5 * ratios
            norm = probabilities.sum() + 1.0e-13
            field = jnp.where(randoms[site] < probabilities[0]/norm, 0, 1)
            chosen_diagonal = diagonal[field]
            overlap = proposed[field]
            log_weight += jnp.log(norm)

            ca = ca.at[site].multiply(hs[field, 0])
            cb = cb.at[site].multiply(hs[field, 1])
            left = jax.ops.segment_sum(chosen_diagonal[layout["physical"]][:, None, None] * local,
                                       layout["dst"], num_segments=layout["n_out"])
        return ca, cb, overlap_in, overlap, jnp.exp(log_weight), nodes
    return sweep


def init_prop_state_typed(**kwargs):
    state = init_prop_state(**kwargs)
    return state._replace(node_encounters=jnp.zeros((), dtype=jnp.int64))


def make_fast_prop_ops(ham_data, walker_kind, overlap_fn, sweep_fn):
    cpmc_ops = make_hubbard_cpmc_ops(ham_data, walker_kind)

    def step(state, *, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ctx):
        key, subkey = jax.random.split(state.rng_key)
        nwalkers = wk.n_walkers(state.walkers)
        randoms = jax.random.uniform(subkey, (nwalkers, cpmc_ops.n_sites()))
        floor, cap = float(params.weight_floor), float(params.weight_cap)

        walkers = cpmc_ops.apply_one_body_half(state.walkers, prop_ctx)
        sweep_many = wk.vmap_chunked(sweep_fn, params.n_chunks, in_axes=(0, 0, 0, None, None))
        ca, cb, overlap_half, overlaps, weight_factor, node_step = sweep_many(
            walkers[0], walkers[1], randoms, prop_ctx.hs_constant, floor)

        ratio = constrain_ratio(jnp.real(overlap_half / state.overlaps), floor)
        nodes = jnp.sum(ratio <= 0.0, dtype=jnp.int64) + jnp.sum(node_step, dtype=jnp.int64)
        weights = state.weights * ratio
        weights = jnp.where(weights > cap, 0.0, weights) * weight_factor
        walkers = (ca, cb)

        walkers = cpmc_ops.apply_one_body_half(walkers, prop_ctx)
        overlap_many = wk.vmap_chunked(overlap_fn, params.n_chunks, in_axes=(0, None))
        overlaps_new = jnp.real(overlap_many(walkers, trial_data))
        ratio = constrain_ratio(jnp.real(overlaps_new / overlaps), floor)
        nodes += jnp.sum(ratio <= 0.0, dtype=jnp.int64)
        weights *= ratio
        weights = jnp.where(weights > cap, 0.0, weights)

        weights *= jnp.exp(prop_ctx.dt * state.pop_control_ene_shift)
        weights = jnp.where(weights > cap, 0.0, weights)
        average = jnp.clip(jnp.mean(weights), min=1.0e-300)
        shift = state.e_estimate - params.pop_control_damping * jnp.log(average) / prop_ctx.dt
        return PropState(walkers, weights, overlaps_new, key, shift,
                         state.e_estimate, state.node_encounters + nodes)

    return PropOps(init_prop_state=init_prop_state_typed,
                   build_prop_ctx=lambda h, _trial, p: _build_prop_ctx(h, p.dt),
                   step=step)


class WalkerOps(NamedTuple):
    convert: Callable
    overlap: Callable
    energy: Callable | None
    sweep: Callable
    walker_charges: tuple[np.ndarray, ...]
    overlap_plan: dict


def make_walker_ops(Ca, Cb, plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, Htrial=None):
    """Walker conversion, overlap, local energy and fast sweep against a fixed trial.

    The reference determinant (Ca, Cb) fixes the static walker bond labels.
    """
    def convert_channels(ca, cb):
        """Orthonormalize an SD walker and convert each spin channel to an MPS."""
        qa, det_ra = qr_with_det(ca)
        qb, det_rb = qr_with_det(cb)
        alpha, qa_charge, gauge_a = channel_mps(qa, plan_a, bond_a)
        beta, qb_charge, gauge_b = channel_mps(qb, plan_b, bond_b)
        prefactor = det_ra * det_rb * gauge_a * gauge_b
        return alpha, qa_charge, beta, qb_charge, prefactor

    _, qa_charge, _, qb_charge, _ = convert_channels(jnp.asarray(Ca), jnp.asarray(Cb))
    walker_charges = combined_charges(qa_charge, qb_charge)
    overlap_plan = make_contraction_plan(walker_charges, trial_charges)
    channel_maps = make_channel_block_maps(overlap_plan, qa_charge, qb_charge)
    trial_blocks = extract_fixed_blocks(trial_np, overlap_plan)
    trial = tuple(jnp.asarray(A) for A in trial_np)

    def overlap_fn(walker, _trial_data=None):
        ca, cb = walker
        alpha_mps, _, beta_mps, _, prefactor = convert_channels(ca, cb)
        blocks_ = extract_channel_blocks(alpha_mps, beta_mps, channel_maps)
        return prefactor * blocked_contract_from_blocks(blocks_, trial_blocks, overlap_plan)

    def energy_fn(walker, _ham=None, _ctx=None, _trial_data=None):
        ca, cb = walker
        alpha, qa, beta, qb, _ = convert_channels(ca, cb)
        tensors, _ = combine_channels(alpha, qa, beta, qb)
        return contract_real(tensors, Htrial) / contract_real(tensors, trial)

    def convert_for_sweep(ca, cb):
        alpha, _, beta, _, prefactor = convert_channels(ca, cb)
        return alpha, beta, prefactor

    sweep_fn = make_fast_sweep(convert_for_sweep, channel_maps, trial_blocks, overlap_plan)
    return WalkerOps(convert_channels, overlap_fn, None if Htrial is None else energy_fn,
                     sweep_fn, walker_charges, overlap_plan)


def make_block_logger(path, n_equilibration, tag="", base_block_fn=blocks.block):
    """Wrap a block function so each block's scalars are appended to a JSONL file
    the moment the block finishes: raw values, before trot's outlier rejection,
    so partial runs survive and every block keeps its index."""
    counter = iter(range(1 << 62))
    start = time.perf_counter()
    result_spec = jax.ShapeDtypeStruct((), jnp.int32)

    def write(energy, weight, e_estimate, nodes):
        block = next(counter)
        record = dict(tag=tag, block=block,
                      phase="equilibration" if block < n_equilibration else "sampling",
                      energy=float(energy), weight=float(weight), e_estimate=float(e_estimate),
                      node_encounters=int(nodes), seconds=time.perf_counter() - start)
        with Path(path).open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return np.int32(0)

    def block_fn(state, **kwargs):
        state, obs = base_block_fn(state, **kwargs)
        jax.experimental.io_callback(write, result_spec, obs.scalars["energy"], obs.scalars["weight"],
                                     state.e_estimate, state.node_encounters, ordered=True)
        return state, obs

    return block_fn


def run_qmc_fixed_chunks(*, sys, params, ham_data, trial_data, meas_ops, trial_ops, prop_ops, block_fn):
    """trot's run_qmc with one chunk size for both phases, gcd(n_eql, n_blocks),
    so the jitted block scan compiles once instead of once per chunk size.
    Same initialisation, block function, outlier rejection and blocking analysis.
    Returns (mean, stderr, block energies, block weights), all blocks unfiltered."""
    prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    state = prop_ops.init_prop_state(sys=sys, ham_data=ham_data, trial_ops=trial_ops,
                                     trial_data=trial_data, meas_ops=meas_ops, params=params)
    run_blocks = make_run_blocks(block_fn=block_fn, sys=sys, params=params, trial_ops=trial_ops,
                                 meas_ops=meas_ops, prop_ops=prop_ops)
    n_eql, total = params.n_eql_blocks, params.n_eql_blocks + params.n_blocks
    chunk = math.gcd(n_eql, params.n_blocks)
    energies, weights, start = [], [], time.perf_counter()
    for done in range(chunk, total + 1, chunk):
        state, scalars, _ = run_blocks(state, ham_data=ham_data, trial_data=trial_data,
                                       meas_ctx=meas_ctx, prop_ctx=prop_ctx, n_blocks=chunk)
        e, w = np.asarray(scalars["energy"]), np.asarray(scalars["weight"])
        energies.extend(e.tolist())
        weights.extend(w.tolist())
        print(f"[{'eql' if done <= n_eql else 'blk'} {done:4d}/{total}]  E_chunk {np.sum(e * w) / np.sum(w):14.10f}"
              f"  W {w.mean():12.6e}  nodes {int(state.node_encounters):10d}"
              f"  t {time.perf_counter() - start:8.1f} s", flush=True)

    sampled = np.column_stack((energies[n_eql:], weights[n_eql:]))
    clean, _ = reject_outliers(sampled, obs=0)
    print(f"\nRejected {len(sampled) - len(clean)} outlier blocks.\n\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(np.asarray(clean[:, 0]), np.asarray(clean[:, 1]), print_q=True)
    return stats["mu"], stats["se_star"], np.asarray(energies), np.asarray(weights)


def save_result(path, record):
    if not path:
        return
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def main(cfg=CFG):

    h1 = hopping_matrix(cfg.L, cfg.hopping)
    ham = HamHubbard(h1=jnp.asarray(h1), u=cfg.interaction)
    system = System(norb=cfg.L, nelec=(cfg.n_up, cfg.n_down), walker_kind="unrestricted")
    orbital_energies, orbitals = np.linalg.eigh(h1)
    Ca, Cb = orbitals[:, :cfg.n_up].copy(), orbitals[:, :cfg.n_down].copy()
    initial_energy = (orbital_energies[:cfg.n_up].sum() + orbital_energies[:cfg.n_down].sum()
                      + cfg.interaction * sum((Ca[i] @ Ca[i]) * (Cb[i] @ Cb[i]) for i in range(cfg.L)))
    trial_data = UhfTrial(mo_coeff_a=jnp.asarray(Ca), mo_coeff_b=jnp.asarray(Cb))
    print(f"L={cfg.L} ({cfg.n_up},{cfg.n_down}), U={cfg.interaction}, initial determinant E={initial_energy:.12f}")

    hamiltonian = build_dmrg_hamiltonian(cfg)
    dmrg_mps, dmrg_energy = run_dmrg(hamiltonian, cfg)
    trial_np, trial_charges = densify_with_charges(dmrg_mps, cfg.L)
    trial = tuple(jnp.asarray(A) for A in trial_np)
    np.testing.assert_allclose(float(contract_real(trial, trial)), 1.0, atol=1e-10)

    Htrial_np = compress_mps(apply_mpo(hubbard_mpo(cfg.L, cfg.hopping, cfg.interaction), trial_np))
    Htrial = tuple(jnp.asarray(A) for A in Htrial_np)
    trial_energy = float(contract_real(Htrial, trial) / contract_real(trial, trial))
    print(f"DMRG Davidson energy={dmrg_energy:.12f}; dense-MPS expectation={trial_energy:.12f}")
    print("trial bonds:", [A.shape[0] for A in trial] + [trial[-1].shape[-1]])
    print("H|trial> compressed bonds:", [A.shape[0] for A in Htrial] + [Htrial[-1].shape[-1]])

    plan_a = make_orbital_plan(Ca, cfg.orbital_plan, cfg.occupation_tolerance)
    plan_b = make_orbital_plan(Cb, cfg.orbital_plan, cfg.occupation_tolerance)

    bond_a = bond_b = None

    if cfg.walker_channel_chi is not None or cfg.walker_cutoff:
        bond_a = plan_bonds(Ca, plan_a, cfg.walker_channel_chi, cfg.walker_cutoff)
        bond_b = plan_bonds(Cb, plan_b, cfg.walker_channel_chi, cfg.walker_cutoff)
        print(f"walker truncation reference discarded weights: {bond_a.reference_discarded_weight:.3e}, "
              f"{bond_b.reference_discarded_weight:.3e}")

    ops = make_walker_ops(Ca, Cb, plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, Htrial)
    report = contraction_report(ops.overlap_plan)
    print("walker bonds:", [len(q) for q in ops.walker_charges])
    print("overlap environment entries:", report,
          f"dense/padded={report['dense']/report['padded_blocks']:.2f}x")

    prop_ops = make_fast_prop_ops(ham, system.walker_kind, ops.overlap, ops.sweep)

    params = QmcParams(dt=cfg.dt, n_walkers=cfg.n_walkers, n_prop_steps=cfg.n_steps,
                       n_blocks=cfg.n_blocks, n_eql_blocks=cfg.n_equilibration,
                       weight_floor=cfg.weight_floor, seed=cfg.seed, n_chunks=cfg.n_chunks)
    trial_ops = make_auto_trial_ops(system, overlap_u=ops.overlap, get_rdm1=uhf_get_rdm1)
    measurement = MeasOps(overlap=ops.overlap, kernels={k_energy: ops.energy})

    block_fn = blocks.block
    if cfg.block_log:
        block_fn = make_block_logger(cfg.block_log, cfg.n_equilibration, cfg.tag)

    start = time.perf_counter()
    mean, error, block_energies, block_weights = run_qmc_fixed_chunks(
        sys=system, params=params, ham_data=ham, trial_data=trial_data,
        meas_ops=measurement, trial_ops=trial_ops, prop_ops=prop_ops,
        block_fn=block_fn)
    elapsed = time.perf_counter() - start

    scalar = lambda x: None if x is None else float(x)
    print(f"CPMC energy = {scalar(mean)} +/- {scalar(error)}; elapsed={elapsed:.1f} s")
    record = asdict(cfg)
    record.update(
        kind="mps", initial_energy=float(initial_energy), dmrg_energy=dmrg_energy,
        trial_energy=trial_energy, cpmc_energy=scalar(mean), cpmc_error=scalar(error),
        seconds=elapsed, walker_d4_chi=max(map(len, ops.walker_charges)),
        gates=int(plan_a.block_sizes.sum() - len(plan_a.block_sizes)))
    save_result(cfg.result_json, record)


if __name__ == "__main__":
    main()
