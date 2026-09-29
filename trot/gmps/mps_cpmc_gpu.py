"""CPMC with a DMRG trial and SD walkers converted to MPS, built for one GPU.

Same algorithm as mps_cpmc_new.py (which this file copies instead of importing):
the walkers stay Slater determinants, every overlap converts each spin channel
to a charge-labelled d=2 MPS with Fishman-White gates, truncating gate by gate
with the orthogonality centre on the gate, and one conversion plus cached right
environments serves the whole diagonal HS sweep. What changes is how the work is
laid out for a GPU, where throughput comes from large batches and few kernels:

* The gate circuit is compiled once on the host (compile_circuit): every centre
  move and split becomes a static gather into padded batches of charge-sector
  blocks. Each sector is factored by the same method as the original
  (_factor_block: closed form for one row or column, QR when exact, the Gram
  eigh when truncating), but on the device (linalg="batched") all sectors of one
  kind are a single call over walkers x spins x sectors. Both spin channels run
  in one batch whenever they share a gate sequence, each with its own sectors and
  kept ranks (spin_batch).
* Walker-trial contractions never form the d=4 walker. The environment is kept as
  (alpha, beta, trial) blocks per shared (N_alpha, N_beta) label and the three
  legs are contracted one at a time (make_factorized_plan), which costs
  Pa*Pb*Pt*(Pa+Pb+Pt) per transition instead of Pa^2*Pb^2*Pt. Sums over incoming
  transitions are gathers, not atomic scatter-adds, so runs are deterministic.
* The local energy uses a charge-labelled, sector-compressed H|trial>
  (compress_mps_qn) with the same factorized contraction. The dense path
  (energy="dense") is kept for validation only.
* Each CPMC step is a scan over half steps with one conversion call site; the
  HS sweep runs under lax.cond on even half steps. The measurement block rescales
  overlaps by det(R) after orthonormalisation and gathers them after the comb, so
  a block needs one conversion per half step plus one for the energy (trot's block
  needs three more).
* Trial blocks are jit arguments, not HLO constants; walkers are chunked with
  lax.map only as far as the memory model requires (n_chunks=0).

The implementation is real-valued and assumes the spatial local basis
|0>, |alpha>, |beta>, |alpha beta>, indexed by n_alpha + 2*n_beta.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import scipy.linalg
from jax import lax

jax.config.update("jax_enable_x64", True)

try:  # DMRG only; a cached trial (Config.trial_cache) runs without pyblock3
    from pyblock3.algebra.mpe import MPE
    from pyblock3.algebra.symmetry import SZ
    from pyblock3.fcidump import FCIDUMP
    from pyblock3.hamiltonian import Hamiltonian
except ImportError:  # pragma: no cover
    MPE = SZ = FCIDUMP = Hamiltonian = None

from trot import walkers as wk
from trot.core.system import System
from trot.ham.hubbard import HamHubbard
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.types import PropState, QmcParams
from trot.stat_utils import blocking_analysis_ratio, reject_outliers
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
    # Determinant that freezes the static structure: the gate circuit (make_orbital_plan)
    # and the per-sector kept counts (plan_bonds). "natural" is the determinant of the
    # trial's most occupied natural orbitals, "rhf" the free-fermion determinant.
    plan_reference: str = "natural"  # natural, rhf
    # Determinant every walker starts from. "natural" follows trot's convention
    # (natural orbitals of the trial's one-body density matrix).
    walker_start: str = "natural"  # natural, rhf
    n_walkers: int = 1024
    n_blocks: int = 40
    n_equilibration: int = 15
    n_steps: int = 20
    dt: float = 0.01
    weight_floor: float = 1.0e-8
    seed: int = 1234
    # --- GPU layout (none of these change the algorithm) ---
    n_chunks: int = 0  # walker chunks per half step; 0 = the fewest the memory model allows
    mem_fraction: float = 0.75  # share of the device memory the chunker may plan for
    linalg: str = "auto"  # batched (all sectors per op in one call per kind, GPUs), native (the original loop), auto
    walker_qr: str = "auto"  # cholesky (CholeskyQR2), native (Householder, trot's _qr), auto
    energy: str = "blocked"  # blocked (charge-labelled H|trial>), dense (original; small sizes)
    spin_batch: bool = True  # convert both spin channels as one batch when their circuits match
    self_check: bool = True  # check one device conversion against the NumPy original at start-up
    trial_cache: str = ""  # directory for cached DMRG trials (skips DMRG when present)
    compile_cache: str = ""  # JAX persistent compilation cache directory
    result_json: str = ""
    block_log: str = ""  # if set, append every block's scalars here as it finishes
    walker_snapshots: str = ""  # if set, every block's walkers go here (.npz, notebook format; see WalkerSnapshots)
    trial_export: str = ""  # if set, the DMRG trial goes here in the notebooks' format (see export_trial)
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


def one_rdm(tensors):
    """Spin-resolved one-body density matrices <c^dag_i,sigma c_j,sigma> of a real
    d=4 MPS in the interleaved (alpha before beta on each site) ordering."""
    create_a = np.zeros((4, 4)); create_a[1, 0] = create_a[3, 2] = 1.0
    create_b = np.zeros((4, 4)); create_b[2, 0] = 1.0; create_b[3, 1] = -1.0
    parity_a, parity_b = np.diag([1.0, -1.0, 1.0, -1.0]), np.diag([1.0, 1.0, -1.0, -1.0])
    operators = {"a": (create_a @ parity_b, create_a.T, np.diag([0.0, 1.0, 0.0, 1.0])),
                 "b": (parity_a @ create_b, create_b.T, np.diag([0.0, 0.0, 1.0, 1.0]))}
    A = [np.asarray(t) for t in tensors]
    L = len(A)
    site = lambda E, x, O: np.einsum("ab,apc,pq,bqd->cd", E, A[x], O, A[x], optimize=True)
    left = [np.ones((1, 1))]
    for x in range(L):
        left.append(site(left[-1], x, np.eye(4)))
    right = [np.ones((1, 1))]
    for x in range(L - 1, -1, -1):
        right.insert(0, np.einsum("apc,bpd,cd->ab", A[x], A[x], right[0], optimize=True))
    gammas = []
    for create_i, annihilate_j, number in operators.values():
        gamma = np.zeros((L, L))
        for i in range(L):
            gamma[i, i] = np.sum(site(left[i], i, number) * right[i + 1])
            E = site(left[i], i, create_i)
            for j in range(i + 1, L):
                gamma[i, j] = gamma[j, i] = np.sum(site(E, j, annihilate_j) * right[j + 1])
                E = site(E, j, parity_a @ parity_b)  # Jordan-Wigner string between i and j
        gammas.append(gamma / left[-1][0, 0])
    return tuple(gammas)


def natural_orbitals(gamma, n):
    """The n most occupied natural orbitals and all occupations, descending."""
    occupations, vectors = np.linalg.eigh(gamma)
    return vectors[:, ::-1][:, :n].copy(), occupations[::-1]


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


def hubbard_dmrg_mpo(hamiltonian, cfg: Config):
    """Hubbard-chain MPO from its ~6L operator terms (bond dimension ~6).
    build_qc_mpo treats g2 as a general two-electron tensor, which gives a bond
    dimension ~L^2/2 and tens of GB of DMRG environments at L=48, chi=200."""
    # Term encoding of pyblock3's flat builder (as in Hamiltonian.build_complex_qc_mpo):
    # operator index = OP * (0 for c+, 1 for c) + SITE * site + SPIN * spin, -1 pads.
    SPIN, SITE, OP = 1, 2, 16384
    C, D = 0 * OP, 1 * OP
    h1 = hopping_matrix(cfg.L, cfg.hopping)
    values, terms = [], []
    for i, j in zip(*np.nonzero(h1)):
        for s in (0, 1):
            values.append(h1[i, j])
            terms.append([C + i * SITE + s * SPIN, D + j * SITE + s * SPIN, -1, -1])
    for i in range(cfg.L):
        # n_up n_down = c+_up c+_down c_down c_up
        values.append(cfg.interaction)
        terms.append([C + i * SITE, C + i * SITE + SPIN, D + i * SITE + SPIN, D + i * SITE])
    gen = (np.array(values, dtype=np.float64), np.array(terms, dtype=np.int32))
    return hamiltonian.build_mpo(gen, cutoff=1.0e-12)


def run_dmrg(hamiltonian, cfg: Config):
    np.random.seed(cfg.dmrg_seed)
    mpo = hubbard_dmrg_mpo(hamiltonian, cfg)
    mps = hamiltonian.build_mps(cfg.trial_chi)
    # Warm up at a larger bond dimension with stronger noise, then truncate to
    # trial_chi: sweeping at chi=8 from a random start got stuck at L=48,
    # U=8 and 12 (energy per site 10x further from chi=200 than elsewhere).
    chi = cfg.trial_chi
    warm = max(chi, min(4 * chi, 64))
    bdims = [warm] * 4 + [max(chi, warm // 2)] * 2 + [chi] * cfg.dmrg_sweeps
    noises = [1.0e-4] * 4 + [1.0e-5] * 2 + [1.0e-6] * (cfg.dmrg_sweeps - 2) + [0.0] * 2
    result = MPE(mps, mpo, mps).dmrg(
        bdims=bdims, noises=noises, dav_thrds=[1.0e-10], iprint=-1, n_sweeps=len(bdims))
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


def constrain_ratio(ratio, weight_floor):
    """trot's cpmc_step rule: zero every overlap ratio at or below the floor.

    This is also the constrained-path condition (a sign change gives ratio <= 0),
    so it cannot be dropped; weight_floor=0 keeps only the constraint.
    """
    return jnp.where(ratio <= weight_floor, 0.0, ratio)


# ============================================================================
# Host: trial cache, reference conversion, charge-labelled H|trial>
# ============================================================================

def _trial_cache_file(cfg):
    name = (f"L{cfg.L}_n{cfg.n_up}-{cfg.n_down}_t{cfg.hopping:g}_U{cfg.interaction:g}"
            f"_chi{cfg.trial_chi}_sw{cfg.dmrg_sweeps}_seed{cfg.dmrg_seed}.npz")
    return Path(cfg.trial_cache) / name


def load_or_run_trial(cfg):
    """DMRG trial as dense tensors, (N_alpha, N_beta) bond labels and its energy.

    With cfg.trial_cache set the trial is read from, or written to, one npz file per
    (L, filling, t, U, chi, sweeps, seed), so GPU jobs and benchmarks skip DMRG.
    """
    path = _trial_cache_file(cfg) if cfg.trial_cache else None
    if path is not None and path.exists():
        with np.load(path) as data:
            tensors = [np.asarray(data[f"A{i}"]) for i in range(cfg.L)]
            charges = tuple(np.asarray(data[f"q{i}"]) for i in range(cfg.L + 1))
            energy = float(data["energy"])
        print(f"trial loaded from {path}")
        return tensors, charges, energy
    if MPE is None:
        raise ImportError("pyblock3 is needed to run DMRG (or point trial_cache at a cached trial)")
    mps, energy = run_dmrg(build_dmrg_hamiltonian(cfg), cfg)
    tensors, charges = densify_with_charges(mps, cfg.L)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, energy=energy, **{f"A{i}": A for i, A in enumerate(tensors)},
                 **{f"q{i}": q for i, q in enumerate(charges)})
        os.replace(tmp, path)
        print(f"trial saved to {path}")
    return tensors, charges, energy


def channel_mps_host(C, plan: OrbitalPlan, bond_plan: BondPlan | None = None):
    """channel_mps run eagerly in NumPy: the original per-sector QR/eigh algorithm,
    used as an independent reference for the device conversion."""
    tensors = [np.eye(2)[int(o)].reshape(1, 2, 1) for o in plan.occupation]
    charges = [np.zeros(1, int)]
    for o in plan.occupation:
        charges.append(charges[-1] + int(o))
    angles, rows = channel_angles(np.asarray(C, float), plan, xp=np)
    centre = None
    for gate_index, (site, theta) in enumerate(reversed(angles)):
        centre = _move_centre(tensors, charges, centre, site, xp=np)
        kept = None if bond_plan is None else bond_plan.kept_per_sector[gate_index]
        pair = gate_pair(tensors[site], tensors[site + 1], theta, xp=np)
        Dl, _, _, Dr = pair.shape
        matrix = pair.reshape(2 * Dl, 2 * Dr)
        split = sector_plan(charges[site], charges[site + 2], kept)
        left, right = [], []
        for r, c, rank in split.sectors:
            q, rr = _factor_block(matrix[np.ix_(r, c)], rank, rank < min(len(r), len(c)), np)
            left.append(q)
            right.append(rr)
        A, B = _assemble(left, right, split, xp=np)
        tensors[site], tensors[site + 1] = A.reshape(Dl, 2, -1), B.reshape(-1, 2, Dr)
        charges[site + 1] = split.middle_charges
        centre = site + 1
    gauge = float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(plan.occupation)])))
    return tensors, charges, gauge


def mps_overlap_host(a, b):
    env = np.ones((1, 1))
    for x, y in zip(a, b):
        env = np.einsum("ab,apc,bpd->cd", env, np.asarray(x), np.asarray(y))
    return float(env.reshape(()))


# Entering MPO bond state k of hubbard_mpo has created this much (N_alpha, N_beta):
# 0 "nothing started" and 5 "done" carry nothing, 1..4 follow cr_a, an_a, cr_b, an_b.
MPO_QN = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])


def apply_mpo_qn(W, tensors, charges):
    """apply_mpo, carrying the bond labels. The output bond flattens (MPO bond, MPS
    bond) with the MPO index major, so its label is MPO_QN[a] + q[c]."""
    out, labels = [], [MPO_QN[:1] + np.asarray(charges[0])]
    last = len(tensors) - 1
    for i, (w, A) in enumerate(zip(W, tensors)):
        if i == 0:
            w = w[:1]
        if i == last:
            w = w[:, :, :, 5:6]
        T = np.einsum("apqb,cqd->acpbd", w, np.asarray(A))
        dl, cl, d, dr, cr = T.shape
        out.append(T.reshape(dl * cl, d, dr * cr))
        mpo_right = MPO_QN[5:6] if i == last else MPO_QN
        labels.append((mpo_right[:, None, :] + np.asarray(charges[i + 1])[None, :, :]).reshape(-1, 2))
    return out, tuple(labels)


def _label_sectors(row_labels, column_labels):
    rows, columns = _charge_index(row_labels), _charge_index(column_labels)
    return [(c, rows[c], columns[c]) for c in sorted(set(rows) & set(columns))]


def compress_mps_qn(tensors, charges, relative_tolerance=1.0e-13):
    """compress_mps that keeps the (N_alpha, N_beta) bond labels.

    The QR sweep and the SVD sweep factor each charge sector separately and the
    rank cut is one relative tolerance per bond applied per sector, so no
    factorisation mixes charges (the dense compress can, at degenerate singular
    values) and the result can be contracted charge-blocked like the trial.
    """
    A = [np.array(t, dtype=float, copy=True) for t in tensors]
    Q = [np.asarray(q, int).reshape(-1, 2) for q in charges]
    for i in range(len(A) - 1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl * d, Dr)
        rows = (Q[i][:, None, :] + PHYSICAL_CHARGE[None]).reshape(-1, 2)
        left, right, labels = [], [], []
        for c, r, k in _label_sectors(rows, Q[i + 1]):
            q, rr = np.linalg.qr(M[np.ix_(r, k)])
            lq = np.zeros((Dl * d, q.shape[1]))
            lq[r] = q
            rq = np.zeros((q.shape[1], Dr))
            rq[:, k] = rr
            left.append(lq)
            right.append(rq)
            labels += [c] * q.shape[1]
        A[i] = np.concatenate(left, axis=1).reshape(Dl, d, -1)
        A[i + 1] = np.tensordot(np.concatenate(right, axis=0), A[i + 1], axes=1)
        Q[i + 1] = np.asarray(labels, int).reshape(-1, 2)
    for i in range(len(A) - 1, 0, -1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl, d * Dr)
        columns = (Q[i + 1][None, :, :] - PHYSICAL_CHARGE[:, None, :]).reshape(-1, 2)
        factors = []
        for c, r, k in _label_sectors(Q[i], columns):
            u, s, vh = np.linalg.svd(M[np.ix_(r, k)], full_matrices=False)
            factors.append((c, r, k, u, s, vh))
        cut = relative_tolerance * max(max(f[4][0] for f in factors), 1e-300)
        left, right, labels = [], [], []
        for c, r, k, u, s, vh in factors:
            n = int(np.sum(s > cut))
            if n == 0:
                continue
            lu = np.zeros((Dl, n))
            lu[r] = u[:, :n] * s[:n]
            rv = np.zeros((n, d * Dr))
            rv[:, k] = vh[:n]
            left.append(lu)
            right.append(rv)
            labels += [c] * n
        A[i] = np.concatenate(right, axis=0).reshape(-1, d, Dr)
        A[i - 1] = np.tensordot(A[i - 1], np.concatenate(left, axis=1), axes=(2, 0))
        Q[i] = np.asarray(labels, int).reshape(-1, 2)
    return A, tuple(Q)


# ============================================================================
# Static circuit: every centre move and split of channel_mps, as gather maps
# ============================================================================
#
# One compiled circuit converts a group of spin channels (both, whenever they
# share a gate sequence) in one batched pass. Each channel keeps its own charge
# sectors and kept ranks: tensors are stored zero-padded to the largest bond of
# the group and every op gathers each channel's sector blocks with its own index
# maps. Every sector is factored by the method mps_cpmc_new._factor_block uses
# for it, batched over all sectors of that kind in the op:
#   row1       single row:                Q = 1, R = block             (closed form)
#   col1       single column:             Q = block/|block|, R = |block| (closed form)
#   qr         exact (moves, full splits): reduced QR
#   eigh_rows  truncated, rows <= cols:   top eigenvectors of block block^T
#   eigh_cols  truncated, rows > cols:    top eigenvectors of block^T block
# Padding leaves each factorisation mathematically unchanged: padded rows and
# columns are zero, and each Gram matrix gets -trace on its padded diagonal
# entries, so padded directions decouple and sort below every real eigenvalue.

SECTOR_KINDS = ("row1", "col1", "qr", "eigh_rows", "eigh_cols")


class SectorClass(NamedTuple):
    kind: str
    gather: np.ndarray  # (spins, S, r, c) flat indices into the padded matrix (sentinel: its size)
    pad: np.ndarray | None  # (spins, S, n, n) ones on the padded diagonal of the Gram matrix (eigh kinds)
    k: int  # columns of Q (rows of R) per sector


class SelectSpec(NamedTuple):
    """Per-walker kept counts inside a padded allocation (dynamic truncation of a gate).

    The static shapes are a padding (at most `caps` states per charge sector); each
    walker keeps its own `chi` largest squared singular values within those caps,
    exactly as allocation_study.gmps(Q, plan, chi=chi, caps=caps) chooses them, and
    zeroes the rest (a zero column of Q times a zero row of R leaves the state intact).
    """
    chi: int
    cand_index: np.ndarray  # (spins, n_cand) flat indices of every sector's candidate values, in the
                            # selection's tie-break order: sector by sector, largest first (sentinel: total)


class FactorOp(NamedTuple):
    """One factorisation M = Q R of the conversion, for every channel of a group.

    kind "move" is the exact QR (step +1) or LQ (step -1) that shifts the
    orthogonality centre; "gate" splits a gated pair, truncating to the kept ranks.
    """
    kind: str
    site: int
    step: int  # +1 / -1 for moves, 0 for gates
    gate: int  # index into the applied gate sequence, -1 for moves
    n_rows: int  # padded matrix shape, shared by the group
    n_cols: int
    K: int  # padded new bond dimension (the largest in the group)
    classes: tuple  # one SectorClass per sector kind present
    q_index: np.ndarray  # (spins, n_rows, K) flat indices into the concatenated Q blocks (sentinel: size)
    r_index: np.ndarray  # (spins, K, n_cols) flat indices into the concatenated R blocks
    select: SelectSpec | None = None  # per-walker truncation inside the padded shapes (gates only)


class Circuit(NamedTuple):
    occupation: np.ndarray  # (spins, L)
    gate_sites: tuple  # gate sites in application order, shared by the group
    ops: tuple
    charges: tuple  # per channel: final bond labels, i.e. the walker channel labels
    pads: tuple  # final padded bond dimensions


def gate_sites(plan: OrbitalPlan):
    """Sites of channel_angles' rotations in planning order (its static output)."""
    sites = []
    for k, B in enumerate(plan.block_sizes):
        for j in range(int(B) - 1, 0, -1):
            sites.append(k + j - 1)
    return sites


def _sector_kind(n_rows, n_cols, truncate):
    """The branch mps_cpmc_new._factor_block takes for a block of this shape."""
    if min(n_rows, n_cols) == 1:
        return "row1" if n_rows == 1 else "col1"
    if not truncate:
        return "qr"
    return "eigh_rows" if n_rows <= n_cols else "eigh_cols"


def _group_op(kind, site, step, gate, specs, n_rows, n_cols, select_chi=None) -> FactorOp:
    """specs: per channel (row_map, col_map, SectorPlan, truncate flags), where the
    maps take the channel's own row/column indices to the padded matrix. With
    select_chi, the SectorPlan's ranks are caps and each walker keeps its own
    select_chi largest values within them (SelectSpec)."""
    spins = len(specs)
    K = max(len(plan.middle_charges) for _, _, plan, _ in specs)
    members = {name: [[] for _ in range(spins)] for name in SECTOR_KINDS}
    order = [[] for _ in range(spins)]  # per channel, in sector order: (kind, index within the kind, rank)
    for sigma, (row_map, col_map, plan, truncate) in enumerate(specs):
        offset = 0
        for (ri, ci, rank), trunc in zip(plan.sectors, truncate):
            name = _sector_kind(len(ri), len(ci), trunc)
            order[sigma].append((name, len(members[name][sigma]), rank))
            members[name][sigma].append((row_map[ri], col_map[ci], rank, offset))
            offset += rank
        assert offset == len(plan.middle_charges)

    classes, layout = [], {}
    q_index = np.full((spins, n_rows, K), -1, np.int64)
    r_index = np.full((spins, K, n_cols), -1, np.int64)
    q_base = r_base = v_base = 0
    for name in SECTOR_KINDS:
        per_spin = members[name]
        S = max(len(x) for x in per_spin)
        if S == 0:
            continue
        found = [m for x in per_spin for m in x]
        r = max(len(m[0]) for m in found)
        c = max(len(m[1]) for m in found)
        k = 1 if name in ("row1", "col1") else max(m[2] for m in found)
        gather = np.full((spins, S, r, c), n_rows * n_cols, np.int64)
        n_gram = r if name == "eigh_rows" else c
        pad = np.zeros((spins, S, n_gram, n_gram)) if name.startswith("eigh") else None
        for sigma, sectors in enumerate(per_spin):
            for s in range(S):
                if s >= len(sectors):  # padding sector of a channel with fewer sectors of this kind
                    if pad is not None:
                        pad[sigma, s] = np.eye(n_gram)
                    continue
                rows, cols, rank, offset = sectors[s]
                gather[sigma, s, :len(rows), :len(cols)] = rows[:, None] * n_cols + cols[None, :]
                if pad is not None:
                    valid = len(rows) if name == "eigh_rows" else len(cols)
                    idx = np.arange(valid, n_gram)
                    pad[sigma, s, idx, idx] = 1.0
                for j in range(rank):
                    q_index[sigma, rows, offset + j] = q_base + s * r * k + np.arange(len(rows)) * k + j
                    r_index[sigma, offset + j, cols] = r_base + s * k * c + j * c + np.arange(len(cols))
        classes.append(SectorClass(name, gather, pad, k))
        layout[name] = (v_base, k)  # each sector's k candidate values, stored (S, k)
        q_base += S * r * k
        r_base += S * k * c
        v_base += S * k
    q_index[q_index < 0] = q_base  # the zero appended after the concatenated blocks
    r_index[r_index < 0] = r_base
    select = None
    if select_chi is not None:
        if "qr" in layout:
            raise AssertionError("a selecting gate needs singular values in every sector")
        cands = [[layout[name][0] + s * layout[name][1] + j for name, s, rank in order[sigma] for j in range(rank)]
                 for sigma in range(spins)]
        cand_index = np.full((spins, max(map(len, cands))), v_base, np.int64)
        for sigma, c in enumerate(cands):
            cand_index[sigma, :len(c)] = c
        select = SelectSpec(int(select_chi), cand_index)
    return FactorOp(kind, site, step, gate, n_rows, n_cols, K, tuple(classes), q_index, r_index, select)


def _move_op(charges, pads, site, step):
    """The static part of _shift_centre for every channel; updates labels and pads."""
    Pl, Pr = pads[site], pads[site + 1]
    specs = []
    for q in charges:
        ql, qr = q[site], q[site + 1]
        Dl, Dr = len(ql), len(qr)
        if step > 0:  # M = A.reshape(2 Dl, Dr): row a*2 + p, column b
            rows, columns = (ql[:, None] + np.arange(2)).ravel(), qr
            row_map, col_map = np.arange(2 * Dl), np.arange(Dr)
        else:  # M = A.reshape(Dl, 2 Dr).T: row p*Dr + b, column a
            rows, columns = (qr[None, :] - np.arange(2)[:, None]).ravel(), ql
            row_map, col_map = (np.arange(2)[:, None] * Pr + np.arange(Dr)[None, :]).ravel(), np.arange(Dl)
        plan = _block_plan(_key(rows), _key(columns))
        specs.append((row_map, col_map, plan, [False] * len(plan.sectors)))
    n_rows, n_cols = (2 * Pl, Pr) if step > 0 else (2 * Pr, Pl)
    op = _group_op("move", site, step, -1, specs, n_rows, n_cols)
    bond = site + 1 if step > 0 else site
    for q, (_, _, plan, _) in zip(charges, specs):
        q[bond] = plan.middle_charges
    pads[bond] = op.K
    return op


def _gate_op(charges, pads, site, gate, kept, select_chi=None):
    """The static part of split_pair for every channel; updates labels and pads.
    With select_chi the kept counts are caps and every sector may be truncated
    per walker, so each goes through the truncating (eigh) branch or a closed form."""
    Pl, Pr = pads[site], pads[site + 2]
    specs = []
    for q, kept_q in zip(charges, kept):
        ql, qr = q[site], q[site + 2]
        Dl, Dr = len(ql), len(qr)
        split = sector_plan(ql, qr, kept_q)
        if select_chi is None:
            truncate = [rank < min(len(r), len(c)) for r, c, rank in split.sectors]
        else:
            truncate = [True] * len(split.sectors)
        # M = pair.reshape(2 Dl, 2 Dr): row a*2 + p, column q*Dr + b
        row_map = np.arange(2 * Dl)
        col_map = (np.arange(2)[:, None] * Pr + np.arange(Dr)[None, :]).ravel()
        specs.append((row_map, col_map, split, truncate))
    op = _group_op("gate", site, 0, gate, specs, 2 * Pl, 2 * Pr, select_chi)
    for q, (_, _, split, _) in zip(charges, specs):
        q[site + 1] = split.middle_charges
    pads[site + 1] = op.K
    return op


def compile_circuit(plans, bond_plans, dynamic_chi=None) -> Circuit:
    """Replay channel_mps symbolically for a group of channels sharing one gate
    sequence. Bond labels depend only on the orbital plans and the frozen kept
    counts, so the whole conversion (which tensor is factored, in which charge
    sectors, by which method, keeping how many vectors) is fixed here once.

    dynamic_chi: the bond plans' kept counts are a padding (the static shapes) and
    every walker keeps its own dynamic_chi largest values within it at each gate
    (the "padded" allocation of allocation_study.py)."""
    if dynamic_chi is not None and any(b is None for b in bond_plans):
        raise ValueError("a dynamic allocation needs a padding (bond plan) for every channel")
    sites = gate_sites(plans[0])[::-1]
    if any(gate_sites(p)[::-1] != sites for p in plans[1:]):
        raise ValueError("channels of one group must share their gate sequence")
    charges = []
    for plan in plans:
        q = [np.zeros(1, int)]
        for o in plan.occupation:
            q.append(q[-1] + int(o))
        charges.append(q)
    pads = [1] * len(charges[0])
    ops, centre = [], None
    for g, site in enumerate(sites):
        if centre is not None:  # None: the initial product state is already canonical
            while centre != site:
                step = -1 if centre > site else 1
                ops.append(_move_op(charges, pads, centre, step))
                centre += step
        kept = [None if b is None else b.kept_per_sector[g] for b in bond_plans]
        ops.append(_gate_op(charges, pads, site, g, kept, dynamic_chi))
        centre = site + 1
    return Circuit(np.stack([np.asarray(p.occupation) for p in plans]), tuple(sites), tuple(ops),
                   tuple(tuple(q) for q in charges), tuple(pads))


def circuit_stats(circuit: Circuit):
    classes = [cls for op in circuit.ops for cls in op.classes]
    eigh = [cls.pad.shape[-1] for cls in classes if cls.pad is not None]
    qr = [max(cls.gather.shape[-2:]) for cls in classes if cls.kind == "qr"]
    return dict(spins=int(circuit.occupation.shape[0]), ops=len(circuit.ops), gates=len(circuit.gate_sites),
                moves=sum(op.kind == "move" for op in circuit.ops),
                closed_form_batches=sum(cls.kind in ("row1", "col1") for cls in classes),
                qr_calls=len(qr), eigh_calls=len(eigh), max_qr=max(qr, default=0), max_eigh=max(eigh, default=0),
                eigh_over_32=sum(n > 32 for n in eigh), max_bond=max(circuit.pads),
                dynamic_chi=next((op.select.chi for op in circuit.ops if op.select is not None), None))


def counts_bond_plan(plan: OrbitalPlan, counts) -> BondPlan:
    """The BondPlan that keeps counts[g][charge] states in each charge sector at
    gate g, capped at the sector's rank: an explicitly given frozen allocation (a
    padding learned on many walkers, say), in plan_bonds' format. The charge of a
    sector is its row label, as in allocation_study.gmps. Labels only, no numerics."""
    charges = [np.zeros(1, int)]
    for o in plan.occupation:
        charges.append(charges[-1] + int(o))
    kept_all, centre = [], None
    for g, site in enumerate(gate_sites(plan)[::-1]):
        if centre is not None:
            while centre != site:
                step = -1 if centre > site else 1
                ql, qr = charges[centre], charges[centre + 1]
                if step > 0:
                    rows, columns, bond = (ql[:, None] + np.arange(2)).ravel(), qr, centre + 1
                else:
                    rows, columns, bond = (qr[None, :] - np.arange(2)[:, None]).ravel(), ql, centre
                charges[bond] = _block_plan(_key(rows), _key(columns)).middle_charges
                centre += step
        ql, qr = charges[site], charges[site + 2]
        row_charge = (ql[:, None] + np.arange(2)).ravel()
        full = sector_plan(ql, qr)
        kept = tuple(min(int(counts[g].get(int(row_charge[r[0]]), 0)), rank) for r, _, rank in full.sectors)
        if not any(kept):
            raise ValueError(f"the allocation keeps nothing at gate {g}")
        kept_all.append(kept)
        charges[site + 1] = sector_plan(ql, qr, kept).middle_charges
        centre = site + 1
    return BondPlan(tuple(kept_all), tuple(charges), max(map(len, charges)), float("nan"))


# ============================================================================
# Device: batched factorisation and conversion
# ============================================================================

ONE_HOT = np.array([[[1.0], [0.0]], [[0.0], [1.0]]])[:, None]  # [occupation] -> (1, 2, 1)


def _gather_flat(x, index, n_axes):
    """Gather from the last n_axes of x, flattened, with a zero appended as sentinel."""
    lead = x.shape[:x.ndim - n_axes]
    flat = x.reshape(lead + (-1,))
    flat = jnp.concatenate((flat, jnp.zeros(lead + (1,), x.dtype)), axis=-1)
    return jnp.take(flat, jnp.asarray(index), axis=-1, mode="clip")


def _take_per_channel(flat, index):
    """flat (spins, n), index (spins, ...): each channel gathers with its own indices."""
    shape = index.shape
    taken = jnp.take_along_axis(flat, jnp.asarray(index.reshape(shape[0], -1)), axis=-1, mode="clip")
    return taken.reshape(shape)


def factor_sectors(M, op: FactorOp):
    """Q and R of M (spins, n_rows, n_cols) for every channel and charge sector,
    each sector by the same method as mps_cpmc_new._factor_block, one batched
    call per sector kind. For a selecting gate (op.select) each walker then keeps
    only its own op.select.chi largest squared singular values within the caps."""
    spins = M.shape[0]
    zero = jnp.zeros((spins, 1), M.dtype)
    flat = jnp.concatenate((M.reshape(spins, -1), zero), axis=-1)
    parts = []  # per class: Q (spins, S, r, k), R (spins, S, k, c), squared singular values (spins, S, k)
    for cls in op.classes:
        B = _take_per_channel(flat, cls.gather)  # (spins, S, r, c)
        values = None
        if cls.kind == "row1":  # _vector_qr, one row
            Q, R = jnp.ones(B.shape[:-2] + (1, 1), B.dtype), B
            values = jnp.sum(B * B, axis=(-2, -1))[..., None]
        elif cls.kind == "col1":  # _vector_qr, one column
            norm = jnp.sqrt(jnp.sum(B * B, axis=(-2, -1), keepdims=True))
            first = jnp.zeros(B.shape[-2:], B.dtype).at[0, 0].set(1.0)
            Q = jnp.where(norm > 0, B / jnp.where(norm > 0, norm, 1.0), first)
            R = norm
            values = (norm * norm)[..., 0]
        elif cls.kind == "qr":  # untruncated: reduced QR
            q, r = jnp.linalg.qr(B, mode="reduced")
            Q, R = q[..., :cls.k], r[..., :cls.k, :]
        else:  # truncated: eigh of the smaller Gram matrix
            rows_side = cls.kind == "eigh_rows"
            Bt = jnp.swapaxes(B, -1, -2)
            G = B @ Bt if rows_side else Bt @ B
            shift = jnp.trace(G, axis1=-2, axis2=-1)[..., None, None] + 1.0e-300
            w, V = jnp.linalg.eigh(G - cls.pad * shift)
            Vk = V[..., ::-1][..., :cls.k]
            values = w[..., ::-1][..., :cls.k]
            if rows_side:
                Q, R = Vk, jnp.swapaxes(Vk, -1, -2) @ B
            else:
                sk = jnp.sqrt(jnp.maximum(values, 0.0))
                Q = (B @ Vk) / jnp.where(sk > 0, sk, 1.0)[..., None, :]
                R = sk[..., :, None] * jnp.swapaxes(Vk, -1, -2)
        parts.append((Q, R, values))
    if op.select is not None:
        parts = _select_per_walker(parts, op.select, spins, M.dtype)
    Qs = [Q.reshape(spins, -1) for Q, _, _ in parts]
    Rs = [R.reshape(spins, -1) for _, R, _ in parts]
    return (_take_per_channel(jnp.concatenate(Qs + [zero], axis=-1), op.q_index),
            _take_per_channel(jnp.concatenate(Rs + [zero], axis=-1), op.r_index))


def _select_per_walker(parts, select: SelectSpec, spins, dtype):
    """Keep each walker's select.chi largest candidate values (every sector's top
    values up to its cap), ties broken by candidate order like NumPy's stable
    argsort in allocation_study.gmps; zero the Q columns and R rows of the rest."""
    values = jnp.concatenate([v.reshape(spins, -1) for _, _, v in parts]
                             + [jnp.full((spins, 1), -jnp.inf, dtype)], axis=-1)
    index = jnp.asarray(select.cand_index)
    candidates = jnp.take_along_axis(values, index, axis=-1)
    order = jnp.argsort(-candidates, axis=-1, stable=True)
    keep = (jnp.argsort(order, axis=-1, stable=True) < select.chi).astype(dtype)
    mask = jnp.zeros_like(values).at[jnp.arange(spins)[:, None], index].set(keep)
    out, offset = [], 0
    for Q, R, v in parts:
        n = v.shape[-2] * v.shape[-1]
        m = mask[:, offset:offset + n].reshape(v.shape)
        offset += n
        out.append((Q * m[..., None, :], R * m[..., :, None], v))
    return out


def gate_pair_batched(A, B, theta):
    """gate_pair with leading batch axes (spin) on the tensors and the angle."""
    c, s = jnp.cos(theta)[..., None, None], jnp.sin(theta)[..., None, None]
    t00 = A[..., :, 0, :] @ B[..., :, 0, :]
    t01 = A[..., :, 0, :] @ B[..., :, 1, :]
    t10 = A[..., :, 1, :] @ B[..., :, 0, :]
    t11 = A[..., :, 1, :] @ B[..., :, 1, :]
    return jnp.stack((jnp.stack((t00, c * t01 + s * t10), axis=-2),
                      jnp.stack((-s * t01 + c * t10, t11), axis=-2)), axis=-3)


def run_circuit(circuit: Circuit, thetas):
    """Apply the compiled conversion to a group of channels; `thetas` are the gate
    angles in application order, each of shape (spins,). Returns each channel's
    tensors cut back to its own bond dimensions."""
    spins = circuit.occupation.shape[0]
    L = circuit.occupation.shape[1]
    tensors = [jnp.asarray(np.stack([ONE_HOT[int(o)] for o in circuit.occupation[:, i]])) for i in range(L)]
    for op in circuit.ops:
        site = op.site
        if op.kind == "gate":
            pair = gate_pair_batched(tensors[site], tensors[site + 1], thetas[op.gate])
            Dl, Dr = pair.shape[1], pair.shape[-1]
            M = pair.reshape(spins, 2 * Dl, 2 * Dr)
        else:
            _, Dl, _, Dr = tensors[site].shape
            if op.step > 0:
                M = tensors[site].reshape(spins, 2 * Dl, Dr)
            else:
                M = jnp.swapaxes(tensors[site].reshape(spins, Dl, 2 * Dr), -1, -2)
        assert M.shape[1:] == (op.n_rows, op.n_cols), (M.shape, op.n_rows, op.n_cols)
        Q, R = factor_sectors(M, op)
        if op.kind == "gate":
            tensors[site] = Q.reshape(spins, Dl, 2, op.K)
            tensors[site + 1] = R.reshape(spins, op.K, 2, Dr)
        elif op.step > 0:
            tensors[site] = Q.reshape(spins, Dl, 2, op.K)
            tensors[site + 1] = jnp.einsum("skd,sdpe->skpe", R, tensors[site + 1])
        else:
            tensors[site] = jnp.swapaxes(Q, -1, -2).reshape(spins, op.K, 2, Dr)
            tensors[site - 1] = jnp.einsum("sapd,skd->sapk", tensors[site - 1], R)
    out = []
    for sigma in range(spins):
        dims = [len(q) for q in circuit.charges[sigma]]
        out.append([A[sigma, :dims[i], :, :dims[i + 1]] for i, A in enumerate(tensors)])
    return out


def device_channel_angles(C, plans):
    """channel_angles for a group of channels C (spins, L, N) that share block
    sizes: the same complete-QR / eigh choice per channel as the original, the
    channel's own occupied/empty choice and reference sign."""
    rows = [C[:, i, :] for i in range(C.shape[-2])]
    angles = []
    for k, B in enumerate(plans[0].block_sizes):
        B = int(B)
        if B == 1:
            continue
        block = jnp.stack(rows[k:k + B], axis=-2)  # (spins, B, N)
        occupied = np.array([bool(p.occupation[k]) for p in plans])
        complete_qr = np.array([bool(p.exact_for_all_walkers and not p.occupation[k]) for p in plans])
        if complete_qr.any():
            # B exceeds the occupied count, so the last complete-QR vector lies
            # exactly in null(block.T): the required empty mode.
            vectors, _ = jnp.linalg.qr(block, mode="complete")
            v_qr = vectors[..., -1]
        if not complete_qr.all():
            _, vectors = jnp.linalg.eigh(block @ jnp.swapaxes(block, -1, -2))
            v_eigh = jnp.where(occupied[:, None], vectors[..., -1], vectors[..., 0])
        if complete_qr.all():
            v = v_qr
        elif not complete_qr.any():
            v = v_eigh
        else:
            v = jnp.where(complete_qr[:, None], v_qr, v_eigh)
        reference = np.stack([np.asarray(p.references[k]) for p in plans])
        v = jnp.where(jnp.sum(v * reference, axis=-1, keepdims=True) < 0, -v, v)
        v = [v[:, j] for j in range(B)]
        for j in range(B - 1, 0, -1):
            theta = jnp.arctan2(v[j], v[j - 1])
            c, s = jnp.cos(theta), jnp.sin(theta)
            v[j - 1] = c * v[j - 1] + s * v[j]
            p = k + j - 1
            c_, s_ = c[:, None], s[:, None]
            rows[p], rows[p + 1] = c_ * rows[p] + s_ * rows[p + 1], -s_ * rows[p] + c_ * rows[p + 1]
            angles.append(theta)
    return angles, rows


class Converter(NamedTuple):
    convert: Callable  # (qa, qb) orthonormal (L, N) -> alpha tensors, beta tensors, (gauge_a, gauge_b)
    charges: tuple  # (alpha bond labels, beta bond labels)
    circuits: tuple  # one Circuit per channel group
    spin_batched: bool
    linalg: str


def _can_group(plan_a, plan_b):
    """Both channels in one batch: same particle number and gate sequence. Kept
    counts and bond labels may differ (e.g. alpha and beta breaking a tie between
    degenerate Schmidt values differently); each keeps its own."""
    return (int(plan_a.occupation.sum()) == int(plan_b.occupation.sum())
            and np.array_equal(plan_a.block_sizes, plan_b.block_sizes)
            and plan_a.exact_for_all_walkers == plan_b.exact_for_all_walkers)


def make_converter(plan_a, plan_b, bond_a=None, bond_b=None, linalg="batched", spin_batch=True,
                   dynamic_chi=None) -> Converter:
    """dynamic_chi: bond_a/bond_b are paddings and each walker keeps its own
    dynamic_chi largest values within them at every gate (batched path only)."""
    if linalg == "eigh":  # the earlier name of the batched path
        linalg = "batched"
    if linalg not in ("batched", "native"):
        raise ValueError("linalg must be 'batched' or 'native'")
    if dynamic_chi is not None and linalg != "batched":
        raise ValueError("per-walker (dynamic) allocation needs linalg='batched'")
    plans, bonds = (plan_a, plan_b), (bond_a, bond_b)
    if linalg == "batched" and spin_batch and _can_group(plan_a, plan_b):
        groups = [((0, 1), compile_circuit(plans, bonds, dynamic_chi))]
    else:
        groups = [((0,), compile_circuit((plan_a,), (bond_a,), dynamic_chi)),
                  ((1,), compile_circuit((plan_b,), (bond_b,), dynamic_chi))]
    labels = [None, None]
    for channels, circuit in groups:
        for j, s in enumerate(channels):
            labels[s] = circuit.charges[j]
            if bonds[s] is not None and not all(np.array_equal(x, y) for x, y in zip(labels[s], bonds[s].charges)):
                raise AssertionError("compiled circuit labels differ from plan_bonds' dry run")

    if linalg == "native":
        def convert(qa, qb):
            alpha, _, gauge_a = channel_mps(qa, plan_a, bond_a)
            beta, _, gauge_b = channel_mps(qb, plan_b, bond_b)
            return alpha, beta, (gauge_a, gauge_b)
    else:
        occupied = [np.flatnonzero(p.occupation) for p in plans]

        def convert(qa, qb):
            C, tensors, gauges = (qa, qb), [None, None], [None, None]
            for channels, circuit in groups:
                group_plans = [plans[s] for s in channels]
                angles, rows = device_channel_angles(jnp.stack([C[s] for s in channels]), group_plans)
                out = run_circuit(circuit, angles[::-1])
                rotated = jnp.stack(rows, axis=-2)  # (spins, L, N)
                for j, s in enumerate(channels):
                    tensors[s] = out[j]
                    gauges[s] = jnp.linalg.det(rotated[j][occupied[s]])
            return tensors[0], tensors[1], (gauges[0], gauges[1])

    return Converter(convert, tuple(labels), tuple(c for _, c in groups), len(groups) == 1 and linalg == "batched",
                     linalg)


def _cholesky_qr(C):
    chol = jnp.linalg.cholesky(jnp.swapaxes(C, -1, -2) @ C)  # C^T C = L L^T, R = L^T
    Q = jnp.swapaxes(jax.scipy.linalg.solve_triangular(chol, jnp.swapaxes(C, -1, -2), lower=True), -1, -2)
    return Q, jnp.prod(jnp.diagonal(chol, axis1=-2, axis2=-1), axis=-1)


def cholesky_qr2(C):
    """C = Q R with diag(R) > 0 and det(R), as trot.walkers._qr, from GEMMs, a
    Cholesky and a triangular solve (all batched on GPUs). The second pass makes Q
    orthonormal to machine precision for condition numbers up to ~1e7."""
    Q1, d1 = _cholesky_qr(C)
    Q2, d2 = _cholesky_qr(Q1)
    return Q2, d1 * d2


def make_batch_qr(mode):
    """Orthonormalise a batch of (L, N) walkers: (Q, det R). CholeskyQR2 falls
    back to Householder QR for the whole batch if any walker is too ill-conditioned."""
    native = jax.vmap(qr_with_det)
    if mode == "native":
        return native
    if mode != "cholesky":
        raise ValueError("walker_qr must be 'cholesky' or 'native'")

    def batch_qr(C):
        Q, d = jax.vmap(cholesky_qr2)(C)
        ok = jnp.all(jnp.isfinite(Q)) & jnp.all(jnp.isfinite(d)) & jnp.all(d > 0)
        return lax.cond(ok, lambda: (Q, d), lambda: native(C))
    return batch_qr


# ============================================================================
# Factorized charge-blocked walker-trial contraction
# ============================================================================

PHYSICAL = ((0, 0), (1, 0), (0, 1), (1, 1))  # p = n_alpha + 2 n_beta
_LEGS = ("aA", "bB", "kK")  # alpha, beta, trial: (lower, upper) block indices


class SitePlan(NamedTuple):
    src: np.ndarray  # (T,) incoming shared-label index of each transition
    dst: np.ndarray  # (T,) outgoing shared-label index
    physical: np.ndarray  # (T,) p = n_alpha + 2 n_beta
    sign: np.ndarray  # (T, 1, 1, 1) alpha/beta reordering sign (-1)^(n_alpha * N_beta)
    onehot: np.ndarray  # (T, 4) physical index one-hot, for the marginal
    gather_a: np.ndarray  # (T, Pa, Pa') flat indices into the alpha tensor (+ zero sentinel)
    gather_b: np.ndarray  # (T, Pb, Pb') flat indices into the beta tensor
    into: np.ndarray  # (n_out, deg) transitions feeding each outgoing label (sentinel T)
    outof: np.ndarray  # (n_in, deg) transitions leaving each incoming label (sentinel T)
    left_order: tuple  # leg order for left environments, cheapest first
    right_order: tuple
    left_fma: int
    right_fma: int
    peak: int  # largest per-transition intermediate, in entries, over both directions


class ContractionPlan(NamedTuple):
    sites: tuple
    shared: tuple  # shared (N_alpha, N_beta) labels per bond
    pads: tuple  # (Pa, Pb, Pt) per bond
    blocks: tuple  # host (T, Pt, Pt') blocks of the fixed MPS (trial or H|trial>)
    stats: dict


def _index_1d(labels):
    grouped = {}
    for i, q in enumerate(np.asarray(labels).tolist()):
        grouped.setdefault(int(q), []).append(i)
    return {q: np.asarray(v, int) for q, v in grouped.items()}


def _leg_order(T, start, end):
    """Cheapest order to contract the alpha, beta and trial legs of a (T, a, b, k)
    environment block from sizes `start` to `end`; returns (order, FMAs, peak)."""
    best = None
    for order in itertools.permutations(range(3)):
        size, cost, peak = list(start), 0, T * int(np.prod(start))
        for leg in order:
            cost += T * int(np.prod(size)) * end[leg]
            size[leg] = end[leg]
            peak = max(peak, T * int(np.prod(size)))
        if best is None or cost < best[1]:
            best = (order, cost, peak)
    return best


def _table(keys, n):
    deg = max(np.bincount(keys, minlength=n).max(), 1)
    table = np.full((n, deg), len(keys), np.int32)
    fill = np.zeros(n, int)
    for t, key in enumerate(keys):
        table[key, fill[key]] = t
        fill[key] += 1
    return table


def make_factorized_plan(qa, qb, fixed_np, fixed_charges) -> ContractionPlan:
    """Layout of <fixed MPS|walker> with the walker kept as its two spin channels.

    Environments at bond i are (n_shared, Pa, Pb, Pt): one padded (alpha, beta,
    fixed) block per shared (N_alpha, N_beta) label. A transition moves label
    (a, b) to (a + n_alpha, b + n_beta) through physical p; its alpha, beta and
    fixed blocks are gathered separately, so the d=4 walker is never formed.
    """
    n = len(qa) - 1
    ia, ib = [_index_1d(q) for q in qa], [_index_1d(q) for q in qb]
    it = [_charge_index(q) for q in fixed_charges]
    shared = [sorted(c for c in it[i] if c[0] in ia[i] and c[1] in ib[i]) for i in range(n + 1)]
    pads = [(max((len(ia[i][c[0]]) for c in shared[i]), default=1),
             max((len(ib[i][c[1]]) for c in shared[i]), default=1),
             max((len(it[i][c]) for c in shared[i]), default=1)) for i in range(n + 1)]

    sites, blocks = [], []
    exact = sum(len(ia[i][c[0]]) * len(ib[i][c[1]]) * len(it[i][c]) for i in range(n + 1) for c in shared[i])
    padded = sum(len(shared[i]) * int(np.prod(pads[i])) for i in range(n + 1))
    for site in range(n):
        (Pa, Pb, Pt), (Pa2, Pb2, Pt2) = pads[site], pads[site + 1]
        out_index = {c: j for j, c in enumerate(shared[site + 1])}
        Dla, Dra = len(qa[site]), len(qa[site + 1])
        Dlb, Drb = len(qb[site]), len(qb[site + 1])
        src, dst, physical, sign, ga, gb, tb = [], [], [], [], [], [], []
        for s, c in enumerate(shared[site]):
            for p, (na, nb) in enumerate(PHYSICAL):
                c2 = (c[0] + na, c[1] + nb)
                if c2 not in out_index:
                    continue
                ra, ca_ = ia[site][c[0]], ia[site + 1][c2[0]]
                rb, cb_ = ib[site][c[1]], ib[site + 1][c2[1]]
                rt, ct = it[site][c], it[site + 1][c2]
                a = np.full((Pa, Pa2), Dla * 2 * Dra, np.int32)
                a[:len(ra), :len(ca_)] = ra[:, None] * 2 * Dra + na * Dra + ca_[None, :]
                b = np.full((Pb, Pb2), Dlb * 2 * Drb, np.int32)
                b[:len(rb), :len(cb_)] = rb[:, None] * 2 * Drb + nb * Drb + cb_[None, :]
                t = np.zeros((Pt, Pt2))
                t[:len(rt), :len(ct)] = np.asarray(fixed_np[site])[np.ix_(rt, [p], ct)][:, 0]
                src.append(s)
                dst.append(out_index[c2])
                physical.append(p)
                sign.append((-1.0) ** (na * c[1]))
                ga.append(a)
                gb.append(b)
                tb.append(t)
        T = len(src)
        if T == 0:
            raise AssertionError(f"no allowed transition at site {site}: the walker and trial share no path")
        src, dst, physical = np.asarray(src), np.asarray(dst), np.asarray(physical)
        left_order, left_fma, left_peak = _leg_order(T, (Pa, Pb, Pt), (Pa2, Pb2, Pt2))
        right_order, right_fma, right_peak = _leg_order(T, (Pa2, Pb2, Pt2), (Pa, Pb, Pt))
        sites.append(SitePlan(
            src=src, dst=dst, physical=physical,
            sign=np.asarray(sign).reshape(T, 1, 1, 1), onehot=np.eye(4)[physical],
            gather_a=np.stack(ga), gather_b=np.stack(gb),
            into=_table(dst, len(shared[site + 1])), outof=_table(src, len(shared[site])),
            left_order=left_order, right_order=right_order, left_fma=left_fma, right_fma=right_fma,
            peak=max(left_peak, right_peak)))
        blocks.append(np.stack(tb))
    stats = dict(exact_entries=exact, padded_entries=padded, transitions=sum(len(s.src) for s in sites),
                 env_entries=padded, contraction_fma=sum(s.left_fma for s in sites),
                 max_pads=tuple(int(max(p[j] for p in pads)) for j in range(3)))
    return ContractionPlan(tuple(sites), tuple(shared), tuple(pads), tuple(blocks), stats)


def walker_blocks(alpha, beta, plan: ContractionPlan):
    """Per-transition (alpha, beta) blocks gathered straight from the channel tensors."""
    return tuple((_gather_flat(A, sp.gather_a, 3), _gather_flat(B, sp.gather_b, 3))
                 for A, B, sp in zip(alpha, beta, plan.sites))


def _legs(X, operands, order, forward):
    """Contract the environment block X with the alpha, beta and fixed blocks one
    leg at a time; forward maps lower to upper indices (left environments),
    backward upper to lower (right environments)."""
    cur = "abk" if forward else "ABK"
    for leg in order:
        lo, up = _LEGS[leg]
        nxt = cur.replace(lo, up) if forward else cur.replace(up, lo)
        X = jnp.einsum(f"t{cur},t{lo}{up}->t{nxt}", X, operands[leg])
        cur = nxt
    return X


def _reduce(Y, table):
    """Sum transitions into labels by gathering (deterministic, no atomics)."""
    Y = jnp.concatenate((Y, jnp.zeros((1,) + Y.shape[1:], Y.dtype)), axis=0)
    return jnp.sum(Y[table], axis=1)


def left_contract(wblocks, fixed_blocks, plan: ContractionPlan):
    """<fixed|walker channels> streaming left to right; no environment is stored."""
    E = jnp.ones((1, 1, 1, 1))
    for (Wa, Wb), F, sp in zip(wblocks, fixed_blocks, plan.sites):
        Y = _legs(E[sp.src], (Wa, Wb, F), sp.left_order, True) * sp.sign
        E = _reduce(Y, sp.into)
    return E[0, 0, 0, 0]


def right_environments(wblocks, fixed_blocks, plan: ContractionPlan):
    right = [jnp.ones((1, 1, 1, 1))]
    for site in range(len(plan.sites) - 1, -1, -1):
        sp = plan.sites[site]
        Wa, Wb = wblocks[site]
        Y = _legs(right[-1][sp.dst], (Wa, Wb, fixed_blocks[site]), sp.right_order, False) * sp.sign
        right.append(_reduce(Y, sp.outof))
    return right[::-1]


def field_sweep(ca, cb, randoms, wblocks, prefactor, right, overlap_in, *, hs, weight_floor,
                fixed_blocks, plan: ContractionPlan):
    """make_fast_sweep's diagonal HS sweep on the factorized environments: at each
    site the four physical marginals give both field proposals, one field is
    sampled, and the left environment absorbs the chosen diagonal factor."""
    diagonal = jnp.stack((jnp.ones(2), hs[:, 0], hs[:, 1], hs[:, 0] * hs[:, 1]), axis=1)
    left = jnp.ones((1, 1, 1, 1))
    overlap, log_weight = overlap_in, jnp.zeros(())
    nodes = jnp.zeros((), jnp.int64)
    fields = []
    for site, ((Wa, Wb), F, sp) in enumerate(zip(wblocks, fixed_blocks, plan.sites)):
        local = _legs(left[sp.src], (Wa, Wb, F), sp.left_order, True) * sp.sign
        by_transition = jnp.sum(local * right[site + 1][sp.dst], axis=(1, 2, 3))
        marginal = by_transition @ sp.onehot

        proposed = prefactor * (diagonal @ marginal)
        ratios = constrain_ratio(proposed / overlap, weight_floor)
        nodes += jnp.sum(ratios <= 0.0, dtype=jnp.int64)
        probabilities = 0.5 * ratios
        norm = probabilities.sum() + 1.0e-13
        field = jnp.where(randoms[site] < probabilities[0] / norm, 0, 1)
        overlap = proposed[field]
        log_weight += jnp.log(norm)
        fields.append(field)
        left = _reduce(diagonal[field][sp.physical][:, None, None, None] * local, sp.into)
    fields = jnp.stack(fields)
    return ca * hs[fields, 0][:, None], cb * hs[fields, 1][:, None], overlap, jnp.exp(log_weight), nodes


# ============================================================================
# Walker ops against a fixed trial
# ============================================================================

class DeviceData(NamedTuple):
    """Everything large the jitted code reads, passed as arguments (not constants)."""
    trial: tuple  # per-site (T, Pt, Pt') trial blocks of the overlap plan
    htrial: tuple  # per-site blocks of the energy plan, or dense H|trial> for energy="dense"
    dense_trial: tuple  # dense trial tensors, energy="dense" only
    exp_h1_half: jax.Array
    hs: jax.Array


class GpuOps(NamedTuple):
    converter: Converter
    overlap_plan: ContractionPlan
    energy_plan: ContractionPlan | None
    data: DeviceData
    batch_qr: Callable  # (n, L, N) -> Q, det R
    overlaps: Callable  # (ca, cb, data) -> overlaps of a walker batch
    energies: Callable  # (ca, cb, data) -> local energies of a walker batch
    step_batch: Callable  # (ca, cb, randoms, sweep, data, weight_floor) -> see make_gpu_ops
    probe: Callable  # (ca, cb, data) for one walker -> dict with MPS, gauges, overlap, energy
    energy_kind: str


def make_gpu_ops(plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, htrial, prop_ctx, *,
                 linalg="batched", walker_qr="cholesky", spin_batch=True, energy="blocked",
                 dynamic_chi=None) -> GpuOps:
    """Conversion, overlap, local energy and the half-step kernel for a fixed trial.

    htrial is (tensors, labels) of the charge-labelled H|trial> for energy="blocked",
    or the dense H|trial> tensors for energy="dense". dynamic_chi: per-walker kept
    counts inside the bond plans' padding (see compile_circuit).
    """
    converter = make_converter(plan_a, plan_b, bond_a, bond_b, linalg, spin_batch, dynamic_chi)
    convert = converter.convert
    qa_labels, qb_labels = converter.charges
    overlap_plan = make_factorized_plan(qa_labels, qb_labels, trial_np, trial_charges)
    if energy == "blocked":
        energy_plan = make_factorized_plan(qa_labels, qb_labels, *htrial)
        hblocks, dense_trial = energy_plan.blocks, ()
    elif energy == "dense":
        energy_plan, hblocks, dense_trial = None, tuple(htrial), tuple(trial_np)
    else:
        raise ValueError("energy must be 'blocked' or 'dense'")
    data = DeviceData(tuple(jnp.asarray(b) for b in overlap_plan.blocks),
                      tuple(jnp.asarray(b) for b in hblocks),
                      tuple(jnp.asarray(b) for b in dense_trial),
                      jnp.asarray(prop_ctx.exp_h1_half), jnp.asarray(prop_ctx.hs_constant))
    batch_qr = make_batch_qr(walker_qr)

    def local_energy(alpha, beta, data):
        if energy == "dense":
            tensors, _ = combine_channels(alpha, qa_labels, beta, qb_labels)
            return contract_real(tensors, data.htrial) / contract_real(tensors, data.dense_trial)
        numerator = left_contract(walker_blocks(alpha, beta, energy_plan), data.htrial, energy_plan)
        denominator = left_contract(walker_blocks(alpha, beta, overlap_plan), data.trial, overlap_plan)
        return numerator / denominator

    def overlaps(ca, cb, data):
        qa, da = batch_qr(ca)
        qb, db = batch_qr(cb)

        def one(qa1, qb1):
            alpha, beta, (ga, gb) = convert(qa1, qb1)
            return ga * gb * left_contract(walker_blocks(alpha, beta, overlap_plan), data.trial, overlap_plan)
        return da * db * jax.vmap(one)(qa, qb)

    def energies(ca, cb, data):
        qa, _ = batch_qr(ca)
        qb, _ = batch_qr(cb)

        def one(qa1, qb1):
            alpha, beta, _ = convert(qa1, qb1)
            return local_energy(alpha, beta, data)
        return jax.vmap(one)(qa, qb)

    def step_batch(ca, cb, randoms, sweep, data, weight_floor):
        """One conversion per walker, right environments and the overlap; with
        sweep (a traced or Python bool) also the HS field sweep. Returns walkers,
        overlap before and after the fields, weight factor and node count."""
        qa, da = batch_qr(ca)
        qb, db = batch_qr(cb)

        def one(qa1, qb1):
            alpha, beta, (ga, gb) = convert(qa1, qb1)
            wb = walker_blocks(alpha, beta, overlap_plan)
            return wb, right_environments(wb, data.trial, overlap_plan), ga * gb
        wb, right, gauge = jax.vmap(one)(qa, qb)
        prefactor = da * db * gauge
        now = prefactor * right[0][:, 0, 0, 0, 0]

        def with_fields():
            sweep_one = partial(field_sweep, hs=data.hs, weight_floor=weight_floor,
                                fixed_blocks=data.trial, plan=overlap_plan)
            return jax.vmap(sweep_one)(ca, cb, randoms, wb, prefactor, right, now)

        def without():
            return ca, cb, now, jnp.ones_like(now), jnp.zeros(now.shape, jnp.int64)

        if isinstance(sweep, bool):
            ca2, cb2, after, factor, nodes = with_fields() if sweep else without()
        else:
            ca2, cb2, after, factor, nodes = lax.cond(sweep, with_fields, without)
        return ca2, cb2, now, after, factor, nodes

    def probe(ca, cb, data):
        qa, da = qr_with_det(ca)
        qb, db = qr_with_det(cb)
        alpha, beta, (ga, gb) = convert(qa, qb)
        overlap = da * db * ga * gb * left_contract(walker_blocks(alpha, beta, overlap_plan), data.trial,
                                                    overlap_plan)
        return dict(qa=qa, qb=qb, alpha=alpha, beta=beta, gauges=(ga, gb), overlap=overlap,
                    energy=local_energy(alpha, beta, data))

    return GpuOps(converter, overlap_plan, energy_plan, data, batch_qr, overlaps, energies, step_batch,
                  probe, energy)


def conversion_self_check(ops: GpuOps, plans, bonds, probe):
    """Relative error of the device channel MPS against channel_mps_host for one
    walker, per spin: |g_dev <host|dev> - g_host <host|host>| / |g_host <host|host>|."""
    errors = []
    for C, plan, bond, dev, g_dev in zip((probe["qa"], probe["qb"]), plans, bonds,
                                         (probe["alpha"], probe["beta"]), probe["gauges"]):
        host, _, g_host = channel_mps_host(np.asarray(C), plan, bond)
        reference = g_host * mps_overlap_host(host, host)
        errors.append(abs(float(g_dev) * mps_overlap_host(host, dev) - reference) / abs(reference))
    return errors


# ============================================================================
# Memory model and walker chunking
# ============================================================================

def memory_model(ops: GpuOps):
    """Rough device bytes per walker of one half step and of one energy evaluation
    (stored right environments, walker blocks, the largest per-site intermediate,
    the conversion), times 2 for XLA's buffer overlap."""
    def blocks_entries(plan):
        return sum(sp.gather_a.size + sp.gather_b.size for sp in plan.sites)

    def peak(plan):
        return max(sp.peak for sp in plan.sites)

    conversion = 0
    for circuit in ops.converter.circuits:  # one per channel group; groups run one after another
        spins, pads = circuit.occupation.shape[0], circuit.pads
        tensors = spins * sum(pads[i] * 2 * pads[i + 1] for i in range(len(pads) - 1))
        largest = max(op.q_index.size + op.r_index.size
                      + sum(cls.gather.size + (0 if cls.pad is None else 2 * cls.pad.size) for cls in op.classes)
                      for op in circuit.ops)
        conversion += 2 * tensors + largest
    plan = ops.overlap_plan
    envs = sum(len(s) * int(np.prod(p)) for s, p in zip(plan.shared, plan.pads))
    step = envs + blocks_entries(plan) + 3 * peak(plan) + conversion
    if ops.energy_plan is not None:
        energy = blocks_entries(plan) + blocks_entries(ops.energy_plan) + 3 * peak(ops.energy_plan) + conversion
    else:
        energy = step + 16 * max(len(q) for q in ops.converter.charges[0]) ** 2 * max(
            b.shape[0] for b in ops.data.htrial)
    return dict(step_bytes_per_walker=16 * step, energy_bytes_per_walker=16 * energy,
                data_bytes=sum(x.nbytes for x in jax.tree_util.tree_leaves(ops.data)))


def device_bytes_limit():
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # pragma: no cover
        stats = {}
    return stats.get("bytes_limit")


def choose_chunks(n_walkers, bytes_per_walker, budget):
    """Fewest walker chunks (a divisor of n_walkers, so there is no remainder
    batch and no second compiled copy) that keep one chunk within budget."""
    if budget is None:
        return 1
    need = max(1, math.ceil(n_walkers * bytes_per_walker / max(budget, 1.0)))
    for chunks in range(need, n_walkers + 1):
        if n_walkers % chunks == 0:
            return chunks
    return n_walkers


def chunked(fn, n_chunks, *args):
    """fn over the leading (walker) axis in n_chunks sequential pieces (lax.map)."""
    if n_chunks == 1:
        return fn(*args)
    split = lambda x: x.reshape((n_chunks, x.shape[0] // n_chunks) + x.shape[1:])
    out = lax.map(lambda xs: fn(*xs), tuple(split(a) for a in args))
    return jax.tree_util.tree_map(lambda y: y.reshape((-1,) + y.shape[2:]), out)


# ============================================================================
# CPMC: half steps, measurement block, driver
# ============================================================================

def make_half_step(ops: GpuOps, params: QmcParams, n_chunks: int):
    """make_fast_prop_ops' step split into its two halves around the one-body
    propagator, with one conversion call site. Even half steps: exp(-dt K/2), the
    overlap ratio, the HS sweep. Odd: exp(-dt K/2), the overlap ratio and the
    population-control shift. Same RNG use, arithmetic and order as the original."""
    floor, cap = float(params.weight_floor), float(params.weight_cap)
    damping, dt = float(params.pop_control_damping), float(params.dt)

    def half_step(state: PropState, index, data: DeviceData) -> PropState:
        even = index % 2 == 0
        nw, L = state.walkers[0].shape[:2]
        key, subkey = jax.random.split(state.rng_key)
        randoms = jax.random.uniform(subkey, (nw, L))
        key = jnp.where(even, key, state.rng_key)
        wu = data.exp_h1_half @ state.walkers[0]
        wd = data.exp_h1_half @ state.walkers[1]
        ca, cb, now, after, factor, node_step = chunked(
            lambda a, b, r: ops.step_batch(a, b, r, even, data, floor), n_chunks, wu, wd, randoms)

        ratio = constrain_ratio(now / state.overlaps, floor)
        nodes = jnp.sum(ratio <= 0.0, dtype=jnp.int64) + jnp.sum(node_step, dtype=jnp.int64)
        weights = state.weights * ratio
        weights = jnp.where(weights > cap, 0.0, weights) * factor

        shifted = weights * jnp.exp(dt * state.pop_control_ene_shift)
        shifted = jnp.where(shifted > cap, 0.0, shifted)
        average = jnp.clip(jnp.mean(shifted), min=1.0e-300)
        shift = state.e_estimate - damping * jnp.log(average) / dt
        return PropState((ca, cb), jnp.where(even, weights, shifted), after, key,
                         jnp.where(even, state.pop_control_ene_shift, shift), state.e_estimate,
                         state.node_encounters + nodes)

    return half_step


def make_block(ops: GpuOps, params: QmcParams, n_chunks: int, energy_chunks: int, record_comb=False):
    """trot.prop.blocks.block for these ops: n_prop_steps steps, orthonormalise,
    measure the energy, comb. Overlaps after orthonormalisation are rescaled by
    det(R) and after the comb gathered, instead of reconverting every walker.
    record_comb: also return the comb's input, per walker: its weight at the
    measurement (pre_comb_weights) and the copy map (comb_index: walker i after the
    comb is a copy of walker comb_index[i] before it)."""
    half_step = make_half_step(ops, params, n_chunks)
    n_half = 2 * params.n_prop_steps

    def block(state: PropState, data: DeviceData):
        state, _ = lax.scan(lambda s, i: (half_step(s, i, data), None), state, jnp.arange(n_half))
        qu, du = ops.batch_qr(state.walkers[0])
        qd, dd = ops.batch_qr(state.walkers[1])
        overlaps = state.overlaps / (du * dd)

        e_samples = chunked(lambda a, b: ops.energies(a, b, data), energy_chunks, qu, qd)
        thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
        e_ref = state.e_estimate
        is_nan = ~jnp.isfinite(e_samples)
        e_samples = jnp.where(is_nan | (jnp.abs(e_samples - e_ref) > thresh), e_ref, e_samples)
        weights = jnp.where(is_nan, 0.0, state.weights)
        w_sum = jnp.sum(weights)
        w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
        e_block = jnp.sum(weights * e_samples) / w_sum_safe
        e_block = jnp.where(w_sum == 0, e_ref, e_block)
        alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
        e_estimate = (1.0 - alpha) * state.e_estimate + alpha * e_block

        key, subkey = jax.random.split(state.rng_key)
        zeta = jax.random.uniform(subkey)
        n = weights.shape[0]
        idx = wk._sr_indices(weights, zeta, n)
        average = jnp.cumsum(jnp.abs(weights))[-1] / n
        state = PropState((qu[idx], qd[idx]), jnp.full((n,), average, weights.dtype), overlaps[idx], key,
                          state.pop_control_ene_shift, e_estimate, state.node_encounters)
        scalars = dict(energy=e_block, weight=w_sum)
        if record_comb:
            scalars.update(pre_comb_weights=weights, comb_index=idx.astype(jnp.int32))
        return state, scalars

    return block


def init_state(ops: GpuOps, system: System, trial_data, params: QmcParams, probe_fn=None):
    """trot's init_prop_state: every walker starts at the natural orbitals of the
    placeholder trial's rdm1; all are identical, so one walker is converted."""
    wu, wd = wk.init_walkers(sys=system, rdm1=uhf_get_rdm1(trial_data), n_walkers=params.n_walkers)
    wu, wd = jnp.real(wu), jnp.real(wd)
    probe = (probe_fn or jax.jit(ops.probe))(wu[0], wd[0], ops.data)
    n = params.n_walkers
    # Two separate buffers (the state is donated), and strong dtypes throughout:
    # a weakly typed scalar would not match the compiled run_blocks after one call.
    e = float(probe["energy"])
    state = PropState((wu.astype(jnp.float64), wd.astype(jnp.float64)), jnp.ones((n,), jnp.float64),
                      jnp.full((n,), probe["overlap"], jnp.float64), jax.random.PRNGKey(int(params.seed)),
                      jnp.asarray(e, jnp.float64), jnp.asarray(e, jnp.float64), jnp.zeros((), jnp.int64))
    return state, probe


def make_block_logger(path, n_equilibration, tag=""):
    """Append each block's scalars to a JSONL file the moment the block finishes:
    raw values, before outlier rejection, so partial runs survive."""
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

    def log(state, scalars):
        jax.experimental.io_callback(write, result_spec, scalars["energy"], scalars["weight"],
                                     state.e_estimate, state.node_encounters, ordered=True)
    return log


def make_run_blocks(block_fn, logger=None):
    donate = () if jax.default_backend() == "cpu" else (0,)

    @partial(jax.jit, static_argnames=("n_blocks",), donate_argnums=donate)
    def run_blocks(state, data, n_blocks):
        def one(state, _):
            state, scalars = block_fn(state, data)
            if logger is not None:
                logger(state, scalars)
            return state, scalars
        return lax.scan(one, state, None, length=n_blocks)
    return run_blocks


def compiled_memory(compiled):
    try:
        m = compiled.memory_analysis()
        return dict(temp_bytes=int(m.temp_size_in_bytes), argument_bytes=int(m.argument_size_in_bytes),
                    output_bytes=int(m.output_size_in_bytes))
    except Exception:  # pragma: no cover - not every backend reports it
        return {}


def peak_device_bytes():
    try:
        return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use")
    except Exception:  # pragma: no cover
        return None


class WalkerSnapshots:
    """Every block's walkers, in the format fixed_block_walkers.ipynb and entanglement_vs_gmps.ipynb load.

    Snapshot 0 is the starting population and snapshot b the population after block b, after its comb (where
    every walker has the same weight): up, dn of shape (n_snap, n_walkers, L, N_sigma), float64, at imaginary
    time tau_snapshots = b * n_steps * dt. Per block b (the one ending at tau_blocks[b] and producing snapshot
    b + 1): energies, weights (the block's energy and total weight), e_estimate and node_encounters (cumulative),
    and the comb's input: pre_comb_weights[b, j] is walker j's weight at the energy measurement, and
    comb_index[b, i] = j says walker i of snapshot b + 1 is a copy of that walker j. So walker i of snapshot
    b + 1 carried the weight pre_comb_weights[b, comb_index[b, i]] before the comb.

    Arrays are written block by block into <name>.parts/ (progress.json says how many are valid), so a crash keeps
    every finished block; finish() packs them with the config into <name>.npz and removes the parts.
    """

    def __init__(self, path, n_snap, n_walkers, L, n_up, n_down, config=None):
        from numpy.lib.format import open_memmap
        self.config = dict(config or {})  # stored in progress.json too, so unfinished runs can be plotted
        self.path = Path(path)
        self.parts = self.path.with_name(self.path.stem + ".parts")
        self.parts.mkdir(parents=True, exist_ok=True)
        memmap = lambda name, shape, dtype=np.float64: open_memmap(self.parts / f"{name}.npy", mode="w+",
                                                                   dtype=dtype, shape=shape)
        self.up = memmap("up", (n_snap, n_walkers, L, n_up))
        self.dn = memmap("dn", (n_snap, n_walkers, L, n_down))
        self.pre_comb_weights = memmap("pre_comb_weights", (n_snap - 1, n_walkers))
        self.comb_index = memmap("comb_index", (n_snap - 1, n_walkers), np.int32)
        self.snapshots, self.blocks = 0, []
        print(f"walker snapshots: {n_snap} x {n_walkers} walkers -> {self.path} "
              f"({(self.up.nbytes + self.dn.nbytes) / 1e9:.1f} GB)", flush=True)

    def add_snapshot(self, state):
        self.up[self.snapshots] = np.asarray(state.walkers[0])
        self.dn[self.snapshots] = np.asarray(state.walkers[1])
        self.snapshots += 1

    def add_block(self, state, scalars):
        """After one block (a run_blocks call with n_blocks=1): its scalars, the comb data and the new walkers."""
        b = len(self.blocks)
        self.pre_comb_weights[b] = np.asarray(scalars["pre_comb_weights"])[-1]
        self.comb_index[b] = np.asarray(scalars["comb_index"])[-1]
        self.blocks.append(dict(energy=float(np.asarray(scalars["energy"])[-1]),
                                weight=float(np.asarray(scalars["weight"])[-1]),
                                e_estimate=float(state.e_estimate), node_encounters=int(state.node_encounters)))
        self.add_snapshot(state)
        for array in (self.up, self.dn, self.pre_comb_weights, self.comb_index):
            array.flush()
        (self.parts / "progress.json").write_text(json.dumps(dict(snapshots=self.snapshots, blocks=self.blocks,
                                                                  config=self.config)))

    def finish(self, config, arrays=None):
        config = {**self.config, **config}
        n, nb = self.snapshots, len(self.blocks)
        step = config["N_PROP"] * config["DT"]
        column = lambda key, dtype=float: np.array([blk[key] for blk in self.blocks], dtype=dtype)
        np.savez(self.path, up=self.up[:n], dn=self.dn[:n], energies=column("energy"), weights=column("weight"),
                 e_estimate=column("e_estimate"), node_encounters=column("node_encounters", np.int64),
                 pre_comb_weights=self.pre_comb_weights[:nb], comb_index=self.comb_index[:nb],
                 tau_snapshots=np.arange(n) * step, tau_blocks=np.arange(1, nb + 1) * step,
                 config=json.dumps(config), **(arrays or {}))
        del self.up, self.dn, self.pre_comb_weights, self.comb_index
        import shutil
        shutil.rmtree(self.parts)
        print(f"saved {self.path}: {n} snapshots, {nb} blocks", flush=True)


def export_trial(path, cfg, trial_np, trial_charges, dmrg_energy, references, start):
    """The DMRG trial in the format fixed_block_walkers.ipynb loads (its DMRG_FILE): T{i} (d=4 tensors, local
    index n_up + 2 n_dn), H{i} (H|trial>, dense-compressed exactly as the notebook builds it), q{i} (the
    (N_up, N_dn) bond labels) and e_dmrg; plus its spin-resolved one-body density matrices (gamma), the plan
    reference determinant (reference_up/dn: the natural orbitals for plan_reference="natural") and the walkers'
    starting determinant (start_up/dn)."""
    H = compress_mps(apply_mpo(hubbard_mpo(cfg.L, cfg.hopping, cfg.interaction), trial_np))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, e_dmrg=dmrg_energy, gamma=np.stack(one_rdm(trial_np)),
                        reference_up=np.asarray(references[0]), reference_dn=np.asarray(references[1]),
                        start_up=np.asarray(start[0]), start_dn=np.asarray(start[1]),
                        **{f"T{i}": np.asarray(A) for i, A in enumerate(trial_np)},
                        **{f"H{i}": np.asarray(A) for i, A in enumerate(H)},
                        **{f"q{i}": np.asarray(q) for i, q in enumerate(trial_charges)})
    print(f"saved the DMRG trial to {path}", flush=True)


def run_qmc(state, data, run_blocks, *, n_eql, n_blocks, n_walkers, n_steps, snapshots=None):
    """run_qmc_fixed_chunks with an explicit AOT compile (timed) and throughput.
    With snapshots (a WalkerSnapshots), one block per call and every block's walkers saved.
    Returns (mean, stderr, energies, weights, collapsed_after_block, timing)."""
    chunk = 1 if snapshots is not None else math.gcd(n_eql, n_blocks)
    total = n_eql + n_blocks
    t0 = time.perf_counter()
    compiled = run_blocks.lower(state, data, n_blocks=chunk).compile()
    compile_seconds = time.perf_counter() - t0
    print(f"compiled run_blocks ({chunk} blocks) in {compile_seconds:.1f} s; memory {compiled_memory(compiled)}",
          flush=True)

    energies, weights, collapsed, chunk_times = [], [], None, []
    if snapshots is not None:
        snapshots.add_snapshot(state)  # snapshot 0, read before the first call donates the state
    start = time.perf_counter()
    for done in range(chunk, total + 1, chunk):
        state, scalars = compiled(state, data)
        if snapshots is not None:
            snapshots.add_block(state, scalars)
        e, w = np.asarray(scalars["energy"]), np.asarray(scalars["weight"])
        chunk_times.append((done, time.perf_counter() - start))
        energies.extend(e.tolist())
        weights.extend(w.tolist())
        print(f"[{'eql' if done <= n_eql else 'blk'} {done:4d}/{total}]  E_chunk {np.sum(e * w) / np.sum(w):14.10f}"
              f"  W {w.mean():12.6e}  nodes {int(state.node_encounters):10d}"
              f"  t {time.perf_counter() - start:8.1f} s", flush=True)
        if not w[-1] > 0.0:
            collapsed = done
            print(f"\nPopulation collapsed: total weight is zero after block {done}. Stopping.", flush=True)
            break

    seconds = chunk_times[-1][1] if chunk_times else 0.0
    rate = n_walkers * n_steps * (chunk_times[-1][0] if chunk_times else 0) / max(seconds, 1e-300)
    timing = dict(compile_seconds=compile_seconds, run_seconds=seconds, walker_steps_per_s=rate,
                  peak_bytes=peak_device_bytes(), **compiled_memory(compiled))
    print(f"throughput {rate:.1f} walker-steps/s over {seconds:.1f} s (compile excluded)", flush=True)
    if collapsed is not None:
        return float("nan"), float("nan"), np.asarray(energies), np.asarray(weights), collapsed, timing
    sampled = np.column_stack((energies[n_eql:], weights[n_eql:]))
    clean, _ = reject_outliers(sampled, obs=0)
    print(f"\nRejected {len(sampled) - len(clean)} outlier blocks.\n\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(np.asarray(clean[:, 0]), np.asarray(clean[:, 1]), print_q=True)
    return stats["mu"], stats["se_star"], np.asarray(energies), np.asarray(weights), None, timing


def save_result(path, record):
    if not path:
        return
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record) + "\n")


# ============================================================================
# Setup and main
# ============================================================================

class Setup(NamedTuple):
    cfg: Config
    system: System
    ham: HamHubbard
    ops: GpuOps
    params: QmcParams
    trial_data: UhfTrial
    plans: tuple
    bonds: tuple
    n_chunks: int
    energy_chunks: int
    memory: dict
    info: dict
    references: tuple  # plan reference determinants (Ra, Rb)
    trial: tuple  # (dense trial tensors, their labels)


def resolve_backend_options(cfg: Config):
    cpu = jax.default_backend() == "cpu"
    linalg = cfg.linalg if cfg.linalg != "auto" else ("native" if cpu else "batched")
    linalg = "batched" if linalg == "eigh" else linalg  # the earlier name
    walker_qr = cfg.walker_qr if cfg.walker_qr != "auto" else ("native" if cpu else "cholesky")
    return linalg, walker_qr


def build(cfg: Config = CFG, verbose=True) -> Setup:
    """Everything up to the initial state: trial, plans, compiled circuit, device
    data and the chunking. main() runs it; the benchmark reuses it."""
    say = print if verbose else (lambda *a, **k: None)
    if cfg.compile_cache:
        jax.config.update("jax_compilation_cache_dir", cfg.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    device = jax.devices()[0]
    linalg, walker_qr = resolve_backend_options(cfg)
    say(f"jax {jax.__version__}, backend {jax.default_backend()}, device {device.device_kind}; "
        f"linalg={linalg}, walker_qr={walker_qr}, energy={cfg.energy}", flush=True)

    h1 = hopping_matrix(cfg.L, cfg.hopping)
    ham = HamHubbard(h1=jnp.asarray(h1), u=cfg.interaction)
    system = System(norb=cfg.L, nelec=(cfg.n_up, cfg.n_down), walker_kind="unrestricted")
    for option in ("plan_reference", "walker_start"):
        if getattr(cfg, option) not in ("rhf", "natural"):
            raise ValueError(f"{option} must be 'rhf' or 'natural'")
    _, orbitals = np.linalg.eigh(h1)
    Ca, Cb = orbitals[:, :cfg.n_up].copy(), orbitals[:, :cfg.n_down].copy()
    say(f"L={cfg.L} ({cfg.n_up},{cfg.n_down}), U={cfg.interaction}")

    trial_np, trial_charges, dmrg_energy = load_or_run_trial(cfg)
    trial = tuple(jnp.asarray(A) for A in trial_np)
    np.testing.assert_allclose(float(contract_real(trial, trial)), 1.0, atol=1e-10)
    mpo = hubbard_mpo(cfg.L, cfg.hopping, cfg.interaction)
    htrial_qn = compress_mps_qn(*apply_mpo_qn(mpo, trial_np, trial_charges))
    Htrial = tuple(jnp.asarray(A) for A in htrial_qn[0])
    trial_energy = float(contract_real(Htrial, trial) / contract_real(trial, trial))
    say(f"DMRG Davidson energy={dmrg_energy:.12f}; dense-MPS expectation={trial_energy:.12f}")
    say("trial bonds:", [A.shape[0] for A in trial_np] + [trial_np[-1].shape[-1]])
    say("H|trial> compressed bonds (charge-labelled):", [A.shape[0] for A in htrial_qn[0]] + [1])

    determinants = {"rhf": (Ca, Cb)}
    if "natural" in (cfg.plan_reference, cfg.walker_start):
        gamma_a, gamma_b = one_rdm(trial_np)
        np.testing.assert_allclose([np.trace(gamma_a), np.trace(gamma_b)], [cfg.n_up, cfg.n_down], atol=1e-8)
        Na, occupations = natural_orbitals(gamma_a, cfg.n_up)
        Nb, _ = natural_orbitals(gamma_b, cfg.n_down)
        determinants["natural"] = (Na, Nb)
        say(f"trial natural orbitals: alpha gap n_N - n_N+1 = {occupations[cfg.n_up - 1] - occupations[cfg.n_up]:.3f}")

    Ra, Rb = determinants[cfg.plan_reference]
    plan_a = make_orbital_plan(Ra, cfg.orbital_plan, cfg.occupation_tolerance)
    plan_b = make_orbital_plan(Rb, cfg.orbital_plan, cfg.occupation_tolerance)
    gates = lambda plan: int(plan.block_sizes.sum() - len(plan.block_sizes))
    say(f"plan reference: {cfg.plan_reference} determinant; gates {gates(plan_a)}, {gates(plan_b)}")

    Sa, Sb = determinants[cfg.walker_start]
    trial_data = UhfTrial(mo_coeff_a=jnp.asarray(Sa), mo_coeff_b=jnp.asarray(Sb))
    Pa, Pb = Sa @ Sa.T, Sb @ Sb.T
    initial_energy = float(np.sum(h1 * (Pa + Pb)) + cfg.interaction * np.diag(Pa) @ np.diag(Pb))

    def plan_fidelity(S, plan):
        _, rows = channel_angles(S, plan, xp=np)
        return float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(plan.occupation)]))) ** 2
    say(f"walkers start from the {cfg.walker_start} determinant, E={initial_energy:.12f}; orbital-plan infidelity "
        f"{1 - plan_fidelity(Sa, plan_a):.1e}, {1 - plan_fidelity(Sb, plan_b):.1e}")

    bond_a = bond_b = None
    if cfg.walker_channel_chi is not None or cfg.walker_cutoff:
        bond_a = plan_bonds(Ra, plan_a, cfg.walker_channel_chi, cfg.walker_cutoff)
        bond_b = plan_bonds(Rb, plan_b, cfg.walker_channel_chi, cfg.walker_cutoff)
        say(f"walker truncation reference discarded weights: {bond_a.reference_discarded_weight:.3e}, "
            f"{bond_b.reference_discarded_weight:.3e}")

    prop_ctx = _build_prop_ctx(ham, cfg.dt)
    htrial = htrial_qn if cfg.energy == "blocked" else tuple(
        jnp.asarray(A) for A in compress_mps(apply_mpo(mpo, trial_np)))
    ops = make_gpu_ops(plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, htrial, prop_ctx,
                       linalg=linalg, walker_qr=walker_qr, spin_batch=cfg.spin_batch, energy=cfg.energy)

    circuit = circuit_stats(ops.converter.circuits[0])
    qa, qb = ops.converter.charges
    say("walker channel bonds:", [len(q) for q in qa])
    say(f"conversion circuit: {circuit}; spin-batched={ops.converter.spin_batched}")
    if linalg == "batched" and not ops.converter.spin_batched:
        say("note: spins converted separately (their orbital plans differ in particle number or gate sequence)")
    if circuit["eigh_over_32"] and linalg == "batched":
        say(f"note: {circuit['eigh_over_32']} factorisations exceed 32x32; check gpu_linalg_bench.py for "
            "whether batched eigh stays batched beyond cuSOLVER's Jacobi limit")
    say("overlap plan:", ops.overlap_plan.stats)
    if ops.energy_plan is not None:
        say("energy plan: ", ops.energy_plan.stats)

    memory = memory_model(ops)
    limit = device_bytes_limit()
    budget = None if limit is None else cfg.mem_fraction * limit - memory["data_bytes"]
    n_chunks = cfg.n_chunks or choose_chunks(cfg.n_walkers, memory["step_bytes_per_walker"], budget)
    energy_chunks = cfg.n_chunks or choose_chunks(cfg.n_walkers, memory["energy_bytes_per_walker"], budget)
    if cfg.n_walkers % n_chunks or cfg.n_walkers % energy_chunks:
        raise ValueError("n_chunks must divide n_walkers")
    say(f"memory model: {memory}; device limit {limit}; n_chunks {n_chunks} (energy {energy_chunks})")

    params = QmcParams(dt=cfg.dt, n_walkers=cfg.n_walkers, n_prop_steps=cfg.n_steps,
                       n_blocks=cfg.n_blocks, n_eql_blocks=cfg.n_equilibration,
                       weight_floor=cfg.weight_floor, seed=cfg.seed, n_chunks=n_chunks)
    info = dict(dmrg_energy=dmrg_energy, trial_energy=trial_energy, initial_energy=initial_energy,
                gates=gates(plan_a), circuit=circuit, linalg=linalg, walker_qr=walker_qr,
                spin_batched=ops.converter.spin_batched, device=device.device_kind,
                backend=jax.default_backend(), walker_d4_chi=max(len(x) * len(y) for x, y in zip(qa, qb)),
                overlap_plan=ops.overlap_plan.stats)
    return Setup(cfg, system, ham, ops, params, trial_data, (plan_a, plan_b), (bond_a, bond_b),
                 n_chunks, energy_chunks, memory, info, (Ra, Rb), (trial_np, trial_charges))


def main(cfg=CFG):
    setup = build(cfg)
    ops, params = setup.ops, setup.params

    start = time.perf_counter()
    state, probe = init_state(ops, setup.system, setup.trial_data, params)
    print(f"initial overlap {float(probe['overlap']):.6e}, local energy {float(probe['energy']):.10f} "
          f"({time.perf_counter() - start:.1f} s incl. compile)", flush=True)
    if cfg.self_check:
        errors = conversion_self_check(ops, setup.plans, setup.bonds, probe)
        print(f"self-check vs NumPy channel_mps: relative error alpha {errors[0]:.1e}, beta {errors[1]:.1e}")
        if max(errors) > 1e-6:
            raise AssertionError(f"device conversion disagrees with the NumPy reference: {errors}")

    start_det = (np.asarray(setup.trial_data.mo_coeff_a), np.asarray(setup.trial_data.mo_coeff_b))
    if cfg.trial_export:
        export_trial(cfg.trial_export, cfg, *setup.trial, setup.info["dmrg_energy"], setup.references, start_det)

    logger = make_block_logger(cfg.block_log, cfg.n_equilibration, cfg.tag) if cfg.block_log else None
    snapshots = None
    if cfg.walker_snapshots:
        # the notebooks' config keys (N_PROP, DT, N_EQL, ...), and this run's settings
        config = dict(L=cfg.L, N_UP=cfg.n_up, N_DN=cfg.n_down, T=cfg.hopping, U=cfg.interaction,
                      N_WALKERS=cfg.n_walkers, N_EQL=cfg.n_equilibration, N_BLOCKS=cfg.n_blocks, N_PROP=cfg.n_steps,
                      DT=cfg.dt, SEED=cfg.seed, DMRG_CHI_T=cfg.trial_chi, DMRG_SWEEPS=cfg.dmrg_sweeps,
                      CHI_PROP=cfg.walker_channel_chi, E_DMRG=setup.info["dmrg_energy"],
                      E_TRIAL=setup.info["trial_energy"], plan_reference=cfg.plan_reference,
                      walker_start=cfg.walker_start, orbital_plan=cfg.orbital_plan, EPS=cfg.occupation_tolerance,
                      trial_file=cfg.trial_export, tag=cfg.tag, module="mps_cpmc_gpu", device=setup.info["device"])
        snapshots = WalkerSnapshots(cfg.walker_snapshots, cfg.n_equilibration + cfg.n_blocks + 1, cfg.n_walkers,
                                    cfg.L, cfg.n_up, cfg.n_down, config)
    block_fn = make_block(ops, params, setup.n_chunks, setup.energy_chunks, record_comb=snapshots is not None)
    run_blocks = make_run_blocks(block_fn, logger)

    start = time.perf_counter()
    mean, error, block_energies, block_weights, collapsed, timing = run_qmc(
        state, ops.data, run_blocks, n_eql=cfg.n_equilibration, n_blocks=cfg.n_blocks,
        n_walkers=cfg.n_walkers, n_steps=cfg.n_steps, snapshots=snapshots)
    elapsed = time.perf_counter() - start

    scalar = lambda x: None if x is None else float(x)
    print(f"CPMC energy = {scalar(mean)} +/- {scalar(error)}; elapsed={elapsed:.1f} s")
    if snapshots is not None:
        snapshots.finish(dict(E_CPMC=scalar(mean), E_CPMC_ERR=scalar(error), collapsed_after_block=collapsed),
                         dict(reference_up=np.asarray(setup.references[0]),
                              reference_dn=np.asarray(setup.references[1]),
                              start_up=start_det[0], start_dn=start_det[1]))
    record = asdict(cfg)
    info = dict(setup.info)
    record.update(
        kind="mps_gpu", initial_energy=info.pop("initial_energy"), dmrg_energy=info.pop("dmrg_energy"),
        trial_energy=info.pop("trial_energy"), cpmc_energy=scalar(mean), cpmc_error=scalar(error),
        seconds=elapsed, collapsed_after_block=collapsed, n_chunks_used=setup.n_chunks,
        energy_chunks=setup.energy_chunks, memory_model=setup.memory, **timing, **info)
    save_result(cfg.result_json, record)


if __name__ == "__main__":
    main()
