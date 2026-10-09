"""Batched engine for MPS-CPMC with Slater-determinant walkers, on CPU and GPU alike.

trot's MPS ops (trot.trial.mps, trot.meas.mps, trot.prop.mps_cpmc) run on it: kernels_for(plan, ...) builds, and
caches on the walker plan, the conversion circuit, the contraction layouts and the jitted kernels; the trial and
H|trial> blocks come from meas_ctx and exp(-dt K/2) and the HS factors from prop_ctx, assembled as DeviceData.

The algorithm is that of the host conversion (trot.gmps.utils.channel_mps, channel_mps_host below, the oracle of the
self-check and the tests): the walkers stay Slater determinants, each spin channel is converted to a charge-labelled
d=2 MPS with Fishman-White gates, truncating gate by gate with the orthogonality centre on the gate. What changes is
the layout, built for batches:

* compile_circuit turns every centre move and split into static gathers into padded batches of charge-sector
  blocks; each sector is factored by the reference method (_factor_block: closed form for one row/column, QR when
  exact, the Gram eigh when truncating), batched over walkers x spins x sectors per kind; both spin channels run
  in one batch when their gate sequences match (spin_batch), and size buckets keep the padding small.
* walker-trial contractions never form the d=4 walker: factorized (alpha, beta, trial) blocks per shared label,
  Pa*Pb*Pt*(Pa+Pb+Pt) per transition (make_factorized_plan, left_contract, right_environments, field_sweep).
  The trial may carry (N_up, N_dn) labels or particle-number labels only (a spin-rotated MpsTrial used as it is,
  see number_labels): then each walker label (N_alpha, N_beta) meets the trial's N_alpha + N_beta sector.
* walker QR by CholeskyQR2 with a Householder fallback (make_batch_qr("cholesky")).
* everything large (trial and H|trial> blocks, exp(-dt K/2), HS factors) is a jit argument (DeviceData), not an
  HLO constant; walkers are chunked with lax.map in params.n_chunks pieces (chunked; trot's run_qmc picks n_chunks
  from the compiled memory).

Host-side primitives (orbital and bond plans, sector plans, block factorisation) come from trot.gmps.utils.
"""
from __future__ import annotations

import itertools
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
from jax import lax

from trot.gmps.utils import (
    BondPlan,
    OrbitalPlan,
    _assemble,
    _block_plan,
    _factor_block,
    _key,
    _move_centre,
    channel_angles,
    combine_channels,
    contract_real,
    gate_pair,
    label_array,
    null_mode,
    sector_plan,
)
from trot.prop.types import PropState, QmcParams
from trot.walkers import _qr as qr_with_det

def constrain_ratio(ratio, weight_floor):
    """trot's cpmc_step rule: zero every overlap ratio at or below the floor.

    This is also the constrained-path condition (a sign change gives ratio <= 0),
    so it cannot be dropped; weight_floor=0 keeps only the constraint.
    """
    return jnp.where(ratio <= weight_floor, 0.0, ratio)


def _charge_index(labels):
    grouped = {}
    for i, charge in enumerate(map(tuple, np.asarray(labels).tolist())):
        grouped.setdefault(charge, []).append(i)
    return {charge: np.asarray(indices, int) for charge, indices in grouped.items()}


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


# ============================================================================
# Static circuit: every centre move and split of channel_mps, as gather maps
# ============================================================================
#
# One compiled circuit converts a group of spin channels (both, whenever they
# share a gate sequence) in one batched pass. Each channel keeps its own charge
# sectors and kept ranks: tensors are stored zero-padded to the largest bond of
# the group and every op gathers each channel's sector blocks with its own index
# maps. Every sector is factored by the method trot.gmps.utils._factor_block uses
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
    """The branch trot.gmps.utils._factor_block takes for a block of this shape."""
    if min(n_rows, n_cols) == 1:
        return "row1" if n_rows == 1 else "col1"
    if not truncate:
        return "qr"
    return "eigh_rows" if n_rows <= n_cols else "eigh_cols"


def _bucket(name, n_rows, n_cols, buckets):
    """The size bucket of a sector of kind `name`: the index of the first bound in `buckets` that its size (the
    Gram dimension for eigh kinds, the larger side for QR) does not exceed, len(buckets) above all of them.
    buckets=None puts every sector of a kind in bucket 0: one class per kind, the original layout."""
    if buckets is None or name in ("row1", "col1"):
        return 0
    size = n_rows if name == "eigh_rows" else n_cols if name == "eigh_cols" else max(n_rows, n_cols)
    return next((i for i, bound in enumerate(buckets) if size <= bound), len(buckets))


def _group_op(kind, site, step, gate, specs, n_rows, n_cols, buckets=None) -> FactorOp:
    """specs: per channel (row_map, col_map, SectorPlan, truncate flags), where the
    maps take the channel's own row/column indices to the padded matrix.

    buckets: increasing size bounds. Each kind's sectors are split by size into one class per bucket, each padded
    to its own largest member instead of the op's largest: the same factorisations on smaller padded matrices
    (and smaller cuSOLVER kernels, e.g. the 16x16 batched Jacobi instead of the 32x32 one)."""
    spins = len(specs)
    K = max(len(plan.middle_charges) for _, _, plan, _ in specs)
    members = {}  # (kind, bucket) -> per channel, its sectors of that class
    for sigma, (row_map, col_map, plan, truncate) in enumerate(specs):
        offset = 0
        for (ri, ci, rank), trunc in zip(plan.sectors, truncate):
            name = _sector_kind(len(ri), len(ci), trunc)
            key = (name, _bucket(name, len(ri), len(ci), buckets))
            per_spin = members.setdefault(key, [[] for _ in range(spins)])
            per_spin[sigma].append((row_map[ri], col_map[ci], rank, offset))
            offset += rank
        assert offset == len(plan.middle_charges)

    classes = []
    q_index = np.full((spins, n_rows, K), -1, np.int64)
    r_index = np.full((spins, K, n_cols), -1, np.int64)
    q_base = r_base = 0
    for key in sorted(members, key=lambda kb: (SECTOR_KINDS.index(kb[0]), kb[1])):
        name, per_spin = key[0], members[key]
        S = max(len(x) for x in per_spin)
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
        q_base += S * r * k
        r_base += S * k * c
    q_index[q_index < 0] = q_base  # the zero appended after the concatenated blocks
    r_index[r_index < 0] = r_base
    return FactorOp(kind, site, step, gate, n_rows, n_cols, K, tuple(classes), q_index, r_index)


def _move_op(charges, pads, site, step, buckets=None):
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
    op = _group_op("move", site, step, -1, specs, n_rows, n_cols, buckets=buckets)
    bond = site + 1 if step > 0 else site
    for q, (_, _, plan, _) in zip(charges, specs):
        q[bond] = plan.middle_charges
    pads[bond] = op.K
    return op


def _gate_op(charges, pads, site, gate, kept, buckets=None):
    """The static part of split_pair for every channel; updates labels and pads."""
    Pl, Pr = pads[site], pads[site + 2]
    specs = []
    for q, kept_q in zip(charges, kept):
        ql, qr = q[site], q[site + 2]
        Dl, Dr = len(ql), len(qr)
        split = sector_plan(ql, qr, kept_q)
        truncate = [rank < min(len(r), len(c)) for r, c, rank in split.sectors]
        # M = pair.reshape(2 Dl, 2 Dr): row a*2 + p, column q*Dr + b
        row_map = np.arange(2 * Dl)
        col_map = (np.arange(2)[:, None] * Pr + np.arange(Dr)[None, :]).ravel()
        specs.append((row_map, col_map, split, truncate))
    op = _group_op("gate", site, 0, gate, specs, 2 * Pl, 2 * Pr, buckets)
    for q, (_, _, split, _) in zip(charges, specs):
        q[site + 1] = split.middle_charges
    pads[site + 1] = op.K
    return op


def compile_circuit(plans, bond_plans, buckets=None) -> Circuit:
    """Replay channel_mps symbolically for a group of channels sharing one gate
    sequence. Bond labels depend only on the orbital plans and the frozen kept
    counts, so the whole conversion (which tensor is factored, in which charge
    sectors, by which method, keeping how many vectors) is fixed here once.

    buckets: size bounds for splitting each factorisation's sector classes (see _group_op); None keeps one
    class per sector kind. The arithmetic per sector is the same either way."""
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
                ops.append(_move_op(charges, pads, centre, step, buckets))
                centre += step
        kept = [None if b is None else b.kept_per_sector[g] for b in bond_plans]
        ops.append(_gate_op(charges, pads, site, g, kept, buckets))
        centre = site + 1
    return Circuit(np.stack([np.asarray(p.occupation) for p in plans]), tuple(sites), tuple(ops),
                   tuple(tuple(q) for q in charges), tuple(pads))


def padding_stats(circuit: Circuit):
    """How much of the batched factorisation work is padding, per walker and conversion.

    Every class pads its sectors (and the channels' missing sectors) to one shape. The work of a matrix is
    n^3 for an eigh of an n x n Gram matrix and r c min(r, c) for a QR of an r x c block; *_waste is the share
    of the padded total that falls on padding. eigh_le16 counts the batched eigh calls whose matrices fit
    cuSOLVER's 16x16 Jacobi kernel (the 32x32 one costs about 6x more per call); eigh_n_* describe the real
    Gram sizes (median, 90th percentile, largest)."""
    eigh_real = eigh_padded = qr_real = qr_padded = 0
    sizes, le16 = [], 0
    for op in circuit.ops:
        sentinel = op.n_rows * op.n_cols
        for cls in op.classes:
            real = cls.gather != sentinel  # (spins, S, r, c): the sectors' own entries
            rows, cols = real.any(axis=-1).sum(axis=-1), real.any(axis=-2).sum(axis=-1)
            spins, S, r, c = cls.gather.shape
            if cls.pad is not None:
                n = cls.pad.shape[-1]
                own = rows if cls.kind == "eigh_rows" else cols
                eigh_padded += spins * S * n ** 3
                eigh_real += int(np.sum(own.astype(np.int64) ** 3))
                sizes += own[own > 0].tolist()
                le16 += n <= 16
            elif cls.kind == "qr":
                qr_padded += spins * S * r * c * min(r, c)
                qr_real += int(np.sum(rows * cols * np.minimum(rows, cols)))
    waste = lambda real, padded: round(1.0 - real / padded, 3) if padded else 0.0
    sizes = np.asarray(sizes if sizes else [0])
    return dict(eigh_waste=waste(eigh_real, eigh_padded), qr_waste=waste(qr_real, qr_padded), eigh_le16=int(le16),
                eigh_n_median=int(np.median(sizes)), eigh_n_p90=int(np.percentile(sizes, 90)),
                eigh_n_max=int(sizes.max()))


def circuit_stats(circuit: Circuit):
    classes = [cls for op in circuit.ops for cls in op.classes]
    eigh = [cls.pad.shape[-1] for cls in classes if cls.pad is not None]
    qr = [max(cls.gather.shape[-2:]) for cls in classes if cls.kind == "qr"]
    return dict(spins=int(circuit.occupation.shape[0]), ops=len(circuit.ops), gates=len(circuit.gate_sites),
                moves=sum(op.kind == "move" for op in circuit.ops),
                closed_form_batches=sum(cls.kind in ("row1", "col1") for cls in classes),
                qr_calls=len(qr), eigh_calls=len(eigh), max_qr=max(qr, default=0), max_eigh=max(eigh, default=0),
                eigh_over_32=sum(n > 32 for n in eigh), max_bond=max(circuit.pads), **padding_stats(circuit))


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
    each sector by the same method as trot.gmps.utils._factor_block, one batched
    call per sector kind."""
    spins = M.shape[0]
    zero = jnp.zeros((spins, 1), M.dtype)
    flat = jnp.concatenate((M.reshape(spins, -1), zero), axis=-1)
    parts = []  # per class: Q (spins, S, r, k), R (spins, S, k, c)
    for cls in op.classes:
        B = _take_per_channel(flat, cls.gather)  # (spins, S, r, c)
        if cls.kind == "row1":  # _vector_qr, one row
            Q, R = jnp.ones(B.shape[:-2] + (1, 1), B.dtype), B
        elif cls.kind == "col1":  # _vector_qr, one column
            norm = jnp.sqrt(jnp.sum(B * B, axis=(-2, -1), keepdims=True))
            first = jnp.zeros(B.shape[-2:], B.dtype).at[0, 0].set(1.0)
            Q = jnp.where(norm > 0, B / jnp.where(norm > 0, norm, 1.0), first)
            R = norm
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
        parts.append((Q, R))
    Qs = [Q.reshape(spins, -1) for Q, _ in parts]
    Rs = [R.reshape(spins, -1) for _, R in parts]
    return (_take_per_channel(jnp.concatenate(Qs + [zero], axis=-1), op.q_index),
            _take_per_channel(jnp.concatenate(Rs + [zero], axis=-1), op.r_index))


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
    sizes: the same null-space / eigh choice per channel as channel_angles, the
    channel's own occupied/empty choice and reference sign."""
    rows = [C[:, i, :] for i in range(C.shape[-2])]
    angles = []
    for k, B in enumerate(plans[0].block_sizes):
        B = int(B)
        if B == 1:
            continue
        block = jnp.stack(rows[k:k + B], axis=-2)  # (spins, B, N)
        occupied = np.array([bool(p.occupation[k]) for p in plans])
        null_step = np.array([bool(p.exact_for_all_walkers and not p.occupation[k]) for p in plans])
        if null_step.any():
            # exact plans: the block is rank-deficient, and any vector of null(block.T) is the required empty mode
            v_null = null_mode(block)
        if not null_step.all():
            _, vectors = jnp.linalg.eigh(block @ jnp.swapaxes(block, -1, -2))
            v_eigh = jnp.where(occupied[:, None], vectors[..., -1], vectors[..., 0])
        if null_step.all():
            v = v_null
        elif not null_step.any():
            v = v_eigh
        else:
            v = jnp.where(null_step[:, None], v_null, v_eigh)
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


def _can_group(plan_a, plan_b):
    """Both channels in one batch: same particle number and gate sequence. Kept
    counts and bond labels may differ (e.g. alpha and beta breaking a tie between
    degenerate Schmidt values differently); each keeps its own."""
    return (int(plan_a.occupation.sum()) == int(plan_b.occupation.sum())
            and np.array_equal(plan_a.block_sizes, plan_b.block_sizes)
            and plan_a.exact_for_all_walkers == plan_b.exact_for_all_walkers)


def make_converter(plan_a, plan_b, bond_a=None, bond_b=None, spin_batch=True, buckets=None) -> Converter:
    """The walker conversion of two spin channels: one compiled circuit per channel group (both channels in one
    batch when spin_batch and their gate sequences match). buckets: size bounds for the sector classes of every
    factorisation (compile_circuit)."""
    plans, bonds = (plan_a, plan_b), (bond_a, bond_b)
    if spin_batch and _can_group(plan_a, plan_b):
        groups = [((0, 1), compile_circuit(plans, bonds, buckets))]
    else:
        groups = [((0,), compile_circuit((plan_a,), (bond_a,), buckets)),
                  ((1,), compile_circuit((plan_b,), (bond_b,), buckets))]
    labels = [None, None]
    for channels, circuit in groups:
        for j, s in enumerate(channels):
            labels[s] = circuit.charges[j]
            if bonds[s] is not None and not all(np.array_equal(x, y) for x, y in zip(labels[s], bonds[s].charges)):
                raise AssertionError("compiled circuit labels differ from plan_bonds' dry run")
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

    return Converter(convert, tuple(labels), tuple(c for _, c in groups), len(groups) == 1)


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
    fixed_rows: np.ndarray  # (T, Pt) rows of the fixed MPS's block of each transition (fill: its left bond dimension)
    fixed_cols: np.ndarray  # (T, Pt') columns (fill: its right bond dimension)
    fixed_keys: tuple  # (left label, p, right label) of each transition, for block-form fixed MPS
    fixed_shapes: tuple  # (rows, columns) of each transition's unpadded block


class ContractionPlan(NamedTuple):
    sites: tuple
    shared: tuple  # shared (N_alpha, N_beta) labels per bond
    pads: tuple  # (Pa, Pb, Pt) per bond
    blocks: tuple | None  # (T, Pt, Pt') blocks of the fixed MPS (trial or H|trial>); None in a bare layout
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


N_PHYSICAL = np.array([0, 1, 1, 2])  # n_alpha + n_beta of p = n_alpha + 2 n_beta


def _fixed_key(width):
    """The fixed MPS's label that a walker label pair c = (N_alpha, N_beta) contracts
    with: c itself for (N_up, N_dn) labels, (N_alpha + N_beta,) for N labels."""
    if width == 2:
        return lambda c: c
    if width == 1:
        return lambda c: (c[0] + c[1],)
    raise ValueError(f"fixed MPS labels must be (N_up, N_dn) pairs or particle numbers, got width {width}")


def make_factorized_layout(qa, qb, fixed_charges) -> ContractionPlan:
    """Layout of <fixed MPS|walker> with the walker kept as its two spin channels.

    Environments at bond i are (n_shared, Pa, Pb, Pt): one padded (alpha, beta,
    fixed) block per shared (N_alpha, N_beta) label. A transition moves label
    (a, b) to (a + n_alpha, b + n_beta) through physical p; its alpha, beta and
    fixed blocks are gathered separately, so the d=4 walker is never formed.

    fixed_charges are (N_up, N_dn) labels, or particle-number labels N (width 1:
    a spin-rotated trial used as it is, see number_labels). With N labels the walker
    label (a, b) meets the fixed MPS's whole N = a + b sector: the walker's channels
    and the interleave sign (-1)^(n_alpha N_beta) keep their own labels, so nothing
    else changes. The layout depends on the labels only; fixed_blocks gathers the
    blocks of any fixed MPS with these labels (blocks is None here).
    """
    n = len(qa) - 1
    ia, ib = [_index_1d(q) for q in qa], [_index_1d(q) for q in qb]
    fixed_charges = [label_array(q) for q in fixed_charges]
    key = _fixed_key(fixed_charges[0].shape[1])
    it = [_charge_index(q) for q in fixed_charges]
    shared = [sorted(c for c in itertools.product(ia[i], ib[i]) if key(c) in it[i]) for i in range(n + 1)]
    pads = [(max((len(ia[i][c[0]]) for c in shared[i]), default=1),
             max((len(ib[i][c[1]]) for c in shared[i]), default=1),
             max((len(it[i][key(c)]) for c in shared[i]), default=1)) for i in range(n + 1)]

    sites = []
    exact = sum(len(ia[i][c[0]]) * len(ib[i][c[1]]) * len(it[i][key(c)]) for i in range(n + 1) for c in shared[i])
    padded = sum(len(shared[i]) * int(np.prod(pads[i])) for i in range(n + 1))
    for site in range(n):
        (Pa, Pb, Pt), (Pa2, Pb2, Pt2) = pads[site], pads[site + 1]
        out_index = {c: j for j, c in enumerate(shared[site + 1])}
        Dla, Dra = len(qa[site]), len(qa[site + 1])
        Dlb, Drb = len(qb[site]), len(qb[site + 1])
        Dlt, Drt = len(fixed_charges[site]), len(fixed_charges[site + 1])
        src, dst, physical, sign, ga, gb, rows, cols, keys, shapes = [], [], [], [], [], [], [], [], [], []
        for s, c in enumerate(shared[site]):
            for p, (na, nb) in enumerate(PHYSICAL):
                c2 = (c[0] + na, c[1] + nb)
                if c2 not in out_index:
                    continue
                ra, ca_ = ia[site][c[0]], ia[site + 1][c2[0]]
                rb, cb_ = ib[site][c[1]], ib[site + 1][c2[1]]
                rt, ct = it[site][key(c)], it[site + 1][key(c2)]
                a = np.full((Pa, Pa2), Dla * 2 * Dra, np.int32)
                a[:len(ra), :len(ca_)] = ra[:, None] * 2 * Dra + na * Dra + ca_[None, :]
                b = np.full((Pb, Pb2), Dlb * 2 * Drb, np.int32)
                b[:len(rb), :len(cb_)] = rb[:, None] * 2 * Drb + nb * Drb + cb_[None, :]
                r = np.full(Pt, Dlt, np.int32)
                r[:len(rt)] = rt
                k = np.full(Pt2, Drt, np.int32)
                k[:len(ct)] = ct
                src.append(s)
                dst.append(out_index[c2])
                physical.append(p)
                sign.append((-1.0) ** (na * c[1]))
                ga.append(a)
                gb.append(b)
                rows.append(r)
                cols.append(k)
                keys.append((key(c), p, key(c2)))
                shapes.append((len(rt), len(ct)))
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
            peak=max(left_peak, right_peak), fixed_rows=np.stack(rows), fixed_cols=np.stack(cols),
            fixed_keys=tuple(keys), fixed_shapes=tuple(shapes)))
    stats = dict(exact_entries=exact, padded_entries=padded, transitions=sum(len(s.src) for s in sites),
                 env_entries=padded, contraction_fma=sum(s.left_fma for s in sites),
                 max_pads=tuple(int(max(p[j] for p in pads)) for j in range(3)))
    return ContractionPlan(tuple(sites), tuple(shared), tuple(pads), None, stats)


def fixed_blocks(fixed, plan: ContractionPlan, xp=jnp) -> tuple:
    """The zero-padded (T, Pt, Pt') blocks of a fixed MPS (the trial, or H|trial>) for a factorized layout.

    Dense site tensors are gathered; with xp=jnp this also works on traced tensors (trial data inside jit),
    with xp=np it stays on the host. Sites in block form ({(left label, p, right label): block}, trot.gmps.trials
    for 6x6 and larger lattices) are assembled on the host.
    """
    out = []
    for A, sp in zip(fixed, plan.sites):
        if isinstance(A, dict):
            B = np.zeros((len(sp.src), sp.fixed_rows.shape[1], sp.fixed_cols.shape[1]))
            for t, (key, (n_rows, n_cols)) in enumerate(zip(sp.fixed_keys, sp.fixed_shapes)):
                block = A.get(key)
                if block is not None:
                    B[t, :n_rows, :n_cols] = block
            out.append(B if xp is np else jnp.asarray(B))
        elif xp is np:
            padded = np.pad(np.asarray(A), ((0, 1), (0, 0), (0, 1)))
            out.append(padded[sp.fixed_rows[:, :, None], sp.physical[:, None, None], sp.fixed_cols[:, None, :]])
        else:
            out.append(jnp.asarray(A).at[sp.fixed_rows[:, :, None], sp.physical[:, None, None],
                                         sp.fixed_cols[:, None, :]].get(mode="fill", fill_value=0.0))
    return tuple(out)


def make_factorized_plan(qa, qb, fixed_np, fixed_charges) -> ContractionPlan:
    """make_factorized_layout with the host blocks of fixed_np (dense or block-form sites)."""
    layout = make_factorized_layout(qa, qb, fixed_charges)
    return layout._replace(blocks=fixed_blocks(fixed_np, layout, xp=np))


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
    """The diagonal HS sweep of one walker on the factorized environments: at each
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
# Spin correlations <trial| S^z_i S^z_j |walker> / <trial|walker>
# ============================================================================

SZ_LOCAL = np.array([0.0, 0.5, -0.5, 0.0])  # S^z on |0>, |up>, |dn>, |up dn> (p = n_alpha + 2 n_beta)


def _symmetric_matrix(diagonal, pairs):
    """(L, L) matrix with `diagonal` on the diagonal and pairs[k] = values at (i, k) for i < k, mirrored."""
    n = len(diagonal)
    C = jnp.zeros((n, n)).at[jnp.arange(n), jnp.arange(n)].set(jnp.stack(diagonal))
    for k, values in enumerate(pairs):
        if k:
            C = C.at[jnp.arange(k), k].set(values).at[k, jnp.arange(k)].set(values)
    return C


def szsz_contract(wblocks, fixed_blocks, plan: ContractionPlan):
    """<fixed|S^z_i S^z_j|walker channels> / <fixed|walker channels> as an (L, L) matrix.

    S^z is diagonal in the local basis, so an insertion is the weight SZ_LOCAL[p] on each transition of the
    factorized layout; the walker, the fixed MPS and their labels (width 2 or 1) are those of left_contract. One
    left sweep carries the plain environment and a stack of environments with one S^z already inserted at an
    earlier site i; closing the stack at site k gives the pair (i, k) against the right environments, and the
    diagonal (k, k) is the double insertion SZ_LOCAL**2. The gauges and det R of the walker cancel in the ratio.
    """
    right = right_environments(wblocks, fixed_blocks, plan)
    sz, sz2 = jnp.asarray(SZ_LOCAL), jnp.asarray(SZ_LOCAL**2)
    left = jnp.ones((1, 1, 1, 1))
    stack = None  # (k, n_shared, Pa, Pb, Pt): left environments holding one S^z at sites 0..k-1
    diagonal, pairs = [], [jnp.zeros(0)]
    for site, ((Wa, Wb), F, sp) in enumerate(zip(wblocks, fixed_blocks, plan.sites)):
        step = lambda X: _legs(X[sp.src], (Wa, Wb, F), sp.left_order, True) * sp.sign
        into = lambda Y: _reduce(Y, sp.into)
        close = right[site + 1][sp.dst]
        local = step(left)
        weight = sz[sp.physical][:, None, None, None]
        diagonal.append(jnp.sum(sz2[sp.physical][:, None, None, None] * local * close))
        if stack is None:
            stack = into(weight * local)[None]
        else:
            moved = jax.vmap(step)(stack)
            pairs.append(jnp.sum(weight[None] * moved * close[None], axis=(1, 2, 3, 4)))
            stack = jnp.concatenate((jax.vmap(into)(moved), into(weight * local)[None]), axis=0)
        left = into(local)
    overlap = right[0][0, 0, 0, 0]
    return _symmetric_matrix(diagonal, pairs) / overlap


def szsz_dense(tensors, fixed):
    """szsz_contract for dense d=4 tensors (the combined walker) against a dense fixed MPS."""
    sz, sz2 = jnp.asarray(SZ_LOCAL), jnp.asarray(SZ_LOCAL**2)
    right = [jnp.ones((1, 1))]
    for A, B in zip(reversed(tensors), reversed(fixed)):
        right.append(jnp.einsum("apr,bps,rs->ab", A, B, right[-1]))
    right = right[::-1]
    left = jnp.ones((1, 1))
    stack = None
    diagonal, pairs = [], [jnp.zeros(0)]
    for site, (A, B) in enumerate(zip(tensors, fixed)):
        close = right[site + 1]
        diagonal.append(jnp.einsum("ab,apr,bps,p,rs->", left, A, B, sz2, close))
        opened = jnp.einsum("ab,apr,bps,p->rs", left, A, B, sz)
        if stack is not None:
            moved = jnp.einsum("mab,apr,bps->mrs", stack, A, B)
            pairs.append(jnp.einsum("mab,apr,bps,p,rs->m", stack, A, B, sz, close))
            stack = jnp.concatenate((moved, opened[None]), axis=0)
        else:
            stack = opened[None]
        left = jnp.einsum("ab,apr,bps->rs", left, A, B)
    return _symmetric_matrix(diagonal, pairs) / right[0][0, 0]


# ============================================================================
# Walker ops against a fixed trial
# ============================================================================

class DeviceData(NamedTuple):
    """Everything large the jitted code reads, passed as arguments (not constants). trot.prop.mps_cpmc assembles it
    from meas_ctx (trial and H|trial> blocks) and prop_ctx (exp(-dt K/2), HS factors); overlaps and energies leave
    the propagation fields None."""
    trial: tuple  # per-site (T, Pt, Pt') trial blocks of the overlap plan
    htrial: tuple  # per-site blocks of the energy plan, or dense H|trial> for energy="dense"
    dense_trial: tuple  # dense trial tensors, energy="dense" only
    exp_h1_half: jax.Array | None
    hs: jax.Array | None


class Kernels(NamedTuple):
    """The batched engine for one walker plan and one trial / H|trial> labelling (make_kernels)."""
    converter: Converter
    overlap_plan: ContractionPlan  # a layout: its blocks come with the DeviceData
    energy_plan: ContractionPlan | None
    energy_kind: str
    batch_qr: Callable  # (n, L, N) -> Q, det R
    overlaps: Callable  # (ca, cb, data) -> overlaps of a walker batch
    energies: Callable  # (ca, cb, data) -> local energies of a walker batch
    step_batch: Callable  # (ca, cb, randoms, sweep, data, weight_floor) -> see make_kernels
    probe: Callable  # (ca, cb, data) for one walker -> dict with MPS, gauges, overlap, energy
    overlap_one: Callable  # (ca, cb, data) -> <trial|walker> of one walker
    energy_one: Callable  # (ca, cb, data) -> local energy of one walker
    jit_probe: Callable  # jax.jit(probe), compiled once per kernels


def make_kernels(converter: Converter, overlap_plan: ContractionPlan, energy_plan: ContractionPlan | None,
                 energy="blocked", walker_qr="cholesky") -> Kernels:
    """Conversion, overlap, local energy and the half-step kernel against a fixed trial whose blocks (and those of
    H|trial>) arrive in the DeviceData argument. energy_plan: the layout of the charge-labelled H|trial> for
    energy="blocked", None for energy="dense" (the d=4 walker against the dense H|trial>, for validation) and for
    energy=None (overlaps only: the trial ops)."""
    if energy not in ("blocked", "dense", None):
        raise ValueError("energy must be 'blocked', 'dense' or None")
    convert = converter.convert
    qa_labels, qb_labels = converter.charges
    batch_qr = make_batch_qr(walker_qr)

    def local_energy(alpha, beta, data):
        if energy is None:
            raise ValueError("these kernels measure overlaps only (energy=None)")
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
                    energy=None if energy is None else local_energy(alpha, beta, data))

    def overlap_one(ca, cb, data):
        qa, da = qr_with_det(ca)
        qb, db = qr_with_det(cb)
        alpha, beta, (ga, gb) = convert(qa, qb)
        return da * db * ga * gb * left_contract(walker_blocks(alpha, beta, overlap_plan), data.trial, overlap_plan)

    def energy_one(ca, cb, data):
        qa, _ = qr_with_det(ca)
        qb, _ = qr_with_det(cb)
        alpha, beta, _ = convert(qa, qb)
        return local_energy(alpha, beta, data)

    return Kernels(converter, overlap_plan, energy_plan, energy, batch_qr, overlaps, energies, step_batch, probe,
                   overlap_one, energy_one, jax.jit(probe))


# ============================================================================
# The engine of a walker plan (trot.trial.mps.MpsWalkerPlan), cached on the plan
# ============================================================================

def resolve_walker_qr(walker_qr="auto") -> str:
    """"auto": CholeskyQR2 on accelerators, Householder (trot's _qr) on CPU."""
    if walker_qr == "auto":
        return "native" if jax.default_backend() == "cpu" else "cholesky"
    if walker_qr not in ("cholesky", "native"):
        raise ValueError(f"walker_qr must be 'auto', 'cholesky' or 'native', got {walker_qr!r}")
    return walker_qr


def converter_for(plan) -> Converter:
    """The plan's compiled conversion circuit (batched sector factorisations, both spins in one batch when their
    gate sequences match, the plan's sector buckets)."""
    converter = plan.caches.get("converter")
    if converter is None:
        (plan_a, plan_b), (bond_a, bond_b) = plan.orbital_plans, plan.bond_plans
        converter = make_converter(plan_a, plan_b, bond_a, bond_b, spin_batch=True,
                                   buckets=tuple(plan.sector_buckets) or None)
        plan.caches["converter"] = converter
    return converter


def layout_for(plan, charges) -> ContractionPlan:
    """The factorized layout pairing the plan's walker channels with an MPS of these (hashable) bond labels."""
    key = ("layout", charges)
    layout = plan.caches.get(key)
    if layout is None:
        layout = make_factorized_layout(*converter_for(plan).charges, charges)
        plan.caches[key] = layout
    return layout


def kernels_for(plan, trial_charges, h_charges=None, energy="blocked") -> Kernels:
    """The plan's kernels for a trial with these bond labels and an H|trial> with h_charges (energy="blocked"; for
    "dense" the dense H|trial> comes with the data, for None there is no energy)."""
    key = ("kernels", trial_charges, h_charges, energy)
    kernels = plan.caches.get(key)
    if kernels is None:
        energy_plan = layout_for(plan, h_charges) if energy == "blocked" else None
        kernels = make_kernels(converter_for(plan), layout_for(plan, trial_charges), energy_plan, energy,
                               resolve_walker_qr(plan.walker_qr))
        plan.caches[key] = kernels
    return kernels


def conversion_self_check(ops, plans, bonds, probe):
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
# Walker chunking
# ============================================================================

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

def make_half_step(ops: Kernels, params: QmcParams, n_chunks: int):
    """The CPMC step (trot.prop.mps_cpmc) split into its two halves around the one-body
    propagator, with one conversion call site. Even half steps: exp(-dt K/2), the
    overlap ratio, the HS sweep. Odd: exp(-dt K/2), the overlap ratio and the
    population-control shift. The RNG use and the order of trot.prop.cpmc's step."""
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


# ============================================================================
# Helpers shared with the trot ops (trot.trial.mps, trot.meas.mps, trot.prop.mps_cpmc)
# ============================================================================

def divisor_at_least(n, k) -> int:
    """The smallest divisor of n that is >= k (chunked needs equal chunks; trot's chunk search may pick any k)."""
    for d in range(max(1, int(k)), n + 1):
        if n % d == 0:
            return d
    return n
