"""Batched device engine for MPS-CPMC with Slater-determinant walkers: the GPU kernels shared by
trot.gmps.mps_cpmc_gpu (the production script) and trot's native MPS-CPMC ops (trot.gmps.driver, engine="batched").

Same algorithm as the reference conversion (trot.gmps.utils.channel_mps, trot.trial.mps): the walkers stay Slater
determinants, each spin channel is converted to a charge-labelled d=2 MPS with Fishman-White gates, truncating gate
by gate with the orthogonality centre on the gate. What changes is the layout for a GPU:

* compile_circuit turns every centre move and split into static gathers into padded batches of charge-sector
  blocks; each sector is factored by the reference method (_factor_block: closed form for one row/column, QR when
  exact, the Gram eigh when truncating), batched over walkers x spins x sectors per kind (linalg="batched"); both
  spin channels run in one batch when their gate sequences match (spin_batch).
* walker-trial contractions never form the d=4 walker: factorized (alpha, beta, trial) blocks per shared label,
  Pa*Pb*Pt*(Pa+Pb+Pt) per transition (make_factorized_plan, left_contract, right_environments, field_sweep).
  The trial may carry (N_up, N_dn) labels (MpsTrial) or particle-number labels only (a RotatedMpsTrial, labelled
  by number_labels): then each walker label (N_alpha, N_beta) meets the trial's N_alpha + N_beta sector.
* walker QR by CholeskyQR2 with a Householder fallback (make_batch_qr("cholesky")).
* everything large (trial and H|trial> blocks, exp(-dt K/2), HS factors) is a jit argument (DeviceData), not an
  HLO constant; walkers are chunked with lax.map as the memory model requires (choose_chunks, chunked).

Host-side primitives (orbital and bond plans, sector plans, block factorisation) come from trot.gmps.utils.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache, partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import scipy.linalg
from jax import lax

from jax import tree_util

from trot import walkers as wk
from trot.core.ops import MeasOps, TrialOps, k_energy
from trot.gmps.utils import (
    BondPlan,
    OrbitalPlan,
    SectorPlan,
    _assemble,
    _block_plan,
    _factor_block,
    _key,
    _move_centre,
    channel_angles,
    channel_mps,
    combine_channels,
    contract_real,
    gate_pair,
    sector_plan,
)
from trot.meas.mps import _dense_overlap, apply_mpo, compress_mps, hubbard_h1, hubbard_mpo_from_h1, trial_times_h
from trot.prop.cpmc import init_prop_state
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.mps_cpmc import constrain_ratio
from trot.prop.types import PropOps, PropState, QmcParams
from trot.trial.mps import compress_mps_qn, get_rdm1, label_array, rhf_orbitals
from trot.walkers import _qr as qr_with_det


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
        complete_qr = np.array([bool(p.exact_for_all_walkers and not p.occupation[k] and B > C.shape[-1])
                                for p in plans])
        if complete_qr.any():
            # B exceeds the number of walker orbitals, so the last complete-QR
            # vector lies exactly in null(block.T): the required empty mode
            # (for B <= N the eigh branch is used, as in utils.channel_angles).
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


N_PHYSICAL = np.array([0, 1, 1, 2])  # n_alpha + n_beta of p = n_alpha + 2 n_beta


def number_labels(tensors):
    """Particle-number bond labels of a real d=4 MPS, read off its nonzero entries.

    Bond 0 has N = 0, and every nonzero entry A[i, p, j] gives index j the label
    N_i + n_up(p) + n_dn(p). A spin rotation acts site by site and keeps N, so a
    rotated trial (RotatedMpsTrial) keeps the input's N labels while its (N_up, N_dn)
    labels are lost. Returns L+1 (D, 1) int arrays, or None if some index is reached
    with two different N (the MPS mixes particle numbers in this gauge). An index
    that no nonzero entry reaches gets -1: its left block is zero, so it pairs with
    no walker label and dropping it is exact.
    """
    labels = [np.zeros(1, int)]
    for A in tensors:
        hit = (np.asarray(A) != 0) & (labels[-1] >= 0)[:, None, None]
        candidate = (labels[-1][:, None] + N_PHYSICAL[None, :])[:, :, None]
        high = np.where(hit, candidate, -1).max(axis=(0, 1))
        low = np.where(hit, candidate, np.iinfo(int).max).min(axis=(0, 1))
        if np.any((high >= 0) & (low != high)):
            return None
        labels.append(high)
    return tuple(q[:, None] for q in labels)


def _fixed_key(width):
    """The fixed MPS's label that a walker label pair c = (N_alpha, N_beta) contracts
    with: c itself for (N_up, N_dn) labels, (N_alpha + N_beta,) for N labels."""
    if width == 2:
        return lambda c: c
    if width == 1:
        return lambda c: (c[0] + c[1],)
    raise ValueError(f"fixed MPS labels must be (N_up, N_dn) pairs or particle numbers, got width {width}")


def _fixed_block(tensor, rows, p, columns, left, right):
    """Block (left label, p, right label) of one site of the fixed MPS: a dense tensor, or
    mps_cpmc_2d_gpu's block form {(left, p, right): block} (rows and columns in bond order)."""
    if isinstance(tensor, dict):
        return tensor.get((left, p, right), 0.0)
    return np.asarray(tensor)[np.ix_(rows, [p], columns)][:, 0]


def make_factorized_plan(qa, qb, fixed_np, fixed_charges) -> ContractionPlan:
    """Layout of <fixed MPS|walker> with the walker kept as its two spin channels.

    Environments at bond i are (n_shared, Pa, Pb, Pt): one padded (alpha, beta,
    fixed) block per shared (N_alpha, N_beta) label. A transition moves label
    (a, b) to (a + n_alpha, b + n_beta) through physical p; its alpha, beta and
    fixed blocks are gathered separately, so the d=4 walker is never formed.

    fixed_charges are (N_up, N_dn) labels, or particle-number labels N (width 1:
    a spin-rotated trial used as it is, see number_labels). With N labels the walker
    label (a, b) meets the fixed MPS's whole N = a + b sector: the walker's channels
    and the interleave sign (-1)^(n_alpha N_beta) keep their own labels, so nothing
    else changes. fixed_np's sites may be dense or in mps_cpmc_2d_gpu's block form.
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

    sites, blocks = [], []
    exact = sum(len(ia[i][c[0]]) * len(ib[i][c[1]]) * len(it[i][key(c)]) for i in range(n + 1) for c in shared[i])
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
                rt, ct = it[site][key(c)], it[site + 1][key(c2)]
                a = np.full((Pa, Pa2), Dla * 2 * Dra, np.int32)
                a[:len(ra), :len(ca_)] = ra[:, None] * 2 * Dra + na * Dra + ca_[None, :]
                b = np.full((Pb, Pb2), Dlb * 2 * Drb, np.int32)
                b[:len(rb), :len(cb_)] = rb[:, None] * 2 * Drb + nb * Drb + cb_[None, :]
                t = np.zeros((Pt, Pt2))
                t[:len(rt), :len(ct)] = _fixed_block(fixed_np[site], rt, p, ct, key(c), key(c2))
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


# ============================================================================
# trot's MPS-CPMC ops on this engine (QmcParamsMps.engine = "batched")
# ============================================================================

def resolve_engine(params) -> str:
    """QmcParamsMps.engine: "auto" is the batched engine on a GPU backend with the "fast" propagator, else the
    reference one (CPU, or propagator="slow", which only the reference implements)."""
    engine = getattr(params, "engine", "auto")
    if engine == "auto":
        fast = getattr(params, "propagator", "fast") == "fast"
        return "batched" if jax.default_backend() == "gpu" and fast else "reference"
    return engine


def resolve_linalg(params) -> tuple[str, str]:
    """(linalg, walker_qr) of QmcParamsMps, "auto" resolved as mps_cpmc_gpu does: batched sector factorisations
    and CholeskyQR2 on accelerators, the per-sector loop and Householder QR on CPU."""
    cpu = jax.default_backend() == "cpu"
    linalg, walker_qr = getattr(params, "linalg", "auto"), getattr(params, "walker_qr", "auto")
    linalg = linalg if linalg != "auto" else ("native" if cpu else "batched")
    walker_qr = walker_qr if walker_qr != "auto" else ("native" if cpu else "cholesky")
    return linalg, walker_qr


def divisor_at_least(n, k) -> int:
    """The smallest divisor of n that is >= k (chunked needs equal chunks; trot's chunk search may pick any k)."""
    for d in range(max(1, int(k)), n + 1):
        if n % d == 0:
            return d
    return n


@tree_util.register_pytree_node_class
class BatchedMeasCtx:
    """meas_ctx of the batched engine: the DeviceData (trial and H|trial> blocks, exp(-dt K/2), HS factors) as jit
    arguments, and static diagnostics in the aux data: the trial's identity (bond labels of an MpsTrial, see
    trial_labels), the energy kernel, <T|H|T> (over all sectors for a RotatedMpsTrial) and the H|trial> bond
    dimensions (the attributes trot.gmps.driver prints, as on trot.meas.mps.MpsMeasCtx)."""

    def __init__(self, data: DeviceData, key: tuple):
        self.data, self.key = data, key

    trial_charges = property(lambda self: self.key[0])
    kernel = property(lambda self: self.key[1])
    trial_energy = property(lambda self: self.key[2])
    h_bond_dims = property(lambda self: self.key[3])

    def tree_flatten(self):
        return (self.data,), self.key

    @classmethod
    def tree_unflatten(cls, key, children):
        return cls(children[0], key)


def trial_identity(trial):
    """What build_meas_ctx compares: an MpsTrial's bond labels, a RotatedMpsTrial's bond dimensions and nelec (the
    rule of trot.meas.mps_rotated.check_rotated_meas_ctx)."""
    from trot.trial.mps_rotation import RotatedMpsTrial

    if isinstance(trial, RotatedMpsTrial):
        return ("rotated", trial.bond_dims, tuple(trial.nelec))
    return trial.charges


def trial_labels(trial):
    """(dense tensors, bond labels) the engine contracts a trial with. An MpsTrial has exact (N_up, N_dn) labels; a
    RotatedMpsTrial (trot.trial.mps_rotation: no definite S_z, bonds as they are) gets the particle-number labels of
    its tensors (number_labels), so its contractions stay blocked."""
    from trot.trial.mps_rotation import RotatedMpsTrial

    tensors = [np.asarray(A) for A in trial.tensors]
    if not isinstance(trial, RotatedMpsTrial):
        return tensors, trial.charge_arrays()
    labels = number_labels(tensors)
    if labels is None:
        raise ValueError("the rotated trial mixes particle numbers at some bond, so it has no N labels to block on; "
                         'use engine="reference" (dense contractions)')
    if int(labels[-1][0, 0]) != sum(trial.nelec):
        raise ValueError(f"the trial holds no N = {sum(trial.nelec)} component (its tensors end at N = "
                         f"{int(labels[-1][0, 0])})")
    return tensors, labels


def make_batched_ops(ham_data, trial, plan, params):
    """trot TrialOps, MeasOps and PropOps for an MpsTrial or a RotatedMpsTrial on this engine.

    plan is trot's MpsWalkerPlan (trot.trial.mps.make_walker_plan, or make_rotated_walker_plan): the orbital and bond
    plans the circuit is compiled from. A RotatedMpsTrial is contracted with its N labels (trial_labels): the walker's
    spin channels stay separate, and H|trial> keeps N labels too. The step is make_half_step twice (the reference
    step's arithmetic, RNG use and order); trot's block measures with these ops. Returns (GpuOps, TrialOps, MeasOps,
    PropOps); trot.gmps.driver.make_mps_cpmc_ops and make_rotated_mps_cpmc_ops wrap them.
    """
    if params.propagator != "fast":
        raise ValueError('the batched engine implements the "fast" propagator; use engine="reference" for "slow"')
    linalg, walker_qr = resolve_linalg(params)
    plan_a, plan_b = plan.orbital_plans
    bond_a, bond_b = plan.bond_plans
    trial_np, trial_q = trial_labels(trial)
    identity = trial_identity(trial)
    W = hubbard_mpo_from_h1(hubbard_h1(ham_data), float(ham_data.u))
    if params.energy_kernel == "blocked":
        htrial = compress_mps_qn(*trial_times_h(W, trial_np, trial_q))
        h_np = htrial[0]
    else:
        h_np = compress_mps(apply_mpo(W, trial_np))
        htrial = tuple(jnp.asarray(A) for A in h_np)
    trial_energy = _dense_overlap(h_np, trial_np) / _dense_overlap(trial_np, trial_np)
    ops = make_gpu_ops(plan_a, plan_b, bond_a, bond_b, trial_np, trial_q, htrial, _build_prop_ctx(ham_data, params.dt),
                       linalg=linalg, walker_qr=walker_qr, spin_batch=True, energy=params.energy_kernel)
    h_bonds = tuple(int(A.shape[0]) for A in h_np) + (int(h_np[-1].shape[-1]),)
    meas_ctx = BatchedMeasCtx(ops.data, (identity, params.energy_kernel, float(trial_energy), h_bonds))

    @jax.jit
    def overlap(walker, trial_data=None):
        """<trial|walker> of one walker; trial blocks from the ops (init and trot's block call this)."""
        ca, cb = walker
        return ops.overlaps(ca[None], cb[None], ops.data)[0]

    @jax.jit
    def energy(walker, ham=None, ctx=None, trial_data=None):
        ca, cb = walker
        return ops.energies(ca[None], cb[None], ops.data if ctx is None else ctx.data)[0]

    def build_meas_ctx(_ham, trial_data):
        if trial_identity(trial_data) != identity:
            raise ValueError("the batched ops were built for a different trial (bond labels differ)")
        return meas_ctx

    nelec = tuple(plan.nelec)

    def init(**kwargs):
        start = getattr(kwargs["params"], "walker_start", "natural")
        if start not in ("natural", "rhf"):
            raise ValueError(f"walker_start must be 'natural' or 'rhf', got {start!r}")
        if start == "rhf" and kwargs.get("rdm1") is None and kwargs.get("initial_walkers") is None:
            Ra, Rb = rhf_orbitals(kwargs["ham_data"], nelec)
            kwargs["rdm1"] = jnp.asarray(np.stack([Ra @ Ra.T, Rb @ Rb.T]))
        state = init_prop_state(**kwargs)
        return state._replace(
            node_encounters=state.node_encounters.astype(jnp.int64),
            e_estimate=state.e_estimate.astype(jnp.result_type(float)),
            pop_control_ene_shift=state.pop_control_ene_shift.astype(jnp.result_type(float)),
        )

    def step(state, *, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ctx):
        if not isinstance(meas_ctx, BatchedMeasCtx):
            raise ValueError("the batched engine needs meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)")
        half = make_half_step(ops, params, divisor_at_least(wk.n_walkers(state.walkers), params.n_chunks))
        state = half(state, 0, meas_ctx.data)
        return half(state, 1, meas_ctx.data)

    trial_ops = TrialOps(overlap=overlap, get_rdm1=get_rdm1)
    meas_ops = MeasOps(overlap=overlap, build_meas_ctx=build_meas_ctx, kernels={k_energy: energy})
    prop_ops = PropOps(init_prop_state=init, build_prop_ctx=lambda ham, _rdm1, p: _build_prop_ctx(ham, p.dt),
                       step=step)
    return ops, trial_ops, meas_ops, prop_ops
