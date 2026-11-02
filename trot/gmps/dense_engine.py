"""CPMC half step with dense walker legs: the walker channels are converted at a fixed bond chi, each walker keeping its
own states (trot.gmps.dense_conversion), and contracted with the trial without any walker sector layout.

The sector engine (trot.gmps.engine) freezes the number of states each electron-number sector keeps at every gate, so
that the walker bonds, the factorized layouts and the contraction blocks are static. Here the static thing is the bond
size chi alone; the walker's bond labels are data (the beta channel's, for the fermionic sign) and the walker enters the
contractions as dense tensors.

Environments are (a, b, t): the alpha and beta bonds of the walker and the trial's bond. A site is applied as GEMMs
with the trial (E (ab, t) times T (t, (p, t')), the walkers batched into the rows), then the alpha tensor, then the
beta tensor dressed with the sign (-1)^(n_alpha N_beta) of combine_channels (N_beta, the beta electrons to the left of
the site, is the label of the beta bond index, rounded from the converter's charges). The trial's side is static, so it
keeps its particle-number structure (DenseSpec.trial):
  "sector"  (sector_layout) each trial bond's reached states sorted by particle number, so every sector is a contiguous
            slice of the environment's t index, and the unreached ones dropped (their environment entries are exactly
            zero). T[t, p, t'] vanishes unless N(t') = N(t) + n(p), so a site is one GEMM per left sector N with the
            blocks of all four physical states side by side (sector_trial_blocks; the right environments likewise per
            right sector): about 1 / n_sectors of the full matrices' multiply-adds.
  "dense"   the full trial matrices (engine.dense_trial_blocks): one GEMM per site, zeros multiplying zeros.
The overlap, the right environments and the HS field sweep (engine.field_sweep) follow the sector engine's structure and
semantics, and the plan is used through the same Kernels interface (engine.kernels_for), so trot's step, block and
driver run on it unchanged. Local energies and observables are not available here: they come from a sector measurement
plan (QmcParamsMps.measure_channel_chi), which a dense plan requires.
"""
from __future__ import annotations

import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from trot.gmps import engine, grouped_gemm
from trot.gmps.dense_conversion import make_dense_converter
from trot.gmps.utils import label_array
from trot.walkers import _qr as qr_with_det

N_PHYSICAL = (0, 1, 1, 2)  # electrons of the local states p = n_alpha + 2 n_beta


# ============================================================================
# The trial in particle-number sectors
# ============================================================================

class SiteLayout(NamedTuple):
    """One site's trial blocks. forward: per left sector with a target, (start, stop, cols) with cols the
    (p, right sector, column offset, width) of each physical state's block in that sector's matrix; backward: the same
    per right sector, cols naming left sectors. left_widths / right_widths: the sectors' sizes."""
    forward: tuple
    backward: tuple
    left_widths: tuple
    right_widths: tuple


class SectorLayout(NamedTuple):
    """The trial bonds' order and sectors: orders[k] the reached states of bond k sorted by particle number (the
    environments' t index), sectors[k] its (N, start, stop) slices; sites[i] the SiteLayout of site i; stats for the
    diagnostics."""
    orders: tuple
    sectors: tuple
    sites: tuple
    stats: dict


def sector_layout(charges) -> SectorLayout:
    """The SectorLayout of a trial with these bond labels ((N_up, N_dn) pairs or N; negative entries mark unreached
    states, dropped)."""
    orders, sectors = [], []
    for q in charges:
        q = label_array(q)
        n = q.sum(axis=1)
        reached = np.flatnonzero(np.all(q >= 0, axis=1))
        order = reached[np.argsort(n[reached], kind="stable")]
        values = n[order]
        bounds = []
        for N in np.unique(values):
            where = np.flatnonzero(values == N)
            bounds.append((int(N), int(where[0]), int(where[-1]) + 1))
        orders.append(order)
        sectors.append(tuple(bounds))
    sites, mac_dense, mac_sector = [], 0, 0
    for i in range(len(charges) - 1):
        left, right = sectors[i], sectors[i + 1]
        right_of = {N: k for k, (N, _, _) in enumerate(right)}
        left_of = {N: k for k, (N, _, _) in enumerate(left)}
        widths = lambda bonds: tuple(stop - start for _, start, stop in bonds)
        lw, rw = widths(left), widths(right)
        forward = []
        for N, start, stop in left:
            cols, offset = [], 0
            for p in range(4):
                k = right_of.get(N + N_PHYSICAL[p])
                if k is not None:
                    cols.append((p, k, offset, rw[k]))
                    offset += rw[k]
            if cols:
                forward.append((start, stop, tuple(cols)))
                mac_sector += (stop - start) * offset
        backward = []
        for N, start, stop in right:
            cols, offset = [], 0
            for p in range(4):
                k = left_of.get(N - N_PHYSICAL[p])
                if k is not None:
                    cols.append((p, k, offset, lw[k]))
                    offset += lw[k]
            if cols:
                backward.append((start, stop, tuple(cols)))
        sites.append(SiteLayout(tuple(forward), tuple(backward), lw, rw))
        mac_dense += len(label_array(charges[i])) * 4 * len(label_array(charges[i + 1]))
    stats = dict(trial_mac_dense=int(mac_dense), trial_mac_sector=int(mac_sector),
                 trial_mac_ratio=round(mac_dense / max(mac_sector, 1), 2),
                 sectors_max=max(len(s) for s in sectors),
                 sector_size_max=max(stop - start for s in sectors for _, start, stop in s),
                 unreached=int(sum(len(label_array(q)) for q in charges) - sum(len(o) for o in orders)))
    return SectorLayout(tuple(orders), tuple(sectors), tuple(sites), stats)


def sector_trial_blocks(fixed, layout: SectorLayout, xp=jnp, device=False) -> tuple:
    """Per site (forward matrices, backward matrices) of the trial tensors A (Dl, 4, Dr) in the layout: for a left
    sector, its rows of every physical state's block side by side (d_N, sum of widths); for a right sector, the
    transposed blocks side by side. xp and device as engine.fixed_blocks."""
    out = []
    for i, A in enumerate(fixed):
        A = xp.take(xp.take(xp.asarray(A), layout.orders[i], axis=0), layout.orders[i + 1], axis=2)
        site, left, right = layout.sites[i], layout.sectors[i], layout.sectors[i + 1]
        forward = tuple(xp.concatenate([A[start:stop, p, right[k][1]:right[k][2]] for p, k, _, _ in cols], axis=1)
                        for start, stop, cols in site.forward)
        backward = tuple(xp.concatenate([A[left[k][1]:left[k][2], p, start:stop].T for p, k, _, _ in cols], axis=1)
                         for start, stop, cols in site.backward)
        blocks = (forward, backward)
        out.append(jax.tree_util.tree_map(jnp.asarray, blocks) if device else blocks)
    return tuple(out)


def dense_overlap_plan(plan, charges) -> engine.DensePlan:
    """The DensePlan of a dense walker plan for a trial with these (hashable) labels, cached on the plan: with the
    trial's SectorLayout (DenseSpec.trial "sector") or without (the full matrices)."""
    key = ("dense_overlap_plan", charges)
    overlap_plan = plan.caches.get(key)
    if overlap_plan is None:
        if plan.dense.trial == "dense":
            overlap_plan = engine.DensePlan(int(plan.norb), dict(trial="dense"))
        else:
            layout = sector_layout(charges)
            overlap_plan = engine.DensePlan(int(plan.norb), dict(trial="sector", **layout.stats), layout)
        plan.caches[key] = overlap_plan
    return overlap_plan


def _assemble(pieces, widths, rows, dtype):
    """(rows, 4, sum of widths) from the pieces {(p, sector): (rows, width)}, zeros where a sector has no source."""
    return jnp.stack([jnp.concatenate([pieces[p, k] if (p, k) in pieces else jnp.zeros((rows, w), dtype)
                                       for k, w in enumerate(widths)], axis=1) for p in range(4)], axis=1)


def _trial_apply(X, matrices, entries, widths):
    """X (rows, D) times the trial's blocks, one GEMM per sector of X's bond: (rows, 4, D') on the other bond."""
    pieces = {}
    for (start, stop, cols), W in zip(entries, matrices):
        Y = X[:, start:stop] @ W
        for p, k, offset, width in cols:
            pieces[p, k] = Y[:, offset:offset + width]
    return _assemble(pieces, widths, X.shape[0], X.dtype)


# "dense" (_trial_apply: every row of an environment through every sector's GEMM) or "grouped" (trot.gmps.grouped_gemm:
# each row only through its own sector's, the rows found from the walker's bond labels; the same products, the zero ones
# skipped). Grouped runs where the walker's labels are passed: the MPO energy (mpo_contract) and szsz (szsz_gemm). 8x8
# chi_T 2048, walker bond 64, one trial GEMM (job 7372094): energy 27-33 -> 11.2-11.8 ms, overlap rows ~1.0 -> 0.5 ms,
# bitwise the same output; a block 1938 -> 1554 s with the energies unchanged (job 7373218). "fused": grouped, and the
# kernel also does what follows the trial GEMM (the MPO gather, szsz's weights, the sign, the walker GEMMs' layout:
# grouped_gemm.fused_trial_apply), so the trial GEMM's output is never written. "blocked": fused, and the walker GEMMs
# on label blocks too (grouped_gemm.contract_left / contract_right): no zero fill and no dense pass over the
# intermediates, only the site's output environment is zero-filled (szsz's stacks, which are read whole; mpo_contract's
# energy environments only at the last site, the next fused kernel reading nothing outside the rows' sectors).
# "cublas": blocked with the trial GEMM in cuBLAS, one DGEMM per sector through a custom call, and the rest of the
# fused kernel reading its products (grouped_gemm.cublas_fused_apply; the library built by cublas_ext/build.sh). 8x8
# blocked step (job 7388160): the Triton trial GEMM is 10.5 of 14.6 ms (energy), 5.3 of 7.9 (szsz transfer).
TRIAL_GEMM = os.environ.get("TROT_TRIAL_GEMM", "dense")
TRIAL_GEMMS = ("dense", "grouped", "fused", "blocked", "cublas")
BLOCKED = ("blocked", "cublas")  # the TRIAL_GEMMs with the walker contractions on label blocks


def _grouped_specs(layout, labels, direction):
    """The layout's TrialSpecs (TRIAL_GEMM grouped) or FusedSpecs (fused, blocked, cublas) in a direction, when the
    trial is in sector blocks and the walker's labels are given, else None."""
    if TRIAL_GEMM not in TRIAL_GEMMS:
        raise ValueError(f"TROT_TRIAL_GEMM must be one of {TRIAL_GEMMS}, not {TRIAL_GEMM!r}")
    if TRIAL_GEMM == "dense" or layout is None or labels is None:
        return None
    return grouped_gemm.layout_specs(layout, direction, fused=TRIAL_GEMM != "grouped")


def _fused(grouped):
    """Whether a site's grouped argument asks for the fused kernel: (n_first, n_second, FusedSpec, offsets, n_groups)
    and for the block contractions also the labels of the output bond's (first, second)."""
    return grouped is not None and isinstance(grouped[2], grouped_gemm.FusedSpec)


def _fused_apply():
    """The fused kernel of TRIAL_GEMM: the trial GEMM in Triton (fused_trial_apply) or, with cublas, in cuBLAS
    (cublas_fused_apply); the same arguments."""
    return grouped_gemm.cublas_fused_apply if TRIAL_GEMM == "cublas" else grouped_gemm.fused_trial_apply


def _site_rows(n_first, n_second, spec, offsets, n_sites, next_rows=None, fill=True):
    """A site's grouped argument: (labels, TrialSpec, offsets) for the grouped trial GEMM; for the fused kernel also the
    bound on the label sums (2 L + 1) and with TRIAL_GEMM blocked (or cublas) the labels next_rows of the output bond,
    in the same (first, second) order as the output's rows, and with fill=False a last False: the energy step's output
    environment is then left undefined outside its rows' sectors (no zero fill; mpo_contract, whose next step reads
    nothing else)."""
    rows = (n_first, n_second, spec, offsets)
    if not isinstance(spec, grouped_gemm.FusedSpec):
        return rows
    if TRIAL_GEMM not in BLOCKED:
        return rows + (2 * n_sites + 1,)
    return rows + (2 * n_sites + 1,) + tuple(next_rows) + (() if fill else (False,))


def _trial_gemm(X, matrices, entries, widths, grouped=None):
    """_trial_apply(X, matrices, entries, widths), or with grouped = (n_first, n_second, spec, offsets) the grouped
    kernel on the rows that can be nonzero: X's rows are (first, second, member) with the walker bond labels n_first,
    n_second and the members' electron-number offsets (the MPO channels'; None: all 0, as for the szsz stacks)."""
    if grouped is None:
        return _trial_apply(X, matrices, entries, widths)
    n_first, n_second, spec, offsets = grouped
    if offsets is None:
        offsets = jnp.zeros(X.shape[0] // (n_first.shape[0] * n_second.shape[0]), jnp.int32)
    g = grouped_gemm.row_sectors(n_first, n_second, spec, offsets)
    return grouped_gemm.grouped_trial_apply(X, g, spec, grouped_gemm.pack_blocks(matrices))  # vmapped: walkers pooled


# ============================================================================
# One site of an environment
# ============================================================================

def _parity_signs(q):
    """(D, 2) factors (-1)^(n_alpha q) of the beta bond labels q (floats holding integers), n_alpha = 0, 1."""
    return jnp.stack((jnp.ones_like(q), 1.0 - 2.0 * jnp.mod(jnp.round(q), 2.0)), axis=1)


def _dressed(B, signs):
    """The beta tensor B (Db, 2, Db') with the fermionic sign for each alpha occupation: (Db, n_beta, n_alpha, Db')."""
    return B[:, :, None, :] * signs[:, None, :, None]


def _forward(E, T, A, Bs, site=None, grouped=None):
    """One site of a left environment: E (Da, Db, Dt) through the trial's blocks T (the full matrices, or the sector
    blocks of SiteLayout `site`) and the walker's tensors to (Da', Db', p, Dt'), p = n_alpha + 2 n_beta. grouped:
    _trial_gemm's (the labels of a and b), or with a FusedSpec (n_a, n_b, FusedSpec, n_a', n_b', n_groups) the unit
    fused kernel and the walker einsums on label blocks (TROT_PROP_GEMM=blocked)."""
    Da, Db, Dt = E.shape
    X = E.reshape(Da * Db, Dt)
    if grouped is not None and isinstance(grouped[2], grouped_gemm.FusedSpec):
        n_a, n_b, fspec, n_c, n_d, n_groups = grouped
        Z = grouped_gemm.fused_trial_apply(X, n_a, n_b, None, grouped_gemm.unit_terms(), None, fspec,
                                           grouped_gemm.pack_blocks(T[0]), n_groups=n_groups, layout="left",
                                           fill=False)
        return grouped_gemm.forward_walker_blocks(Z, A, Bs[:, :, 0, :], _parity_signs(n_b), n_a, n_b, n_c, n_d,
                                                  fspec)  # Bs[:, :, 0, :]: B itself (the sign of n_alpha = 0 is 1)
    Z = X @ T[0] if site is None else _trial_gemm(X, T[0], site.forward, site.right_widths, grouped)
    Z = Z.reshape(Da, Db, 2, 2, -1)  # (a, b, n_beta, n_alpha, t')
    H = jnp.einsum("abjit,aic->cbjit", Z, A)
    Y = jnp.einsum("cbjit,bjid->cdjit", H, Bs)
    return Y.reshape(Y.shape[0], Y.shape[1], 4, Y.shape[4])


def _backward(R, T, A, Bs, site=None, grouped=None):
    """One site of a right environment: R (Da', Db', Dt') to (Da, Db, p, Dt), with the trial's T[1] (the transposed full
    matrix (Dt', 4 Dt), or the sector blocks of `site`). grouped: _trial_gemm's (the labels of a' and b'), or with a
    FusedSpec (n_a', n_b', FusedSpec, n_a, n_b, n_groups) the unit fused kernel and the walker einsums on label blocks
    (TROT_PROP_GEMM=blocked)."""
    Dc, Dd, Dt = R.shape
    X = R.reshape(Dc * Dd, Dt)
    if grouped is not None and isinstance(grouped[2], grouped_gemm.FusedSpec):
        n_c, n_d, fspec, n_a, n_b, n_groups = grouped
        U = grouped_gemm.fused_trial_apply(X, n_c, n_d, None, grouped_gemm.unit_terms(), None, fspec,
                                           grouped_gemm.pack_blocks(T[1]), n_groups=n_groups, layout="right",
                                           fill=False)
        return grouped_gemm.backward_walker_blocks(U, A, Bs[:, :, 0, :], _parity_signs(n_b), n_a, n_b, n_c, n_d,
                                                   fspec)
    U = X @ T[1] if site is None else _trial_gemm(X, T[1], site.backward, site.left_widths, grouped)
    U = U.reshape(Dc, Dd, 2, 2, -1)  # (c, d, n_beta, n_alpha, t)
    H = jnp.einsum("cdjit,aic->adjit", U, A)
    X = jnp.einsum("adjit,bjid->abjit", H, Bs)
    return X.reshape(X.shape[0], X.shape[1], 4, X.shape[4])


def _starts(widths):
    return np.concatenate(([0], np.cumsum(widths)))[:-1]


def _backward_summed(R, T, A, Bs, site=None, grouped=None):
    """_backward(R, T, A, Bs, site).sum(axis=2), the walker tensors first (right_environments by default,
    TROT_RIGHT_ENVS=walker): R (Dc, Dd, Dt') through the alpha tensor (one GEMM: (a, n_alpha, d, t')) and the dressed
    beta tensor (one GEMM batched over (a, n_alpha): (a, n_alpha, b, n_beta, t'), 4x the environment), then the trial's
    forward blocks with the sum over p inside the GEMM: per left sector one GEMM whose inner dimension is the right
    sectors of the four physical states side by side, gathered in the order of sector_trial_blocks' forward matrix (the
    dense trial: one GEMM with the full (Dt, 4 Dt') matrix). _backward instead writes the trial GEMM's 4x output, copies
    it into the zero-padded (rows, 4, Dt) layout and runs both walker einsums on 4x operands, which XLA transposes. The
    same products summed in another order. grouped = (n_a, n_b, SummedSpec): the labels of the left bond's (a, b) and
    the site's summed spec; each row (a, b) then goes through its own sector's GEMM only, read in place
    (grouped_gemm.summed_trial_apply; TROT_PROP_GEMM grouped or envs)."""
    Da, Db = A.shape[0], Bs.shape[0]
    H = jnp.tensordot(A, R, axes=([2], [0]))  # (a, n_alpha, d, t')
    Bb = jnp.broadcast_to(Bs.transpose(2, 0, 1, 3), (Da,) + Bs.shape[2:3] + Bs.shape[:2] + Bs.shape[3:])
    X = lax.dot_general(Bb, H, (((4,), (2,)), ((0, 1), (0, 1))))  # (a, n_alpha, b, n_beta, t')
    if grouped is not None:
        n_a, n_b, sspec = grouped
        g = grouped_gemm.row_sectors(n_a, n_b, sspec)
        packed = grouped_gemm.pack_blocks([W.T for W in T[0]])  # each block (K, width): read along the output
        return grouped_gemm.summed_trial_apply(X, g, sspec, packed).reshape(Da, Db, -1)
    if site is None:
        X = X.transpose(0, 2, 3, 1, 4).reshape(Da * Db, -1)  # columns (p, t'), p = n_alpha + 2 n_beta
        return (X @ T[0].T).reshape(Da, Db, -1)
    right = _starts(site.right_widths)
    blocks = {}
    for (start, _, cols), W in zip(site.forward, T[0]):
        G = jnp.concatenate([X[:, p % 2, :, p // 2, right[k]:right[k] + width] for p, k, _, width in cols], axis=2)
        blocks[start] = G.reshape(Da * Db, -1) @ W.T
    out = jnp.concatenate([blocks[s] if s in blocks else jnp.zeros((Da * Db, w), X.dtype)
                           for s, w in zip(_starts(site.left_widths), site.left_widths)], axis=1)
    return out.reshape(Da, Db, -1)


def _sites(layout, n):
    return layout.sites if layout is not None else (None,) * n


# ============================================================================
# <trial|H|walker> through the MPO: no stored H|trial>
# ============================================================================
#
# The local energy is <T|H|W>/<T|W>. The blocked kernels contract the walker with H|T> stored as an MPS (compressed, or
# uncompressed: bond (2 + 4 Ly) chi_T), whose distinct blocks are 18.9 GB at the doped 6x6's chi_T 512 and ~76 GB at 1024.
# Here H stays the automaton MPO of hubbard_mpo_from_h1 (bond 26 on the 6x6, a few MB) and is applied on the fly: the
# environments carry its bond, E (a, b, w, t), and a site is the trial's sector blocks on E's rows (a b w), then the MPO
# on (w, q) -> (p, w'), then the walker's alpha and beta tensors on (a, p) and (b, p). W[w, p, q, w'] acts on the trial's
# physical index q and leaves p for the walker, as apply_mpo / trial_times_h build H|T>, so the value is <H T|W> = <T|H|W>
# for the real symmetric H. The memory is the environments, ~436 MB per walker at the walker bond 64 and chi_T 512.
#
# Three contractions of a site: _forward_mpo applies the MPO as a dense (w, q) x (p, w') GEMM and the walker tensors as
# einsums (the reference, TROT_MPO_KERNEL=einsum); _forward_mpo_gather (the default) applies the MPO's few nonzero terms
# as one gather and the walker tensors as two GEMMs without transposes, and carries the overlap in the MPO's identity
# channel, so mpo_contract gives numerator and overlap in one sweep; _forward_mpo_walker (TROT_MPO_KERNEL=walker) does
# the same with the walker tensors before the trial (8x8, chi_T 2048, bond 64: the trial GEMMs' inner dimension is one
# sector, 230-540, and they ran at ~29 TFLOPS, job 7363725).

MPO_KERNEL = os.environ.get("TROT_MPO_KERNEL", "gather")  # "gather", "walker" (_forward_mpo_walker) or "einsum"


def mpo_site_tensors(W):
    """The MPO's site tensors W[w, p, q, w'] with the channels each bond never uses dropped (trot.meas.mps.trial_times_h's
    `active`): the first site keeps channel 0 on its left, the last the finished-term channel on its right."""
    W = np.asarray(W)
    n, D = len(W), W.shape[1]
    active = ([np.zeros(1, int)] + [np.flatnonzero(np.any(W[b - 1] != 0, axis=(0, 1, 2))) for b in range(1, n)]
              + [np.full(1, D - 1)])
    return tuple(W[k][active[k]][..., active[k + 1]] for k in range(n))


def _forward_mpo(E, T, A, Bs, W, site=None):
    """One site of the left environment of <trial|H|walker>: E (Da, Db, Dw, Dt) to (Da', Db', Dw', Dt')."""
    Da, Db, Dw, Dt = E.shape
    X = E.reshape(Da * Db * Dw, Dt)
    Z = X @ T[0] if site is None else _trial_apply(X, T[0], site.forward, site.right_widths)
    Z = Z.reshape(Da, Db, Dw, 4, -1)  # (a, b, w, q, t')
    Y = jnp.einsum("abwqt,wpqv->abpvt", Z, W)  # the MPO: (w, q) -> (p, w')
    Y = Y.reshape(Da, Db, 2, 2, W.shape[-1], -1)  # (a, b, n_beta, n_alpha, w', t'), p = n_alpha + 2 n_beta
    H = jnp.einsum("abjivt,aic->cbjivt", Y, A)
    return jnp.einsum("cbjivt,bjid->cdvt", H, Bs)


def mpo_numerator(alpha, dressed, trial, mpo, layout=None):
    """<trial|H|walker channels> (the walker's own scale: divide by left_contract for the local energy)."""
    E = jnp.ones((1, 1, 1, 1))
    for A, Bs, T, W, site in zip(alpha, dressed, trial, mpo, _sites(layout, len(alpha))):
        E = _forward_mpo(E, T, A, Bs, W, site)
    return E[0, 0, 0, 0]


class MpoGather(NamedTuple):
    """One site of the MPO as gathers (mpo_gathers). The output (p, v) of a channel with at most one term per p is
    single_coef[p, j] times the column single_src[p, j] = 4 w + q of the trial GEMM's (w, q) columns, j the channel's
    position among those; a channel with more terms sums multi_coef[k, p, j] times the column multi_src[k, p, j] over k
    (zero coefficients pad the shorter sums). The site's right bond lists the single channels, then the multi ones."""
    single_src: jax.Array  # (4, n1) int32
    single_coef: jax.Array  # (4, n1)
    multi_src: jax.Array  # (K, 4, n2) int32
    multi_coef: jax.Array  # (K, 4, n2)


class WalkerMpoGather(NamedTuple):
    """MpoGather from the walker's side, for _forward_mpo_walker: the output (q, v), q the trial's physical state, from
    the columns src = 4 w + p of the walker GEMMs' (w, p), the same channels in the same order."""
    single_src: jax.Array  # (4, n1) int32
    single_coef: jax.Array  # (4, n1)
    multi_src: jax.Array  # (K, 4, n2) int32
    multi_coef: jax.Array  # (K, 4, n2)


def mpo_gathers(mpo, device=True, side="trial") -> tuple:
    """The site tensors W[w, p, q, v] of mpo_site_tensors as one MpoGather per site, for _forward_mpo_gather.

    An output (p, v) is the sum over the (w, q) with W[w, p, q, v] != 0: one term for most channels of
    hubbard_mpo_from_h1 (the identity, a string carried on by Pa Pb, a string opened), none for half the outputs of an
    opening, a few for the finished-term channel (the closings, the on-site terms, the finished identity). Each site's
    right channels are put in the order single, then multi, and the next site's rows follow that order. The last site
    also keeps channel 0, the identity channel (W[0, p, q, 0] = delta_pq and nothing else feeds it, checked at every
    site), whose environment is <trial|walker>: the last bond is (overlap, finished term), so one sweep gives both
    (mpo_contract).

    side "walker": one WalkerMpoGather per site instead, the outputs (q, v) from the (w, p), for _forward_mpo_walker,
    with the same channel order. The automaton's operators have at most one nonzero per row and per column, so a
    channel that is single from the walker's side is single from the trial's (checked)."""
    if side not in ("trial", "walker"):
        raise ValueError(f"side must be 'trial' or 'walker', not {side!r}")
    n, out = len(mpo), []
    order, ident = np.arange(np.asarray(mpo[0]).shape[0]), 0  # the left bond's channels, the identity's position
    for k, W in enumerate(mpo):
        W = np.asarray(W, dtype=np.float64)[order]
        if k == n - 1:
            keep = np.zeros(W.shape[:3] + (1,))
            keep[ident, :, :, 0] = np.eye(4)
            W = np.concatenate([keep, W], axis=3)
        feed = np.zeros(W.shape[:3])
        feed[ident] = np.eye(4)
        if not np.array_equal(W[..., 0], feed):
            raise ValueError(f"site {k}: channel 0 is not the identity channel of hubbard_mpo_from_h1's automaton")
        counts = np.array([[np.count_nonzero(W[:, p, :, v]) for p in range(4)] for v in range(W.shape[3])])  # (Dv, 4)
        single, multi = np.flatnonzero(counts.max(axis=1) <= 1), np.flatnonzero(counts.max(axis=1) > 1)
        # terms[v][x]: the (w, y, coefficient) of output x, from the source physical state y
        if side == "trial":
            terms = [[[(int(w), int(q), W[w, p, q, v]) for w, q in np.argwhere(W[:, p, :, v])] for p in range(4)]
                     for v in range(W.shape[3])]
        else:
            terms = [[[(int(w), int(p), W[w, p, q, v]) for w, p in np.argwhere(W[:, :, q, v])] for q in range(4)]
                     for v in range(W.shape[3])]
            if any(len(terms[v][q]) > 1 for v in single for q in range(4)):
                raise ValueError(f"site {k}: a channel with one term per walker state has more per trial state")
        K = max((len(t) for v in multi for t in terms[v]), default=0)
        src = (np.zeros((4, len(single)), np.int32), np.zeros((K, 4, len(multi)), np.int32))
        coef = (np.zeros((4, len(single))), np.zeros((K, 4, len(multi))))
        for j, v in enumerate(single):
            for x in range(4):
                for w, y, c in terms[v][x]:
                    src[0][x, j], coef[0][x, j] = 4 * w + y, c
        for j, v in enumerate(multi):
            for x in range(4):
                for t, (w, y, c) in enumerate(terms[v][x]):
                    src[1][t, x, j], coef[1][t, x, j] = 4 * w + y, c
        order = np.concatenate([single, multi])
        ident = int(np.flatnonzero(order == 0)[0])
        out.append((MpoGather if side == "trial" else WalkerMpoGather)(src[0], coef[0], src[1], coef[1]))
    if order.tolist() != [0, 1]:
        raise ValueError(f"the last bond's channels came out as {order.tolist()}, not (overlap, finished term)")
    return tuple(type(g)(*map(jnp.asarray, g)) for g in out) if device else tuple(out)


def mpo_device_blocks(mpo) -> tuple:
    """DeviceData.htrial of the MPO energy for the site tensors of mpo_site_tensors: their gathers (mpo_gathers), with
    TROT_MPO_KERNEL=walker the walker-side gathers (_forward_mpo_walker), or with TROT_MPO_KERNEL=einsum the tensors
    themselves, for _forward_mpo."""
    if MPO_KERNEL == "einsum":
        return tuple(jnp.asarray(W) for W in mpo)
    if MPO_KERNEL not in ("gather", "walker"):
        raise ValueError(f"TROT_MPO_KERNEL must be 'gather', 'walker' or 'einsum', not {MPO_KERNEL!r}")
    return mpo_gathers(mpo, side="trial" if MPO_KERNEL == "gather" else "walker")


def is_mpo_gather(block) -> bool:
    """Whether a DeviceData.htrial entry is the MPO as gathers (mpo_contract), from either side."""
    return isinstance(block, (MpoGather, WalkerMpoGather))


def _forward_mpo_gather(E, T, A, B, signs, gather, site=None, grouped=None):
    """One site of the left environment of <trial|H|walker> with the MPO as gathers: E (Da, Db, Dw, Dt) to
    (Da', Db', Dv, Dt'). B is the beta tensor undressed, signs its (Db, 2) sign factors (_parity_signs). grouped:
    _trial_gemm's, for the trial GEMM on E's rows (a, b, w).

    _forward_mpo applies the MPO as a dense (w, q) x (p, v) GEMM (136 x 136 on the 8x8, nearly all zeros) and the walker
    tensors as einsums whose operands XLA transposes first: each of those passes reads and writes an intermediate of
    Da Db Dw 4 Dt' entries (4.6 GB per walker at walker bond 64, chi_T 1024, MPO bond 34). Here, after the same trial
    GEMM, one gather pass makes every (p, v) from its terms, applies the fermionic sign signs[b, n_alpha] and writes
    the layout (a, n_alpha, b, n_beta, v, t') that the next GEMM reads as it is. With the sign off the beta tensor the
    alpha tensor contracts (a, n_alpha) in one GEMM, and the beta tensor (b, n_beta) in one GEMM batched over a', which
    writes the next environment's layout. The same products, summed in another order."""
    Da, Db, Dw, Dt = E.shape
    X = E.reshape(Da * Db * Dw, Dt)
    if _fused(grouped):  # the gather and the sign inside the kernel, written in the layout below
        n_a, n_b, fspec, offsets, n_groups = grouped[:5]
        offsets = jnp.zeros(Dw, jnp.int32) if offsets is None else offsets
        offsets_out = grouped_gemm.channel_offsets(gather, offsets)  # the next bond's channels
        blocked = len(grouped) > 5
        Y = _fused_apply()(X, n_a, n_b, offsets, grouped_gemm.mpo_terms(gather), signs, fspec,
                           grouped_gemm.pack_blocks(T[0]), n_groups=n_groups, layout="left",
                           offsets_out=offsets_out, fill=not blocked)
        if blocked:  # the walker GEMMs on label blocks too (grouped[7] False: no zero fill of the output)
            return grouped_gemm.contract_left(Y, A, B, n_a, n_b, *grouped[5:7], offsets_out, fspec,
                                              fill=len(grouped) <= 7 or bool(grouped[7]))
    else:
        Z = X @ T[0] if site is None else _trial_gemm(X, T[0], site.forward, site.right_widths, grouped)
        Z = Z.reshape(Da, Db, Dw * 4, -1)  # columns (w, q)

        def take(src):  # (a, b) + src.shape[-2:] + (t',)
            return jnp.take(Z, src.reshape(-1), axis=2, mode="clip").reshape((Da, Db) + src.shape[-2:] + (-1,))

        parts = []
        if gather.single_src.shape[-1]:
            parts.append(take(gather.single_src) * gather.single_coef[..., None])
        if gather.multi_src.shape[-1]:
            parts.append(sum(take(s) * c[..., None] for s, c in zip(gather.multi_src, gather.multi_coef)))
        Y = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=3)  # (a, b, p, v, t')
        Y = Y.reshape(Da, Db, 2, 2, Y.shape[3], -1) * signs[None, :, None, :, None, None]  # (a, b, j, i, v, t')
        Y = Y.transpose(0, 3, 1, 2, 4, 5)  # (a, n_alpha, b, n_beta, v, t')
    H = jnp.tensordot(A, Y, axes=([0, 1], [0, 1]))  # (a', b, n_beta, v, t')
    B = jnp.broadcast_to(B, (H.shape[0],) + B.shape)  # (a', b, n_beta, b'): a' is a batch dimension of the GEMM
    return lax.dot_general(B, H, (((1, 2), (1, 2)), ((0,), (0,))))  # (a', b', v, t')


def _forward_mpo_walker(E, T, A, B, signs, gather, site=None):
    """_forward_mpo_gather with the walker tensors first (TROT_MPO_KERNEL=walker; gather a WalkerMpoGather): E (Da, Db,
    Dw, Dt) through the alpha tensor ((n_alpha, a', b, w, t), one GEMM) and the beta tensor with the sign
    signs[b, n_alpha] on it ((n_alpha, a', n_beta, b', w, t), one GEMM batched over (n_alpha, a')), then the MPO from
    the walker's side, (w, p) -> (q, v), as one gather per right trial sector that writes that sector's GEMM operand
    ((a' b' v), (q, t)) with the left sectors of the four q side by side, in the order of sector_trial_blocks' backward
    matrix, and per right sector one GEMM with it. The trial GEMMs' inner dimension is then the four physical states'
    sectors (the forward kernel's is one sector) and they write the environment (Da', Db', Dv, Dt') itself instead of
    a 4x intermediate. The same products summed in another order."""
    Dw, Dt = E.shape[2:]
    Dc, Dd = A.shape[2], B.shape[2]
    Dv = gather.single_src.shape[-1] + gather.multi_src.shape[-1]
    H = jnp.tensordot(A, E, axes=([0], [0]))  # (n_alpha, a', b, w, t)
    Bt = B[None] * signs.T[:, :, None, None]  # (n_alpha, b, n_beta, b'): the sign on the beta tensor
    Bt = jnp.broadcast_to(Bt[:, None], (2, Dc) + B.shape)  # (n_alpha, a', b, n_beta, b'): a' a batch dimension
    X = lax.dot_general(Bt, H, (((2,), (2,)), ((0, 1), (0, 1))))  # (n_alpha, a', n_beta, b', w, t)
    X = X.transpose(1, 3, 4, 2, 0, 5).reshape(Dc, Dd, 4 * Dw, Dt)  # (a', b', (w, p), t), p = n_alpha + 2 n_beta

    def outputs(q, start, stop):  # the MPO's outputs (a', b', v, t) of the trial's physical state q, t in [start, stop)
        part = X[..., start:stop]
        take = lambda src: jnp.take(part, src, axis=2, mode="clip")
        parts = []
        if gather.single_src.shape[-1]:
            parts.append(take(gather.single_src[q]) * gather.single_coef[q][:, None])
        if gather.multi_src.shape[-1]:
            parts.append(sum(take(s[q]) * c[q][:, None] for s, c in zip(gather.multi_src, gather.multi_coef)))
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=2)

    if site is None:
        G = jnp.stack([outputs(q, 0, Dt) for q in range(4)], axis=3)  # (a', b', v, q, t)
        return (G.reshape(Dc * Dd * Dv, 4 * Dt) @ T[1].T).reshape(Dc, Dd, Dv, -1)
    left = _starts(site.left_widths)
    blocks = {}
    for (start, _, cols), W in zip(site.backward, T[1]):
        G = jnp.concatenate([outputs(q, left[k], left[k] + width) for q, k, _, width in cols], axis=3)
        blocks[start] = G.reshape(Dc * Dd * Dv, -1) @ W.T
    out = jnp.concatenate([blocks[s] if s in blocks else jnp.zeros((Dc * Dd * Dv, w), X.dtype)
                           for s, w in zip(_starts(site.right_widths), site.right_widths)], axis=1)
    return out.reshape(Dc, Dd, Dv, -1)


def mpo_contract(alpha, beta, signs, trial, gathers, layout=None, labels=None):
    """(<trial|walker channels>, <trial|H|walker channels>) in one sweep through the MPO's gathers (mpo_gathers), at
    the walker's own scale (it cancels in the local energy): the last bond holds the identity channel, whose
    environment is the overlap's, and the finished-term channel. beta and signs as dense_converter(plan, signed=True)
    gives them. Walker-side gathers (WalkerMpoGather) run the sites with _forward_mpo_walker. labels: the walker's bond
    labels (dense_converter(plan, labelled=True)); with TROT_TRIAL_GEMM=grouped the trial GEMMs then skip the rows
    that cannot be nonzero, each row (a, b, w) in its sector n_a + n_b + the channel's offset (grouped_gemm)."""
    step = _forward_mpo_walker if isinstance(gathers[0], WalkerMpoGather) else _forward_mpo_gather
    specs = _grouped_specs(layout, labels, "forward") if step is _forward_mpo_gather else None
    E, offsets = jnp.ones((1, 1, 1, 1)), jnp.zeros(1, jnp.int32)
    for i, (A, B, s, T, G, site) in enumerate(zip(alpha, beta, signs, trial, gathers, _sites(layout, len(alpha)))):
        if specs is None:
            E = step(E, T, A, B, s, G, site)
        else:
            E = step(E, T, A, B, s, G, site, grouped=_site_rows(labels[0][i], labels[1][i], specs[i], offsets,
                                                                len(alpha), (labels[0][i + 1], labels[1][i + 1]),
                                                                fill=i == len(alpha) - 1))  # the last one is read
            offsets = grouped_gemm.channel_offsets(G, offsets)  # the channels of the next bond
    return E[0, 0, 0, 0], E[0, 0, 1, 0]


def mpo_trial_energy(mpo, trial):
    """<T|H|T> / <T|T> of the dense trial tensors (Dl, 4, Dr) and the MPO's site tensors, on the device."""
    E, N = jnp.ones((1, 1, 1)), jnp.ones((1, 1))
    for W, A in zip(mpo, trial):
        A, W = jnp.asarray(A), jnp.asarray(W)
        S = jnp.tensordot(E, A, axes=([0], [0]))  # (w, u, p, t')
        S = jnp.tensordot(S, W, axes=([0, 2], [0, 1]))  # (u, t', q, w')
        E = jnp.tensordot(S, A, axes=([0, 2], [0, 1]))  # (t', w', u')
        N = jnp.tensordot(jnp.tensordot(N, A, axes=([0], [0])), A, axes=([0, 1], [0, 1]))  # (t', u')
    return float(E[0, 0, 0] / N[0, 0])


def left_contract(alpha, dressed, trial, layout=None):
    """<trial|walker channels> streaming left to right (trial: per site blocks of the layout, or full matrices)."""
    E = jnp.ones((1, 1, 1))
    for A, Bs, T, site in zip(alpha, dressed, trial, _sites(layout, len(alpha))):
        E = _forward(E, T, A, Bs, site).sum(axis=2)
    return E[0, 0, 0]


SZSZ_GROUP = int(os.environ.get("TROT_SZSZ_GROUP", 4))  # stack members one site step moves at a time (szsz_contract)


def _move_stack(stack, step, closing, sz, group):
    """A stack (k, Da, Db, Dt) of environments one site on: (the next stack, the pair values (k,) of S^z at the members'
    sites and S^z at this site against `closing`). step is _forward (left environments, closing the right environment
    beyond the site) or _backward (right environments, closing the left environment before it); both keep the physical
    index p. `group` members at a time (a lax.map over the groups, the remainder in one vmap): all k at once holds k
    copies of every intermediate of the step (the trial GEMM's (rows, 4 Dt') output and its padded sector copies, both
    walker einsums), which peaked at 45.9 GB at chi_T 2048 and walker bond 16 with 5 walkers per chunk."""
    def one(E):
        Y = step(E)  # (a, b, p, t) on the other bond of the site
        return Y.sum(axis=2), jnp.einsum("cdpt,cdt,p->", Y, closing, sz)

    k = stack.shape[0]
    main = (k // group) * group
    moved, values = [], []
    if main:
        grouped = stack[:main].reshape((main // group, group) + stack.shape[1:])
        nxt, val = lax.map(jax.vmap(one), grouped)
        moved.append(nxt.reshape((main,) + nxt.shape[2:]))
        values.append(val.reshape(main))
    if main < k:
        nxt, val = jax.vmap(one)(stack[main:])
        moved.append(nxt)
        values.append(val)
    return jnp.concatenate(moved, axis=0), jnp.concatenate(values, axis=0)


def szsz_contract(alpha, dressed, trial, layout=None, group=None):
    """<trial|S^z_i S^z_j|walker> / <trial|walker> as an (L, L) matrix, on dense walker legs (engine.szsz_contract).

    S^z is diagonal in the local basis, so an insertion is the weight SZ_LOCAL[p] on the physical index p that _forward
    and _backward keep. Every pair (i, j) needs an environment with S^z at i carried to j, so a single sweep costs
    L (L - 1) / 2 site steps (630 on the 6x6, against 36 for an overlap). Here the two halves meet at the middle bond m:
    - a left sweep carries the plain environment (all the diagonals (k, k), the double insertion SZ_LOCAL**2 against the
      right environments) and, up to m, a stack of left environments with one S^z at sites i < m: moving it through site
      k gives the pairs (i, k), i < k < m;
    - a right sweep from the last site down to m carries a stack of right environments with one S^z at sites j >= m:
      moving it through site k gives the pairs (k, j), m <= k < j, against the plain left environment at bond k;
    - every pair (i, j), i < m <= j, is then the inner product of the two stacks at bond m.
    That is about L**2 / 4 + 3 L site steps (about 400 on the 6x6) instead of L**2 / 2 + L, with stacks of at most L / 2
    members; measured 1.29x faster at trial bond 2048 / measurement bond 16 and 1.37x at 1024 / 64 (the steps saved were
    near the chain's right end, where the bonds are small). An optimization barrier runs the right sweep after the left
    one. `group` stack members are moved at a time (default SZSZ_GROUP, _move_stack: it bounds the intermediates, not
    the result). The gauges of the walker cancel in the ratio, as in the local energy."""
    group = max(int(SZSZ_GROUP if group is None else group), 1)
    sz, sz2 = jnp.asarray(engine.SZ_LOCAL), jnp.asarray(engine.SZ_LOCAL ** 2)
    n = len(alpha)
    middle = n // 2
    sites = _sites(layout, n)
    right = right_environments(alpha, dressed, trial, layout)
    upper, diagonal = jnp.zeros((n, n)), []
    left, lefts = jnp.ones((1, 1, 1)), {}  # lefts[k]: the plain left environment at bond k, for k >= middle
    stack = None  # (k, Da, Db, Dt): left environments with one S^z at sites 0..k-1, k <= middle
    for site, (A, Bs, T, sl) in enumerate(zip(alpha, dressed, trial, sites)):
        if site >= middle:
            lefts[site] = left
        Y = _forward(left, T, A, Bs, sl)  # (a', b', p, t')
        diagonal.append(jnp.einsum("cdpt,cdt,p->", Y, right[site + 1], sz2))
        if site < middle:
            opened = jnp.einsum("cdpt,p->cdt", Y, sz)
            if stack is None:
                stack = opened[None]
            else:
                moved, values = _move_stack(stack, lambda E, T=T, A=A, Bs=Bs, sl=sl: _forward(E, T, A, Bs, sl),
                                            right[site + 1], sz, group)
                upper = upper.at[jnp.arange(site), site].set(values)
                stack = jnp.concatenate((moved, opened[None]), axis=0)
        left = Y.sum(axis=2)
    # The right sweep needs nothing of the left one but the plain left environments, so XLA may run both at once and
    # hold both sweeps' intermediates (the peak at walker bond 64, trial bond 1024, chunk 4 went 25.3 -> 48.5 GB). The
    # barrier starts the right sweep (all of whose inputs below come out of it) once the left one is done.
    stack, lefts, right, upper, diagonal = lax.optimization_barrier((stack, lefts, right, upper, diagonal))
    stack_right = None  # (n - k, Da, Db, Dt): right environments with one S^z at sites k..n-1, ascending
    for site in range(n - 1, middle - 1, -1):
        A, Bs, T, sl = alpha[site], dressed[site], trial[site], sites[site]
        opened = jnp.einsum("abpt,p->abt", _backward(right[site + 1], T, A, Bs, sl), sz)
        if stack_right is None:
            stack_right = opened[None]
        else:
            moved, values = _move_stack(stack_right, lambda R, T=T, A=A, Bs=Bs, sl=sl: _backward(R, T, A, Bs, sl),
                                        lefts[site], sz, group)
            upper = upper.at[site, jnp.arange(site + 1, n)].set(values)
            stack_right = jnp.concatenate((opened[None], moved), axis=0)
    upper = upper.at[:middle, middle:].set(jnp.einsum("iabt,jabt->ij", stack, stack_right))
    return (upper + upper.T + jnp.diag(jnp.stack(diagonal))) / right[0][0, 0, 0]


# ============================================================================
# <S^z_i S^z_j> with GEMMs: szsz_contract's stacks without the physical index
# ============================================================================
#
# szsz_contract moves every stack member with _forward / _backward, which keep the physical index p (4x the
# environment) through einsums XLA transposes first, and gets each pair value from that p-resolved output. Here a
# member only needs the plain transfer: the S^z of the pair's second site is a right (left) environment computed once
# per site (Rsz, Lsz below), and the pair value is the inner product of the member with it. The transfers apply the
# fermionic sign to the intermediate, as _forward_mpo_gather does, so each site is the trial GEMM, one elementwise pass
# and two GEMMs. Environments carry a member axis k: left (a, b, k, t), right (b, a, k, t), the layouts the batched
# GEMMs write.

SZSZ_KERNEL = os.environ.get("TROT_SZSZ_KERNEL", "gemm")  # "gemm" (szsz_gemm) or "einsum" (szsz_contract)
SZ_WEIGHTS = np.stack([np.ones(4), engine.SZ_LOCAL, engine.SZ_LOCAL ** 2])  # on p: 1, S^z, (S^z)^2


def _left_transfer(E, T, A, B, signs, weights, site=None, grouped=None):
    """Left environments E (Da, Db, k, Dt) through one site, each physical state p weighted by weights[w, p] (W, 4) and
    summed: (Da', Db', k W, Dt'), member m with weight w at m W + w. B is the beta tensor undressed, signs its (Db, 2)
    sign factors (_parity_signs). The trial GEMM on the rows (a, b, k); one elementwise pass applies the weights and
    signs[b, n_alpha] and writes (a, n_alpha, b, n_beta, k, W, t'); one GEMM contracts (a, n_alpha) with the alpha
    tensor, and one, batched over a', contracts (b, n_beta) with the beta tensor in the output's layout. grouped:
    _trial_gemm's (the labels of a and b)."""
    Da, Db, k, Dt = E.shape
    X = E.reshape(Da * Db * k, Dt)
    n_weights = weights.shape[0]
    if _fused(grouped):  # the weights and the sign inside the kernel, written in the layout below
        n_a, n_b, fspec, _, n_groups = grouped[:5]
        blocked = len(grouped) > 5
        Y = _fused_apply()(X, n_a, n_b, None, grouped_gemm.weight_terms(k, weights), signs, fspec,
                           grouped_gemm.pack_blocks(T[0]), n_groups=n_groups, layout="left",
                           fill=not blocked)
        if blocked:  # the walker GEMMs on label blocks too: (a', b', k W, t')
            return grouped_gemm.contract_left(Y, A, B, n_a, n_b, *grouped[5:7], None, fspec)
        Y = Y.reshape(Da, 2, Db, 2, k, n_weights, -1)
    else:
        Z = X @ T[0] if site is None else _trial_gemm(X, T[0], site.forward, site.right_widths, grouped)
        Z = Z.reshape(Da, Db, k, 2, 2, 1, -1)  # (a, b, k, n_beta, n_alpha, 1, t'), p = n_alpha + 2 n_beta
        f = weights.reshape(n_weights, 2, 2).transpose(1, 2, 0)[None, None, None, :, :, :, None]  # (n_beta, n_alpha, W)
        Y = (Z * f * signs[None, :, None, None, :, None, None]).transpose(0, 4, 1, 3, 2, 5, 6)  # (a, i, b, j, k, W, t')
    H = jnp.tensordot(A, Y, axes=([0, 1], [0, 1]))  # (a', b, n_beta, k, W, t')
    Bc = jnp.broadcast_to(B, (H.shape[0],) + B.shape)  # (a', b, n_beta, b')
    out = lax.dot_general(Bc, H, (((1, 2), (1, 2)), ((0,), (0,))))  # (a', b', k, W, t')
    return out.reshape(out.shape[0], out.shape[1], k * n_weights, -1)


def _right_transfer(R, T, A, B, signs, weights, site=None, grouped=None):
    """Right environments R (Db', Da', k, Dt') through one site, weighted as _left_transfer: (Db, Da, k W, Dt), the
    beta bond first. The trial GEMM on the rows (b', a', k); one elementwise pass applies the weights and writes
    (n_beta, b', a', n_alpha, k, W, t); one GEMM contracts (n_beta, b') with the beta tensor; the sign
    signs[b, n_alpha] goes on the alpha tensor (a (Db, Da, Da', 2) array) and one GEMM, batched over b, contracts
    (a', n_alpha) in the output's layout. grouped: _trial_gemm's (the labels of b' and a', in that order)."""
    Dd, Dc, k, Dt = R.shape
    X = R.reshape(Dd * Dc * k, Dt)
    n_weights = weights.shape[0]
    if _fused(grouped):  # the weights inside the kernel, written in the layout below (the sign goes on A below)
        n_bp, n_ap, fspec, _, n_groups = grouped[:5]
        blocked = len(grouped) > 5
        V = _fused_apply()(X, n_bp, n_ap, None, grouped_gemm.weight_terms(k, weights), None, fspec,
                           grouped_gemm.pack_blocks(T[1]), n_groups=n_groups, layout="right",
                           fill=not blocked)
        if blocked:  # the walker GEMMs on label blocks too: (b, a, k W, t); next rows (n_b, n_a) of the left bond
            n_b, n_a = grouped[5:7]
            return grouped_gemm.contract_right(V, A, B, signs, n_a, n_b, n_ap, n_bp, fspec)
        V = V.reshape(2, Dd, Dc, 2, k, n_weights, -1)
    else:
        U = X @ T[1] if site is None else _trial_gemm(X, T[1], site.backward, site.left_widths, grouped)
        U = U.reshape(Dd, Dc, k, 2, 2, 1, -1)  # (b', a', k, n_beta, n_alpha, 1, t)
        f = weights.reshape(n_weights, 2, 2).transpose(1, 2, 0)[None, None, None, :, :, :, None]
        V = (U * f).transpose(3, 0, 1, 4, 2, 5, 6)  # (n_beta, b', a', n_alpha, k, W, t)
    V = jnp.tensordot(B, V, axes=([1, 2], [0, 1]))  # (b, a', n_alpha, k, W, t)
    As = A.transpose(0, 2, 1)[None] * signs[:, None, None, :]  # (b, a, a', n_alpha)
    out = lax.dot_general(As, V, (((2, 3), (1, 2)), ((0,), (0,))))  # (b, a, k, W, t)
    return out.reshape(out.shape[0], out.shape[1], k * n_weights, -1)


def _move_members(stack, transfer, group):
    """transfer (plain, one weight) of a stack's members (axis 2), `group` at a time: a lax.map over the groups and the
    remainder at once (it bounds the intermediates, ~4 environments per member, not the result)."""
    k = stack.shape[2]
    main = (k // group) * group
    parts = []
    if main:
        grouped = jnp.moveaxis(stack[:, :, :main].reshape(stack.shape[:2] + (main // group, group) + stack.shape[3:]),
                               2, 0)
        moved = lax.map(transfer, grouped)  # (groups, d1, d2, group, t')
        parts.append(jnp.moveaxis(moved, 0, 2).reshape(moved.shape[1:3] + (main,) + moved.shape[4:]))
    if main < k:
        parts.append(transfer(stack[:, :, main:]))
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=2)


def szsz_gemm(alpha, beta, signs, trial, layout=None, group=None, labels=None):
    """<trial|S^z_i S^z_j|walker> / <trial|walker> as an (L, L) matrix, szsz_contract's meet-in-the-middle with the
    transfers above (beta and signs as dense_converter(plan, signed=True) gives them; labels, the walker's bond labels
    of dense_converter(plan, labelled=True): with TROT_TRIAL_GEMM=grouped the trial GEMMs skip the rows that cannot be
    nonzero).

    A right sweep stores, at every bond k, Rsz[k] and Rsz2[k]: the right environment with S^z (and (S^z)^2) on site k,
    one transfer with three weights per site. A left sweep then gives the diagonal <Lplain[k], Rsz2[k]>, moves the
    stack of left environments with S^z at sites i < k up to the middle bond m (pairs (i, k): <stack, Rsz[k]>) and
    keeps Lsz[k], the left environment at bond k + 1 with S^z on site k, for k >= m. A right sweep from the last site
    down to m moves the stack of right environments with S^z at sites j > k (pairs (k, j): <Lsz[k], stack>), opened
    with Rsz[k]; the pairs across m are the inner products of the two stacks. Every pair is the same contraction as in
    szsz_contract, summed in another order. `group` stack members move at a time (default SZSZ_GROUP)."""
    group = max(int(SZSZ_GROUP if group is None else group), 1)
    weights, plain = jnp.asarray(SZ_WEIGHTS), jnp.ones((1, 4))
    n = len(alpha)
    middle = n // 2
    sites = _sites(layout, n)
    forward, backward = _grouped_specs(layout, labels, "forward"), _grouped_specs(layout, labels, "backward")

    def rows(transfer, k):  # the grouped argument of site k: left rows (a, b, member), right rows (b', a', member)
        if transfer is _left_transfer:
            return None if forward is None else _site_rows(labels[0][k], labels[1][k], forward[k], None, n,
                                                           (labels[0][k + 1], labels[1][k + 1]))
        return None if backward is None else _site_rows(labels[1][k + 1], labels[0][k + 1], backward[k], None, n,
                                                        (labels[1][k], labels[0][k]))

    R, Rsz, Rsz2 = jnp.ones((1, 1, 1, 1)), [None] * n, [None] * n
    for k in range(n - 1, -1, -1):
        out = _right_transfer(R, trial[k], alpha[k], beta[k], signs[k], weights, sites[k],
                              rows(_right_transfer, k))  # (b, a, 3, t)
        R, Rsz[k], Rsz2[k] = out[:, :, 0:1], out[:, :, 1:2], out[:, :, 2:3]
    overlap = R[0, 0, 0, 0]

    def mover(transfer, k):
        return lambda S: transfer(S, trial[k], alpha[k], beta[k], signs[k], plain, sites[k], rows(transfer, k))

    upper, diagonal = jnp.zeros((n, n)), []
    left, stack, Lsz = jnp.ones((1, 1, 1, 1)), None, {}  # stack: (a, b, k, t), S^z at sites 0..k-1, at bond k
    for k in range(n):
        diagonal.append(jnp.einsum("abt,bat->", left[:, :, 0], Rsz2[k][:, :, 0]))
        if stack is not None and k < middle:
            upper = upper.at[jnp.arange(k), k].set(jnp.einsum("abmt,bat->m", stack, Rsz[k][:, :, 0]))
        out = _left_transfer(left, trial[k], alpha[k], beta[k], signs[k], weights[:2], sites[k],
                             rows(_left_transfer, k))  # (a, b, 2, t')
        left, opened = out[:, :, 0:1], out[:, :, 1:2]
        if k < middle:
            stack = opened if stack is None else jnp.concatenate(
                (_move_members(stack, mover(_left_transfer, k), group), opened), axis=2)
        else:
            Lsz[k] = opened
    # As in szsz_contract: the right stack needs only stored environments, so the barrier keeps XLA from running it
    # alongside the left sweep (both sweeps' intermediates at once).
    stack, Lsz, Rsz, upper, diagonal = lax.optimization_barrier((stack, Lsz, Rsz, upper, diagonal))
    stack_right = None  # (b, a, k, t): S^z at sites k..n-1, ascending, at bond k
    for k in range(n - 1, middle - 1, -1):
        if stack_right is None:
            stack_right = Rsz[k]
            continue
        upper = upper.at[k, jnp.arange(k + 1, n)].set(jnp.einsum("abt,bamt->m", Lsz[k][:, :, 0], stack_right))
        stack_right = jnp.concatenate((Rsz[k], _move_members(stack_right, mover(_right_transfer, k), group)), axis=2)
    upper = upper.at[:middle, middle:].set(jnp.einsum("abit,bajt->ij", stack, stack_right))
    return (upper + upper.T + jnp.diag(jnp.stack(diagonal))) / overlap


# "walker" (_backward_summed) or "trial" (_backward, then the sum). Walker-first by default since 2026-10-29: 4 half
# steps of 200 walkers on the 8x8 at chi_T 2048 9.73 -> 8.24 s (jobs 7364937-38), the compile 636 -> 1137 s.
RIGHT_ENVS = os.environ.get("TROT_RIGHT_ENVS", "walker")
# "dense" or "grouped": the propagation's trial GEMMs (right environments, field sweep) on the rows that can be nonzero
# (grouped_gemm, as TROT_TRIAL_GEMM=grouped does for the measurement; the walker's labels come from
# dense_converter(plan, labelled=True)). Grouped right environments follow TROT_RIGHT_ENVS: walker-first through
# grouped_gemm.summed_trial_apply (the sum over the physical index inside the kernel, X read in place), or trial-first
# (_backward with the grouped trial GEMM, then the sum). "envs": the right environments grouped, the field sweep dense.
# 8x8 at walker bond 16: about 21% of the trial GEMMs' multiply-adds can be nonzero (job 7366493).
# "blocked": the unit fused kernel (no zero fill) and the walker einsums on label blocks too
# (grouped_gemm.forward_walker_blocks / backward_walker_blocks), the right environments trial-first.
# 8x8, 4 half steps of 200 walkers (jobs 7375439-41): dense 8.23 s, grouped (trial-first) 8.87, blocked 10.23, each
# walker's rows in tiles of their own (a few dozen rows per sector). Grouped trial-first with the walkers' rows pooled
# under vmap (grouped_gemm.grouped_trial_apply): 8.79 s (job 7388161). In the dense half step the trial GEMMs are 55% of
# the kernel time, the eigh 17%, the rest XLA fusions (trace 7388162). The walker-first grouped step is not timed yet.
PROP_GEMM = os.environ.get("TROT_PROP_GEMM", "dense")
PROP_GEMMS = ("dense", "grouped", "envs", "blocked")


def _prop_mode(layout, labels):
    """PROP_GEMM where it applies (the trial in sector blocks, the walker's labels given), else None."""
    if PROP_GEMM not in PROP_GEMMS:
        raise ValueError(f"TROT_PROP_GEMM must be one of {PROP_GEMMS}, not {PROP_GEMM!r}")
    return None if PROP_GEMM == "dense" or layout is None or labels is None else PROP_GEMM


def _prop_specs(layout, labels, direction):
    """The layout's TrialSpecs (PROP_GEMM grouped, envs) or FusedSpecs (blocked) in a direction when the propagation's
    trial GEMMs run grouped, else None."""
    mode = _prop_mode(layout, labels)
    return None if mode is None else grouped_gemm.layout_specs(layout, direction, fused=mode == "blocked")


def _prop_rows(labels, spec, bond, other, n_sites):
    """A propagation step's grouped argument: rows on `bond` (labels (alpha, beta)), the step's output on `other`:
    (n_alpha, n_beta, TrialSpec, None) for the grouped trial GEMM, (n_alpha, n_beta, FusedSpec, n_alpha', n_beta',
    2 L + 1) for the blocked step."""
    if isinstance(spec, grouped_gemm.FusedSpec):
        return labels[0][bond], labels[1][bond], spec, labels[0][other], labels[1][other], 2 * n_sites + 1
    return labels[0][bond], labels[1][bond], spec, None


def right_environments(alpha, dressed, trial, layout=None, labels=None):
    """The right environments, right[k] for the sites k.. (right[L] = 1): each site the walker tensors first and the
    sum over the physical index inside the trial GEMM (_backward_summed), or with TROT_RIGHT_ENVS=trial the trial first
    and the physical index summed after the walker tensors (_backward). labels: the walker's bond labels; with
    TROT_PROP_GEMM=grouped or envs the trial GEMMs on the rows that can be nonzero: walker-first each row (a, b) through
    the left sector of n_a + n_b (summed_trial_apply), trial-first each row (a', b') through the right sector of
    n_a' + n_b'; blocked: trial-first on label blocks."""
    if RIGHT_ENVS not in ("trial", "walker"):
        raise ValueError(f"TROT_RIGHT_ENVS must be 'trial' or 'walker', not {RIGHT_ENVS!r}")
    mode = _prop_mode(layout, labels)
    summed = (grouped_gemm.layout_specs(layout, "summed")
              if mode in ("grouped", "envs") and RIGHT_ENVS == "walker" else None)
    specs = None if summed is not None else _prop_specs(layout, labels, "backward")
    right = [jnp.ones((1, 1, 1))]
    sites = _sites(layout, len(alpha))
    for k in range(len(alpha) - 1, -1, -1):
        A, Bs, T, site = alpha[k], dressed[k], trial[k], sites[k]
        if summed is not None:  # rows (a, b) on bond k
            right.append(_backward_summed(right[-1], T, A, Bs, site, (labels[0][k], labels[1][k], summed[k])))
        elif specs is not None:
            grouped = _prop_rows(labels, specs[k], k + 1, k, len(alpha))
            right.append(_backward(right[-1], T, A, Bs, site, grouped).sum(axis=2))
        elif RIGHT_ENVS == "walker":
            right.append(_backward_summed(right[-1], T, A, Bs, site))
        else:
            right.append(_backward(right[-1], T, A, Bs, site).sum(axis=2))
    return right[::-1]


def field_sweep(ca, cb, randoms, alpha, dressed, prefactor, right, overlap_in, labels=None, *, hs, weight_floor, trial,
                layout=None):
    """The diagonal HS sweep of one walker (engine.field_sweep) on dense environments: at each site the four physical
    marginals give both field proposals, one field is sampled, and the left environment absorbs the chosen diagonal
    factor. labels: the walker's bond labels, for the grouped trial GEMMs of TROT_PROP_GEMM=grouped or blocked (envs
    keeps the sweep dense)."""
    diagonal = jnp.stack((jnp.ones(2), hs[:, 0], hs[:, 1], hs[:, 0] * hs[:, 1]), axis=1)
    specs = None if _prop_mode(layout, labels) == "envs" else _prop_specs(layout, labels, "forward")
    left = jnp.ones((1, 1, 1))
    overlap, log_weight = overlap_in, jnp.zeros(())
    nodes = jnp.zeros((), jnp.int64)
    fields = []
    for site, (A, Bs, T, sl) in enumerate(zip(alpha, dressed, trial, _sites(layout, len(alpha)))):
        grouped = None if specs is None else _prop_rows(labels, specs[site], site, site + 1, len(alpha))
        Y = _forward(left, T, A, Bs, sl, grouped)  # (a', b', p, t')
        marginal = jnp.einsum("cdpt,cdt->p", Y, right[site + 1])
        proposed = prefactor * (diagonal @ marginal)
        ratios = engine.constrain_ratio(proposed / overlap, weight_floor)
        nodes += jnp.sum(ratios <= 0.0, dtype=jnp.int64)
        probabilities = 0.5 * ratios
        norm = probabilities.sum() + 1.0e-13
        field = jnp.where(randoms[site] < probabilities[0] / norm, 0, 1)
        overlap = proposed[field]
        log_weight += jnp.log(norm)
        fields.append(field)
        left = jnp.einsum("p,cdpt->cdt", diagonal[field], Y)
    fields = jnp.stack(fields)
    return ca * hs[fields, 0][:, None], cb * hs[fields, 1][:, None], overlap, jnp.exp(log_weight), nodes


# ============================================================================
# The kernels of a dense walker plan
# ============================================================================

def dense_converter(plan, signed=False, labelled=False):
    """convert(qa, qb) of a dense walker plan for one orthonormal walker: the alpha tensors, the beta tensors dressed
    with the fermionic sign of their labels, and the gauge product. signed=True: (alpha, beta, signs, gauge) instead,
    the beta tensors undressed and the (Db, 2) sign factors of their left labels (_parity_signs), which the MPO energy
    applies to its intermediate (_forward_mpo_gather). labelled=True: (alpha, beta, signs, gauge, labels), labels the
    bond labels (n_alpha per bond, n_beta per bond) for the grouped trial GEMMs (mpo_contract, szsz_gemm). Cached on
    the plan, all sharing the channel converters."""
    key = "dense_converter_labelled" if labelled else "dense_converter_signed" if signed else "dense_converter"
    convert = plan.caches.get(key)
    if convert is None:
        converters = plan.caches.get("dense_channel_converters")
        if converters is None:
            spec = plan.dense
            converters = tuple(make_dense_converter(p, spec.chi, spec.move, cans_lo=spec.cans_lo,
                                                    gram_floor=spec.gram_floor, charge_gap=spec.gap(), eigh=spec.eigh,
                                                    snap=spec.snap) for p in plan.orbital_plans)
            plan.caches["dense_channel_converters"] = converters

        def convert(qa, qb):
            alpha, alpha_charges, ga = converters[0].convert(qa)
            beta, charges, gb = converters[1].convert(qb)
            signs = [_parity_signs(q) for q in charges[:-1]]  # the labels left of each site
            if labelled:
                return alpha, beta, signs, ga * gb, (list(alpha_charges), list(charges))
            if signed:
                return alpha, beta, signs, ga * gb
            return alpha, [_dressed(B, s) for B, s in zip(beta, signs)], ga * gb

        plan.caches[key] = convert
    return convert


def make_dense_kernels(plan, overlap_plan: engine.DensePlan | None = None, energy=None) -> engine.Kernels:
    """The engine.Kernels of a dense walker plan (plan.dense a DenseSpec) for a trial whose blocks follow overlap_plan
    (dense_overlap_plan; None: the full matrices): batch_qr, overlaps, step_batch, probe and the one-walker overlap.
    energy "mpo": also the local energies <T|H|W>/<T|W> through the MPO (DeviceData.htrial holds mpo_device_blocks: the
    MPO's gathers, or its site tensors); None: the energies raise (use a measurement plan)."""
    overlap_plan = engine.DensePlan(int(plan.norb), dict(trial="dense")) if overlap_plan is None else overlap_plan
    layout = overlap_plan.layout
    convert = dense_converter(plan)
    labelled = dense_converter(plan, labelled=True)  # the labels feed the grouped trial GEMMs (TROT_TRIAL_GEMM)
    batch_qr = engine.make_batch_qr(engine.resolve_walker_qr(plan.walker_qr))

    def overlaps(ca, cb, data):
        qa, da = batch_qr(ca)
        qb, db = batch_qr(cb)

        def one(qa1, qb1):
            alpha, dressed, gauge = convert(qa1, qb1)
            return gauge * left_contract(alpha, dressed, data.trial, layout)
        return da * db * jax.vmap(one)(qa, qb)

    def step_batch(ca, cb, randoms, sweep, data, weight_floor):
        """As engine.make_kernels' step_batch: one conversion per walker, right environments and the overlap, with
        sweep also the HS field sweep. Returns walkers, overlap before and after the fields, weight factor, nodes."""
        qa, da = batch_qr(ca)
        qb, db = batch_qr(cb)

        def one(qa1, qb1):  # the labelled conversion: the labels feed the grouped trial GEMMs (TROT_PROP_GEMM)
            alpha, beta, signs, gauge, labels = labelled(qa1, qb1)
            dressed = [_dressed(B, s) for B, s in zip(beta, signs)]
            return alpha, dressed, labels, right_environments(alpha, dressed, data.trial, layout, labels), gauge
        alpha, dressed, labels, right, gauge = jax.vmap(one)(qa, qb)
        prefactor = da * db * gauge
        now = prefactor * right[0][:, 0, 0, 0]

        def with_fields():
            sweep_one = partial(field_sweep, hs=data.hs, weight_floor=weight_floor, trial=data.trial, layout=layout)
            return jax.vmap(sweep_one)(ca, cb, randoms, alpha, dressed, prefactor, right, now, labels)

        def without():
            return ca, cb, now, jnp.ones_like(now), jnp.zeros(now.shape, jnp.int64)

        if isinstance(sweep, bool):
            ca2, cb2, after, factor, nodes = with_fields() if sweep else without()
        else:
            ca2, cb2, after, factor, nodes = lax.cond(sweep, with_fields, without)
        return ca2, cb2, now, after, factor, nodes

    def local_energy(alpha, beta, signs, data, labels=None):
        """One converted walker's <T|H|W>/<T|W>: one sweep through the MPO's gathers (the trial GEMMs grouped with
        TROT_TRIAL_GEMM=grouped), or with its site tensors (TROT_MPO_KERNEL=einsum) the einsum numerator over
        left_contract."""
        if is_mpo_gather(data.htrial[0]):
            overlap, numerator = mpo_contract(alpha, beta, signs, data.trial, data.htrial, layout, labels)
            return numerator / overlap
        dressed = [_dressed(B, s) for B, s in zip(beta, signs)]
        return (mpo_numerator(alpha, dressed, data.trial, data.htrial, layout)
                / left_contract(alpha, dressed, data.trial, layout))

    def probe(ca, cb, data):
        qa, da = qr_with_det(ca)
        qb, db = qr_with_det(cb)
        alpha, beta, signs, gauge, labels = labelled(qa, qb)
        dressed = [_dressed(B, s) for B, s in zip(beta, signs)]
        overlap = da * db * gauge * left_contract(alpha, dressed, data.trial, layout)
        return dict(qa=qa, qb=qb, alpha=alpha, beta=dressed, gauges=gauge, overlap=overlap,
                    energy=None if energy is None else local_energy(alpha, beta, signs, data, labels))

    def overlap_one(ca, cb, data):
        return probe(ca, cb, data)["overlap"]

    def no_energy(*_):
        raise ValueError("this dense walker plan has no local energy: set measure_channel_chi (a sector measurement "
                         "plan) or energy_kernel='mpo'")

    if energy is None:
        energies = energy_one = no_energy
    else:
        def energies(ca, cb, data):
            qa, _ = batch_qr(ca)
            qb, _ = batch_qr(cb)

            def one(qa1, qb1):
                alpha, beta, signs, _, labels = labelled(qa1, qb1)
                return local_energy(alpha, beta, signs, data, labels)
            return jax.vmap(one)(qa, qb)

        def energy_one(ca, cb, data):
            qa, _ = qr_with_det(ca)
            qb, _ = qr_with_det(cb)
            alpha, beta, signs, _, labels = labelled(qa, qb)
            return local_energy(alpha, beta, signs, data, labels)

    return engine.Kernels(None, overlap_plan, None, energy, batch_qr, overlaps, energies, step_batch, probe,
                          overlap_one, energy_one, jax.jit(probe))
