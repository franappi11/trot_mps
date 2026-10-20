"""Dense-bond conversion of a Slater-determinant channel to a charge-labelled MPS: every walker keeps its own states.

The batched engine (trot.gmps.engine) freezes, at every gate, how many states each electron-number sector keeps
(BondPlan.kept_per_sector), so that the shapes are static. Here the shapes are static in another way: the bonds are dense
and have a fixed size chi, and each walker keeps the chi largest Schmidt states of the gate's two-site matrix whatever
their electron number. That is the truncation of channel_mps with plan_bonds(..., chi_max=chi) walker by walker (the
"own counts" of trot.gmps.allocation.walker_allocation), without counts to train or to freeze.

The bond labels become data: bond index k of walker w carries the electron number charges[bond][k] (float array of
integers, one per bond index). They are what the contraction with a number-labelled trial needs (the fermionic sign of
combine_channels depends on the beta bond's number). A two-site matrix is block diagonal in the electron number; the
factorisations below keep every kept state inside one sector:

  gate   eigh of the smaller Gram matrix, as engine.factor_sectors, after adding a shift charge_gap * trace(G) * charge
         to each index: the shift separates eigenvalues of different sectors (so the eigenvectors cannot mix across
         sectors even where the spectra are degenerate) and leaves the eigenvectors of each sector unchanged. The states
         are ranked by their true Rayleigh quotients, the label of a kept state is the rounded mean charge of its
         eigenvector. Measured on the H100: with matrices up to 32 x 32 (chi <= 16, the Jacobi eigh) the labels are exact
         (impurity exactly 0 at every factorisation of every walker); with 64 x 64 matrices (chi = 32) mislabeled states
         are common (a quarter of the factorisations), under a gap of 1e-7 and also of 2 (disjoint sector bands, which
         rules out accidental coincidences of shifted eigenvalues), and the large gap also corrupts the states (8x8
         overlap error 0.16). So the failure is not an accident the shift can remove. Snapping (snap, the default of
         DenseSpec) restores the blocks' zeros after every factorisation and makes the labels exact at bond 32.
  move   the exact factorisation M = Q R that shifts the orthogonality centre, three ways:
           "qr"    dense Householder QR. Exact, but a dependent column (zero-weight padding, a sector with more columns
                   than rows) gets an arbitrary completion that later columns are orthogonalised against: the labels
                   of weighted states can be wrong. Kept as the speed and accuracy floor of the moves.
           "gram"  polar factor W = M (M^T M)^(-1/2) from one eigh of the Gram matrix, Q = W, R = W^T M. W is a function
                   of M, so it is block structured whatever the eigenvectors do inside degenerate eigenspaces.
           "cans"  the same polar factor by an odd cubic polynomial iteration (Chebyshev-optimal coefficients on
                   [lo, 1], polar_coefficients; the idea of https://github.com/fallnlove/cans_svd): matrix products
                   only, a fixed number of steps, zero singular values stay exactly zero.
         Q = W has the charges of M's columns, so the labels carry over exactly. A move that cannot lose rank (fewer rows
         than columns) is Q = 1, R = M for every method.
Precision: the polar factors resolve singular values down to a relative lo (cans_lo, gram_floor) and drop what lies
below, an amplitude error of that size; the Gram eigh of a gate has the squared-value precision of the engine's
truncating eigh (about 1e-8 in amplitude).

make_dense_converter(plan, chi, move) returns convert(C) for one orthonormal L x N channel: (tensors, charges, gauge),
padded zero-weight states included; vmap it over walkers. The orbital plan (gates) is that of the frozen walker plan, so
the angles, gauge and the state before truncation are the engine's.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from trot.gmps.engine import device_channel_angles, gate_sites
from trot.gmps.utils import OrbitalPlan, gate_pair

MOVES = ("qr", "gram", "cans")
EIGH = ("default", "jacobi")  # jnp.linalg.eigh, or jax.lax.linalg.eigh(implementation=JACOBI) (GPU only)


class DenseSpec(NamedTuple):
    """The settings of a dense walker plan (MpsWalkerPlan.dense): bond size, centre-move method, the gate eigh's charge
    gap (None: 1e-7), the polar-factor precision options, the eigh implementation and snapping (make_dense_converter).
    Measured on the H100 (bench_dense_conversion.py, 6x6 and 8x8 walkers): at chi <= 16 (gate matrices <= 32 x 32, the
    Jacobi eigh) the carried charge labels are exact; at chi = 32 they are not under either gap, and a gap of 2 (disjoint
    sector bands) also damages the states themselves (8x8 overlap error 0.16). Snapping (the default) or eigh="jacobi"
    makes them exact at bond 32 too (bench_dense_conversion.py --variants, 6x6 and 8x8: state error 1.6e-7 / 1.1e-7,
    the sign check at its floor), snapping at no measurable cost."""
    chi: int
    move: str = "gram"
    charge_gap: float | None = None
    gram_floor: float = 1.0e-10
    cans_lo: float = 1.0e-9
    eigh: str = "default"
    snap: bool = True  # exact sector blocks at any bond (no-op up to 16; the bond-32 fix, at no measurable cost)
    trial: str = "sector"  # the trial in the contractions: "sector" blocks (dense_engine.sector_layout) or "dense" matrices

    def gap(self) -> float:
        return float(self.charge_gap) if self.charge_gap is not None else 1.0e-7


WEIGHT_FLOOR = 1.0e-20  # a kept state lighter than this (squared norm) is a zero mode: its label is irrelevant
BAD_IMPURITY = 1.0e-9  # a kept state mixing charges by more than this (sum Q^2 (label - charge)^2) is mislabeled
MISLABEL = 1.0e-6  # impurity above which a kept state's sign in a contraction is wrong by more than ~1e-3 of its amplitude
STATS = ("impurity", "offlabel", "mislabeled_weight", "input_leak")  # per factorisation, see _diagnostics


def _eigh(A, eigh):
    """(eigenvalues ascending, eigenvectors) of a symmetric matrix: jnp.linalg.eigh (XLA picks the solver: batched
    Jacobi up to 32 x 32 on GPUs, a QR-type solver above, LAPACK on CPUs) or the Jacobi solver forced at any size."""
    if eigh == "jacobi":
        V, w = lax.linalg.eigh(A, implementation=lax.linalg.EighImplementation.JACOBI)
        return w, V
    return jnp.linalg.eigh(A)


# ============================================================================
# Polar factor
# ============================================================================

def polar_coefficients(lo=1.0e-9, tol=1.0e-14, max_steps=64):
    """Coefficients (a_k, c_k) of the odd cubics p_k(x) = a_k x - c_k x^3 whose composition sends [lo, 1] to 1 within tol.

    Each cubic is the one that minimises max |1 - p| on the interval [l, u] its predecessor left (closed form by
    equioscillation: p(l) = p(u) = 1 - E, p(sqrt((u^2 + u l + l^2) / 3)) = 1 + E); the next interval is [1 - E, 1 + E].
    From lo = 1e-9 about 15 steps (the first steps multiply small values by about 5, the last ones are Newton-Schulz)."""
    l, u = float(lo), 1.0
    a, c = [], []
    for _ in range(max_steps):
        S = u * u + u * l + l * l
        m = np.sqrt(S / 3.0)
        ck = 2.0 / (2.0 * S * m / 3.0 + S * l - l ** 3)
        ak = ck * S
        error = 2.0 * ak * m / 3.0 - 1.0
        a.append(ak)
        c.append(ck)
        l, u = 1.0 - error, 1.0 + error
        if error < tol:
            break
    return np.asarray(a), np.asarray(c)


def polar_cans(M, a, c, unroll=1):
    """The partial-isometry polar factor of a tall matrix M by the cubic iteration X <- a X - c X (X^T X) of
    polar_coefficients, from M / ||M||_F. Singular values below lo ||M||_F stay below 1 (an amplitude error of at most
    that); zero singular values stay zero."""
    norm = jnp.sqrt(jnp.sum(M * M))
    X = M / jnp.where(norm > 0, norm, 1.0)

    def step(X, coefficients):
        ak, ck = coefficients
        return ak * X - ck * (X @ (X.T @ X)), None

    X, _ = lax.scan(step, X, (jnp.asarray(a), jnp.asarray(c)), unroll=unroll)
    return X


def polar_gram(M, floor, eigh="default"):
    """The polar factor of a tall matrix M from the eigh of M^T M; singular values below floor times the largest are
    dropped (zero modes included)."""
    w, V = _eigh(M.T @ M, eigh)
    s = jnp.sqrt(jnp.maximum(w, 0.0))
    keep = s > floor * jnp.max(s)
    inverse = jnp.where(keep, 1.0 / jnp.where(keep, s, 1.0), 0.0)
    return (M @ (V * inverse)) @ V.T


# ============================================================================
# Factorisations
# ============================================================================

class Options(NamedTuple):
    move: str
    a: np.ndarray
    c: np.ndarray
    unroll: int
    gram_floor: float
    charge_gap: float
    eigh: str
    snap: bool


def _leak(M, c_row, c_col):
    """The share of M's squared entries outside its sector blocks (row label != column label): exactly 0 when the input
    of a factorisation respects the labels it carries."""
    outside = c_row[:, None] != c_col[None, :]
    return jnp.sum(jnp.where(outside, M * M, 0.0)) / (jnp.sum(M * M) + 1.0e-300)


def _diagnostics(Q, labels, new, weight):
    """One factorisation's new states before snapping, as a dict of scalars:
      impurity           the largest sum_i Q_ik^2 (label_i - new_k)^2 over the kept states heavier than WEIGHT_FLOOR;
      offlabel           the largest |Q_ik| with label_i != new_k over every kept state: exactly 0 while every step keeps
                         the zeros of the sector blocks, the seed of a mislabeling when not;
      mislabeled_weight  the largest weight (squared norm) of a kept state whose impurity exceeds MISLABEL. Every
                         factorisation acts at the orthogonality centre, so this is a share of the walker's norm."""
    outside = labels[:, None] != new[None, :]
    spread = jnp.sum(Q * Q * (labels[:, None] - new[None, :]) ** 2, axis=0)
    weight = jnp.maximum(weight, 0.0)
    return dict(impurity=jnp.max(jnp.where(weight > WEIGHT_FLOOR, spread, 0.0)),
                offlabel=jnp.max(jnp.where(outside, jnp.abs(Q), 0.0)),
                mislabeled_weight=jnp.max(jnp.where(spread > MISLABEL, weight, 0.0)))


def _snap(Q, labels, new):
    """Q with every entry outside its column's sector set to zero: the sector blocks' zeros restored exactly."""
    return jnp.where(labels[:, None] == new[None, :], Q, 0.0)


def _factor_move(M, c_row, c_col, options: Options):
    """M = Q R exactly, the new bond's labels and the diagnostics of the new states (with options.snap, Q is snapped to
    its sectors and R = Q^T M)."""
    r, c = M.shape
    if r <= c:  # the rows span everything: no rank is lost
        zero = jnp.zeros((), M.dtype)
        return jnp.eye(r, dtype=M.dtype), M, c_row, dict(impurity=zero, offlabel=zero, mislabeled_weight=zero)
    if options.move == "qr":
        Q, R = jnp.linalg.qr(M, mode="reduced")
        new = jnp.round(jnp.sum(Q * Q * c_row[:, None], axis=0))
    else:
        if options.move == "gram":
            Q = polar_gram(M, options.gram_floor, options.eigh)
        else:
            Q = polar_cans(M, options.a, options.c, options.unroll)
        R = Q.T @ M
        new = c_col
    info = _diagnostics(Q, c_row, new, jnp.sum(R * R, axis=1))
    if options.snap:
        Q = _snap(Q, c_row, new)
        R = Q.T @ M
    return Q, R, new, info


def _factor_gate(M, c_row, c_col, K, options: Options):
    """The top K Schmidt states of M (the smaller Gram matrix's eigenvectors, sector-pure through the charge shift):
    M ~ Q R with Q isometric, the new bond's labels and the diagnostics of the kept states (with options.snap, the
    eigenvectors are snapped to their sectors first)."""
    r, c = M.shape
    rows_side = r <= c
    B = M if rows_side else M.T
    labels = c_row if rows_side else c_col
    G = B @ B.T
    trace = jnp.trace(G)
    unit = options.charge_gap * jnp.where(trace > 0, trace, 1.0)
    _, V = _eigh(G + jnp.diag(labels * unit), options.eigh)
    rayleigh = jnp.sum(V * (G @ V), axis=0)  # the true eigenvalues, free of the shift
    top, index = lax.top_k(rayleigh, K)
    Vk = V[:, index]
    new = jnp.round(jnp.sum(Vk * Vk * labels[:, None], axis=0))
    info = _diagnostics(Vk, labels, new, top)
    if options.snap:
        Vk = _snap(Vk, labels, new)
    if rows_side:
        Q, R = Vk, Vk.T @ M
    else:
        s = jnp.sqrt(jnp.maximum(top, 0.0))
        keep = s > options.gram_floor * jnp.max(s)
        inverse = jnp.where(keep, 1.0 / jnp.where(keep, s, 1.0), 0.0)
        Q, R = (M @ Vk) * inverse, (s * keep)[:, None] * Vk.T
    return Q, R, new, info


# ============================================================================
# The converter of one channel
# ============================================================================

class DenseConverter(NamedTuple):
    convert: Callable  # (C, stats=False) -> tensors, charges, gauge [, stats: {STATS name: (n_ops,), "leakage": ()}]
    ops: tuple  # (kind, site, step, gate) in application order
    move: str
    chi: int
    polar_steps: int  # iterations of the cans polar factor
    shapes: tuple  # per op, the (rows, columns) of the matrix it factors (a gate's eigh has the smaller size)


def dense_shapes(plan: OrbitalPlan, chi: int, ops=None) -> tuple:
    """The (rows, columns) of the matrix each factorisation of make_dense_converter factors, from the bond sizes it
    produces (a host dry run of convert's shapes)."""
    ops = dense_ops(plan) if ops is None else ops
    dims = [1] * (len(plan.occupation) + 1)
    shapes = []
    for kind, site, step, _ in ops:
        if kind == "gate":
            r, c = 2 * dims[site], 2 * dims[site + 2]
            dims[site + 1] = min(int(chi), r, c)
        elif step > 0:
            r, c = 2 * dims[site], dims[site + 1]
            dims[site + 1] = min(r, c)
        else:
            r, c = 2 * dims[site + 1], dims[site]
            dims[site] = min(r, c)
        shapes.append((r, c))
    return tuple(shapes)


def dense_ops(plan: OrbitalPlan):
    """The factorisations of channel_mps in application order: ("move", site, step, -) shifting the orthogonality
    centre onto the next gate (engine.compile_circuit's sequence) and ("gate", site, 0, index)."""
    ops, centre = [], None
    for gate, site in enumerate(gate_sites(plan)[::-1]):
        if centre is not None:
            while centre != site:
                step = -1 if centre > site else 1
                ops.append(("move", centre, step, -1))
                centre += step
        ops.append(("gate", site, 0, gate))
        centre = site + 1
    return tuple(ops)


def make_dense_converter(plan: OrbitalPlan, chi: int, move: str = "cans", *, cans_lo=1.0e-9, cans_tol=1.0e-14,
                         gram_floor=1.0e-10, charge_gap=1.0e-7, unroll=1, eigh="default",
                         snap=False) -> DenseConverter:
    """convert(C) of one orthonormal L x N channel for the gates of `plan`, bonds of size at most chi (see the module
    docstring for move, the options and the labels; charge_gap is the per-charge shift of the gate's eigh in units of
    trace(G)).

    eigh: "default" (XLA's choice: Jacobi up to 32 x 32 on GPUs) or "jacobi" (forced at every size; GPU only), for the
      gates and the gram moves.
    snap: zero every entry of each new isometry outside its column's sector (and recompute R = Q^T M), so that every
      factorisation hands on exactly sector-structured tensors whatever the eigensolver does (at the price of the
      dropped entries, which are the solver's off-sector noise when the labels are right).

    stats=True also returns a dict of per-factorisation arrays, in the order of ops: _diagnostics' impurity, offlabel and
    mislabeled_weight (of the new states, before snapping), and input_leak, the share of the factored matrix outside
    its carried sector blocks (_leak); and "leakage", the share of the final tensors' squared entries that violates
    charge conservation (charge_leakage, unweighted)."""
    if move not in MOVES:
        raise ValueError(f"move must be one of {MOVES}, got {move!r}")
    if eigh not in EIGH:
        raise ValueError(f"eigh must be one of {EIGH}, got {eigh!r}")
    occupation = np.asarray(plan.occupation, int)
    ops = dense_ops(plan)
    a, c = polar_coefficients(cans_lo, cans_tol)
    options = Options(move, a, c, unroll, float(gram_floor), float(charge_gap), eigh, bool(snap))
    start = [np.array([float(q)]) for q in np.concatenate(([0], np.cumsum(occupation)))]
    occupied = np.flatnonzero(occupation)
    chi = int(chi)

    def convert(C, stats=False):
        angles, rows = device_channel_angles(C[None], [plan])
        thetas = [t[0] for t in angles[::-1]]
        tensors = [jnp.asarray(np.eye(2)[int(o)].reshape(1, 2, 1)) for o in occupation]
        charges = [jnp.asarray(q) for q in start]
        two = jnp.arange(2.0)
        infos, leaks = [], []
        for kind, site, step, gate in ops:
            if kind == "gate":
                pair = gate_pair(tensors[site], tensors[site + 1], thetas[gate])
                Dl, Dr = pair.shape[0], pair.shape[3]
                M = pair.reshape(2 * Dl, 2 * Dr)
                c_row = (charges[site][:, None] + two).ravel()  # row (a, p): the bond's charge after the site
                c_col = (charges[site + 2][None, :] - two[:, None]).ravel()  # column (q, b)
                Q, R, new, info = _factor_gate(M, c_row, c_col, min(chi, 2 * Dl, 2 * Dr), options)
                tensors[site] = Q.reshape(Dl, 2, -1)
                tensors[site + 1] = R.reshape(-1, 2, Dr)
                charges[site + 1] = new
            elif step > 0:
                Dl, Dr = tensors[site].shape[0], tensors[site].shape[2]
                M = tensors[site].reshape(2 * Dl, Dr)
                c_row, c_col = (charges[site][:, None] + two).ravel(), charges[site + 1]
                Q, R, new, info = _factor_move(M, c_row, c_col, options)
                tensors[site] = Q.reshape(Dl, 2, -1)
                tensors[site + 1] = jnp.einsum("kd,dpe->kpe", R, tensors[site + 1])
                charges[site + 1] = new
            else:
                Dl, Dr = tensors[site].shape[0], tensors[site].shape[2]
                M = tensors[site].reshape(Dl, 2 * Dr).T  # row (p, b)
                c_row, c_col = (charges[site + 1][None, :] - two[:, None]).ravel(), charges[site]
                Q, R, new, info = _factor_move(M, c_row, c_col, options)
                tensors[site] = Q.T.reshape(-1, 2, Dr)
                tensors[site - 1] = jnp.einsum("apd,kd->apk", tensors[site - 1], R)
                charges[site] = new
            infos.append(info)
            leaks.append(_leak(M, c_row, c_col))
        gauge = jnp.linalg.det(jnp.stack(rows, axis=-2)[0][occupied])
        if not stats:
            return tensors, charges, gauge
        out = {name: jnp.stack([info[name] for info in infos]) for name in STATS if name != "input_leak"}
        out["input_leak"] = jnp.stack(leaks)
        out["leakage"] = charge_leakage(tensors, charges)
        return tensors, charges, gauge, out

    return DenseConverter(convert, ops, move, chi, len(a), dense_shapes(plan, chi, ops))


# ============================================================================
# Checks
# ============================================================================

def mps_norm2(tensors):
    """<MPS|MPS> of a real d=2 MPS (1 minus the discarded weight of a conversion of a normalised channel)."""
    env = jnp.ones((1, 1))
    for A in tensors:
        env = jnp.einsum("ab,apc,bpd->cd", env, A, A)
    return env[0, 0]


def charge_leakage(tensors, charges):
    """The share of the tensors' squared entries that break charge conservation under the carried labels: A[a, p, b]
    must vanish unless charges[i][a] + p == charges[i + 1][b] (zero-weight states have zero rows, whatever their label)."""
    wrong = total = 0.0
    for i, A in enumerate(tensors):
        expected = charges[i][:, None, None] + jnp.arange(2.0)[None, :, None]
        violating = jnp.round(expected) != jnp.round(charges[i + 1])[None, None, :]
        wrong = wrong + jnp.sum(jnp.where(violating, A * A, 0.0))
        total = total + jnp.sum(A * A)
    return wrong / total
