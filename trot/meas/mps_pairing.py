"""Pairing correlations of MPS trials: the mixed estimator <T|Delta^dag_a(j) Delta_a(l)|walker> / <T|walker>.

Delta_a(l) = sum_delta f_a(delta) [c_{l up} c_{l+delta dn} - c_{l dn} c_{l+delta up}], delta = (+-1, 0), (0, +-1),
with f_s = 1 (extended s) and f_d = +1 for x bonds, -1 for y bonds (d_{x^2-y^2}). The lattice is square_hopping_matrix's:
site = x * Ly + y, x bonds (i, i + Ly), y bonds (i, i + 1), with open, periodic or antiperiodic sides.

The bond singlet S_uv = c_{u up} c_{v dn} - c_{u dn} c_{v up} is symmetric in u and v, and Delta_a(l) = sum_b G_a[l, b] S_b with
G_a[l, b] = f_a(b) (times the wrap sign, with the multiplicity of the bond at l). So with
B(b', b) = <T|S_b'^dag S_b|walker> / <T|walker>   the estimator is   P_a = G_a B G_a^T,  P_a[j, l] = <Delta_a^dag(j) Delta_a(l)>
(the usual P_a(l) of a reference site 0 is the column l = 0). The result of the observable is (2, n_rows, L), a = (s, d).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from trot.gmps.utils import contract_real

# ---------------------------------------------------------------------------------------------
# Lattice: bonds and form factors
# ---------------------------------------------------------------------------------------------


class PairingSpec(NamedTuple):
    """The lattice of the pairing operators and how the blocked kernel evaluates them (hashable: it goes into the
    measurement context's static key).

    rows: sites j of Delta^dag(j) to measure, or None for all (the observable is (2, len(rows), L)).
    method: "windows" (engine.pair_window_values: the contractions outside every bra bond's window are shared by all
      bra terms) or "terms" (engine.pair_term_values: every bra term over the whole lattice, a lax.map over the terms).
    term_batch: "terms" only, bra terms evaluated together (lax.map batch_size; 1: one after the other).
    walker_batch: "windows" only, walkers evaluated together (lax.map batch_size), None: the whole batch at once.
      The windowed kernel keeps the shared right objects of every bond of a walker alive, so this bounds the memory.
    """

    Lx: int
    Ly: int
    boundary_x: str = "open"
    boundary_y: str = "open"
    rows: tuple | None = None
    method: str = "windows"
    term_batch: int = 1
    walker_batch: int | None = None

    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly


_WRAP = {"open": 0.0, "periodic": 1.0, "antiperiodic": -1.0}


def bond_structure(spec: PairingSpec):
    """(bonds, G): bonds is a list of (u, v), u < v, one per distinct bond operator S_uv; G (2, L, n_bonds) has
    G[a, l, b] = sum of f_a(delta) (times the antiperiodic wrap sign) over the neighbours delta of site l that form
    bond b. A periodic side of length 2 reaches the same neighbour in both directions: the entry counts twice."""
    if spec.boundary_x not in _WRAP or spec.boundary_y not in _WRAP:
        raise ValueError("boundaries must be 'open', 'periodic' or 'antiperiodic'")
    Lx, Ly, n = spec.Lx, spec.Ly, spec.n_sites
    entries = []  # (l, m, direction: 0 = x, 1 = y, wrap sign)
    for x in range(Lx):
        for y in range(Ly):
            l = x * Ly + y
            for direction, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                axis = direction // 2  # 0: x, 1: y
                nx, ny = x + dx, y + dy
                sign, wrapped = 1.0, False
                if not 0 <= nx < Lx:
                    nx, wrapped = nx % Lx, True
                    if not _WRAP[spec.boundary_x] or Lx == 1:
                        continue
                    sign = _WRAP[spec.boundary_x]
                if not 0 <= ny < Ly:
                    ny, wrapped = ny % Ly, True
                    if not _WRAP[spec.boundary_y] or Ly == 1:
                        continue
                    sign = _WRAP[spec.boundary_y]
                m = nx * Ly + ny
                if m != l:
                    entries.append((l, m, axis, sign))
    bonds = sorted({(min(l, m), max(l, m)) for l, m, _, _ in entries})
    index = {b: k for k, b in enumerate(bonds)}
    G = np.zeros((2, n, len(bonds)))
    for l, m, axis, sign in entries:
        k = index[(min(l, m), max(l, m))]
        G[0, l, k] += sign  # extended s: f = 1
        G[1, l, k] += sign * (1.0 if axis == 0 else -1.0)  # d_{x^2-y^2}: +1 on x bonds, -1 on y bonds
    return bonds, G


# ---------------------------------------------------------------------------------------------
# Local operators in the basis |0>, |up>, |dn>, |up dn> (p = n_up + 2 n_dn); see trot.meas.mps.hubbard_mpo_from_h1
# ---------------------------------------------------------------------------------------------

_CREATE_A = np.zeros((4, 4))
_CREATE_A[1, 0] = _CREATE_A[3, 2] = 1.0
_CREATE_B = np.zeros((4, 4))
_CREATE_B[2, 0] = 1.0
_CREATE_B[3, 1] = -1.0  # |up dn> = c+_up c+_dn |0>: c+_dn on |up> passes the up mode
FERMION = (_CREATE_A.T, _CREATE_B.T)  # c_up, c_dn on one site, the sign of the same-site up mode included
PARITY = np.diag([1.0, -1.0, -1.0, 1.0])  # (-1)^(n_up + n_dn): the Jordan-Wigner string of one site


def _site_operator(site, u, v, s_u, s_v):
    """The local factor at `site` of c_{u s_u} c_{v s_v} (u < v): c_v acts first with the string over the sites before
    it, then c_u with the string over the sites before u, so: before u nothing (the strings cancel), at u c_u P,
    strictly between P, at v c_v, after v nothing."""
    if site == u:
        return FERMION[s_u] @ PARITY
    if site == v:
        return FERMION[s_v]
    if u < site < v:
        return PARITY
    return None


def pair_annihilate(tensors, u, s_u, v, s_v, xp=jnp):
    """c_{u s_u} c_{v s_v} applied to a d=4 MPS (u < v; s = 0 up, 1 down): the same bonds, local factors only."""
    if not u < v:
        raise ValueError("pair_annihilate needs u < v")
    out = []
    for site, A in enumerate(tensors):
        op = _site_operator(site, u, v, s_u, s_v)
        out.append(A if op is None else xp.einsum("pq,lqr->lpr", xp.asarray(op), A))
    return out


# ---------------------------------------------------------------------------------------------
# Dense kernel (the d=4 walker MPS against the dense trial): for validation, and the "dense" energy kernel
# ---------------------------------------------------------------------------------------------

SINGLET = ((0, 1.0), (1, -1.0))  # S_uv = c_{u up} c_{v dn} - c_{u dn} c_{v up}: (spin at u, coefficient)


def singlet_terms(tensors, bonds, xp=jnp):
    """S_b |tensors> for every bond, as the two terms (coefficient included) of each: a list over bonds of 2 MPS."""
    out = []
    for u, v in bonds:
        out.append([[xp.asarray(coefficient) * A if site == 0 else A
                     for site, A in enumerate(pair_annihilate(tensors, u, s, v, 1 - s, xp))]
                    for s, coefficient in SINGLET])
    return out


def pair_matrix_dense(walker, trial, bonds):
    """B(b', b) = <T|S_b'^dag S_b|walker> / <T|walker> for d=4 tensors, as (n_bonds, n_bonds)."""
    bra = singlet_terms(trial, bonds)  # S_b'|T>: <T|S_b'^dag = (S_b'|T>)^T for real tensors
    ket = singlet_terms(walker, bonds)
    overlap = contract_real(trial, walker)
    B = jnp.stack([jnp.stack([sum(contract_real(t, w) for t in bra[i] for w in ket[j])
                              for j in range(len(bonds))]) for i in range(len(bonds))])
    return B / overlap


def pairing_dense(walker, trial, spec: PairingSpec):
    """(2, n_rows, L): the mixed pairing correlations P_a[j, l] = G_a B G_a^T of d=4 tensors (walker against trial)."""
    bonds, G = bond_structure(spec)
    B = pair_matrix_dense(walker, trial, bonds)
    rows = list(range(spec.n_sites)) if spec.rows is None else list(spec.rows)
    G = jnp.asarray(G)
    return jnp.einsum("ajb,bc,alc->ajl", G[:, rows], B, G)


# ---------------------------------------------------------------------------------------------
# Blocked kernel: the pair space shared by the bra terms, and the walker-side insertions
# ---------------------------------------------------------------------------------------------

import jax
from jax import lax

from trot.gmps import engine
from trot.gmps.utils import combine_channels, label_array
from trot.trial.mps import _hashable_charges

DECREMENT = (1, 2)  # the local index of |up> / |dn> is n_up + 2 n_dn: removing a spin s lowers p by this


def has_spin(p, s):
    return bool((p >> s) & 1)


def opening_moves(s):
    """c_{u s} on the walker at the lower site: moves (p_walker, p_fixed, coefficient, jw) with the singlet's sign
    (+1 for the up-down term, -1 for the down-up term) in the coefficient."""
    sign = SINGLET[s][1]
    return tuple((p, p - DECREMENT[s], sign * FERMION[s][p - DECREMENT[s], p], True) for p in range(4) if has_spin(p, s))


def closing_moves(s):
    """c_{v, 1-s} on the walker at the partner site."""
    t = 1 - s
    return tuple((p, p - DECREMENT[t], FERMION[t][p - DECREMENT[t], p], True) for p in range(4) if has_spin(p, t))


class PairingStatic(NamedTuple):
    """The host-side structure of a PairingSpec: bonds, form factors, the bra terms (bond index, spin) and the
    (n_bonds, n_terms) matrix that sums a bond's two terms."""

    bonds: tuple
    G: np.ndarray  # (2, L, n_bonds)
    rows: tuple
    terms: tuple
    assemble: np.ndarray


PAIRING_METHODS = ("windows", "terms")


def pairing_static(spec: PairingSpec) -> PairingStatic:
    if spec.method not in PAIRING_METHODS:
        raise ValueError(f"PairingSpec.method must be one of {PAIRING_METHODS}, got {spec.method!r}")
    bonds, G = bond_structure(spec)
    rows = tuple(range(spec.n_sites)) if spec.rows is None else tuple(int(j) for j in spec.rows)
    active = [b for b in range(len(bonds)) if np.any(G[:, list(rows), b] != 0.0)]
    terms = tuple((b, s) for b in active for s in (0, 1))
    assemble = np.zeros((len(bonds), len(terms)))
    for t, (b, _) in enumerate(terms):
        assemble[b, t] = 1.0
    return PairingStatic(tuple(bonds), G, rows, terms, assemble)


def removed_electrons(width, k, u, v, s):
    """The electrons the bra term c_{u s} c_{v, 1-s}|T> has removed left of bond k, in the label's units: nothing up
    to the bond right of site u, one of spin s up to the bond right of v, both after (width 2: (n_up, n_dn); width 1: N)."""
    if k <= u:
        return np.zeros(width, int)
    unit = np.eye(2, dtype=int)[s] if width == 2 else np.array([1])
    return unit if k <= v else (np.ones(2, int) if width == 2 else np.array([2]))


class PairingData(NamedTuple):
    """Device data of the pairing observable (a MpsMeasCtx child): the dense trial tensors and, per bra term, the slot
    maps and local remaps that turn them into the term on the shared pair space."""

    trial: tuple
    slot_to_trial: tuple  # per bond: (n_terms, n_slots + 1) int32, the trial's bond dimension where a slot is unused
    phys_src: np.ndarray  # (n_terms, L, 4) int32
    phys_sign: np.ndarray  # (n_terms, L, 4)
    # per bond (n_slots + 1,): the slot maps of the trial with nothing / both electrons removed, which every bra term
    # equals left / right of its window (the windowed kernel); valid where some bra term is so (bonds <= max u,
    # bonds > min v), elsewhere unused
    slot_plain: tuple
    slot_both: tuple


def phys_maps(site, u, v, s):
    """(src, sign) of the bra term's local operator at `site`: src[p] the trial's local state that the operator maps to
    p, sign[p] the matrix element (0 if p has no source)."""
    op = _site_operator(site, u, v, s, 1 - s)
    if op is None:
        return np.arange(4), np.ones(4)
    src, sign = np.zeros(4, int), np.zeros(4)
    for p in range(4):
        nonzero = np.flatnonzero(op[p])
        if len(nonzero):
            src[p], sign[p] = nonzero[0], op[p, nonzero[0]]
    return src, sign


def build_pairing_data(trial_tensors, trial_charges, spec: PairingSpec):
    """(PairingData, plabels): the pair space's labels per bond (hashable) and the per-term maps.

    A bra term's true label of trial index i at bond k is q_k[i] - removed_electrons(k): the slots of the pair space
    hold every true label as often as the term that has it most (indices with a negative component are zero blocks and
    dropped). All terms then share the same layouts."""
    static = pairing_static(spec)
    L = spec.n_sites
    labels = [label_array(q) for q in trial_charges]
    width = labels[0].shape[1]

    def grouped(k, removed):
        """True label -> trial indices at bond k of the trial with `removed` electrons removed left of it."""
        g = {}
        for i, lab in enumerate(map(tuple, (labels[k] - removed).tolist())):
            if min(lab) >= 0:
                g.setdefault(lab, []).append(i)
        return g

    groups = []  # per term, per bond: true label -> trial indices
    for b, s in static.terms:
        u, v = static.bonds[b]
        groups.append([grouped(k, removed_electrons(width, k, u, v, s)) for k in range(L + 1)])
    both = np.ones(2, int) if width == 2 else np.array([2])
    plain_groups = [grouped(k, np.zeros(width, int)) for k in range(L + 1)]
    both_groups = [grouped(k, both) for k in range(L + 1)]
    plabels, slot_maps, slot_plain, slot_both = [], [[] for _ in range(L + 1)], [], []
    for k in range(L + 1):
        counts = {}
        for per_bond in groups:
            for lab, idx in per_bond[k].items():
                counts[lab] = max(counts.get(lab, 0), len(idx))
        order = sorted(counts)
        plabels.append(np.array([lab for lab in order for _ in range(counts[lab])], int).reshape(-1, width))
        D_T = labels[k].shape[0]

        def slots_of(group):
            # a label's indices fill its first slots; the plain / both maps are cut to the slots where no bra term
            # has them (bonds where they are unused), the terms' never are
            slots = np.full(len(plabels[k]) + 1, D_T, np.int32)
            at = 0
            for lab in order:
                idx = group.get(lab, [])[:counts[lab]]
                slots[at:at + len(idx)] = idx
                at += counts[lab]
            return slots

        for per_bond in groups:
            slot_maps[k].append(slots_of(per_bond[k]))
        slot_plain.append(slots_of(plain_groups[k]))
        slot_both.append(slots_of(both_groups[k]))
    src = np.zeros((len(static.terms), L, 4), np.int32)
    sign = np.zeros((len(static.terms), L, 4))
    for t, (b, s) in enumerate(static.terms):
        u, v = static.bonds[b]
        for site in range(L):
            src[t, site], sign[t, site] = phys_maps(site, u, v, s)
        sign[t, u] *= SINGLET[s][1]
    data = PairingData(tuple(jnp.asarray(A) for A in trial_tensors),
                       tuple(jnp.asarray(np.stack(m)) for m in slot_maps), jnp.asarray(src), jnp.asarray(sign),
                       tuple(jnp.asarray(m) for m in slot_plain), tuple(jnp.asarray(m) for m in slot_both))
    return data, _hashable_charges(plabels)


def pair_kernel_for(plan, plabels_key, spec: PairingSpec) -> engine.PairKernel:
    """The static contraction structure of the "terms" kernel for a walker plan and a pair space (cached on the plan)."""
    lattice = spec._replace(method="terms", term_batch=1, walker_batch=None)  # the batch sizes do not change it
    key = ("pair_kernel", plabels_key, lattice)
    kernel = plan.caches.get(key)
    if kernel is None:
        qa, qb = engine.converter_for(plan).charges
        plabels = [np.asarray(q, int).reshape(len(q), -1) for q in plabels_key]
        kernel = engine.make_pair_kernel(qa, qb, plabels, list(pairing_static(lattice).bonds),
                                         (opening_moves(0), opening_moves(1)), (closing_moves(0), closing_moves(1)))
        plan.caches[key] = kernel
    return kernel


def pair_windows_for(plan, plabels_key, spec: PairingSpec) -> engine.PairWindows:
    """The static structure of the "windows" kernel for a walker plan and a pair space (cached on the plan)."""
    lattice = spec._replace(method="windows", term_batch=1, walker_batch=None)
    key = ("pair_windows", plabels_key, lattice)
    kernel = plan.caches.get(key)
    if kernel is None:
        static = pairing_static(lattice)
        if static.terms[1::2] != tuple((b, 1) for b, _ in static.terms[::2]):
            raise AssertionError("the windowed kernel expects the bra terms as (bond, 0), (bond, 1) pairs")
        qa, qb = engine.converter_for(plan).charges
        plabels = [np.asarray(q, int).reshape(len(q), -1) for q in plabels_key]
        kernel = engine.make_pair_windows(qa, qb, plabels, list(static.bonds),
                                          [static.bonds[b] for b, _ in static.terms[::2]],
                                          (opening_moves(0), opening_moves(1)), (closing_moves(0), closing_moves(1)))
        plan.caches[key] = kernel
    return kernel


def _converted(kernels, data, qa, qb):
    """The walkers' channel MPS and their overlaps with the trial (no gauges: the ratios do not need them)."""
    alpha, beta, _ = jax.vmap(kernels.converter.convert)(qa, qb)
    overlap = jax.vmap(lambda a, b: engine.left_contract(engine.walker_blocks(a, b, kernels.overlap_plan),
                                                         data.trial, kernels.overlap_plan))(alpha, beta)
    return alpha, beta, overlap


def _pairing_from(static, B):
    """P_a = G_a B G_a^T for every walker: (n_walkers, 2, n_rows, L)."""
    G = jnp.asarray(static.G)
    return jnp.einsum("ajb,wbc,alc->wajl", G[:, list(static.rows)], B, G)


def pairing_batch_blocked(kernel, static, kernels, data, pdata, spec, qa, qb):
    """(n_walkers, 2, n_rows, L) pairing correlations of orthonormalised walkers (qa, qb), "terms" kernel: a lax.map
    over the bra terms, spec.term_batch of them at a time."""
    alpha, beta, overlap = _converted(kernels, data, qa, qb)
    wblocks = jax.vmap(lambda a, b: engine.pair_walker_blocks(a, b, kernel))(alpha, beta)

    def one_term(xs):
        slot, src, sign = xs
        fixed = engine.pair_fixed_blocks(kernel, pdata.trial, slot, src, sign)
        return jax.vmap(lambda wb: engine.pair_term_values(kernel, wb, fixed))(wblocks)  # (n_walkers, n_bonds)

    batch = min(int(spec.term_batch), len(static.terms)) if spec.term_batch and spec.term_batch > 1 else None
    values = lax.map(one_term, (pdata.slot_to_trial, pdata.phys_src, pdata.phys_sign),
                     batch_size=batch)  # (n_terms, n_walkers, n_bonds)
    B = jnp.einsum("bt,twc->wbc", jnp.asarray(static.assemble), values) / overlap[:, None, None]
    return _pairing_from(static, B)


def pairing_batch_windows(kernel, static, kernels, data, pdata, spec, qa, qb):
    """(n_walkers, 2, n_rows, L) pairing correlations of orthonormalised walkers (qa, qb), "windows" kernel
    (engine.pair_window_values), spec.walker_batch walkers at a time."""
    alpha, beta, overlap = _converted(kernels, data, qa, qb)
    L = len(pdata.trial)
    identity = (jnp.tile(jnp.arange(4, dtype=jnp.int32), (L, 1)), jnp.ones((L, 4)))
    plain_maps, both_maps = (pdata.slot_plain,) + identity, (pdata.slot_both,) + identity
    window_maps = tuple((tuple(m[2 * i:2 * i + 2] for m in pdata.slot_to_trial), pdata.phys_src[2 * i:2 * i + 2],
                         pdata.phys_sign[2 * i:2 * i + 2]) for i in range(len(kernel.windows)))

    def one(walker):
        wblocks = engine.pair_walker_blocks(*walker, kernel)
        return engine.pair_window_values(kernel, wblocks, pdata.trial, plain_maps, both_maps, window_maps)

    n = qa.shape[0]
    if spec.walker_batch and spec.walker_batch < n:
        values = lax.map(one, (alpha, beta), batch_size=int(spec.walker_batch))  # (n_walkers, n_terms, n_bonds)
    else:
        values = jax.vmap(one)((alpha, beta))
    B = jnp.einsum("bt,wtc->wbc", jnp.asarray(static.assemble), values) / overlap[:, None, None]
    return _pairing_from(static, B)


class PairingFns(NamedTuple):
    batch: object  # (ca, cb, data) -> (n_walkers, 2, n_rows, L)
    one: object  # (ca, cb, data) -> (2, n_rows, L)


def pairing_fns(meas_ctx) -> PairingFns:
    """The pairing kernels of a measurement context. The static structure (layouts, schedule) is cached on the walker
    plan; the closures capture the context's device data, so they are built per call and used inside the same trace."""
    spec, plabels_key = meas_ctx.pairing_key
    plan, kernels, pdata = meas_ctx.plan, meas_ctx.kernels, meas_ctx.pairing
    static = pairing_static(spec)
    qa_labels, qb_labels = kernels.converter.charges

    if kernels.energy_kind == "dense":
        def batch(ca, cb, data):
            qa, _ = kernels.batch_qr(ca)
            qb, _ = kernels.batch_qr(cb)

            def value(a, b):
                alpha, beta, _ = kernels.converter.convert(a, b)
                tensors, _ = combine_channels(alpha, qa_labels, beta, qb_labels)
                return pairing_dense(tensors, list(pdata.trial), spec)
            return jax.vmap(value)(qa, qb)
    else:
        if spec.method == "windows":
            kernel, blocked = pair_windows_for(plan, plabels_key, spec), pairing_batch_windows
        else:
            kernel, blocked = pair_kernel_for(plan, plabels_key, spec), pairing_batch_blocked

        def batch(ca, cb, data):
            qa, _ = kernels.batch_qr(ca)
            qb, _ = kernels.batch_qr(cb)
            return blocked(kernel, static, kernels, data, pdata, spec, qa, qb)

    return PairingFns(batch, lambda ca, cb, data: batch(ca[None], cb[None], data)[0])
