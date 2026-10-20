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

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from trot.gmps import engine
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


# ============================================================================
# One site of an environment
# ============================================================================

def _parity_signs(q):
    """(D, 2) factors (-1)^(n_alpha q) of the beta bond labels q (floats holding integers), n_alpha = 0, 1."""
    return jnp.stack((jnp.ones_like(q), 1.0 - 2.0 * jnp.mod(jnp.round(q), 2.0)), axis=1)


def _dressed(B, signs):
    """The beta tensor B (Db, 2, Db') with the fermionic sign for each alpha occupation: (Db, n_beta, n_alpha, Db')."""
    return B[:, :, None, :] * signs[:, None, :, None]


def _forward(E, T, A, Bs, site=None):
    """One site of a left environment: E (Da, Db, Dt) through the trial's blocks T (the full matrices, or the sector
    blocks of SiteLayout `site`) and the walker's tensors to (Da', Db', p, Dt'), p = n_alpha + 2 n_beta."""
    Da, Db, Dt = E.shape
    X = E.reshape(Da * Db, Dt)
    Z = X @ T[0] if site is None else _trial_apply(X, T[0], site.forward, site.right_widths)
    Z = Z.reshape(Da, Db, 2, 2, -1)  # (a, b, n_beta, n_alpha, t')
    H = jnp.einsum("abjit,aic->cbjit", Z, A)
    Y = jnp.einsum("cbjit,bjid->cdjit", H, Bs)
    return Y.reshape(Y.shape[0], Y.shape[1], 4, Y.shape[4])


def _backward(R, T, A, Bs, site=None):
    """One site of a right environment: R (Da', Db', Dt') to (Da, Db, p, Dt), with the trial's T[1] (the transposed full
    matrix (Dt', 4 Dt), or the sector blocks of `site`)."""
    Dc, Dd, Dt = R.shape
    X = R.reshape(Dc * Dd, Dt)
    U = X @ T[1] if site is None else _trial_apply(X, T[1], site.backward, site.left_widths)
    U = U.reshape(Dc, Dd, 2, 2, -1)  # (c, d, n_beta, n_alpha, t)
    H = jnp.einsum("cdjit,aic->adjit", U, A)
    X = jnp.einsum("adjit,bjid->abjit", H, Bs)
    return X.reshape(X.shape[0], X.shape[1], 4, X.shape[4])


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


def right_environments(alpha, dressed, trial, layout=None):
    """The right environments, right[k] for the sites k.. (right[L] = 1)."""
    right = [jnp.ones((1, 1, 1))]
    sites = _sites(layout, len(alpha))
    for A, Bs, T, site in zip(alpha[::-1], dressed[::-1], trial[::-1], sites[::-1]):
        right.append(_backward(right[-1], T, A, Bs, site).sum(axis=2))
    return right[::-1]


def field_sweep(ca, cb, randoms, alpha, dressed, prefactor, right, overlap_in, *, hs, weight_floor, trial, layout=None):
    """The diagonal HS sweep of one walker (engine.field_sweep) on dense environments: at each site the four physical
    marginals give both field proposals, one field is sampled, and the left environment absorbs the chosen diagonal
    factor."""
    diagonal = jnp.stack((jnp.ones(2), hs[:, 0], hs[:, 1], hs[:, 0] * hs[:, 1]), axis=1)
    left = jnp.ones((1, 1, 1))
    overlap, log_weight = overlap_in, jnp.zeros(())
    nodes = jnp.zeros((), jnp.int64)
    fields = []
    for site, (A, Bs, T, sl) in enumerate(zip(alpha, dressed, trial, _sites(layout, len(alpha)))):
        Y = _forward(left, T, A, Bs, sl)  # (a', b', p, t')
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

def dense_converter(plan):
    """convert(qa, qb) of a dense walker plan for one orthonormal walker: the alpha tensors, the beta tensors dressed
    with the fermionic sign of their labels, and the gauge product; cached on the plan."""
    convert = plan.caches.get("dense_converter")
    if convert is None:
        spec = plan.dense
        converters = tuple(make_dense_converter(p, spec.chi, spec.move, cans_lo=spec.cans_lo,
                                                gram_floor=spec.gram_floor, charge_gap=spec.gap(), eigh=spec.eigh,
                                                snap=spec.snap) for p in plan.orbital_plans)

        def convert(qa, qb):
            alpha, _, ga = converters[0].convert(qa)
            beta, charges, gb = converters[1].convert(qb)
            dressed = [_dressed(B, _parity_signs(q)) for B, q in zip(beta, charges[:-1])]  # the labels left of each site
            return alpha, dressed, ga * gb

        plan.caches["dense_converter"] = convert
    return convert


def make_dense_kernels(plan, overlap_plan: engine.DensePlan | None = None, energy=None) -> engine.Kernels:
    """The engine.Kernels of a dense walker plan (plan.dense a DenseSpec) for a trial whose blocks follow overlap_plan
    (dense_overlap_plan; None: the full matrices): batch_qr, overlaps, step_batch, probe and the one-walker overlap.
    energy "mpo": also the local energies <T|H|W>/<T|W> through the MPO (DeviceData.htrial holds the site tensors of
    mpo_site_tensors); None: the energies raise (use a measurement plan)."""
    overlap_plan = engine.DensePlan(int(plan.norb), dict(trial="dense")) if overlap_plan is None else overlap_plan
    layout = overlap_plan.layout
    convert = dense_converter(plan)
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

        def one(qa1, qb1):
            alpha, dressed, gauge = convert(qa1, qb1)
            return alpha, dressed, right_environments(alpha, dressed, data.trial, layout), gauge
        alpha, dressed, right, gauge = jax.vmap(one)(qa, qb)
        prefactor = da * db * gauge
        now = prefactor * right[0][:, 0, 0, 0]

        def with_fields():
            sweep_one = partial(field_sweep, hs=data.hs, weight_floor=weight_floor, trial=data.trial, layout=layout)
            return jax.vmap(sweep_one)(ca, cb, randoms, alpha, dressed, prefactor, right, now)

        def without():
            return ca, cb, now, jnp.ones_like(now), jnp.zeros(now.shape, jnp.int64)

        if isinstance(sweep, bool):
            ca2, cb2, after, factor, nodes = with_fields() if sweep else without()
        else:
            ca2, cb2, after, factor, nodes = lax.cond(sweep, with_fields, without)
        return ca2, cb2, now, after, factor, nodes

    def local_energy(alpha, dressed, data):
        return (mpo_numerator(alpha, dressed, data.trial, data.htrial, layout)
                / left_contract(alpha, dressed, data.trial, layout))

    def probe(ca, cb, data):
        qa, da = qr_with_det(ca)
        qb, db = qr_with_det(cb)
        alpha, dressed, gauge = convert(qa, qb)
        overlap = da * db * gauge * left_contract(alpha, dressed, data.trial, layout)
        return dict(qa=qa, qb=qb, alpha=alpha, beta=dressed, gauges=gauge, overlap=overlap,
                    energy=None if energy is None else local_energy(alpha, dressed, data))

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
                alpha, dressed, _ = convert(qa1, qb1)
                return local_energy(alpha, dressed, data)
            return jax.vmap(one)(qa, qb)

        def energy_one(ca, cb, data):
            qa, _ = qr_with_det(ca)
            qb, _ = qr_with_det(cb)
            alpha, dressed, _ = convert(qa, qb)
            return local_energy(alpha, dressed, data)

    return engine.Kernels(None, overlap_plan, None, energy, batch_qr, overlaps, energies, step_batch, probe,
                          overlap_one, energy_one, jax.jit(probe))
