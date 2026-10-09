"""Hubbard measurements for MPS trials (trot.trial.mps).

The local energy is <trial|H|walker> / <trial|walker> with H|trial> built once on the host from the automaton MPO
of any real symmetric h1 plus on-site U (hubbard_mpo_from_h1). Two kernels:

- "blocked": H|trial> keeps exact bond labels ((N_up, N_dn) or N, as the trial's; trial_times_h), is compressed
  sector by sector (compress_mps_qn) and contracted charge-blocked against the walker channels, without ever
  forming the d=4 walker MPS.
- "dense": the d=4 walker MPS is formed (combine_channels) and contracted with the densely compressed H|trial>
  (compress_mps); for validation.

Both give the same number up to rounding. build_meas_ctx (make_mps_meas_ops_hubbard) gathers the padded trial and
H|trial> blocks of the plan's engine (trot.gmps.engine) once; the propagation step reads the trial blocks from the
same context. A precomputed H|trial> (trot.gmps.trials: block form for 6x6 and larger lattices) can be passed in.

The observable "szsz" is the mixed estimator <trial|S^z_i S^z_j|walker> / <trial|walker> as an (L, L) matrix, for a
trial with (N_up, N_dn) bond labels or with particle-number labels (a spin-rotated trial used as it is). It reuses
the trial blocks of the overlap layout, so it needs no extra context, and follows the energy kernel ("blocked" or
"dense"). The MPS block measures it for the whole walker batch at once (BATCHED_OBSERVABLES).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from trot.core.ops import MeasOps, k_energy, o_szsz
from trot.gmps import engine
from trot.gmps.utils import combine_channels
from trot.trial.mps import (
    MpsWalkerPlan,
    _hashable_charges,
    check_trial,
    check_walker,
    compress_mps_qn,
    label_array,
    mps_overlap_fn,
)
from trot.walkers import _qr as qr_with_det

ENERGY_KERNELS = ("blocked", "dense")


# ---------------------------------------------------------------------------------------------
# MPOs and H|trial>
# ---------------------------------------------------------------------------------------------


def hubbard_mpo(L, hopping, interaction):
    """Open-chain Hubbard MPO in the spatial local basis, virtual dimension six."""
    eye = np.eye(4)
    create_a = np.zeros((4, 4))
    create_a[1, 0] = create_a[3, 2] = 1.0
    create_b = np.zeros((4, 4))
    create_b[2, 0] = 1.0
    create_b[3, 1] = -1.0
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


def hubbard_mpo_from_h1(h1, interaction):
    """Hubbard MPO for any real symmetric one-body matrix, in the spatial local basis.

    A finite-state automaton. Channel 0 has placed nothing and the last channel holds a
    finished term. A hopping opened on site i travels in four channels, one per opening
    operator, and carries the Jordan-Wigner string Pa Pb to its partner site. Bonds are
    padded to a common width. A nearest-neighbour chain gives hubbard_mpo exactly.
    """
    h1 = np.asarray(h1)
    if np.iscomplexobj(h1) or not np.array_equal(h1, h1.T):
        raise ValueError("h1 must be real symmetric")
    n = len(h1)
    eye = np.eye(4)
    create_a = np.zeros((4, 4))
    create_a[1, 0] = create_a[3, 2] = 1.0
    create_b = np.zeros((4, 4))
    create_b[2, 0] = 1.0
    create_b[3, 1] = -1.0
    annihilate_a, annihilate_b = create_a.T, create_b.T
    parity_a = np.diag([1.0, -1.0, 1.0, -1.0])
    parity_b = np.diag([1.0, 1.0, -1.0, -1.0])
    double = np.diag([0.0, 0.0, 0.0, 1.0])
    number = np.diag([0.0, 1.0, 1.0, 2.0])
    opening = (
        create_a @ parity_b,
        annihilate_a @ parity_b,
        parity_a @ create_b,
        parity_a @ annihilate_b,
    )
    closing = (annihilate_a, create_a, annihilate_b, create_b)

    upper = np.triu(h1, 1) != 0
    reach = [np.flatnonzero(row).max(initial=i) for i, row in enumerate(upper)]
    slots = [
        {i: 1 + 4 * rank for rank, i in enumerate(i for i in range(b) if reach[i] >= b)}
        for b in range(n + 1)
    ]
    D = 2 + 4 * max(map(len, slots))

    W = np.zeros((n, D, 4, 4, D))
    for k in range(n):
        left, right = slots[k], slots[k + 1]
        W[k, 0, :, :, 0] = W[k, -1, :, :, -1] = eye
        W[k, 0, :, :, -1] = interaction * double + h1[k, k] * number
        for op in range(4):
            if k in right:
                W[k, 0, :, :, right[k] + op] = opening[op]
            for i, slot in left.items():
                if h1[i, k]:
                    W[k, slot + op, :, :, -1] = h1[i, k] * closing[op]
                if i in right:
                    W[k, slot + op, :, :, right[i] + op] = parity_a @ parity_b
    return W


def apply_mpo(W, tensors):
    """W|tensors>, uncompressed. The finished-term channel is the last one (5 for hubbard_mpo)."""
    out = []
    for i, (operator, A) in enumerate(zip(W, tensors)):
        if i == 0:
            operator = operator[:1]
        if i == len(tensors) - 1:
            operator = operator[..., -1:]
        T = np.einsum("apqb,cqd->acpbd", operator, np.asarray(A))
        dl, cl, d, dr, cr = T.shape
        out.append(T.reshape(dl * cl, d, dr * cr))
    return out


def compress_mps(tensors, relative_tolerance=1.0e-13):
    """Compress a fixed MPS by a host QR/SVD sweep."""
    tensors = [np.array(A, copy=True) for A in tensors]
    for i in range(len(tensors) - 1):
        Dl, d, Dr = tensors[i].shape
        q, r = np.linalg.qr(tensors[i].reshape(Dl * d, Dr))
        tensors[i] = q.reshape(Dl, d, -1)
        tensors[i + 1] = np.tensordot(r, tensors[i + 1], axes=1)
    for i in range(len(tensors) - 1, 0, -1):
        Dl, d, Dr = tensors[i].shape
        u, s, vh = np.linalg.svd(tensors[i].reshape(Dl, d * Dr), full_matrices=False)
        rank = max(1, int(np.sum(s > relative_tolerance * max(s[0], 1e-300))))
        tensors[i] = vh[:rank].reshape(rank, d, Dr)
        tensors[i - 1] = np.tensordot(tensors[i - 1], u[:, :rank] * s[:rank], axes=1)
    return tensors


CHANNEL_CHARGE = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # c†a, ca, c†b, cb left open


def trial_times_h(W, trial_np, trial_charges):
    """H|trial>, uncompressed, with exact (N_alpha, N_beta) bond labels, or exact particle-number
    labels N for a trial labelled by N alone (labels of width 1, e.g. a spin-rotated MPS).

    Bond index (channel, trial) is labelled with the trial's label plus the charge that
    the channel's open operator has put left of the cut (the opening order of
    hubbard_mpo_from_h1). That lets <H trial|walker> use the blocked overlap machinery.
    Padding channels that a bond never uses are dropped.
    """
    n, D = len(trial_np), W.shape[1]
    trial_charges = [label_array(q) for q in trial_charges]
    width = trial_charges[0].shape[1]
    delta = np.zeros((D, 2), int)
    delta[1:-1] = np.tile(CHANNEL_CHARGE, ((D - 2) // 4, 1))
    if width == 1:
        delta = delta.sum(axis=1, keepdims=True)
    active = (
        [np.zeros(1, int)]
        + [np.flatnonzero(np.any(W[b - 1] != 0, axis=(0, 1, 2))) for b in range(1, n)]
        + [np.full(1, D - 1)]
    )
    tensors = []
    for k in range(n):
        T = np.einsum(
            "apqb,cqd->acpbd", W[k][active[k]][..., active[k + 1]], np.asarray(trial_np[k])
        )
        dl, cl, d, dr, cr = T.shape
        tensors.append(T.reshape(dl * cl, d, dr * cr))
    charges = tuple(
        (delta[active[b]][:, None, :] + trial_charges[b][None]).reshape(-1, width)
        for b in range(n + 1)
    )
    return tensors, charges


def hubbard_h1(ham_data) -> np.ndarray:
    """ham_data.h1 as a real symmetric NumPy matrix (symmetrised exactly after a 1e-12 check)."""
    h1 = np.asarray(ham_data.h1)
    if np.iscomplexobj(h1):
        raise ValueError("MPS-CPMC supports real h1 only")
    h1 = np.asarray(h1, dtype=float)
    if h1.ndim != 2 or h1.shape[0] != h1.shape[1]:
        raise ValueError(f"h1 must be a square matrix, got shape {h1.shape}")
    if not np.allclose(h1, h1.T, atol=1.0e-12, rtol=0.0):
        raise ValueError("h1 must be symmetric")
    return 0.5 * (h1 + h1.T)


# ---------------------------------------------------------------------------------------------
# Measurement context and the energy kernel
# ---------------------------------------------------------------------------------------------


@tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class MpsMeasCtx:
    """Per-run measurement data for MPS-CPMC, built once by build_meas_ctx.

    trial_blocks: padded charge blocks of the trial for the plan's overlap layout (also read by the propagation).
    h_blocks: padded blocks of the labelled, compressed H|trial> ("blocked"), or the densely compressed H|trial>
      tensors ("dense").
    dense_trial: "dense" kernel only, the trial tensors.
    key: static aux (plan, trial charges, H|trial> charges or None, kernel, trial energy, H|trial> bond
      dimensions). Kernels and the step check plan and charges at trace time.
    """

    trial_blocks: tuple
    h_blocks: tuple
    dense_trial: tuple
    key: tuple

    @property
    def plan(self) -> MpsWalkerPlan:
        return self.key[0]

    @property
    def trial_charges(self) -> tuple:
        return self.key[1]

    @property
    def h_charges(self) -> tuple | None:
        return self.key[2]

    @property
    def kernel(self) -> str:
        return self.key[3]

    @property
    def trial_energy(self) -> float:
        """<trial|H|trial> / <trial|trial> (over every sector for a trial used as it is)."""
        return self.key[4]

    @property
    def h_bond_dims(self) -> tuple:
        return self.key[5]

    @property
    def kernels(self) -> engine.Kernels:
        """The plan's engine for this trial and H|trial> (trot.gmps.engine.kernels_for, cached on the plan)."""
        return engine.kernels_for(self.plan, self.trial_charges, self.h_charges, self.kernel)

    def data(self, prop_ctx=None) -> engine.DeviceData:
        """The engine's DeviceData: these blocks, and exp(-dt K/2) and the HS factors of a HubbardCpmcCtx."""
        return engine.DeviceData(self.trial_blocks, self.h_blocks, self.dense_trial,
                              None if prop_ctx is None else prop_ctx.exp_h1_half,
                              None if prop_ctx is None else prop_ctx.hs_constant)

    def tree_flatten(self):
        return (self.trial_blocks, self.h_blocks, self.dense_trial), self.key

    @classmethod
    def tree_unflatten(cls, key, children):
        trial_blocks, h_blocks, dense_trial = children
        return cls(trial_blocks=trial_blocks, h_blocks=h_blocks, dense_trial=dense_trial, key=key)


def _dense_overlap(a, b) -> float:
    env = np.ones((1, 1))
    for x, y in zip(a, b):
        env = np.einsum("ab,apc,bpd->cd", env, np.asarray(x), np.asarray(y), optimize=True)
    return float(env.reshape(()))


def _device(blocks) -> tuple:
    return tuple(jnp.asarray(b) for b in blocks)


def build_mps_meas_ctx(ham_data, trial_data, *, plan: MpsWalkerPlan, kernel: str, htrial=None) -> MpsMeasCtx:
    """Host-side precomputation (called eagerly by trot's drivers): H|trial> and the padded blocks.

    htrial: optional precomputed (tensors, labels, trial_energy) of the compressed, labelled H|trial> ("blocked"
    only); the tensors may be in block form (trot.gmps.trials, 6x6 and larger lattices).
    """
    if kernel not in ENERGY_KERNELS:
        raise ValueError(f"energy_kernel must be one of {ENERGY_KERNELS}, got {kernel!r}")
    check_trial(trial_data, plan)
    trial_np = [np.asarray(A) for A in trial_data.tensors]
    # padded on the host and copied once (a device gather per site would compile one kernel per block shape)
    trial_blocks = _device(engine.fixed_blocks(trial_np, engine.layout_for(plan, trial_data.charges), xp=np))
    dense_trial: tuple = ()
    if htrial is not None:
        if kernel != "blocked":
            raise ValueError('a precomputed H|trial> needs energy_kernel="blocked"')
        h_np, h_q, energy = htrial
    else:
        W = hubbard_mpo_from_h1(hubbard_h1(ham_data), float(ham_data.u))
        if kernel == "blocked":
            h_np, h_q = compress_mps_qn(*trial_times_h(W, trial_np, trial_data.charge_arrays()))
        else:
            h_np, h_q = compress_mps(apply_mpo(W, trial_np)), None
        energy = _dense_overlap(h_np, trial_np) / _dense_overlap(trial_np, trial_np)
    if kernel == "blocked":
        h_charges = _hashable_charges(h_q)
        h_blocks = _device(engine.fixed_blocks(h_np, engine.layout_for(plan, h_charges), xp=np))
        bonds = tuple(len(label_array(q)) for q in h_q)
    else:
        h_charges = None
        h_blocks = tuple(jnp.asarray(A) for A in h_np)
        dense_trial = tuple(jnp.asarray(A) for A in trial_np)
        bonds = tuple(int(A.shape[0]) for A in h_np) + (int(h_np[-1].shape[-1]),)
    key = (plan, trial_data.charges, h_charges, kernel, float(energy), bonds)
    return MpsMeasCtx(trial_blocks=trial_blocks, h_blocks=h_blocks, dense_trial=dense_trial, key=key)


def check_meas_ctx(meas_ctx, plan: MpsWalkerPlan, trial_data, kernel: str | None = None) -> None:
    if not isinstance(meas_ctx, MpsMeasCtx):
        raise ValueError(
            "MPS-CPMC needs meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data) "
            f"(got {type(meas_ctx).__name__})"
        )
    if meas_ctx.plan is not plan:
        raise ValueError("meas_ctx was built for a different walker plan")
    if meas_ctx.trial_charges != trial_data.charges:
        raise ValueError("meas_ctx was built for a different trial (bond labels differ)")
    if kernel is not None and meas_ctx.kernel != kernel:
        raise ValueError(f"meas_ctx holds the {meas_ctx.kernel!r} kernel, not {kernel!r}")


def mps_energy(walker, ham_data, meas_ctx: MpsMeasCtx, trial_data, plan: MpsWalkerPlan):
    """<H trial|walker> / <trial|walker> for one SD walker."""
    check_meas_ctx(meas_ctx, plan, trial_data)
    check_walker(walker)
    return meas_ctx.kernels.energy_one(walker[0], walker[1], meas_ctx.data())


def mps_energy_fn(plan: MpsWalkerPlan):
    """The jitted local-energy kernel of a plan (cached on the plan)."""
    fn = plan.caches.get("energy")
    if fn is None:
        fn = jax.jit(lambda walker, ham, ctx, trial: mps_energy(walker, ham, ctx, trial, plan))
        plan.caches["energy"] = fn
    return fn


class SzszFns(NamedTuple):
    batch: Callable  # (ca, cb, data) -> (n, L, L) <S^z_i S^z_j> of a walker batch
    one: Callable  # (ca, cb, data) -> (L, L) of one walker


def szsz_fns(meas_ctx: MpsMeasCtx) -> SzszFns:
    """The <trial|S^z_i S^z_j|walker> / <trial|walker> kernels of a context, cached on the plan.

    Like the energy: the walker is QR-orthonormalised and converted, then contracted against the trial blocks
    ("blocked", engine.szsz_contract) or as the d=4 MPS against the dense trial ("dense", engine.szsz_dense).
    """
    plan, kernels = meas_ctx.plan, meas_ctx.kernels
    key = ("szsz", meas_ctx.trial_charges, meas_ctx.kernel)
    fns = plan.caches.get(key)
    if fns is None:
        qa_labels, qb_labels = kernels.converter.charges

        def value(qa, qb, data):
            alpha, beta, _ = kernels.converter.convert(qa, qb)
            if kernels.energy_kind == "dense":
                tensors, _ = combine_channels(alpha, qa_labels, beta, qb_labels)
                return engine.szsz_dense(tensors, data.dense_trial)
            blocks = engine.walker_blocks(alpha, beta, kernels.overlap_plan)
            return engine.szsz_contract(blocks, data.trial, kernels.overlap_plan)

        def batch(ca, cb, data):
            qa, _ = kernels.batch_qr(ca)
            qb, _ = kernels.batch_qr(cb)
            return jax.vmap(lambda a, b: value(a, b, data))(qa, qb)

        def one(ca, cb, data):
            qa, _ = qr_with_det(ca)
            qb, _ = qr_with_det(cb)
            return value(qa, qb, data)

        fns = plan.caches[key] = SzszFns(batch, one)
    return fns


def mps_szsz(walker, ham_data, meas_ctx: MpsMeasCtx, trial_data, plan: MpsWalkerPlan):
    """<trial|S^z_i S^z_j|walker> / <trial|walker> (L, L) for one SD walker."""
    check_meas_ctx(meas_ctx, plan, trial_data)
    check_walker(walker)
    return szsz_fns(meas_ctx).one(walker[0], walker[1], meas_ctx.data())


def mps_szsz_fn(plan: MpsWalkerPlan):
    """The jitted per-walker szsz observable of a plan (cached on the plan)."""
    fn = plan.caches.get("szsz")
    if fn is None:
        fn = jax.jit(lambda walker, ham, ctx, trial: mps_szsz(walker, ham, ctx, trial, plan))
        plan.caches["szsz"] = fn
    return fn


# observables the MPS block (trot.prop.mps_cpmc) measures for the whole walker batch: name -> meas_ctx -> batch fn
BATCHED_OBSERVABLES = {o_szsz: lambda meas_ctx: szsz_fns(meas_ctx).batch}


def make_mps_meas_ops_hubbard(plan: MpsWalkerPlan, *, energy_kernel: str = "blocked", htrial=None) -> MeasOps:
    """MeasOps for an MpsTrial and a HamHubbard: overlap, build_meas_ctx, the energy kernel and the szsz observable.

    htrial: optional precomputed H|trial> for build_meas_ctx (see build_mps_meas_ctx).
    """
    if energy_kernel not in ENERGY_KERNELS:
        raise ValueError(f"energy_kernel must be one of {ENERGY_KERNELS}, got {energy_kernel!r}")
    return MeasOps(
        overlap=mps_overlap_fn(plan),
        build_meas_ctx=partial(build_mps_meas_ctx, plan=plan, kernel=energy_kernel, htrial=htrial),
        kernels={k_energy: mps_energy_fn(plan)},
        observables={o_szsz: mps_szsz_fn(plan)},
    )
