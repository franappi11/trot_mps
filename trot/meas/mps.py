"""Hubbard measurements for MPS trials (trot.trial.mps).

The local energy is <trial|H|walker> / <trial|walker> with H|trial> built once on the host from
the automaton MPO of any real symmetric h1 plus on-site U (hubbard_mpo_from_h1). Two kernels:

- "blocked": H|trial> keeps exact (N_up, N_dn) bond labels (trial_times_h), is compressed sector
  by sector (compress_mps_qn) and contracted charge-blocked against the walker channels, without
  ever forming the d=4 walker MPS.
- "dense": the d=4 walker MPS is formed (combine_channels) and contracted with the densely
  compressed H|trial> (compress_mps), as trot/gmps/mps_cpmc_new.py did.

Both give the same number up to rounding. build_mps_meas_ctx also precomputes the padded trial
blocks used by the propagation step, so the hot loop never gathers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from trot.core.ops import MeasOps, k_energy
from trot.gmps.utils import combine_channels, contract_real
from trot.trial.mps import (
    MpsWalkerPlan,
    _hashable_charges,
    blocked_contract_from_blocks,
    check_trial,
    compress_mps_qn,
    contraction_layout,
    convert_walker,
    extract_channel_blocks,
    extract_fixed_blocks,
    mps_overlap_fn,
)

ENERGY_KERNELS = ("blocked", "dense")


# ---------------------------------------------------------------------------------------------
# MPOs and H|trial> (moved from trot/gmps/mps_cpmc_new.py and trot/gmps/mps_cpmc_2d.py)
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
    """H|trial>, uncompressed, with exact (N_alpha, N_beta) bond labels.

    Bond index (channel, trial) is labelled with the trial's label plus the charge that
    the channel's open operator has put left of the cut (the opening order of
    hubbard_mpo_from_h1). That lets <H trial|walker> use the blocked overlap machinery.
    Padding channels that a bond never uses are dropped.
    """
    n, D = len(trial_np), W.shape[1]
    delta = np.zeros((D, 2), int)
    delta[1:-1] = np.tile(CHANNEL_CHARGE, ((D - 2) // 4, 1))
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
        (delta[active[b]][:, None, :] + np.asarray(trial_charges[b])[None]).reshape(-1, 2)
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
# Measurement context and energy kernels
# ---------------------------------------------------------------------------------------------


@tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class MpsMeasCtx:
    """Per-run measurement data for MPS-CPMC, built once by build_mps_meas_ctx.

    trial_blocks: padded charge blocks of the trial for the overlap layout (also used by the
      propagation step).
    h_blocks: "blocked" kernel only, padded charge blocks of the labelled, compressed H|trial>.
    h_tensors: "dense" kernel only, the densely compressed H|trial> tensors.
    key: static aux (plan, trial charges, H|trial> charges or None, kernel, trial energy,
      H|trial> bond dimensions). Kernels and the step check plan and charges at trace time.
    """

    trial_blocks: tuple
    h_blocks: tuple | None
    h_tensors: tuple | None
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
        """<trial|H|trial> / <trial|trial> of the (projected) trial."""
        return self.key[4]

    @property
    def h_bond_dims(self) -> tuple:
        return self.key[5]

    def tree_flatten(self):
        return (self.trial_blocks, self.h_blocks, self.h_tensors), self.key

    @classmethod
    def tree_unflatten(cls, key, children):
        trial_blocks, h_blocks, h_tensors = children
        return cls(trial_blocks=trial_blocks, h_blocks=h_blocks, h_tensors=h_tensors, key=key)


def _dense_overlap(a, b) -> float:
    env = np.ones((1, 1))
    for x, y in zip(a, b):
        env = np.einsum("ab,apc,bpd->cd", env, np.asarray(x), np.asarray(y), optimize=True)
    return float(env.reshape(()))


def build_mps_meas_ctx(ham_data, trial_data, *, plan: MpsWalkerPlan, kernel: str) -> MpsMeasCtx:
    """Host-side precomputation (called eagerly by trot's drivers): H|trial> and padded blocks."""
    if kernel not in ENERGY_KERNELS:
        raise ValueError(f"energy_kernel must be one of {ENERGY_KERNELS}, got {kernel!r}")
    check_trial(trial_data, plan)
    trial_np = [np.asarray(A) for A in trial_data.tensors]
    W = hubbard_mpo_from_h1(hubbard_h1(ham_data), float(ham_data.u))
    layout = contraction_layout(plan, trial_data.charges)
    trial_blocks = extract_fixed_blocks(trial_np, layout.contraction)
    norm2 = _dense_overlap(trial_np, trial_np)
    if kernel == "blocked":
        h_np, h_q = compress_mps_qn(*trial_times_h(W, trial_np, trial_data.charge_arrays()))
        h_charges = _hashable_charges(h_q)
        h_layout = contraction_layout(plan, h_charges)
        h_blocks = extract_fixed_blocks(h_np, h_layout.contraction)
        h_tensors = None
    else:
        h_np = compress_mps(apply_mpo(W, trial_np))
        h_charges, h_blocks = None, None
        h_tensors = tuple(jnp.asarray(A) for A in h_np)
    energy = _dense_overlap(h_np, trial_np) / norm2
    bonds = tuple(int(A.shape[0]) for A in h_np) + (int(h_np[-1].shape[-1]),)
    key = (plan, trial_data.charges, h_charges, kernel, energy, bonds)
    return MpsMeasCtx(trial_blocks=trial_blocks, h_blocks=h_blocks, h_tensors=h_tensors, key=key)


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


def energy_blocked(walker, ham_data, meas_ctx: MpsMeasCtx, trial_data, plan: MpsWalkerPlan):
    """<H trial|walker> / <trial|walker>, both through charge-blocked contractions."""
    check_meas_ctx(meas_ctx, plan, trial_data, "blocked")
    trial_layout = contraction_layout(plan, meas_ctx.trial_charges)
    h_charges = meas_ctx.h_charges
    if h_charges is None:
        raise ValueError("meas_ctx has no labelled H|trial> (built for the dense kernel?)")
    h_layout = contraction_layout(plan, h_charges)
    alpha, _, beta, _, _ = convert_walker(walker, plan)
    numerator = blocked_contract_from_blocks(
        extract_channel_blocks(alpha, beta, h_layout.channel_maps),
        meas_ctx.h_blocks,
        h_layout.contraction,
    )
    denominator = blocked_contract_from_blocks(
        extract_channel_blocks(alpha, beta, trial_layout.channel_maps),
        meas_ctx.trial_blocks,
        trial_layout.contraction,
    )
    return numerator / denominator


def energy_dense(walker, ham_data, meas_ctx: MpsMeasCtx, trial_data, plan: MpsWalkerPlan):
    """<H trial|walker> / <trial|walker> with the dense d=4 walker MPS."""
    check_meas_ctx(meas_ctx, plan, trial_data, "dense")
    alpha, qa, beta, qb, _ = convert_walker(walker, plan)
    tensors, _ = combine_channels(alpha, qa, beta, qb)
    return contract_real(tensors, meas_ctx.h_tensors) / contract_real(
        tensors, tuple(trial_data.tensors)
    )


def mps_energy_fn(plan: MpsWalkerPlan, energy_kernel: str):
    """The jitted local-energy kernel of a plan (cached on the plan)."""
    if energy_kernel not in ENERGY_KERNELS:
        raise ValueError(f"energy_kernel must be one of {ENERGY_KERNELS}, got {energy_kernel!r}")
    key = ("energy", energy_kernel)
    fn = plan.caches.get(key)
    if fn is None:
        kernel = energy_blocked if energy_kernel == "blocked" else energy_dense
        fn = jax.jit(lambda walker, ham, ctx, trial: kernel(walker, ham, ctx, trial, plan))
        plan.caches[key] = fn
    return fn


def make_mps_meas_ops_hubbard(plan: MpsWalkerPlan, *, energy_kernel: str = "blocked") -> MeasOps:
    """MeasOps for an MpsTrial and a HamHubbard: overlap, build_meas_ctx and the energy kernel."""
    return MeasOps(
        overlap=mps_overlap_fn(plan),
        build_meas_ctx=partial(build_mps_meas_ctx, plan=plan, kernel=energy_kernel),
        kernels={k_energy: mps_energy_fn(plan, energy_kernel)},
    )
