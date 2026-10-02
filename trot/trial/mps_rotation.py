"""Spin rotations of MPS trials as an S_z-conserving MPO.

rotate_spin (trot.trial.mps) applies the site operator M = spin_rotation_unitary(R) on every site.
The result spans every (N_up, N_dn) sector with the same N, and make_mps_trial projects it back
onto the walkers' sector. Here the projection is part of the operator. M splits into S_z-covariant
pieces,

    M = u0 + u_plus + u_minus,  u0 = diag(1, R00, R11, det R),  u_plus = R01 |up><dn|,
                                u_minus = R10 |dn><up|,

which change the site's (N_up, N_dn) by (0, 0), (+1, -1) and (-1, +1). The product of M over the
sites is a sum of 3^L products of pieces, and a target sector keeps the products whose net number
of down->up moves is shift = N_up(target) - N_up(input). A bond that counts the moves made so far
selects exactly those: the MPO

    W[s] = sum_d S_d (x) u_d,   (S_d)[k, k'] = 1 if k' = k + d,

with k = 0 on the left and k = shift on the right equals P_target (M x M x ... x M). Applied to an
MPS with (N_up, N_dn) bond labels, it gives an MPS whose labels are exact by construction (bond
index (k, a) carries charges[a] + (k, -k)), so make_mps_trial uses it as it is, without projecting.

RotatedMpsTrial is the third option: the rotated MPS used as it is, with the input's bond dimension
and no definite (N_up, N_dn). Walkers are S_z eigenstates, so <T|W> = <P T|W> picks the walkers'
sector by itself, and H conserves S_z, so <T|H|W> = <P T|H|W>: overlaps are those of the projected
trial times the constant sqrt(<P T|P T>), local energies are the same, and so is the CPMC
trajectory. Without bond labels the contractions with walkers are dense (the walker's spin
channels combined into one d=4 MPS).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from trot.core.ops import TrialOps
from trot.gmps.utils import combine_channels, contract_real
from trot.trial.mps import (
    MpsTrial,
    MpsWalkerPlan,
    _norm2,
    _real_tensors,
    _require_x64,
    _valid_sz_labels,
    compress_mps_qn,
    convert_walker,
    make_mps_trial,
    make_walker_plan_from_reference,
    natural_orbitals,
    one_rdm,
    rhf_orbitals,
    rotate_spin,
    spin_rotation_unitary,
)


def spin_rotation_pieces(R) -> dict[int, np.ndarray]:
    """The S_z-covariant pieces of the site operator M = spin_rotation_unitary(R).

    pieces[d] changes the site's (N_up, N_dn) by (d, -d), and pieces[0] + pieces[1] + pieces[-1] = M:
      pieces[0]  = diag(1, R00, R11, det R)  keeps the occupation,
      pieces[1]  = R01 |up><dn|              turns a down electron up,
      pieces[-1] = R10 |dn><up|              turns an up electron down.
    """
    M = spin_rotation_unitary(R)
    up, down = np.zeros((4, 4)), np.zeros((4, 4))
    up[1, 2] = M[1, 2]
    down[2, 1] = M[2, 1]
    return {0: np.diag(np.diag(M)), 1: up, -1: down}


def spin_rotation_mpo(R, L, shift=0):
    """MPO of P (M x M x ... x M), with M = spin_rotation_unitary(R) and P the projector onto
    N_up(out) - N_up(in) = shift at fixed N.

    The bond after site s carries k, the net number of down->up moves on sites 0..s. k is 0 before
    the first site and must be shift after the last, so bond c keeps only the values that can still
    get there: max(-c, shift - (L - c)) <= k <= min(c, shift + (L - c)). Each site tensor is the
    sum over the pieces of the rotation, W[s] = sum_d S_d (x) pieces[d] with (S_d)[k, k'] = 1 if
    k' = k + d.

    Returns (W, ks): W[s] of shape (len(ks[s]), 4, 4, len(ks[s + 1])), indexed (k, out, in, k'),
    and ks[c] the counter values on bond c (ks[0] = [0], ks[L] = [shift]).
    """
    if not abs(shift) <= L:
        raise ValueError(f"|shift| = {abs(shift)} exceeds the number of sites {L}")
    pieces = spin_rotation_pieces(R)
    ks = [np.arange(max(-c, shift - (L - c)), min(c, shift + (L - c)) + 1) for c in range(L + 1)]
    W = []
    for s in range(L):
        step = ks[s + 1][None, :] - ks[s][:, None]  # k' - k for every pair of bond values
        T = np.zeros((len(ks[s]), 4, 4, len(ks[s + 1])))
        for d, piece in pieces.items():
            T += np.einsum("ij,pq->ipqj", (step == d).astype(float), piece)
        W.append(T)
    return W, ks


def apply_rotation_mpo(W, ks, tensors, charges):
    """W|T> for an MPS with (N_up, N_dn) bond labels, exactly (no truncation).

    Bond index (i, a) of the result, MPO index major, pairs the counter value k = ks[c][i] with MPS
    index a and carries the label charges[c][a] + (k, -k). Returns (tensors, charges).
    """
    out = []
    for w, A in zip(W, tensors):
        T = np.einsum("ipqj,aqb->iapjb", w, np.asarray(A, dtype=float))
        i, a, d, j, b = T.shape
        out.append(T.reshape(i * a, d, j * b))
    labels = []
    for k, q in zip(ks, charges):
        moves = np.stack([k, -k], axis=1)
        labels.append((moves[:, None, :] + np.asarray(q, int).reshape(-1, 2)[None]).reshape(-1, 2))
    return out, tuple(labels)


def _labelled_sector(tensors, charges):
    """(N_up, N_dn) of an MPS whose charges are valid (N_up, N_dn) bond labels, else None."""
    try:
        last = np.asarray(charges[-1], int).reshape(-1, 2)
        nelec = (int(last[0, 0]), int(last[0, 1]))
    except (TypeError, ValueError, IndexError):
        return None
    return nelec if _valid_sz_labels(tensors, charges, nelec) else None


def _structurally_zero(tensors) -> bool:
    """True if no chain of nonzero entries joins the two boundary bonds (the MPS is exactly 0)."""
    reach = np.ones(1, bool)
    for A in tensors:
        reach = np.any(reach[:, None, None] & (np.asarray(A) != 0), axis=(0, 1))
    return not reach[0]


def rotate_spin_mpo(tensors, charges, R, *, nelec=None, relative_tolerance=1.0e-13):
    """Spin-rotate an MPS with (N_up, N_dn) bond labels straight into one (N_up, N_dn) sector.

    Gives the state of project_to_sector(rotate_spin(tensors, R), nelec) without a projection: the
    rotation MPO keeps only the parts of the rotation that end in nelec, and its output carries
    exact labels. nelec defaults to the input's sector; another sector must have the same total N.
    The result is compressed sector by sector (compress_mps_qn); nothing else is truncated.

    Returns (tensors, charges, norm2) like project_to_sector, with norm2 = <P U T|P U T>. Raises
    ValueError for input without valid (N_up, N_dn) labels, and when the sector holds less than
    1e-12 of the norm.
    """
    A = _real_tensors(tensors)
    L = len(A)
    nelec_in = _labelled_sector(A, charges)
    if nelec_in is None:
        raise ValueError(
            "rotate_spin_mpo needs (N_up, N_dn) bond labels that the tensors respect; for other "
            "input use make_mps_trial(rotate_spin(tensors, R), nelec=...), which projects"
        )
    nup, ndn = nelec_in if nelec is None else tuple(int(x) for x in nelec)
    if nup + ndn != sum(nelec_in):
        raise ValueError(
            f"a spin rotation keeps N = {sum(nelec_in)}: cannot rotate {nelec_in} into {(nup, ndn)}"
        )
    if not (0 <= nup <= L and 0 <= ndn <= L):
        raise ValueError(f"{(nup, ndn)} is not a sector of {L} sites")
    W, ks = spin_rotation_mpo(R, L, nup - nelec_in[0])
    rotated, labels = apply_rotation_mpo(W, ks, A, charges)
    norm_total = _norm2(A)
    norm2 = 0.0
    if not _structurally_zero(rotated):
        rotated, labels = compress_mps_qn(rotated, labels, relative_tolerance)
        norm2 = _norm2(rotated)
    if not norm2 > 1.0e-12 * norm_total:
        raise ValueError(
            f"the rotated state has no weight in the sector {(nup, ndn)} "
            f"(norm^2 {norm2:.3e} of {norm_total:.3e})"
        )
    return rotated, labels, norm2


def rotated_rdm1(rdm1, R) -> np.ndarray:
    """Spin-diagonal 1-RDM blocks of U(R)|T> before any projection, from those of an S_z
    eigenstate |T>.

    U(R)^dag c+_{i,s} U(R) = sum_t R[s, t] c+_{i,t} and the up-down blocks of |T> vanish, so
    rho'_s = sum_t R[s, t]^2 rho_t. This is what make_mps_trial computes by default for
    rotate_spin output, and trot's get_rdm1_block_diag of GhfTrial(kron(R, I_L) @ block_diag(Ca, Cb))
    for |T> = SD(Ca, Cb).
    """
    R = np.asarray(R, dtype=float)
    return np.einsum("st,tij->sij", R**2, np.asarray(rdm1, dtype=float))


def rotate_mps_trial(trial: MpsTrial, R, *, nelec=None, rdm1=None) -> MpsTrial:
    """Spin-rotated copy of an MPS trial, rotated straight into one (N_up, N_dn) sector.

    nelec: target sector, the trial's by default (another one must have the same total N).
    rdm1: walker-start 1-RDM. The default, rotated_rdm1(one_rdm(trial.tensors), R), is what
      make_mps_trial(rotate_spin(trial.tensors, R), nelec=...) uses.
    The result's sector_weight is <P U T|P U T>/<T|T>, as for the projected trial.
    """
    tensors, charges, norm2 = rotate_spin_mpo(trial.tensors, trial.charge_arrays(), R, nelec=nelec)
    if rdm1 is None:
        rdm1 = rotated_rdm1(np.stack(one_rdm(trial.tensors)), R)
    target = trial.nelec if nelec is None else nelec
    rotated = make_mps_trial(tensors, charges, nelec=target, rdm1=rdm1)
    return dataclasses.replace(rotated, sector_weight=float(norm2 / _norm2(trial.tensors)))


# ---------------------------------------------------------------------------------------------
# The rotated trial used as it is: no definite (N_up, N_dn), dense contractions with walkers
# ---------------------------------------------------------------------------------------------


@tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class RotatedMpsTrial:
    """A real d=4 MPS trial without a definite (N_up, N_dn), such as rotate_spin output.

    tensors: L site tensors (D_left, 4, D_right), local index n_up + 2*n_dn, normalised as a whole
      (all sectors together); no projection and no compression, so the bonds are the input's.
    rdm1: (2, L, L) spin-diagonal blocks <c+_i,s c_j,s> of the trial's 1-RDM; trot starts walkers
      from its natural orbitals.
    nelec: static aux, the walkers' (N_up, N_dn).
    """

    tensors: tuple
    rdm1: Any
    nelec: tuple

    @property
    def norb(self) -> int:
        return len(self.tensors)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return tuple(int(A.shape[0]) for A in self.tensors) + (int(self.tensors[-1].shape[2]),)

    def tree_flatten(self):
        return (tuple(self.tensors), self.rdm1), self.nelec

    @classmethod
    def tree_unflatten(cls, nelec, children):
        tensors, rdm1 = children
        return cls(tensors=tuple(tensors), rdm1=rdm1, nelec=nelec)


def make_rotated_mps_trial(tensors, R=None, *, nelec, rdm1=None) -> RotatedMpsTrial:
    """trial_data from a real d=4 MPS that is used as it is (no S_z projection).

    tensors: L arrays (D_left, 4, D_right) in the local basis of trot.trial.mps.
    R: optional real orthogonal 2x2 spin rotation, applied first with rotate_spin.
    nelec: (N_up, N_dn) of the walkers.
    rdm1: optional (2, L, L); defaults to one_rdm of the (rotated) tensors, the same walker start
      as make_mps_trial(rotate_spin(tensors, R), nelec=...) and as the rotated GhfTrial.
    """
    _require_x64()
    A = _real_tensors(tensors)
    if R is not None:
        A = rotate_spin(A, R)
    nup, ndn = (int(x) for x in nelec)
    L = len(A)
    if not (0 <= nup <= L and 0 <= ndn <= L):
        raise ValueError(f"{(nup, ndn)} is not a sector of {L} sites")
    if rdm1 is None:
        rdm1 = np.stack(one_rdm(A))
    else:
        if np.iscomplexobj(rdm1):
            raise TypeError("rdm1 must be real")
        rdm1 = np.asarray(rdm1, dtype=float)
        if rdm1.shape != (2, L, L):
            raise ValueError(f"rdm1 must have shape (2, {L}, {L}), got {rdm1.shape}")
    norm2 = _norm2(A)
    if not norm2 > 0.0:
        raise ValueError("the trial MPS has zero norm")
    if abs(norm2 - 1.0) > 1.0e-12:
        A = [A[0] / np.sqrt(norm2)] + A[1:]
    return RotatedMpsTrial(
        tensors=tuple(jnp.asarray(t) for t in A), rdm1=jnp.asarray(rdm1), nelec=(nup, ndn)
    )


def check_rotated_trial(trial, plan: MpsWalkerPlan) -> None:
    if not isinstance(trial, RotatedMpsTrial):
        raise TypeError(f"trial_data must be a RotatedMpsTrial, got {type(trial).__name__}")
    if tuple(trial.nelec) != tuple(plan.nelec) or trial.norb != plan.norb:
        raise ValueError(
            f"trial (norb={trial.norb}, nelec={trial.nelec}) does not match the walker plan "
            f"(norb={plan.norb}, nelec={tuple(plan.nelec)})"
        )


def make_rotated_walker_plan(ham_data, trial: RotatedMpsTrial, sys_, params) -> MpsWalkerPlan:
    """make_walker_plan (trot.trial.mps) for a RotatedMpsTrial: the same reference choices
    ("natural": natural orbitals of trial.rdm1, "rhf": eigenvectors of h1) and settings."""
    _require_x64()
    if not isinstance(trial, RotatedMpsTrial):
        raise TypeError("trial must be a RotatedMpsTrial (see make_rotated_mps_trial)")
    if sys_.walker_kind != "unrestricted":
        raise ValueError("MPS-CPMC needs walker_kind='unrestricted'")
    nelec = (int(sys_.nelec[0]), int(sys_.nelec[1]))
    if tuple(trial.nelec) != nelec or trial.norb != sys_.norb:
        raise ValueError(
            f"trial (norb={trial.norb}, nelec={trial.nelec}) does not match the system "
            f"(norb={sys_.norb}, nelec={nelec})"
        )
    if params.plan_reference == "natural":
        rdm1 = np.asarray(trial.rdm1)
        Ra = natural_orbitals(rdm1[0], nelec[0])[0]
        Rb = natural_orbitals(rdm1[1], nelec[1])[0]
    elif params.plan_reference == "rhf":
        Ra, Rb = rhf_orbitals(ham_data, nelec)
    else:
        raise ValueError(
            f"plan_reference must be 'natural' or 'rhf', got {params.plan_reference!r}"
        )
    return make_walker_plan_from_reference(
        Ra,
        Rb,
        orbital_plan=params.orbital_plan,
        occupation_tolerance=params.occupation_tolerance,
        walker_channel_chi=params.walker_channel_chi,
        walker_cutoff=params.walker_cutoff,
    )


def dense_walker_mps(walker, plan: MpsWalkerPlan):
    """The walker as one d=4 MPS (spin channels combined, with the interleave sign) and the
    prefactor that multiplies its amplitudes."""
    alpha, qa, beta, qb, prefactor = convert_walker(walker, plan)
    tensors, _ = combine_channels(alpha, qa, beta, qb)
    return tensors, prefactor


def rotated_overlap(walker, trial: RotatedMpsTrial, plan: MpsWalkerPlan):
    """<trial|walker> for one SD walker by a dense contraction, a real float64 scalar."""
    check_rotated_trial(trial, plan)
    tensors, prefactor = dense_walker_mps(walker, plan)
    return jnp.real(prefactor * contract_real(tensors, tuple(trial.tensors)))


def rotated_overlap_fn(plan: MpsWalkerPlan):
    """The jitted rotated-trial overlap of a plan, shared by its trial and measurement ops."""
    fn = plan.caches.get("rotated_overlap")
    if fn is None:
        fn = jax.jit(lambda walker, trial_data: rotated_overlap(walker, trial_data, plan))
        plan.caches["rotated_overlap"] = fn
    return fn


def _trial_rdm1(trial_data: RotatedMpsTrial) -> jax.Array:
    return trial_data.rdm1


def make_rotated_trial_ops(plan: MpsWalkerPlan) -> TrialOps:
    """TrialOps for a RotatedMpsTrial: dense overlap and rdm1."""
    return TrialOps(overlap=rotated_overlap_fn(plan), get_rdm1=_trial_rdm1)
