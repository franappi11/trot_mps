"""Spin rotations of MPS trials as an S_z-conserving MPO.

rotate_spin (trot.trial.mps) applies the site operator M = spin_rotation_unitary(R) on every site. The result spans
every (N_up, N_dn) sector with the same N; make_mps_trial uses it as it is, with particle-number labels (the walkers
pick their sector by themselves). Here the projection onto one sector is part of the operator. M splits into
S_z-covariant pieces,

    M = u0 + u_plus + u_minus,  u0 = diag(1, R00, R11, det R),  u_plus = R01 |up><dn|,
                                u_minus = R10 |dn><up|,

which change the site's (N_up, N_dn) by (0, 0), (+1, -1) and (-1, +1). The product of M over the sites is a sum of
3^L products of pieces, and a target sector keeps the products whose net number of down->up moves is
shift = N_up(target) - N_up(input). A bond that counts the moves made so far selects exactly those: the MPO

    W[s] = sum_d S_d (x) u_d,   (S_d)[k, k'] = 1 if k' = k + d,

with k = 0 on the left and k = shift on the right equals P_target (M x M x ... x M). Applied to an MPS with
(N_up, N_dn) bond labels, it gives an MPS whose labels are exact by construction (bond index (k, a) carries
charges[a] + (k, -k)): rotate_mps_trial returns it as an MpsTrial with (N_up, N_dn) labels and its sector weight.
Both trials give the same local energies and CPMC trajectory; the projected one has exact (N_up, N_dn) blocks.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from trot.trial.mps import (
    MpsTrial,
    _norm2,
    _real_tensors,
    _valid_sz_labels,
    compress_mps_qn,
    make_mps_trial,
    one_rdm,
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

    Gives P_nelec rotate_spin(tensors, R), the rotated state projected onto nelec: the rotation MPO
    keeps only the parts of the rotation that end in nelec, and its output carries exact labels.
    nelec defaults to the input's sector; another sector must have the same total N.
    The result is compressed sector by sector (compress_mps_qn); nothing else is truncated.

    Returns (tensors, charges, norm2) with norm2 = <P U T|P U T>. Raises
    ValueError for input without valid (N_up, N_dn) labels, and when the sector holds less than
    1e-12 of the norm.
    """
    A = _real_tensors(tensors)
    L = len(A)
    nelec_in = _labelled_sector(A, charges)
    if nelec_in is None:
        raise ValueError(
            "rotate_spin_mpo needs (N_up, N_dn) bond labels that the tensors respect; for other "
            "input use make_mps_trial(rotate_spin(tensors, R), nelec=...): the rotated MPS as it is, N labels"
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
    The result's sector_weight is <P U T|P U T>/<T|T>; make_mps_trial(rotate_spin(...)) is the same trial
    used as it is (particle-number labels, sector_weight None).
    """
    tensors, charges, norm2 = rotate_spin_mpo(trial.tensors, trial.charge_arrays(), R, nelec=nelec)
    if rdm1 is None:
        rdm1 = rotated_rdm1(np.stack(one_rdm(trial.tensors)), R)
    target = trial.nelec if nelec is None else nelec
    rotated = make_mps_trial(tensors, charges, nelec=target, rdm1=rdm1)
    return dataclasses.replace(rotated, sector_weight=float(norm2 / _norm2(trial.tensors)))
