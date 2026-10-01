"""MPS trial wave functions for CPMC with Slater-determinant walkers (Hubbard model).

The trial is a dense, charge-labelled d=4 MPS in the local basis |0>, |up>, |dn>, |up dn>,
indexed n_up + 2*n_dn, with |up dn> = c+_up c+_dn |0> and fermion operators ordered site by
site with up before down. Bond labels are (N_up, N_dn) counted from the left. Walkers stay
Slater determinants: each spin channel is converted to a charge-labelled Gaussian MPS
(Fishman-White, trot.gmps.utils) whenever an overlap is needed, and the overlap is a
charge-blocked contraction against the trial.

Trials that do not conserve N_up and N_dn separately (for example spin-rotated ones, see
rotate_spin) are projected exactly onto the walkers' (N_up, N_dn) sector. The walkers are S_z
eigenstates and the Hubbard Hamiltonian conserves S_z, so overlaps and local energies, and hence
the whole CPMC trajectory, are unchanged by the projection.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from trot.core.ops import TrialOps
from trot.gmps.utils import (
    channel_mps,
    combined_charges,
    make_orbital_plan,
    plan_bonds,
    sd_to_gmps,
)
from trot.walkers import _qr as qr_with_det


""""
Convetions for the particle charges:(N_up,N_down)
"""
PHYSICAL_CHARGE = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


def _require_x64() -> None:
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "MPS-CPMC needs float64: enable jax_enable_x64 (trot.config.configure_once() does)."
        )


def one_rdm(tensors):
    """Spin-resolved one-body density matrices <c^dag_i,sigma c_j,sigma> of a real
    d=4 MPS in the interleaved (alpha before beta on each site) ordering."""
    create_a = np.zeros((4, 4))
    create_a[1, 0] = create_a[3, 2] = 1.0
    create_b = np.zeros((4, 4))
    create_b[2, 0] = 1.0
    create_b[3, 1] = -1.0
    parity_a, parity_b = np.diag([1.0, -1.0, 1.0, -1.0]), np.diag([1.0, 1.0, -1.0, -1.0])
    operators = {
        "a": (create_a @ parity_b, create_a.T, np.diag([0.0, 1.0, 0.0, 1.0])),
        "b": (parity_a @ create_b, create_b.T, np.diag([0.0, 0.0, 1.0, 1.0])),
    }
    A = [np.asarray(t) for t in tensors]
    L = len(A)
    site = lambda E, x, O: np.einsum("ab,apc,pq,bqd->cd", E, A[x], O, A[x], optimize=True)
    left = [np.ones((1, 1))]
    for x in range(L):
        left.append(site(left[-1], x, np.eye(4)))
    right = [np.ones((1, 1))]
    for x in range(L - 1, -1, -1):
        right.insert(0, np.einsum("apc,bpd,cd->ab", A[x], A[x], right[0], optimize=True))
    gammas = []
    for create_i, annihilate_j, number in operators.values():
        gamma = np.zeros((L, L))
        for i in range(L):
            gamma[i, i] = np.sum(site(left[i], i, number) * right[i + 1])
            E = site(left[i], i, create_i)
            for j in range(i + 1, L):
                gamma[i, j] = gamma[j, i] = np.sum(site(E, j, annihilate_j) * right[j + 1])
                E = site(E, j, parity_a @ parity_b)  # Jordan-Wigner string between i and j
        gammas.append(gamma / left[-1][0, 0])
    return tuple(gammas)


def natural_orbitals(gamma, n):
    """The n most occupied natural orbitals and all occupations, descending."""
    occupations, vectors = np.linalg.eigh(gamma)
    return vectors[:, ::-1][:, :n].copy(), occupations[::-1]


def _charge_index(labels):
    grouped = {}
    for i, charge in enumerate(map(tuple, np.asarray(labels).tolist())):
        grouped.setdefault(charge, []).append(i)
    return {charge: np.asarray(indices, int) for charge, indices in grouped.items()}


def make_contraction_plan(walker_charges, trial_charges):
    """Prepare dense padded charge blocks for repeated walker/trial contractions."""
    n = len(walker_charges) - 1
    walker_index = [_charge_index(q) for q in walker_charges]
    trial_index = [_charge_index(q) for q in trial_charges]
    shared, walker_pad, trial_pad = [], [], []
    for bond in range(n + 1):
        charges = sorted(set(walker_index[bond]) & set(trial_index[bond]))
        shared.append(charges)
        walker_pad.append(max(map(lambda q: len(walker_index[bond][q]), charges), default=0))
        trial_pad.append(max(map(lambda q: len(trial_index[bond][q]), charges), default=0))

    sites = []
    for site in range(n):
        incoming = {q: i for i, q in enumerate(shared[site])}
        outgoing = {q: i for i, q in enumerate(shared[site + 1])}
        src, dst, rows, columns, masks, physical = [], [], [], [], [], []
        for charge in shared[site]:
            left_indices = walker_index[site][charge]
            for p, delta in enumerate(PHYSICAL_CHARGE):
                next_charge = tuple(np.asarray(charge) + delta)
                if next_charge not in outgoing:
                    continue
                right_indices = walker_index[site + 1][next_charge]
                shape = (walker_pad[site], walker_pad[site + 1])
                r = np.zeros(shape, int)
                c = np.zeros(shape, int)
                mask = np.zeros(shape)
                r[: len(left_indices), : len(right_indices)] = left_indices[:, None]
                c[: len(left_indices), : len(right_indices)] = right_indices[None, :]
                mask[: len(left_indices), : len(right_indices)] = 1.0
                src.append(incoming[charge])
                dst.append(outgoing[next_charge])
                rows.append(r)
                columns.append(c)
                masks.append(mask)
                physical.append(p)
        sites.append(
            dict(
                src=np.asarray(src, int),
                dst=np.asarray(dst, int),
                rows=np.stack(rows),
                columns=np.stack(columns),
                mask=np.stack(masks),
                physical=np.asarray(physical, int),
                n_out=len(shared[site + 1]),
            )
        )
    return dict(
        sites=tuple(sites),
        shared=shared,
        walker_pad=walker_pad,
        trial_pad=trial_pad,
        walker_index=walker_index,
        trial_index=trial_index,
        n=n,
    )


def extract_fixed_blocks(tensors, plan):
    blocks = []
    for site, layout in enumerate(plan["sites"]):
        out = np.zeros((len(layout["src"]), plan["trial_pad"][site], plan["trial_pad"][site + 1]))
        for t, (qin, qout, p) in enumerate(zip(layout["src"], layout["dst"], layout["physical"])):
            charge_in = plan["shared"][site][qin]
            charge_out = plan["shared"][site + 1][qout]
            rows = plan["trial_index"][site][charge_in]
            columns = plan["trial_index"][site + 1][charge_out]
            out[t, : len(rows), : len(columns)] = np.asarray(tensors[site])[
                np.ix_(rows, [p], columns)
            ][:, 0]
        blocks.append(jnp.asarray(out))
    return tuple(blocks)


def make_channel_block_maps(plan, alpha_charges, beta_charges):
    """Map allowed combined-spin blocks back to the two channel tensors."""
    maps = []
    for site, layout in enumerate(plan["sites"]):
        left_beta = len(beta_charges[site])
        right_beta = len(beta_charges[site + 1])
        physical = layout["physical"]
        beta_left = layout["rows"] % left_beta
        maps.append(
            dict(
                alpha_left=layout["rows"] // left_beta,
                beta_left=beta_left,
                alpha_right=layout["columns"] // right_beta,
                beta_right=layout["columns"] % right_beta,
                n_alpha=(physical % 2)[:, None, None],
                n_beta=(physical // 2)[:, None, None],
                sign=layout["mask"]
                * (-1.0) ** ((physical % 2)[:, None, None] * beta_charges[site][beta_left]),
            )
        )
    return tuple(maps)


def extract_channel_blocks(alpha_mps, beta_mps, channel_maps):
    """Build only the charge blocks needed by the overlap, never the dense d=4 MPS."""
    blocks = []
    for Aa, Ab, m in zip(alpha_mps, beta_mps, channel_maps):
        a = Aa[m["alpha_left"], m["n_alpha"], m["alpha_right"]]
        b = Ab[m["beta_left"], m["n_beta"], m["beta_right"]]
        blocks.append(m["sign"] * a * b)
    return tuple(blocks)


def blocked_contract_from_blocks(walker_blocks, trial_blocks, plan):
    env = jnp.ones((1, plan["walker_pad"][0], plan["trial_pad"][0]))
    for wb, tb, layout in zip(walker_blocks, trial_blocks, plan["sites"]):
        incoming = env[layout["src"]]
        temp = jnp.einsum("tij,tik->tjk", wb, incoming)
        contributions = jnp.einsum("tjk,tkl->tjl", temp, tb)
        env = jax.ops.segment_sum(contributions, layout["dst"], num_segments=layout["n_out"])
    return env.reshape(())


def contraction_report(plan):
    exact = dense = padded = 0
    for bond in range(plan["n"] + 1):
        for charge in plan["shared"][bond]:
            exact += len(plan["walker_index"][bond][charge]) * len(
                plan["trial_index"][bond][charge]
            )
        dense += sum(map(len, plan["walker_index"][bond].values())) * sum(
            map(len, plan["trial_index"][bond].values())
        )
        padded += len(plan["shared"][bond]) * plan["walker_pad"][bond] * plan["trial_pad"][bond]
    return dict(
        dense=dense,
        exact_blocks=exact,
        padded_blocks=padded,
        transitions=sum(len(site["src"]) for site in plan["sites"]),
    )


def _label_sectors(row_labels, column_labels):
    rows, columns = _charge_index(row_labels), _charge_index(column_labels)
    return [(c, rows[c], columns[c]) for c in sorted(set(rows) & set(columns))]


def compress_mps_qn(tensors, charges, relative_tolerance=1.0e-13):
    """compress_mps that keeps the (N_alpha, N_beta) bond labels.

    The QR sweep and the SVD sweep factor each charge sector separately and the
    rank cut is one relative tolerance per bond applied per sector, so no
    factorisation mixes charges (the dense compress can, at degenerate singular
    values) and the result can be contracted charge-blocked like the trial.
    """
    A = [np.array(t, dtype=float, copy=True) for t in tensors]
    Q = [np.asarray(q, int).reshape(-1, 2) for q in charges]
    for i in range(len(A) - 1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl * d, Dr)
        rows = (Q[i][:, None, :] + PHYSICAL_CHARGE[None]).reshape(-1, 2)
        left, right, labels = [], [], []
        for c, r, k in _label_sectors(rows, Q[i + 1]):
            q, rr = np.linalg.qr(M[np.ix_(r, k)])
            lq = np.zeros((Dl * d, q.shape[1]))
            lq[r] = q
            rq = np.zeros((q.shape[1], Dr))
            rq[:, k] = rr
            left.append(lq)
            right.append(rq)
            labels += [c] * q.shape[1]
        A[i] = np.concatenate(left, axis=1).reshape(Dl, d, -1)
        A[i + 1] = np.tensordot(np.concatenate(right, axis=0), A[i + 1], axes=1)
        Q[i + 1] = np.asarray(labels, int).reshape(-1, 2)
    for i in range(len(A) - 1, 0, -1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl, d * Dr)
        columns = (Q[i + 1][None, :, :] - PHYSICAL_CHARGE[:, None, :]).reshape(-1, 2)
        factors = []
        for c, r, k in _label_sectors(Q[i], columns):
            u, s, vh = np.linalg.svd(M[np.ix_(r, k)], full_matrices=False)
            factors.append((c, r, k, u, s, vh))
        cut = relative_tolerance * max(max(f[4][0] for f in factors), 1e-300)
        left, right, labels = [], [], []
        for c, r, k, u, s, vh in factors:
            n = int(np.sum(s > cut))
            if n == 0:
                continue
            lu = np.zeros((Dl, n))
            lu[r] = u[:, :n] * s[:n]
            rv = np.zeros((n, d * Dr))
            rv[:, k] = vh[:n]
            left.append(lu)
            right.append(rv)
            labels += [c] * n
        A[i] = np.concatenate(right, axis=0).reshape(-1, d, Dr)
        A[i - 1] = np.tensordot(A[i - 1], np.concatenate(left, axis=1), axes=(2, 0))
        Q[i] = np.asarray(labels, int).reshape(-1, 2)
    return A, tuple(Q)


# ---------------------------------------------------------------------------------------------
# Sector projection and spin rotations
# ---------------------------------------------------------------------------------------------


def _norm2(tensors) -> float:
    env = np.ones((1, 1))
    for A in tensors:
        env = np.einsum("ab,apc,bpd->cd", env, A, A, optimize=True)
    return float(env.reshape(()))


def _valid_sz_labels(tensors, charges, nelec) -> bool:
    """True if charges are (N_up, N_dn) bond labels that every nonzero entry respects."""
    try:
        Q = [np.asarray(q, int) for q in charges]
    except (TypeError, ValueError):
        return False
    L = len(tensors)
    if len(Q) != L + 1 or any(q.ndim != 2 or q.shape[1] != 2 for q in Q):
        return False
    if any(len(Q[b]) != tensors[b].shape[0] for b in range(L)) or len(Q[L]) != 1:
        return False
    if not (np.all(Q[0] == 0) and np.array_equal(Q[L][0], np.asarray(nelec, int))):
        return False
    for s, A in enumerate(tensors):
        for p, delta in enumerate(PHYSICAL_CHARGE):
            consistent = np.all(Q[s][:, None, :] + delta == Q[s + 1][None, :, :], axis=-1)
            if np.any((A[:, p, :] != 0) & ~consistent):
                return False
    return True


def project_to_sector(tensors, nelec, charges=None):
    """Exact projection of a real d=4 MPS onto the (N_up, N_dn) = nelec sector.

    Works for any input labelling (SZ labels, total-N labels or none): every bond index is
    split into (old index, N_up so far, N_dn so far), entries are kept only where the counts
    match the physical charge, and labels that cannot reach nelec at the right end are pruned.
    The result is compressed sector by sector (compress_mps_qn), so it carries exact SZ labels
    and contracts charge-blocked. Valid SZ input that already ends at nelec is returned
    unchanged.

    Returns (tensors, charges, norm2) with norm2 = <P T|P T>. Raises ValueError when the
    sector holds less than 1e-12 of the norm.
    """
    A = [np.asarray(t, dtype=float) for t in tensors]
    nup, ndn = (int(x) for x in nelec)
    if charges is not None and _valid_sz_labels(A, charges, (nup, ndn)):
        Q = tuple(np.asarray(q, int).reshape(-1, 2) for q in charges)
        return A, Q, _norm2(A)

    L = len(A)
    nonzero = [a != 0 for a in A]
    shape = (nup + 1, ndn + 1)

    def shifted(R, delta, sign):
        """R shifted by sign*delta on the (u, d) axes, zero filled."""
        out = np.zeros_like(R)
        du, dd = sign * delta
        u0, u1 = max(0, du), min(shape[0], shape[0] + du)
        d0, d1 = max(0, dd), min(shape[1], shape[1] + dd)
        out[:, u0:u1, d0:d1] = R[:, u0 - du : u1 - du, d0 - dd : d1 - dd]
        return out

    reach = [np.zeros((1,) + shape, bool)]
    reach[0][0, 0, 0] = True
    for s in range(L):
        nxt = np.zeros((A[s].shape[2],) + shape, bool)
        for p, delta in enumerate(PHYSICAL_CHARGE):
            moved = shifted(reach[s].astype(float), delta, +1)
            nxt |= np.einsum("auv,ab->buv", moved, nonzero[s][:, p, :].astype(float)) > 0
        reach.append(nxt)
    alive: list[np.ndarray] = [np.zeros((0,) + shape, bool)] * (L + 1)
    alive[L] = np.zeros((1,) + shape, bool)
    alive[L][0, nup, ndn] = reach[L][0, nup, ndn]
    for s in range(L - 1, -1, -1):
        back = np.zeros_like(reach[s])
        for p, delta in enumerate(PHYSICAL_CHARGE):
            ahead = np.einsum("ab,buv->auv", nonzero[s][:, p, :].astype(float), alive[s + 1])
            back |= shifted(ahead, delta, -1) > 0
        alive[s] = reach[s] & back

    norm_total = _norm2(A)
    states = [np.argwhere(a) for a in alive]  # rows (index, n_up, n_dn), sorted
    if len(states[0]) == 0:
        raise ValueError(f"the trial has no weight in the walkers' sector {(nup, ndn)}")
    order = [np.lexsort((st[:, 0], st[:, 2], st[:, 1])) for st in states]  # group by label
    states = [st[o] for st, o in zip(states, order)]
    projected = []
    for s in range(L):
        left, right = states[s], states[s + 1]
        B = np.zeros((len(left), 4, len(right)))
        for p, delta in enumerate(PHYSICAL_CHARGE):
            match = np.all(left[:, None, 1:] + delta == right[None, :, 1:], axis=-1)
            B[:, p, :] = A[s][left[:, 0][:, None], p, right[:, 0][None, :]] * match
        projected.append(B)
    labels = tuple(st[:, 1:].copy() for st in states)
    projected, labels = compress_mps_qn(projected, labels)
    norm2 = _norm2(projected)
    if not norm2 > 1.0e-12 * norm_total:
        raise ValueError(
            f"the trial has no weight in the walkers' sector {(nup, ndn)} "
            f"(projected norm^2 {norm2:.3e} of {norm_total:.3e})"
        )
    return projected, labels, norm2


def spin_rotation_unitary(R) -> np.ndarray:
    """Site operator of a real orthogonal spin rotation R in the local basis |0>,|up>,|dn>,|up dn>.

    Column sigma of R is the image of |sigma>: M[1:3, 1:3] = R. The doubly occupied state
    c+_up c+_dn|0> picks up det(R). The many-body rotation of a determinant with spin
    orbitals C (rows 0..L-1 spin up, L..2L-1 spin down) is the determinant of kron(R, I_L) @ C,
    i.e. GhfTrial(kron(R, I_L) @ block_diag(Ca, Cb)) for an unrestricted one.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (2, 2) or not np.allclose(R @ R.T, np.eye(2), atol=1.0e-12, rtol=0.0):
        raise ValueError("R must be a real orthogonal 2x2 matrix")
    M = np.zeros((4, 4))
    M[0, 0] = 1.0
    M[1:3, 1:3] = R
    M[3, 3] = np.linalg.det(R)
    return M


def rotate_spin(tensors, R):
    """Apply the spin rotation R on every site: A'[a, p, b] = sum_s M[p, s] A[a, s, b].

    The result conserves the total particle number but in general not N_up and N_dn; pass it
    to make_mps_trial, which projects it onto the walkers' sector. Note that
    np.einsum('ijk,jl', A, M) applies M^T instead (wrong unless M is symmetric), and that a
    4x4 matrix without the det(R) entry on |up dn> is not a fermionic spin rotation.
    """
    M = spin_rotation_unitary(R)
    return [np.einsum("ps,asb->apb", M, np.asarray(A, dtype=float)) for A in tensors]


# ---------------------------------------------------------------------------------------------
# The trial
# ---------------------------------------------------------------------------------------------


def _hashable_charges(charges) -> tuple:
    return tuple(
        tuple(tuple(pair) for pair in np.asarray(q, int).reshape(-1, 2).tolist()) for q in charges
    )


@tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class MpsTrial:
    """Charge-labelled d=4 MPS trial (trial_data for MPS-CPMC).

    tensors: L site tensors (D_left, 4, D_right), local index n_up + 2*n_dn, normalised.
    rdm1: (2, L, L) spin-diagonal blocks <c+_i,s c_j,s> of the trial's 1-RDM (of the
      unprojected trial for spin-rotated input); trot starts walkers from its natural orbitals.
    charges: static aux, L+1 bond labels, each a tuple of (N_up, N_dn) pairs.
    sector_weight: static aux, <P T|P T>/<T|T> kept by the sector projection (1.0 for SZ input).
    """

    tensors: tuple
    rdm1: Any
    charges: tuple
    sector_weight: float = 1.0

    @property
    def norb(self) -> int:
        return len(self.tensors)

    @property
    def nelec(self) -> tuple[int, int]:
        nup, ndn = self.charges[-1][0]
        return int(nup), int(ndn)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return tuple(len(q) for q in self.charges)

    def charge_arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(q, int).reshape(-1, 2) for q in self.charges)

    def tree_flatten(self):
        return (tuple(self.tensors), self.rdm1), (self.charges, self.sector_weight)

    @classmethod
    def tree_unflatten(cls, aux, children):
        tensors, rdm1 = children
        charges, sector_weight = aux
        return cls(tensors=tuple(tensors), rdm1=rdm1, charges=charges, sector_weight=sector_weight)


def _real_tensors(tensors) -> list[np.ndarray]:
    out = []
    for i, A in enumerate(tensors):
        if np.iscomplexobj(A):
            raise TypeError("MPS trials must be real; complex tensors are not supported")
        A = np.asarray(A, dtype=float)
        if A.ndim != 3 or A.shape[1] != 4:
            raise ValueError(f"site {i}: expected a (D_left, 4, D_right) tensor, got {A.shape}")
        out.append(A)
    if not out:
        raise ValueError("an MPS trial needs at least one site")
    if out[0].shape[0] != 1 or out[-1].shape[2] != 1:
        raise ValueError("the boundary bonds of an MPS trial must have dimension 1")
    for i in range(len(out) - 1):
        if out[i].shape[2] != out[i + 1].shape[0]:
            raise ValueError(
                f"bond {i + 1}: site {i} and site {i + 1} disagree on the bond dimension"
            )
    return out


def make_mps_trial(tensors, charges=None, *, nelec, rdm1=None) -> MpsTrial:
    """Build trial_data from a real d=4 MPS.

    tensors: L arrays (D_left, 4, D_right) in the local basis described in the module docstring.
    charges: optional L+1 bond labels. Valid (N_up, N_dn) labels ending at nelec are used as
      they are; anything else (total-N labels, None) triggers the exact sector projection.
    nelec: (N_up, N_dn) of the walkers.
    rdm1: optional (2, L, L); defaults to one_rdm of the unprojected tensors.
    """
    _require_x64()
    if isinstance(charges, (tuple, list)) and len(charges) == 2:
        if all(isinstance(x, (int, np.integer)) for x in charges):
            raise TypeError(
                "make_mps_trial(tensors, charges=None, *, nelec, rdm1=None): pass nelec by keyword"
            )
    A = _real_tensors(tensors)
    nup, ndn = (int(x) for x in nelec)
    L = len(A)
    if rdm1 is None:
        rdm1 = np.stack(one_rdm(A))
    else:
        if np.iscomplexobj(rdm1):
            raise TypeError("rdm1 must be real")
        rdm1 = np.asarray(rdm1, dtype=float)
        if rdm1.shape != (2, L, L):
            raise ValueError(f"rdm1 must have shape (2, {L}, {L}), got {rdm1.shape}")
    norm_total = _norm2(A)
    projected, labels, norm2 = project_to_sector(A, (nup, ndn), charges)
    if abs(norm2 - 1.0) > 1.0e-12:
        projected = [projected[0] / np.sqrt(norm2)] + list(projected[1:])
    return MpsTrial(
        tensors=tuple(jnp.asarray(t) for t in projected),
        rdm1=jnp.asarray(rdm1),
        charges=_hashable_charges(labels),
        sector_weight=float(norm2 / norm_total),
    )


def mps_trial_from_sd(Ca, Cb, *, rdm1=None, mode="maximal") -> MpsTrial:
    """Unrestricted Slater determinant SD(Ca, Cb) as an MPS trial, converted exactly.

    The default rdm1 is the pair of projectors onto the column spaces (after QR), which is what
    trot's UhfTrial/GhfTrial report for orthonormal orbitals. Pass rdm1 explicitly (e.g.
    ghf_trial_ops.get_rdm1(ghf_trial)) to reproduce another trial's walker start bit for bit.
    """
    if np.iscomplexobj(Ca) or np.iscomplexobj(Cb):
        raise TypeError("orbitals must be real")
    Ca, Cb = np.asarray(Ca, dtype=float), np.asarray(Cb, dtype=float)
    gmps = sd_to_gmps(Ca, Cb, chi=None, cutoff=0.0, mode=mode)
    if rdm1 is None:
        Qa, Qb = np.linalg.qr(Ca)[0], np.linalg.qr(Cb)[0]
        rdm1 = np.stack([Qa @ Qa.T, Qb @ Qb.T])
    return make_mps_trial(
        [np.asarray(t) for t in gmps.tensors],
        gmps.charges,
        nelec=(Ca.shape[1], Cb.shape[1]),
        rdm1=rdm1,
    )


def mps_trial_from_pyblock3(mps, *, nelec=None, rdm1=None) -> MpsTrial:
    """A pyblock3 SZ MPS (e.g. a DMRG ground state) as an MPS trial."""
    from trot.gmps.utils import densify

    dense = densify(mps)
    if nelec is None:
        nelec = tuple(int(x) for x in np.asarray(dense.charges[-1])[0])
    return make_mps_trial(dense.tensors, dense.charges, nelec=nelec, rdm1=rdm1)


def as_mps_trial(x, *, nelec=None) -> MpsTrial:
    """Accept an MpsTrial, a pyblock3 MPS, a Gmps/DenseMps-like object or (tensors, charges).

    Only MpsTrial is checked with isinstance. The rest is duck-typed, so objects created by a
    second (bare-imported) copy of trot/gmps/utils.py are accepted too.
    """
    if isinstance(x, MpsTrial):
        if nelec is not None and tuple(int(v) for v in nelec) != x.nelec:
            raise ValueError(f"trial has nelec {x.nelec}, the walkers have {tuple(nelec)}")
        return x
    pyblock3_mps = sys.modules.get("pyblock3.algebra.mps")
    if pyblock3_mps is not None and isinstance(x, pyblock3_mps.MPS):
        return mps_trial_from_pyblock3(x, nelec=nelec)
    if hasattr(x, "tensors") and hasattr(x, "charges"):
        tensors, charges = x.tensors, x.charges
    elif isinstance(x, (tuple, list)) and len(x) == 2:
        tensors, charges = x
    else:
        raise TypeError(
            "trial must be an MpsTrial, a pyblock3 MPS, a Gmps/DenseMps or (tensors, charges)"
        )
    if nelec is None:
        last = np.asarray(charges[-1])
        if last.shape != (1, 2):
            raise ValueError("nelec is needed when the bond labels are not (N_up, N_dn) labels")
        nelec = tuple(int(v) for v in last[0])
    return make_mps_trial([np.asarray(t) for t in tensors], charges, nelec=nelec)


def get_rdm1(trial_data: MpsTrial) -> jax.Array:
    return trial_data.rdm1


# ---------------------------------------------------------------------------------------------
# Walker plan: the static layout shared by every walker
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MpsWalkerPlan:
    """Static walker-to-MPS layout, frozen on one reference determinant.

    orbital_plans and bond_plans fix the gate circuit and the kept count of every charge block
    for each spin channel (trot.gmps.utils.make_orbital_plan / plan_bonds). channel_charges and
    walker_charges are the resulting static bond labels. Lives in the ops closures (never a jit
    argument). caches holds per-plan layouts and jitted kernels; they are freed with the plan.
    """

    orbital_plans: tuple
    bond_plans: tuple
    channel_charges: tuple
    walker_charges: tuple
    reference: tuple
    nelec: tuple
    norb: int
    caches: dict = field(default_factory=dict, compare=False, repr=False)


def rhf_orbitals(ham_data, nelec):
    """Lowest eigenvectors of h1 for each spin: the free-fermion determinant."""
    _, orbitals = np.linalg.eigh(np.asarray(ham_data.h1, dtype=float))
    return orbitals[:, : nelec[0]].copy(), orbitals[:, : nelec[1]].copy()


def make_walker_plan_from_reference(
    Ra,
    Rb,
    *,
    orbital_plan="adaptive",
    occupation_tolerance=1.0e-10,
    walker_channel_chi=None,
    walker_cutoff=0.0,
) -> MpsWalkerPlan:
    """Freeze the gate circuit and the per-sector kept counts on the determinant (Ra, Rb)."""
    Ra, Rb = np.asarray(Ra, dtype=float), np.asarray(Rb, dtype=float)
    if Ra.shape[1] == 0 or Rb.shape[1] == 0:
        raise ValueError("MPS walkers need at least one electron of each spin")
    plan_a = make_orbital_plan(Ra, orbital_plan, occupation_tolerance)
    plan_b = make_orbital_plan(Rb, orbital_plan, occupation_tolerance)
    bond_a = bond_b = None
    if walker_channel_chi is not None or walker_cutoff:
        bond_a = plan_bonds(Ra, plan_a, walker_channel_chi, walker_cutoff)
        bond_b = plan_bonds(Rb, plan_b, walker_channel_chi, walker_cutoff)
    qa, _ = qr_with_det(jnp.asarray(Ra))
    qb, _ = qr_with_det(jnp.asarray(Rb))
    _, qa_charge, _ = channel_mps(qa, plan_a, bond_a)
    _, qb_charge, _ = channel_mps(qb, plan_b, bond_b)
    return MpsWalkerPlan(
        orbital_plans=(plan_a, plan_b),
        bond_plans=(bond_a, bond_b),
        channel_charges=(tuple(qa_charge), tuple(qb_charge)),
        walker_charges=combined_charges(qa_charge, qb_charge),
        reference=(Ra, Rb),
        nelec=(Ra.shape[1], Rb.shape[1]),
        norb=Ra.shape[0],
    )


def make_walker_plan(ham_data, trial: MpsTrial, sys_, params) -> MpsWalkerPlan:
    """The walker plan for a run: reference determinant chosen by params.plan_reference.

    "natural": the most occupied natural orbitals of trial.rdm1; 
    "rhf": the lowest eigenvectors of ham_data.h1. 
    Settings come from QmcParamsMps (orbital_plan, occupation_tolerance,
    walker_channel_chi, walker_cutoff).
    """
    _require_x64()
    if not isinstance(trial, MpsTrial):
        raise TypeError("trial must be an MpsTrial (see make_mps_trial / as_mps_trial)")
    if sys_.walker_kind != "unrestricted":
        raise ValueError("MPS-CPMC needs walker_kind='unrestricted'")
    nelec = (int(sys_.nelec[0]), int(sys_.nelec[1]))
    if trial.nelec != nelec or trial.norb != sys_.norb:
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


# ---------------------------------------------------------------------------------------------
# Contraction layouts, gathers and the overlap
# ---------------------------------------------------------------------------------------------


class ContractionLayout(NamedTuple):
    contraction: dict  # make_contraction_plan(walker charges, other charges)
    channel_maps: tuple  # make_channel_block_maps for the plan's channel labels
    gather: tuple  # per site (rows (T, pad_l), physical (T,), cols (T, pad_r)), int32


def _separable_gather(contraction, dims):
    out = []
    for site, layout in enumerate(contraction["sites"]):
        n_transitions = len(layout["src"])
        rows = np.full((n_transitions, contraction["trial_pad"][site]), dims[site], np.int32)
        cols = np.full(
            (n_transitions, contraction["trial_pad"][site + 1]), dims[site + 1], np.int32
        )
        for t, (qin, qout) in enumerate(zip(layout["src"], layout["dst"])):
            r = contraction["trial_index"][site][contraction["shared"][site][qin]]
            c = contraction["trial_index"][site + 1][contraction["shared"][site + 1][qout]]
            rows[t, : len(r)] = r
            cols[t, : len(c)] = c
        out.append((rows, np.asarray(layout["physical"], np.int32), cols))
    return tuple(out)


def contraction_layout(plan: MpsWalkerPlan, charges: tuple) -> ContractionLayout:
    """Static layout pairing the plan's walker labels with another MPS's labels (cached)."""
    key = ("layout", charges)
    layout = plan.caches.get(key)
    if layout is None:
        other = tuple(np.asarray(q, int).reshape(-1, 2) for q in charges)
        contraction = make_contraction_plan(plan.walker_charges, other)
        channel_maps = make_channel_block_maps(contraction, *plan.channel_charges)
        gather = _separable_gather(contraction, [len(q) for q in other])
        layout = ContractionLayout(contraction, channel_maps, gather)
        plan.caches[key] = layout
    return layout


def gather_blocks(tensors, gather):
    """Padded charge blocks of an MPS; the values equal extract_fixed_blocks'."""
    return tuple(
        jnp.asarray(A)
        .at[rows[:, :, None], phys[:, None, None], cols[:, None, :]]
        .get(mode="fill", fill_value=0.0)
        for A, (rows, phys, cols) in zip(tensors, gather)
    )


def convert_walker(walker, plan: MpsWalkerPlan):
    """Orthonormalise an SD walker and convert each spin channel to an MPS (static labels)."""
    ca, cb = walker
    for c in (ca, cb):
        if not jnp.issubdtype(jnp.result_type(c), jnp.floating):
            raise TypeError(
                "MPS-CPMC walkers must be real floating-point arrays (cpmc.init_prop_state "
                "applies jnp.real; do the same for state= or initial walkers)"
            )
    plan_a, plan_b = plan.orbital_plans
    bond_a, bond_b = plan.bond_plans
    qa, det_ra = qr_with_det(ca)
    qb, det_rb = qr_with_det(cb)
    alpha, qa_charge, gauge_a = channel_mps(qa, plan_a, bond_a)
    beta, qb_charge, gauge_b = channel_mps(qb, plan_b, bond_b)
    prefactor = det_ra * det_rb * gauge_a * gauge_b
    return alpha, qa_charge, beta, qb_charge, prefactor


def overlap_from_blocks(walker, trial_blocks, layout: ContractionLayout, plan: MpsWalkerPlan):
    alpha, _, beta, _, prefactor = convert_walker(walker, plan)
    walker_blocks = extract_channel_blocks(alpha, beta, layout.channel_maps)
    value = prefactor * blocked_contract_from_blocks(
        walker_blocks, trial_blocks, layout.contraction
    )
    return jnp.real(value)


def check_trial(trial, plan: MpsWalkerPlan) -> None:
    if not isinstance(trial, MpsTrial):
        raise TypeError(f"MPS-CPMC trial_data must be an MpsTrial, got {type(trial).__name__}")
    if trial.nelec != tuple(plan.nelec) or trial.norb != plan.norb:
        raise ValueError(
            f"trial (norb={trial.norb}, nelec={trial.nelec}) does not match the walker plan "
            f"(norb={plan.norb}, nelec={tuple(plan.nelec)})"
        )


def mps_overlap(walker, trial: MpsTrial, plan: MpsWalkerPlan):
    """<trial|walker> for one SD walker, a real float64 scalar."""
    check_trial(trial, plan)
    layout = contraction_layout(plan, trial.charges)
    return overlap_from_blocks(walker, gather_blocks(trial.tensors, layout.gather), layout, plan)


def mps_overlap_fn(plan: MpsWalkerPlan):
    """The jitted overlap of a plan, shared by its trial and measurement ops."""
    fn = plan.caches.get("overlap")
    if fn is None:
        fn = jax.jit(lambda walker, trial_data: mps_overlap(walker, trial_data, plan))
        plan.caches["overlap"] = fn
    return fn


def make_mps_trial_ops(plan: MpsWalkerPlan) -> TrialOps:
    """TrialOps for an MpsTrial: overlap and rdm1. CPMC fast updates live in the prop step."""
    return TrialOps(overlap=mps_overlap_fn(plan), get_rdm1=get_rdm1)
