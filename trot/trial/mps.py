"""MPS trial wave functions for CPMC with Slater-determinant walkers (Hubbard model).

The trial (MpsTrial, trial_data) is a dense, charge-labelled d=4 MPS in the local basis |0>, |up>, |dn>, |up dn>,
indexed n_up + 2*n_dn, with |up dn> = c+_up c+_dn |0> and fermion operators ordered site by site with up before
down. Its bond labels, counted from the left, are (N_up, N_dn) pairs when it has the walkers' definite (N_up, N_dn),
and particle numbers N when it has not (a spin-rotated MPS used as it is): the walkers are S_z eigenstates and the
Hubbard Hamiltonian conserves S_z, so <T|W> and <T|H|W> pick the walkers' sector by themselves, and the CPMC
trajectory is that of the trial projected onto it. make_mps_trial reads the labels off the tensors when none are
given; it never projects. rotate_mps_trial (trot.trial.mps_rotation) rotates a trial straight into one sector.

Walkers stay Slater determinants. The walker plan (make_walker_plan) freezes the gate circuit that converts each
spin channel to a charge-labelled Gaussian MPS (Fishman-White, trot.gmps.utils) on one reference determinant; the
batched engine of the plan (trot.gmps.engine, cached on the plan) does the conversions and the charge-blocked
contractions. make_mps_trial_ops gives trot's TrialOps; trot.meas.mps and trot.prop.mps_cpmc give the measurement
and propagation ops on the same plan.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from trot.core.ops import TrialOps
from trot.gmps import engine
from trot.gmps.utils import (
    PHYSICAL_CHARGE,
    combined_charges,
    label_array,
    make_orbital_plan,
    number_labels,
    plan_bonds,
    sd_to_gmps,
    sz_labels,
)


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



def _label_sectors(row_labels, column_labels):
    rows, columns = _charge_index(row_labels), _charge_index(column_labels)
    return [(c, rows[c], columns[c]) for c in sorted(set(rows) & set(columns))]



def physical_charge(width: int) -> np.ndarray:
    """What each local state |0>, |up>, |dn>, |up dn> adds to a bond label of the given width:
    (n_up, n_dn) for (N_up, N_dn) labels, n_up + n_dn for N labels."""
    if width == 2:
        return PHYSICAL_CHARGE
    if width == 1:
        return PHYSICAL_CHARGE.sum(axis=1, keepdims=True)
    raise ValueError(f"bond labels must be (N_up, N_dn) pairs or particle numbers N, got width {width}")


def compress_mps_qn(tensors, charges, relative_tolerance=1.0e-13):
    """compress_mps that keeps the bond labels: (N_alpha, N_beta) pairs, or particle numbers N
    (labels of width 1, as for a spin-rotated MPS without definite S_z).

    The QR sweep and the SVD sweep factor each charge sector separately and the
    rank cut is one relative tolerance per bond applied per sector, so no
    factorisation mixes charges (the dense compress can, at degenerate singular
    values) and the result can be contracted charge-blocked like the trial.

    The R factors (QR sweep) and U S factors (SVD sweep) are block diagonal in the
    charge, so each is multiplied into the neighbouring tensor one sector at a time,
    on that sector's rows or columns only. A dense product (the earlier code) gives
    the same numbers up to summation order and costs about one factor of the number
    of sectors more: minutes on the host at bond 18 x 256 (4x4 lattice).
    """
    A = [np.array(t, dtype=float, copy=True) for t in tensors]
    Q = [label_array(q) for q in charges]
    width = Q[0].shape[1]
    physical = physical_charge(width)
    for i in range(len(A) - 1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl * d, Dr)
        rows = (Q[i][:, None, :] + physical[None]).reshape(-1, width)
        factors = [(c, r, k, *np.linalg.qr(M[np.ix_(r, k)])) for c, r, k in _label_sectors(rows, Q[i + 1])]
        new = sum(q.shape[1] for *_, q, _ in factors)
        following = A[i + 1].reshape(Dr, -1)
        left, right, labels, start = np.zeros((Dl * d, new)), np.zeros((new, following.shape[1])), [], 0
        for c, r, k, q, rr in factors:
            m = q.shape[1]
            left[r, start:start + m] = q
            right[start:start + m] = rr @ following[k]
            labels += [c] * m
            start += m
        A[i] = left.reshape(Dl, d, new)
        A[i + 1] = right.reshape((new,) + A[i + 1].shape[1:])
        Q[i + 1] = np.asarray(labels, int).reshape(-1, width)
    for i in range(len(A) - 1, 0, -1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl, d * Dr)
        columns = (Q[i + 1][None, :, :] - physical[:, None, :]).reshape(-1, width)
        factors = []
        for c, r, k in _label_sectors(Q[i], columns):
            u, s, vh = np.linalg.svd(M[np.ix_(r, k)], full_matrices=False)
            factors.append((c, r, k, u, s, vh))
        cut = relative_tolerance * max(max(f[4][0] for f in factors), 1e-300)
        kept = [(c, r, k, u, s, vh, int(np.sum(s > cut))) for c, r, k, u, s, vh in factors]
        kept = [f for f in kept if f[-1] > 0]
        new = sum(f[-1] for f in kept)
        preceding = A[i - 1].reshape(-1, Dl)
        left, right, labels, start = np.zeros((preceding.shape[0], new)), np.zeros((new, d * Dr)), [], 0
        for c, r, k, u, s, vh, n in kept:
            right[start:start + n, k] = vh[:n]
            left[:, start:start + n] = preceding[:, r] @ (u[:, :n] * s[:n])
            labels += [c] * n
            start += n
        A[i] = right.reshape(new, d, Dr)
        A[i - 1] = left.reshape(A[i - 1].shape[:2] + (new,))
        Q[i] = np.asarray(labels, int).reshape(-1, width)
    return A, tuple(Q)


# ---------------------------------------------------------------------------------------------
# Labels and spin rotations
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

    The result conserves the total particle number but in general not N_up and N_dn: make_mps_trial uses it as it
    is, with particle-number labels, and trot.trial.mps_rotation.rotate_mps_trial rotates a trial straight into
    one (N_up, N_dn) sector. Note that
    np.einsum('ijk,jl', A, M) applies M^T instead (wrong unless M is symmetric), and that a
    4x4 matrix without the det(R) entry on |up dn> is not a fermionic spin rotation.
    """
    M = spin_rotation_unitary(R)
    return [np.einsum("ps,asb->apb", M, np.asarray(A, dtype=float)) for A in tensors]


def spin_rotation_y(beta_deg) -> np.ndarray:
    """R for rotate_spin of exp(-i beta S^y): |up> -> cos(beta/2)|up> + sin(beta/2)|dn> (beta in degrees).

    Projected onto an S_z sector, a rotation by beta multiplies the spin-S part of an S_z = 0 state by
    P_S(cos beta): beta = 90 removes every odd total spin (trot/gmps/notes/spin_projection.tex).
    """
    b = np.deg2rad(float(beta_deg)) / 2.0
    return np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])



# ---------------------------------------------------------------------------------------------
# The trial
# ---------------------------------------------------------------------------------------------


def _hashable_charges(charges) -> tuple:
    return tuple(tuple(tuple(row) for row in label_array(q).tolist()) for q in charges)


@tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class MpsTrial:
    """Charge-labelled d=4 MPS trial (trial_data for MPS-CPMC).

    tensors: L site tensors (D_left, 4, D_right), local index n_up + 2*n_dn, normalised (all sectors together).
    rdm1: (2, L, L) spin-diagonal blocks <c+_i,s c_j,s> of the trial's 1-RDM (before any projection for a rotated
      trial); trot starts walkers from its natural orbitals.
    charges: static aux, L+1 bond labels: tuples of (N_up, N_dn) pairs for a trial in the walkers' sector, of
      particle numbers (N,) for a trial without a definite (N_up, N_dn), used as it is.
    nelec: static aux, the walkers' (N_up, N_dn).
    sector_weight: static aux, <P T|P T>/<T|T> of the walkers' sector when known: 1.0 for (N_up, N_dn) input, the
      kept weight for rotate_mps_trial, None for a trial used as it is (overlaps are then sqrt(sector_weight) times
      those of the projected trial, local energies the same).
    """

    tensors: tuple
    rdm1: Any
    charges: tuple
    nelec: tuple
    sector_weight: float | None = 1.0

    @property
    def norb(self) -> int:
        return len(self.tensors)

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return tuple(len(q) for q in self.charges)

    @property
    def label_width(self) -> int:
        """2 for (N_up, N_dn) labels, 1 for particle-number labels."""
        return len(self.charges[0][0])

    def charge_arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(q, int).reshape(-1, self.label_width) for q in self.charges)

    def tree_flatten(self):
        return (tuple(self.tensors), self.rdm1), (self.charges, self.nelec, self.sector_weight)

    @classmethod
    def tree_unflatten(cls, aux, children):
        tensors, rdm1 = children
        charges, nelec, sector_weight = aux
        return cls(tensors=tuple(tensors), rdm1=rdm1, charges=charges, nelec=nelec, sector_weight=sector_weight)


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
    """Build trial_data from a real d=4 MPS; nothing is projected or truncated.

    tensors: L arrays (D_left, 4, D_right) in the local basis described in the module docstring.
    charges: optional L+1 bond labels. (N_up, N_dn) labels that the tensors respect and that end at nelec are used
      as they are; otherwise (N_up, N_dn) labels are read off the nonzero entries, and when the MPS has no definite
      (N_up, N_dn) (a spin-rotated MPS), particle-number labels: the trial is then used as it is.
    nelec: (N_up, N_dn) of the walkers.
    rdm1: optional (2, L, L); defaults to one_rdm of the tensors.
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
    if charges is not None and _valid_sz_labels(A, charges, (nup, ndn)):
        labels, weight = tuple(label_array(q) for q in charges), 1.0
    else:
        labels, weight = sz_labels(A), 1.0
        if labels is not None and tuple(labels[-1][0]) != (nup, ndn):
            raise ValueError(
                f"the trial has (N_up, N_dn) = {tuple(int(x) for x in labels[-1][0])} and the walkers {(nup, ndn)}: "
                "their overlaps vanish (rotate it with trot.trial.mps_rotation.rotate_mps_trial)"
            )
        if labels is None:
            labels, weight = number_labels(A), None
            if labels is None:
                raise ValueError("the MPS mixes particle numbers at some bond, so it has no labels to block on")
            if int(labels[-1][0, 0]) != nup + ndn:
                raise ValueError(f"the trial has N = {int(labels[-1][0, 0])} and the walkers {nup + ndn}")
    return MpsTrial(
        tensors=tuple(jnp.asarray(t) for t in A),
        rdm1=jnp.asarray(rdm1),
        charges=_hashable_charges(labels),
        nelec=(nup, ndn),
        sector_weight=weight,
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
    walker_charges are the resulting static bond labels. sector_buckets and walker_qr set how the
    batched engine factors the sector blocks and orthonormalises the walkers (QmcParamsMps). Lives
    in the ops closures (never a jit argument). caches holds the plan's engine (trot.gmps.engine:
    converter, layouts, kernels, jitted functions); it is freed with the plan.
    """

    orbital_plans: tuple
    bond_plans: tuple
    channel_charges: tuple
    walker_charges: tuple
    reference: tuple
    nelec: tuple
    norb: int
    sector_buckets: tuple = ()
    walker_qr: str = "auto"
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
    sector_buckets=(),
    walker_qr="auto",
) -> MpsWalkerPlan:
    """Freeze the gate circuit and the per-sector kept counts on the determinant (Ra, Rb)."""
    engine.resolve_walker_qr(walker_qr)  # validates
    buckets = tuple(int(b) for b in sector_buckets)
    if any(b < 1 for b in buckets) or list(buckets) != sorted(set(buckets)):
        raise ValueError(f"sector_buckets must be increasing positive sizes, got {sector_buckets!r}")
    Ra, Rb = np.asarray(Ra, dtype=float), np.asarray(Rb, dtype=float)
    if Ra.shape[1] == 0 or Rb.shape[1] == 0:
        raise ValueError("MPS walkers need at least one electron of each spin")
    plan_a = make_orbital_plan(Ra, orbital_plan, occupation_tolerance)
    plan_b = make_orbital_plan(Rb, orbital_plan, occupation_tolerance)
    bond_a = bond_b = None
    if walker_channel_chi is not None or walker_cutoff:
        bond_a = plan_bonds(Ra, plan_a, walker_channel_chi, walker_cutoff)
        bond_b = plan_bonds(Rb, plan_b, walker_channel_chi, walker_cutoff)
    # the walker labels are those of the compiled conversion circuit (host bookkeeping, no numerics); the circuit is
    # what the plan's engine runs, so it goes into the plan's cache
    converter = engine.make_converter(plan_a, plan_b, bond_a, bond_b, spin_batch=True, buckets=buckets or None)
    qa_charge, qb_charge = converter.charges
    plan = MpsWalkerPlan(
        orbital_plans=(plan_a, plan_b),
        bond_plans=(bond_a, bond_b),
        channel_charges=(tuple(qa_charge), tuple(qb_charge)),
        walker_charges=combined_charges(qa_charge, qb_charge),
        reference=(Ra, Rb),
        nelec=(Ra.shape[1], Rb.shape[1]),
        norb=Ra.shape[0],
        sector_buckets=buckets,
        walker_qr=walker_qr,
    )
    plan.caches["converter"] = converter
    return plan


def make_walker_plan(ham_data, trial: MpsTrial, sys_, params) -> MpsWalkerPlan:
    """The walker plan for a run: reference determinant chosen by params.plan_reference.

    "natural": the most occupied natural orbitals of trial.rdm1;
    "rhf": the lowest eigenvectors of ham_data.h1.
    Settings come from QmcParamsMps (orbital_plan, occupation_tolerance, walker_channel_chi,
    walker_cutoff, sector_buckets, walker_qr).
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
        sector_buckets=getattr(params, "sector_buckets", ()),
        walker_qr=getattr(params, "walker_qr", "auto"),
    )


# ---------------------------------------------------------------------------------------------
# The overlap and the trial ops
# ---------------------------------------------------------------------------------------------


def check_trial(trial, plan: MpsWalkerPlan) -> None:
    if not isinstance(trial, MpsTrial):
        raise TypeError(f"MPS-CPMC trial_data must be an MpsTrial, got {type(trial).__name__}")
    if tuple(trial.nelec) != tuple(plan.nelec) or trial.norb != plan.norb:
        raise ValueError(
            f"trial (norb={trial.norb}, nelec={trial.nelec}) does not match the walker plan "
            f"(norb={plan.norb}, nelec={tuple(plan.nelec)})"
        )


def check_walker(walker) -> None:
    for c in walker:
        if not jnp.issubdtype(jnp.result_type(c), jnp.floating):
            raise TypeError(
                "MPS-CPMC walkers must be real floating-point arrays (init_prop_state applies jnp.real; "
                "do the same for state= or initial walkers)"
            )


def mps_overlap(walker, trial: MpsTrial, plan: MpsWalkerPlan):
    """<trial|walker> for one SD walker, a real float64 scalar: the plan's engine with the trial's blocks
    gathered from trial.tensors."""
    # check_trial(trial, plan)
    # check_walker(walker)
    kernels = engine.kernels_for(plan, trial.charges, energy=None)
    data = engine.DeviceData(engine.fixed_blocks(trial.tensors, kernels.overlap_plan), (), (), None, None)
    return kernels.overlap_one(walker[0], walker[1], data)


def mps_overlap_fn(plan: MpsWalkerPlan):
    """The jitted overlap of a plan, shared by its trial and measurement ops."""
    fn = plan.caches.get("overlap")
    if fn is None:
        fn = jax.jit(lambda walker, trial_data: mps_overlap(walker, trial_data, plan))
        plan.caches["overlap"] = fn
    return fn


def make_mps_trial_ops(plan: MpsWalkerPlan) -> TrialOps:
    """TrialOps for an MpsTrial: overlap and rdm1. CPMC updates live in the prop step (trot.prop.mps_cpmc)."""
    return TrialOps(overlap=mps_overlap_fn(plan), get_rdm1=get_rdm1)
