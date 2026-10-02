"""UHF trial for trot's CPMC on the Hubbard model: the SCF of uhf_trial_cpmc.ipynb (and an RHF one) and fast-update ops.

trot.prop.cpmc propagates with fast updates (one Green's function per walker and step, then per site
the determinant-lemma overlap ratio and a Sherman-Morrison update) but needs the trial's calc_green,
calc_overlap_ratio and update_green, which trot.trial.uhf does not provide (only the GHF trials do).
The ones here are trot.trial.ghf's for the trial whose coefficients are spin-block diagonal: its G is
block diagonal, so it is kept as its two blocks, (2, norb, norb), and the rank-2 update of a Hubbard
site (one up and one down diagonal entry) is one rank-1 update per spin, with half the memory and the
flops of the (2 norb, 2 norb) GHF matrix. The guard against a vanishing ratio and the replacement of
non-finite entries are the GHF ones.

Conventions as trot.trial.ghf: G_s = (W_s (C_s^+ W_s)^-1 C_s^+)^T for spin s; update_indices =
[[0, i], [1, j]] (spin, site: an up update at site i and a down update at site j, which is what
trot.prop.cpmc passes, with i = j); update_constants = (c_up - 1, c_dn - 1).
"""
import jax
import jax.numpy as jnp
import numpy as np

from trot.core.ops import TrialOps
from trot.core.system import System
from trot.trial.uhf import UhfTrial, make_uhf_trial_ops


def uhf_scf(h1, u, na, nb, guess=0.3, mix=0.5, tol=1e-12, iters=5000):
    """Self-consistent UHF, F_s = h1 + u diag(n_-s), from a Neel guess, with density mixing."""
    n = h1.shape[0]
    stag = (-1.0) ** np.arange(n)
    da, db = na / n + guess * stag, nb / n - guess * stag
    for it in range(iters):
        _, va = np.linalg.eigh(h1 + u * np.diag(db))
        _, vb = np.linalg.eigh(h1 + u * np.diag(da))
        Ca, Cb = va[:, :na], vb[:, :nb]
        new_a, new_b = np.einsum("ik,ik->i", Ca, Ca), np.einsum("ik,ik->i", Cb, Cb)
        change = max(abs(new_a - da).max(), abs(new_b - db).max())
        da, db = (1 - mix) * da + mix * new_a, (1 - mix) * db + mix * new_b
        if change < tol:
            break
    Pa, Pb = Ca @ Ca.T, Cb @ Cb.T
    energy = np.sum(h1 * (Pa + Pb)) + u * np.diag(Pa) @ np.diag(Pb)
    return Ca, Cb, energy, it, change


def rhf_scf(h1, u, n, mix=0.5, tol=1e-12, iters=5000):
    """Self-consistent RHF (the same n orbitals for both spins), F = h1 + u diag(n_s), from a uniform density.

    At half filling on a bipartite lattice the density stays uniform, so this is the free-electron determinant.
    """
    m = h1.shape[0]
    d = np.full(m, n / m)
    for it in range(iters):
        _, v = np.linalg.eigh(h1 + u * np.diag(d))
        C = v[:, :n]
        new = np.einsum("ik,ik->i", C, C)
        change = abs(new - d).max()
        d = (1 - mix) * d + mix * new
        if change < tol:
            break
    P = C @ C.T
    energy = 2 * np.sum(h1 * P) + u * np.diag(P) @ np.diag(P)
    return C, energy, it, change


def calc_green(walker: tuple[jax.Array, jax.Array], trial_data: UhfTrial) -> jax.Array:
    """G of an unrestricted walker, (2, norb, norb): the diagonal blocks of trot.trial.ghf.calc_green_u."""
    def block(w, c):
        x = jnp.linalg.solve(c.conj().T @ w, c.conj().T)  # (nocc, norb)
        return (w @ x).T

    wu, wd = walker
    return jnp.stack([block(wu, trial_data.mo_coeff_a), block(wd, trial_data.mo_coeff_b)])


def calc_overlap_ratio(greens: jax.Array, update_indices: jax.Array, update_constants: jax.Array) -> jax.Array:
    """<T|W'>/<T|W> for row i of the up and row j of the down determinant scaled by 1 + u0 and 1 + u1."""
    i, j = update_indices[0, 1], update_indices[1, 1]
    u0, u1 = update_constants[0], update_constants[1]
    return (1.0 + u0 * greens[0, i, i]) * (1.0 + u1 * greens[1, j, j])


def update_green(greens: jax.Array, update_indices: jax.Array, update_constants: jax.Array,
                 *, eps: float = 1.0e-8) -> jax.Array:
    """G after that update: trot.trial.ghf._update_full_rank2 on the block-diagonal G, block by block."""
    i, j = update_indices[0, 1], update_indices[1, 1]
    u0, u1 = update_constants[0], update_constants[1]
    g_up, g_dn = greens[0], greens[1]
    d_up, d_dn = 1.0 + u0 * g_up[i, i], 1.0 + u1 * g_dn[j, j]
    r = d_up * d_dn
    r_safe = jnp.where(jnp.abs(r) < eps, jnp.asarray(1.0, dtype=r.dtype), r)
    # GHF: G + (u0/r) col_i (x) term_i + (u1/r) col_j (x) term_j with term_i = -d_dn (G[i] - e_i), term_j = -d_up (G[j] - e_j)
    new_up = g_up - (u0 * d_dn / r_safe) * jnp.outer(g_up[:, i], g_up[i].at[i].add(-1))
    new_dn = g_dn - (u1 * d_up / r_safe) * jnp.outer(g_dn[:, j], g_dn[j].at[j].add(-1))
    new = jnp.stack([new_up, new_dn])
    return jnp.where(jnp.isfinite(new), new, jnp.asarray(0.0, dtype=new.dtype))


def make_uhf_cpmc_trial_ops(sys: System) -> TrialOps:
    """trot's UHF trial ops (overlap, rdm1) with the fast-update ops, for trot.prop.cpmc."""
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError(f"the UHF fast-update ops need unrestricted walkers, not {sys.walker_kind}")
    return make_uhf_trial_ops(sys)._replace(calc_green=calc_green, calc_overlap_ratio=calc_overlap_ratio,
                                            update_green=update_green)
