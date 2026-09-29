"""Tests for trot/gmps/mps_cpmc_2d.py: the square-lattice hopping and general-h1 MPO.

The MPO is checked as an operator identity on 8 sites, against H|psi> built by
exact enumeration of all (alpha, beta) determinant pairs, so it is independent
of DMRG and of the chain MPO it generalises.
"""
from itertools import combinations

import numpy as np
import pytest

pytest.importorskip("pyblock3")

from trot.gmps import mps_cpmc_2d as s
from trot.gmps import mps_cpmc_new as m
from trot.lattices import TwoDimensionalGrid

L, N, U = 8, 4, 4.0


def _all_occupations():
    rows = np.array(list(combinations(range(L), N)))
    occ = np.zeros((len(rows), L), int)
    occ[np.arange(len(rows))[:, None], rows] = 1
    return occ


def _signed_amplitudes(tensors, occ):
    """<n_alpha, n_beta|MPS> times the reordering sign: all alpha modes before beta."""
    amp = np.empty((len(occ), len(occ)))
    for a, oa in enumerate(occ):
        for b, ob in enumerate(occ):
            v = np.ones((1, 1))
            for i, A in enumerate(tensors):
                v = v @ np.asarray(A)[:, oa[i] + 2 * ob[i], :]
            amp[a, b] = v[0, 0]
    lower = np.tril(np.ones((L, L), int), -1)
    return (1 - 2 * ((occ @ lower @ occ.T) & 1)) * amp


def _hopping_operators(occ):
    """<a|c^dag_i c_j|a'> for one spin species, by enumeration."""
    index = {tuple(o): k for k, o in enumerate(occ)}
    hop = np.zeros((L, L, len(occ), len(occ)))
    for k, o in enumerate(occ):
        for j in np.flatnonzero(o):
            removed = o.copy(); removed[j] = 0
            for i in np.flatnonzero(removed == 0):
                added = removed.copy(); added[i] = 1
                hop[i, j, index[tuple(added)], k] = (-1) ** (o[:j].sum() + removed[:i].sum())
    return hop


@pytest.fixture(scope="module")
def state():
    """A random 8-site MPS in the (4, 4) sector, with its enumerated amplitudes."""
    cfg = s.Config(Lx=4, Ly=2, n_up=N, n_down=N, interaction=U)
    np.random.seed(7)
    mps = s.build_dmrg_hamiltonian(cfg, s.lattice_hopping(cfg)).build_mps(12)
    tensors, _ = m.densify_with_charges(mps, L)
    occ = _all_occupations()
    return tensors, occ, _signed_amplitudes(tensors, occ), _hopping_operators(occ)


def test_chain_limit_matches_mps_cpmc_new():
    for t in (1.0, 0.7):
        np.testing.assert_array_equal(s.square_hopping_matrix(L, 1, t), m.hopping_matrix(L, t))
        np.testing.assert_array_equal(s.hubbard_mpo_from_h1(m.hopping_matrix(L, t), U),
                                      m.hubbard_mpo(L, t, U))


@pytest.mark.parametrize("Lx,Ly,ours,theirs", [
    (4, 2, "open", "obc"), (4, 3, "open", "obc"), (4, 3, "periodic", "pbc")])
def test_square_lattice_matches_trot_grid(Lx, Ly, ours, theirs):
    """TwoDimensionalGrid(l_x=Ly, l_y=Lx) numbers sites x*Ly + y too. Its pbc does not
    double a length-2 wrap bond, so no periodic side of length 2 is compared."""
    adjacency = TwoDimensionalGrid(l_x=Ly, l_y=Lx, boundary=theirs).create_adjacency_matrix()
    np.testing.assert_array_equal(-s.square_hopping_matrix(Lx, Ly, 1.0, ours, ours), adjacency)


def test_boundary_signs():
    periodic = s.square_hopping_matrix(4, 3, 1.0, "periodic", "open")
    anti = s.square_hopping_matrix(4, 3, 1.0, "antiperiodic", "open")
    wrap = (0, 9)  # (x=0, y=0) and (x=3, y=0)
    assert periodic[wrap] == -1.0 and anti[wrap] == 1.0
    assert s.square_hopping_matrix(2, 1, 1.0, "periodic")[0, 1] == -2.0


def _random_h1():
    rng = np.random.default_rng(3)
    h1 = rng.normal(size=(L, L))
    return h1 + h1.T


@pytest.mark.parametrize("h1", [
    s.square_hopping_matrix(4, 2, 1.0),
    s.square_hopping_matrix(4, 2, 1.0, "periodic", "periodic"),
    s.square_hopping_matrix(4, 2, 1.0, "antiperiodic", "open"),
    s.square_hopping_matrix(2, 4, 1.0, "periodic", "antiperiodic"),
    _random_h1(),
], ids=["4x2-open", "4x2-periodic", "4x2-antiperiodic-x", "2x4-mixed", "random-dense"])
def test_mpo_operator_identity(state, h1):
    """MPO|psi> equals sum_ij h_ij c^dag_i c_j |psi> + U sum_i n_i,up n_i,dn |psi>."""
    tensors, occ, amp, hop = state
    expected = np.einsum("ij,ijab,bc->ac", h1, hop, amp) + np.einsum("ij,ijab,cb->ca", h1, hop, amp)
    expected += U * (occ @ occ.T) * amp
    applied = m.compress_mps(s.apply_mpo(s.hubbard_mpo_from_h1(h1, U), tensors))
    got = _signed_amplitudes(applied, occ)
    np.testing.assert_allclose(got, expected, atol=1e-10 * np.abs(expected).max())


@pytest.mark.parametrize("boundary,walker_chi", [("open", None), ("open", 4), ("periodic", 8)])
def test_blocked_energy_matches_dense(boundary, walker_chi):
    """The charge-blocked local energy against make_walker_ops' dense one, walker by walker."""
    import jax.numpy as jnp
    cfg = s.Config(Lx=4, Ly=2, boundary_x=boundary, boundary_y=boundary, n_up=N, n_down=N,
                   interaction=U, trial_chi=16, dmrg_sweeps=8)
    h1 = s.lattice_hopping(cfg)
    trial_np, trial_charges, *_ = s.densified_trial(cfg, h1)
    Htrial_np, Htrial_charges = s.trial_times_h(s.hubbard_mpo_from_h1(h1, U), trial_np, trial_charges)
    Na, _ = m.natural_orbitals(m.one_rdm(trial_np)[0], N)
    Nb, _ = m.natural_orbitals(m.one_rdm(trial_np)[1], N)
    plans = [m.make_orbital_plan(C, "rank_exact") for C in (Na, Nb)]
    bonds = [None if walker_chi is None else m.plan_bonds(C, p, walker_chi) for C, p in zip((Na, Nb), plans)]
    dense = tuple(map(jnp.asarray, m.compress_mps(Htrial_np)))
    ops = m.make_walker_ops(Na, Nb, *plans, *bonds, trial_np, trial_charges, dense)
    blocked = s.make_blocked_energy(ops, Na, Nb, trial_np, Htrial_np, Htrial_charges)
    rng = np.random.default_rng(1)
    for _ in range(4):
        walker = tuple(jnp.asarray(np.linalg.qr(C + 0.3 * rng.normal(size=C.shape))[0]) for C in (Na, Nb))
        np.testing.assert_allclose(float(blocked(walker)), float(ops.energy(walker)), rtol=1e-10)
