"""Tests for the mixed pairing correlations of MPS-CPMC (trot.meas.mps_pairing, trot.gmps.engine pair kernels).

References are exact Fock-space vectors (tests/helpers/hubbard_fock.py: annihilate_full, pair_singlet_full), applied to
the walker and the trial literally: P_a[j, l] = <Delta_a^dag(j) T | Delta_a(l) W> / <T|W> with Delta built here from
the neighbours of every site (independently of bond_structure), plus trot's GHF Wick contractions for determinants.
"""

from trot import config

config.configure_once()

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers import hubbard_fock as hf
from trot.meas import mps_pairing as mp

U = 4.0


def _random_mps(L, rng, bond=3):
    bonds = [1, *([bond] * (L - 1)), 1]
    return [rng.standard_normal((bonds[i], 4, bonds[i + 1])) for i in range(L)]


def _neighbours(spec):
    """(l, m, direction axis, wrap sign) of every delta of every site, written independently of bond_structure."""
    out = []
    wrap = {"open": 0, "periodic": 1, "antiperiodic": -1}
    for x, y in itertools.product(range(spec.Lx), range(spec.Ly)):
        for axis, step in itertools.product((0, 1), (1, -1)):
            nx, ny = (x + step, y) if axis == 0 else (x, y + step)
            L_axis, boundary = (spec.Lx, spec.boundary_x) if axis == 0 else (spec.Ly, spec.boundary_y)
            c = nx if axis == 0 else ny
            sign = 1
            if not 0 <= c < L_axis:
                if wrap[boundary] == 0:
                    continue
                sign = wrap[boundary]
                nx, ny = (nx % spec.Lx, ny) if axis == 0 else (nx, ny % spec.Ly)
            m = nx * spec.Ly + ny
            if m != x * spec.Ly + y:
                out.append((x * spec.Ly + y, m, axis, sign))
    return out


def delta_full(vector, spec, l, kind):
    """Delta_kind(l)|vector> = sum_delta f(delta) [c_{l up} c_{m dn} - c_{l dn} c_{m up}]|vector>, term by term."""
    L, out = spec.n_sites, 0.0
    for a, m, axis, sign in _neighbours(spec):
        if a != l:
            continue
        f = 1.0 if kind == "s" else (1.0 if axis == 0 else -1.0)
        term = (hf.annihilate_full(hf.annihilate_full(vector, L, m, 1), L, l, 0)
                - hf.annihilate_full(hf.annihilate_full(vector, L, m, 0), L, l, 1))
        out = out + sign * f * term
    return out


def pairing_by_enumeration(trial_vec, walker_vec, spec):
    """(2, L, L): P_a[j, l] = <Delta_a(j) T | Delta_a(l) W> / <T|W>, applying the operators literally."""
    L = spec.n_sites
    P = np.zeros((2, L, L))
    for a, kind in enumerate(("s", "d")):
        bra = [delta_full(trial_vec, spec, j, kind) for j in range(L)]
        ket = [delta_full(walker_vec, spec, l, kind) for l in range(L)]
        P[a] = np.array([[bra[j] @ ket[l] for l in range(L)] for j in range(L)]) / (trial_vec @ walker_vec)
    return P


@pytest.mark.parametrize("u, v", [(0, 1), (1, 3), (0, 4), (2, 4)])
@pytest.mark.parametrize("s_u, s_v", [(0, 1), (1, 0), (0, 0), (1, 1)])
def test_pair_annihilate_is_the_product_of_two_fermion_operators(u, v, s_u, s_v):
    """c_u c_v on a random d=4 MPS equals the Fock-space c_u c_v (v acts first), for every spin combination."""
    L = 5
    tensors = _random_mps(L, np.random.default_rng(u * 7 + v))
    vec = hf.mps_full_vector(tensors)
    want = hf.annihilate_full(hf.annihilate_full(vec, L, v, s_v), L, u, s_u)
    got = hf.mps_full_vector([np.asarray(t) for t in mp.pair_annihilate(tensors, u, s_u, v, s_v, xp=np)])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_singlet_signs_and_symmetry():
    """S_uv = c_{u up} c_{v dn} - c_{u dn} c_{v up} is symmetric in u, v; the two terms of singlet_terms add to it."""
    L = 4
    tensors = _random_mps(L, np.random.default_rng(1))
    vec = hf.mps_full_vector(tensors)
    S = hf.pair_singlet_full(vec, L, 1, 3)
    np.testing.assert_allclose(hf.pair_singlet_full(vec, L, 3, 1), S, rtol=0, atol=1e-12)
    (terms,) = mp.singlet_terms(tensors, [(1, 3)], xp=np)
    got = sum(hf.mps_full_vector([np.asarray(t) for t in term]) for term in terms)
    np.testing.assert_allclose(got, S, rtol=0, atol=1e-12)
    assert np.abs(S).max() > 1e-3


@pytest.mark.parametrize("boundaries", [("open", "open"), ("periodic", "open"), ("open", "periodic"),
                                        ("periodic", "periodic"), ("antiperiodic", "open")])
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (2, 2)])
def test_bond_structure_reproduces_the_literal_pair_operators(shape, boundaries):
    """Delta_a(l) = sum_b G_a[l, b] S_b on a random state, for every lattice and boundary, doubled bonds included."""
    spec = mp.PairingSpec(*shape, *boundaries)
    L = spec.n_sites
    vec = np.random.default_rng(5).standard_normal(4**L)
    bonds, G = mp.bond_structure(spec)
    for a, kind in enumerate(("s", "d")):
        for l in range(L):
            want = delta_full(vec, spec, l, kind)
            got = sum(G[a, l, k] * hf.pair_singlet_full(vec, L, u, v) for k, (u, v) in enumerate(bonds))
            np.testing.assert_allclose(got, want if np.ndim(want) else np.zeros_like(got), rtol=0, atol=1e-12)


@pytest.mark.parametrize("boundaries", [("open", "open"), ("periodic", "open")])
def test_pairing_dense_matches_literal_enumeration(boundaries):
    """The dense kernel (d=4 walker MPS against the dense trial) equals <Delta^dag(j) T|Delta(l) W>/<T|W> evaluated
    literally on Fock vectors, for the exact ground state of a 2x3 lattice and a determinant walker."""
    spec = mp.PairingSpec(2, 3, *boundaries)
    h1 = hf  # placeholder to keep the module import used below
    from trot.ham.hubbard import square_hopping_matrix

    h1 = square_hopping_matrix(2, 3, 1.0, *boundaries)
    L, nelec = 6, (3, 3)
    _, psi = hf.ground_state(h1, U, *nelec)
    tensors, _ = hf.exact_mps_from_sector_state(psi, L, *nelec)
    trial_vec = hf.mps_full_vector(tensors)
    Ca, Cb = hf.staggered_determinant(h1, *nelec, field=0.3)
    Wa, Wb = hf.random_field_walkers(h1, U, 0.1, Ca, Cb, n=2, steps=6, seed=3)[1]
    walker_vec = hf.full_from_sector(hf.sd_amplitudes(Wa, Wb), L, *nelec)
    # the walker as a d=4 MPS: the exact MPS of its Fock vector
    walker_tensors, _ = hf.exact_mps_from_sector_state(hf.sd_amplitudes(Wa, Wb), L, *nelec)
    want = pairing_by_enumeration(trial_vec, walker_vec, spec)
    got = np.asarray(mp.pairing_dense([jnp.asarray(t) for t in walker_tensors], [jnp.asarray(t) for t in tensors], spec))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-11)
    assert np.abs(want).max() > 1e-2


# ---------------------------------------------------------------------------------------------
# The blocked and dense MPS-CPMC kernels
# ---------------------------------------------------------------------------------------------

import jax

from tests.test_mps_meas import (  # noqa: F401,E402  (the shared fixtures and builders)
    ENERGY_CASES,
    KERNELS,
    case,
    ed_2x3,
    rotated_sd_chain,
    sd_chain,
)
from trot.meas.ghf import calc_green_u
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.trial.ghf import GhfTrial


def _spec(case):
    """The lattice of a shared case: the 2x3 ED case is periodic in x (doubled rungs), the others are the 6-site chain."""
    return mp.PairingSpec(2, 3, "periodic", "open") if case.E0 is not None else mp.PairingSpec(1, 6)


def _pairing(case, plan_kind, kernel, spec=None):
    """(n_walkers, 2, n_rows, L) from the per-walker observable of ops built with the pairing spec."""
    spec = spec or _spec(case)
    ops = make_mps_meas_ops_hubbard(case.plans[plan_kind], energy_kernel=kernel, pairing=spec)
    ctx = ops.build_meas_ctx(case.ham, case.trial)
    observable = ops.observables["pairing"]
    return np.array([np.asarray(observable(w, case.ham, ctx, case.trial)) for w in case.walkers]), ops, ctx


def _enumerated(case, spec=None):
    spec = spec or _spec(case)
    trial_vec = hf.mps_full_vector(case.trial.tensors)
    return np.array([
        pairing_by_enumeration(trial_vec, hf.full_from_sector(hf.sd_amplitudes(np.asarray(wa), np.asarray(wb)),
                                                              spec.n_sites, *case.trial.nelec), spec)
        for wa, wb in case.walkers])


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_pairing_matches_enumeration(case, kernel):
    """With exact walkers the observable is <Delta^dag(j) T|Delta(l) W>/<T|W> evaluated literally on Fock vectors, for
    the exact ground state of a lattice with doubled periodic bonds, a determinant and the rotated (N-labelled) one."""
    got, _, _ = _pairing(case, "exact", kernel)
    want = _enumerated(case)
    assert got.shape == want.shape == (len(case.walkers), 2, case.trial.norb, case.trial.norb)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)
    assert np.abs(want).max() > 1e-2 and np.abs(want - want.mean(axis=0)).max() > 1e-3


@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_truncated_walkers_blocked_equals_dense(case):
    """With chi=2 adaptive walker channels the blocked and dense kernels agree, and truncation is visible."""
    blocked = _pairing(case, "truncated", "blocked")[0]
    np.testing.assert_allclose(blocked, _pairing(case, "truncated", "dense")[0], rtol=0, atol=1e-10)
    if case.E0 is None:
        assert np.abs(blocked - _pairing(case, "exact", "blocked")[0]).max() > 1e-6


@pytest.mark.parametrize("case", ["sd_chain", "rotated_sd_chain"], indirect=True)
def test_pairing_matches_ghf_wick(case):
    """For (spin-rotated) determinant trials: Wick's theorem with trot's mixed Green function, G[p, q] = <c+_p c_q>,
    gives <c+_{x up} c+_{y dn} c_{z dn} c_{w up}> = G[x, w] G[L+y, L+z] - G[x, L+z] G[L+y, w]."""
    spec = _spec(case)
    L = spec.n_sites
    bonds, G = mp.bond_structure(spec)
    ghf = GhfTrial(mo_coeff=jnp.asarray(case.C))
    want = []
    for w in case.walkers:
        g = np.asarray(calc_green_u(w, ghf))
        D = lambda x, y, z, v: g[x, v] * g[L + y, L + z] - g[x, L + z] * g[L + y, v]
        B = np.array([[sum(D(x, y, c, a) for x, y in ((ub, vb), (vb, ub)) for a, c in ((u, v), (v, u)))
                       for (u, v) in bonds] for (ub, vb) in bonds])
        want.append(np.einsum("ajb,bc,alc->ajl", G, B, G))
    np.testing.assert_allclose(_pairing(case, "exact", "blocked")[0], np.array(want), rtol=0, atol=1e-10)


@pytest.mark.parametrize("kernel", KERNELS)
def test_pairing_rows_are_the_rows_of_the_full_matrix(ed_2x3, kernel):
    """spec.rows restricts Delta^dag(j): the result is those rows of the full matrix."""
    spec = mp.PairingSpec(2, 3, "periodic", "open")
    full = _pairing(ed_2x3, "exact", kernel, spec)[0]
    sub = _pairing(ed_2x3, "exact", kernel, spec._replace(rows=(1, 4)))[0]
    assert sub.shape == (len(ed_2x3.walkers), 2, 2, 6)
    np.testing.assert_allclose(sub, full[:, :, [1, 4], :], rtol=0, atol=1e-11)


def test_pairing_without_a_spec_is_not_an_observable(ed_2x3):
    """Ops built without a pairing spec do not offer the observable, and a context without data refuses it."""
    ops = make_mps_meas_ops_hubbard(ed_2x3.plans["exact"])
    assert "pairing" not in ops.observables
    with pytest.raises(ValueError, match="pairing"):
        from trot.meas.mps import BATCHED_OBSERVABLES

        BATCHED_OBSERVABLES["pairing"](ed_2x3.ctx["exact", "blocked"])


# ---------------------------------------------------------------------------------------------
# The block, run_qmc and run_qmc_mps
# ---------------------------------------------------------------------------------------------

import scipy.linalg

from tests.test_mps_cpmc import _determinant, _diffused_state, _exact_trial, _noisy_trial, _params, _system
from trot.gmps.driver import make_mps_cpmc_ops, run_qmc_mps
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.prop import blocks, mps_cpmc
from trot.trial.mps import mps_trial_from_sd


def _block_setup(walker_channel_chi, energy_kernel, pairing):
    L, nelec = 6, (3, 2)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(orbital_plan="adaptive", walker_channel_chi=walker_channel_chi, energy_kernel=energy_kernel,
                     n_walkers=6)
    Ca, Cb = _determinant("uhf", h1, nelec)
    trial = mps_trial_from_sd(Ca, Cb)
    ops = make_mps_cpmc_ops(ham, trial, sys_, params, pairing=pairing)
    state = _diffused_state(trial, ops, h1, Ca, Cb, params.n_walkers, seed=4)
    rng = np.random.default_rng(4)
    walkers = tuple(jnp.asarray(np.stack([hf.nonorthonormal(np.asarray(w), rng) for w in spin])) for spin in state.walkers)
    state = state._replace(
        walkers=walkers, weights=jnp.asarray(rng.uniform(0.5, 1.5, params.n_walkers)),
        overlaps=jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))(walkers, trial))
    kwargs = dict(sys=sys_, params=params, ham_data=ham, trial_data=trial, trial_ops=ops.trial_ops,
                  meas_ops=ops.meas_ops, meas_ctx=ops.meas_ops.build_meas_ctx(ham, trial), prop_ops=ops.prop_ops,
                  prop_ctx=ops.prop_ops.build_prop_ctx(ham, None, params))
    return state, kwargs


@pytest.mark.parametrize("walker_channel_chi, energy_kernel", [(None, "blocked"), (2, "blocked"), (None, "dense")])
def test_mps_block_pairing_matches_trots_block(walker_channel_chi, energy_kernel):
    """The MPS block's pairing (batched kernel, det-R rescaled walkers) equals blocks.block's (per-walker observable,
    reconverted walkers); the energy and the walkers do not change when it is asked for."""
    spec = mp.PairingSpec(1, 6)
    state, kwargs = _block_setup(walker_channel_chi, energy_kernel, spec)
    got, got_obs = jax.jit(lambda s: mps_cpmc.block(s, observable_names=("pairing",), **kwargs))(state)
    want, want_obs = jax.jit(lambda s: blocks.block(s, observable_names=("pairing",), **kwargs))(state)
    plain, plain_obs = jax.jit(lambda s: mps_cpmc.block(s, **kwargs))(state)
    P = np.asarray(got_obs.observables["pairing"])
    assert P.shape == (2, 6, 6) and plain_obs.observables == {}
    np.testing.assert_allclose(P, np.asarray(want_obs.observables["pairing"]), rtol=0, atol=1e-9)
    assert np.abs(P).max() > 1e-3
    assert float(got_obs.scalars["energy"]) == float(plain_obs.scalars["energy"])
    for x, y in zip(got.walkers, plain.walkers):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_mps_block_masks_dead_walkers(monkeypatch):
    """A walker with weight 0 whose observable sample is NaN must not poison the weighted mean (0 * NaN), for any
    batched observable; the other walkers' mean is unchanged."""
    state, kwargs = _block_setup(None, "blocked", None)
    n = state.walkers[0].shape[0]

    def nan_observable(meas_ctx):
        def batch(ca, cb, data):
            samples = jnp.arange(ca.shape[0], dtype=float)[:, None] * jnp.ones((1, 3))
            return samples.at[0].set(jnp.nan)  # walker 0 is dead
        return batch

    monkeypatch.setitem(mps_cpmc.BATCHED_OBSERVABLES, "nan_obs", nan_observable)
    weights = jnp.asarray(np.linspace(1.0, 2.0, n)).at[0].set(0.0)
    _, obs = mps_cpmc.block(state._replace(weights=weights), observable_names=("nan_obs",), **kwargs)
    got = np.asarray(obs.observables["nan_obs"])
    assert np.all(np.isfinite(got))


def test_mps_block_rejects_unknown_observables_and_missing_pairing_data():
    state, kwargs = _block_setup(None, "blocked", None)
    with pytest.raises(ValueError, match="rdm1"):
        mps_cpmc.block(state, observable_names=("rdm1",), **kwargs)
    with pytest.raises(ValueError, match="pairing"):
        mps_cpmc.block(state, observable_names=("pairing",), **kwargs)


def test_run_qmc_mps_returns_pairing_means_and_errors():
    """run_qmc_mps(..., pairing=spec, observable_names=("pairing",)): block values, means and standard errors of the
    (2, L, L) pair correlations, and the energies equal a run without observables."""
    h1 = hopping_matrix(4, 1.0)
    nelec = (2, 2)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(4, nelec)
    _, trial = _exact_trial(h1, nelec)
    noisy = _noisy_trial(trial, nelec, seed=4)
    params = _params(n_walkers=6, n_prop_steps=3, n_eql_blocks=5, n_blocks=20, seed=3)
    spec = mp.PairingSpec(1, 4)
    with_obs = run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=noisy, pairing=spec,
                           observable_names=("pairing",))
    without = run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=noisy)
    np.testing.assert_array_equal(np.asarray(with_obs.block_energies), np.asarray(without.block_energies))
    series = np.asarray(with_obs.block_observables["pairing"])
    mean, err = np.asarray(with_obs.observable_means["pairing"]), np.asarray(with_obs.observable_stderrs["pairing"])
    assert series.shape[1:] == mean.shape == err.shape == (2, 4, 4)
    assert np.all(np.isfinite(mean)) and np.ptp(series, axis=0).max() > 1e-6
