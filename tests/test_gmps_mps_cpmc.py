"""Regression tests for trot/gmps/mps_cpmc_new.py.

SD walkers are converted to charge-labelled MPS (Fishman-White gates) and
contracted with an MPS trial.  Everything is checked at L=8 against exact
enumeration of all (alpha, beta) determinant pairs, and the fast CPMC step is
checked step for step against trot's reference cpmc_step.
"""
from itertools import combinations
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

pytest.importorskip("pyblock3")

from trot.core.ops import MeasOps
from trot.core.system import System
from trot.gmps import mps_cpmc_new as m
from trot.ham.hubbard import HamHubbard
from trot.prop import cpmc as trot_cpmc
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.types import PropState, QmcParams
from trot.trial.ghf import GhfTrial, make_ghf_trial_ops

L, N, U, DT = 8, 4, 4.0, 0.01
CHI_TRUNC = 4  # per-channel bond cap that truncates visibly at L=8


def _all_occupations():
    rows = np.array(list(combinations(range(L), N)))
    occ = np.zeros((len(rows), L), int)
    occ[np.arange(len(rows))[:, None], rows] = 1
    return rows, occ


def _signed_amplitudes(tensors, occ):
    """<n_alpha, n_beta|MPS> times the reordering sign, so that
    <MPS|SD> = det_alpha @ result @ det_beta."""
    amp = np.empty((len(occ), len(occ)))
    for a, oa in enumerate(occ):
        for b, ob in enumerate(occ):
            v = np.ones((1, 1))
            for i, A in enumerate(tensors):
                v = v @ np.asarray(A)[:, oa[i] + 2 * ob[i], :]
            amp[a, b] = v[0, 0]
    lower = np.tril(np.ones((L, L), int), -1)
    return (1 - 2 * ((occ @ lower @ occ.T) & 1)) * amp


def _random_field_walkers(C, n, steps, seed):
    """HF propagated by the CPMC propagator with unguided random fields."""
    rng = np.random.default_rng(seed)
    half = scipy.linalg.expm(-0.5 * DT * m.hopping_matrix(L, 1.0))
    gamma = np.arccosh(np.exp(0.5 * DT * U))
    out = []
    for _ in range(n):
        ca, cb = C.copy(), C.copy()
        for _ in range(steps):
            field = rng.integers(0, 2, L) * 2 - 1
            ca = np.linalg.qr(half @ (np.exp(gamma * field)[:, None] * (half @ ca)))[0]
            cb = np.linalg.qr(half @ (np.exp(-gamma * field)[:, None] * (half @ cb)))[0]
        out.append((ca, cb))
    return out


def _overlap(a, b):
    env = np.ones((1, 1))
    for x, y in zip(a, b):
        env = np.einsum("ab,asc,bsd->cd", env, np.asarray(x), np.asarray(y))
    return float(env.reshape(()))


def _infidelity(exact, approx):
    return 1.0 - _overlap(exact, approx) ** 2 / (_overlap(exact, exact) * _overlap(approx, approx))


def _last_centre(C, plan):
    angles, _ = m.channel_angles(np.asarray(C), plan, xp=np)
    return angles[0][0] + 1  # the last gate applied is the first one planned


@pytest.fixture(scope="module")
def model():
    cfg = m.Config(L=L, n_up=N, n_down=N, interaction=U, trial_chi=16, dmrg_sweeps=8)
    C = np.linalg.eigh(m.hopping_matrix(L, 1.0))[1][:, :N]
    mps, _ = m.run_dmrg(m.build_dmrg_hamiltonian(cfg), cfg)
    trial_np, trial_charges = m.densify_with_charges(mps, L)
    Htrial_np = m.compress_mps(m.apply_mpo(m.hubbard_mpo(L, 1.0, U), trial_np))
    rows, occ = _all_occupations()
    return SimpleNamespace(
        C=C, trial_np=trial_np, trial_charges=trial_charges,
        Htrial=tuple(jnp.asarray(A) for A in Htrial_np),
        rows=rows, occ=occ,
        amp=_signed_amplitudes(trial_np, occ), hamp=_signed_amplitudes(Htrial_np, occ),
        walkers=_random_field_walkers(C, n=6, steps=150, seed=0))


def _exact_overlap(model, ca, cb):
    return np.linalg.det(ca[model.rows]) @ model.amp @ np.linalg.det(cb[model.rows])


def _exact_energy(model, ca, cb):
    da, db = np.linalg.det(ca[model.rows]), np.linalg.det(cb[model.rows])
    return (da @ model.hamp @ db) / (da @ model.amp @ db)


def test_one_rdm_matches_enumeration(model):
    """one_rdm of the trial against <c^dag_i c_j> built from its determinant amplitudes."""
    index = {tuple(o): k for k, o in enumerate(model.occ)}
    hop = np.zeros((L, L, len(model.occ), len(model.occ)))  # <a|c^dag_i c_j|a'>
    for k, o in enumerate(model.occ):
        for j in np.flatnonzero(o):
            removed = o.copy(); removed[j] = 0
            for i in np.flatnonzero(removed == 0):
                added = removed.copy(); added[i] = 1
                hop[i, j, index[tuple(added)], k] = (-1) ** (o[:j].sum() + removed[:i].sum())
    amp = model.amp / np.linalg.norm(model.amp)
    gamma_a = np.einsum("ab,ijac,cb->ij", amp, hop, amp)
    gamma_b = np.einsum("ba,ijac,bc->ij", amp, hop, amp)
    got_a, got_b = m.one_rdm(model.trial_np)
    np.testing.assert_allclose(got_a, gamma_a, atol=1e-10)
    np.testing.assert_allclose(got_b, gamma_b, atol=1e-10)


@pytest.mark.parametrize("mode,tol", [("rank_exact", 1e-12), ("maximal", 1e-12), ("adaptive", 1e-8)])
def test_channel_conversion_is_exact(model, mode, tol):
    """Without truncation the channel MPS times the gauge reproduces every determinant."""
    plan = m.make_orbital_plan(model.C, mode)
    for ca, _ in model.walkers:
        tensors, _, gauge = m.channel_mps(jnp.asarray(ca), plan)
        for rows, occ in zip(model.rows, model.occ):
            v = np.ones((1, 1))
            for A, n in zip(tensors, occ):
                v = v @ np.asarray(A)[:, n, :]
            assert abs(float(gauge) * v[0, 0] - np.linalg.det(ca[rows])) < tol


@pytest.mark.parametrize("chi", [None, CHI_TRUNC])
def test_conversion_ends_in_mixed_canonical_form(model, chi):
    """Every split must happen with the orthogonality centre on the gate; the
    conversion therefore ends left-isometric before the centre and
    right-isometric after it, with charge-conserving tensors and the labels the
    dry run predicted."""
    plan = m.make_orbital_plan(model.C)
    bond = None if chi is None else m.plan_bonds(model.C, plan, chi)
    centre = _last_centre(model.C, plan)
    for ca, _ in model.walkers:
        tensors, charges, _ = m.channel_mps(jnp.asarray(ca), plan, bond)
        for i, A in enumerate(map(np.asarray, tensors)):
            if i < centre:
                M = A.reshape(-1, A.shape[2])
                np.testing.assert_allclose(M.T @ M, np.eye(M.shape[1]), atol=1e-12)
            elif i > centre:
                M = A.reshape(A.shape[0], -1)
                np.testing.assert_allclose(M @ M.T, np.eye(M.shape[0]), atol=1e-12)
            for a, n, b in np.argwhere(np.abs(A) > 1e-13):
                assert charges[i][a] + n == charges[i + 1][b]
        if bond is not None:
            assert all(np.array_equal(x, y) for x, y in zip(charges, bond.charges))


def test_truncation_keeps_the_largest_schmidt_values(model):
    """On the reference, where the frozen sector counts are the optimal ones,
    gate-by-gate truncation matches compressing the exact MPS once in canonical
    form, and the dry run's discarded weight is the norm actually lost."""
    plan = m.make_orbital_plan(model.C)
    bond = m.plan_bonds(model.C, plan, CHI_TRUNC)
    exact, _, _ = m.channel_mps(jnp.asarray(model.C), plan)
    truncated, charges, _ = m.channel_mps(jnp.asarray(model.C), plan, bond)

    best = [np.asarray(A).copy() for A in exact]
    for i in range(L - 1):
        Dl, d, Dr = best[i].shape
        q, r = np.linalg.qr(best[i].reshape(Dl * d, Dr))
        best[i], best[i + 1] = q.reshape(Dl, d, -1), np.tensordot(r, best[i + 1], (1, 0))
    for i in range(L - 1, 0, -1):
        Dl, d, Dr = best[i].shape
        u, s, vt = np.linalg.svd(best[i].reshape(Dl, d * Dr), full_matrices=False)
        k = len(charges[i])
        best[i], best[i - 1] = vt[:k].reshape(k, d, Dr), np.tensordot(best[i - 1], u[:, :k] * s[:k], (2, 0))

    lost, lost_best = _infidelity(exact, truncated), _infidelity(exact, best)
    assert lost > 1e-4  # the test must actually truncate
    assert lost <= 1.05 * lost_best
    assert abs(bond.reference_discarded_weight - lost) <= 0.1 * lost


def test_overlap_and_local_energy_match_enumeration(model):
    """Production conversion + blocked contraction, including the det(R) gauge
    of non-orthonormal walkers and the alpha/beta reordering sign."""
    plan = m.make_orbital_plan(model.C)
    ops = m.make_walker_ops(model.C, model.C, plan, plan, None, None,
                            model.trial_np, model.trial_charges, model.Htrial)
    rng = np.random.default_rng(1)
    for ca, cb in model.walkers:
        ca = ca @ (np.eye(N) + 0.3 * rng.standard_normal((N, N)))
        cb = cb @ (np.eye(N) + 0.3 * rng.standard_normal((N, N)))
        walker = (jnp.asarray(ca), jnp.asarray(cb))
        exact = _exact_overlap(model, ca, cb)
        assert abs(float(ops.overlap(walker)) - exact) <= 1e-10 * abs(exact)
        assert abs(float(ops.energy(walker)) - _exact_energy(model, ca, cb)) <= 1e-10


def test_fast_sweep_matches_fresh_conversions(model):
    """With exact conversion the environment sweep must agree with converting the
    walker before and after the sweep, and apply one HS field per site."""
    plan = m.make_orbital_plan(model.C)
    ops = m.make_walker_ops(model.C, model.C, plan, plan, None, None,
                            model.trial_np, model.trial_charges)
    hs = _build_prop_ctx(HamHubbard(h1=jnp.asarray(m.hopping_matrix(L, 1.0)), u=U), DT).hs_constant
    randoms = jnp.asarray(np.random.default_rng(2).random(L))
    for ca, cb in model.walkers:
        ca, cb = jnp.asarray(ca), jnp.asarray(cb)
        ca2, cb2, before, after, _, _ = ops.sweep(ca, cb, randoms, hs, 1e-8)
        np.testing.assert_allclose(float(before), float(ops.overlap((ca, cb))), rtol=1e-10)
        np.testing.assert_allclose(float(after), float(ops.overlap((ca2, cb2))), rtol=1e-10)
        scale_a = np.asarray(ca2[:, 0] / ca[:, 0])
        scale_b = np.asarray(cb2[:, 0] / cb[:, 0])
        fields = np.isclose(scale_a, float(hs[1, 0])).astype(int)
        np.testing.assert_allclose(scale_a, np.asarray(hs[fields, 0]), rtol=1e-12)
        np.testing.assert_allclose(scale_b, np.asarray(hs[fields, 1]), rtol=1e-12)


@pytest.mark.parametrize("weight_floor", [1e-8, 0.5])
def test_fast_step_matches_trot_cpmc_step(model, weight_floor):
    """With a determinant trial converted exactly to an MPS, make_fast_prop_ops
    must reproduce trot's cpmc_step: same fields, weights, nodes and overlap
    ratios from the same RNG state.  weight_floor=0.5 separates trot's rule
    (floor on the ratio) from flooring half the ratio."""
    C = model.C
    plan = m.make_orbital_plan(C)
    alpha, qa, _ = m.channel_mps(jnp.asarray(C), plan)
    beta, qb, _ = m.channel_mps(jnp.asarray(C), plan)
    trial_np, trial_charges = m.combine_channels(alpha, qa, beta, qb)
    ops = m.make_walker_ops(C, C, plan, plan, None, None, [np.asarray(A) for A in trial_np], trial_charges)

    ham = HamHubbard(h1=jnp.asarray(m.hopping_matrix(L, 1.0)), u=U)
    ghf = GhfTrial(mo_coeff=jnp.asarray(scipy.linalg.block_diag(C, C)))
    ghf_ops = make_ghf_trial_ops(System(norb=L, nelec=(N, N), walker_kind="unrestricted"))
    params = QmcParams(dt=DT, n_walkers=6, n_prop_steps=1, n_blocks=1, n_eql_blocks=0,
                       weight_floor=weight_floor, seed=0)
    prop_ctx = _build_prop_ctx(ham, DT)
    fast = m.make_fast_prop_ops(ham, "unrestricted", ops.overlap, ops.sweep)
    reference = trot_cpmc.make_prop_ops(ham, "unrestricted", ghf_ops)

    rng = np.random.default_rng(3)
    walkers = tuple(jnp.asarray(np.stack([np.linalg.qr(C + 0.15 * rng.standard_normal(C.shape))[0]
                                          for _ in range(6)])) for _ in range(2))

    def initial(overlap):
        return PropState(walkers=walkers, weights=jnp.ones(6),
                         overlaps=jax.vmap(overlap, in_axes=(0, None))(walkers, ghf),
                         rng_key=jax.random.PRNGKey(0), pop_control_ene_shift=jnp.asarray(-6.0),
                         e_estimate=jnp.asarray(-6.0), node_encounters=jnp.zeros((), jnp.int64))

    def run(prop, trial_ops, overlap):
        step = jax.jit(lambda s: prop.step(s, params=params, ham_data=ham, trial_data=ghf,
                                           trial_ops=trial_ops, meas_ops=MeasOps(overlap=overlap, kernels={}),
                                           meas_ctx=None, prop_ctx=prop_ctx))
        state = initial(overlap)
        for _ in range(3):
            state = step(state)
        return state

    got = run(fast, ghf_ops, ops.overlap)
    want = run(reference, ghf_ops, ghf_ops.overlap)
    for g, w in zip(got.walkers, want.walkers):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.weights), np.asarray(want.weights), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(float(got.pop_control_ene_shift), float(want.pop_control_ene_shift), rtol=1e-9)
    assert int(got.node_encounters) == int(want.node_encounters)
    ratio = np.asarray(got.overlaps) / np.asarray(want.overlaps)  # trial MPS = SD up to a constant
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)
