"""Regression tests for trot/gmps/mps_cpmc_gpu.py (CPU or GPU backend).

The GPU script re-implements the device side of mps_cpmc_new.py: batched sector
factorisations from a compiled circuit, factorized charge-blocked contractions,
a charge-labelled H|trial>, half-step scans and a measurement block without
extra conversions. Everything is checked at L=8 against exact enumeration and
against mps_cpmc_new / trot, which remain the references. The tests marked gpu
only run on a GPU backend and check that the step stays on the device and
batches walkers efficiently.

    python -m pytest tests/test_gmps_mps_cpmc_gpu.py
"""
import time
from itertools import combinations
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from jax import lax

pytest.importorskip("pyblock3")

from trot.core.ops import MeasOps, k_energy
from trot.core.system import System
from trot.gmps import mps_cpmc_gpu as g
from trot.gmps import mps_cpmc_new as ref
from trot.ham.hubbard import HamHubbard
from trot.prop import blocks
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.types import PropState, QmcParams
from trot.trial.auto import make_auto_trial_ops
from trot.trial.uhf import UhfTrial, get_rdm1 as uhf_get_rdm1
from trot.walkers import _qr as qr_with_det

L, N, U, DT = 8, 4, 4.0, 0.01
CHI_TRUNC = 4  # per-channel bond cap that truncates visibly at L=8
gpu = pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU backend")


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
    half = scipy.linalg.expm(-0.5 * DT * ref.hopping_matrix(L, 1.0))
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


def _last_centre(C, plan):
    angles, _ = ref.channel_angles(np.asarray(C), plan, xp=np)
    return angles[0][0] + 1  # the last gate applied is the first one planned


@pytest.fixture(scope="module")
def model():
    cfg = ref.Config(L=L, n_up=N, n_down=N, interaction=U, trial_chi=16, dmrg_sweeps=8)
    C = np.linalg.eigh(ref.hopping_matrix(L, 1.0))[1][:, :N]
    mps, _ = ref.run_dmrg(ref.build_dmrg_hamiltonian(cfg), cfg)
    trial_np, trial_charges = ref.densify_with_charges(mps, L)
    mpo = ref.hubbard_mpo(L, 1.0, U)
    Htrial_np = ref.compress_mps(ref.apply_mpo(mpo, trial_np))
    ham = HamHubbard(h1=jnp.asarray(ref.hopping_matrix(L, 1.0)), u=U)
    rows, occ = _all_occupations()
    return SimpleNamespace(
        C=C, trial_np=trial_np, trial_charges=trial_charges, mpo=mpo, Htrial_np=Htrial_np,
        H_qn=g.compress_mps_qn(*g.apply_mpo_qn(mpo, trial_np, trial_charges)),
        ham=ham, prop_ctx=_build_prop_ctx(ham, DT), rows=rows, occ=occ,
        amp=_signed_amplitudes(trial_np, occ), hamp=_signed_amplitudes(Htrial_np, occ),
        walkers=_random_field_walkers(C, n=6, steps=150, seed=0))


def _exact_overlap(model, ca, cb):
    return np.linalg.det(ca[model.rows]) @ model.amp @ np.linalg.det(cb[model.rows])


def _exact_energy(model, ca, cb):
    da, db = np.linalg.det(ca[model.rows]), np.linalg.det(cb[model.rows])
    return (da @ model.hamp @ db) / (da @ model.amp @ db)


def _plans(model, chi=None, mode="rank_exact"):
    plan = ref.make_orbital_plan(model.C, mode)
    return plan, None if chi is None else ref.plan_bonds(model.C, plan, chi)


def _ops(model, chi=None, linalg="batched", walker_qr="cholesky", spin_batch=True, energy="blocked"):
    plan, bond = _plans(model, chi)
    htrial = model.H_qn if energy == "blocked" else tuple(jnp.asarray(A) for A in model.Htrial_np)
    ops = g.make_gpu_ops(plan, plan, bond, bond, model.trial_np, model.trial_charges, htrial, model.prop_ctx,
                         linalg=linalg, walker_qr=walker_qr, spin_batch=spin_batch, energy=energy)
    return ops, plan, bond


def _walker_batch(model, n=None, mix=0.0, seed=1):
    """The fixture walkers (repeated to n), optionally made non-orthonormal."""
    rng = np.random.default_rng(seed)
    ca = np.stack([w[0] for w in model.walkers])
    cb = np.stack([w[1] for w in model.walkers])
    if n is not None:
        reps = -(-n // len(ca))
        ca, cb = np.concatenate([ca] * reps)[:n], np.concatenate([cb] * reps)[:n]
    if mix:
        ca = ca @ (np.eye(N) + mix * rng.standard_normal((len(ca), N, N)))
        cb = cb @ (np.eye(N) + mix * rng.standard_normal((len(cb), N, N)))
    return ca, cb


def _state(ops, ca, cb, key=0):
    overlaps = jax.jit(ops.overlaps)(jnp.asarray(ca), jnp.asarray(cb), ops.data)
    return PropState(walkers=(jnp.asarray(ca), jnp.asarray(cb)), weights=jnp.ones(len(ca), jnp.float64),
                     overlaps=overlaps, rng_key=jax.random.PRNGKey(key),
                     pop_control_ene_shift=jnp.asarray(-6.0, jnp.float64),
                     e_estimate=jnp.asarray(-6.0, jnp.float64), node_encounters=jnp.zeros((), jnp.int64))


def _propagate(ops, params, n_chunks, n_half):
    half = g.make_half_step(ops, params, n_chunks)
    return jax.jit(lambda s, data: lax.scan(lambda st, i: (half(st, i, data), None), s, jnp.arange(n_half))[0])


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

@pytest.mark.parametrize("chi", [None, CHI_TRUNC])
def test_compiled_circuit_labels_match_channel_mps(model, chi):
    """The host circuit compiler predicts exactly the bond labels channel_mps ends with."""
    plan, bond = _plans(model, chi)
    circuit = g.compile_circuit((plan,), (bond,))
    _, charges, _ = ref.channel_mps(jnp.asarray(model.C), plan, bond)
    assert len(circuit.charges[0]) == len(charges)
    assert all(np.array_equal(a, b) for a, b in zip(circuit.charges[0], charges))
    assert sum(op.kind == "gate" for op in circuit.ops) == len(circuit.gate_sites)
    if bond is not None:
        assert all(np.array_equal(a, b) for a, b in zip(circuit.charges[0], bond.charges))


def test_sector_methods_follow_the_original(model):
    """Every sector is routed to the branch of mps_cpmc_new._factor_block that the
    original takes for it: centre moves never truncate, and only truncated gate
    sectors (rank below the block's rank) use an eigh."""
    plan, bond = _plans(model, CHI_TRUNC)
    circuit = g.compile_circuit((plan,), (bond,))
    kinds = {"move": set(), "gate": set()}
    for op in circuit.ops:
        kinds[op.kind] |= {cls.kind for cls in op.classes}
    assert not kinds["move"] & {"eigh_rows", "eigh_cols"}
    assert kinds["gate"] & {"eigh_rows", "eigh_cols"}  # CHI_TRUNC must truncate somewhere
    exact, _ = _plans(model)
    assert not any(cls.kind.startswith("eigh") for op in g.compile_circuit((exact,), (None,)).ops
                   for cls in op.classes)
    assert g.circuit_stats(circuit)["closed_form_batches"] > 0


@pytest.mark.parametrize("mode,tol", [("rank_exact", 1e-12), ("maximal", 1e-12), ("adaptive", 1e-8)])
def test_batched_conversion_is_exact(model, mode, tol):
    """Without truncation, the batched conversion times its gauge reproduces every
    determinant of both spin channels (the tolerances of the original's test)."""
    plan, _ = _plans(model, mode=mode)
    convert = jax.jit(g.make_converter(plan, plan, linalg="batched", spin_batch=True).convert)
    for ca, cb in model.walkers:
        alpha, beta, gauges = convert(jnp.asarray(ca), jnp.asarray(cb))
        for tensors, gauge, C in ((alpha, gauges[0], ca), (beta, gauges[1], cb)):
            for rows, occ in zip(model.rows, model.occ):
                v = np.ones((1, 1))
                for A, n in zip(tensors, occ):
                    v = v @ np.asarray(A)[:, n, :]
                assert abs(float(gauge) * v[0, 0] - np.linalg.det(C[rows])) < tol


@pytest.mark.parametrize("chi", [None, CHI_TRUNC])
def test_batched_conversion_ends_in_mixed_canonical_form(model, chi):
    """Every batched factor is an isometry and conserves charge, so the conversion
    ends left-isometric before the centre and right-isometric after it."""
    plan, bond = _plans(model, chi)
    converter = g.make_converter(plan, plan, bond, bond, linalg="batched", spin_batch=True)
    convert = jax.jit(converter.convert)
    centre = _last_centre(model.C, plan)
    for ca, cb in model.walkers:
        alpha, beta, _ = convert(jnp.asarray(ca), jnp.asarray(cb))
        for tensors, charges in ((alpha, converter.charges[0]), (beta, converter.charges[1])):
            for i, A in enumerate(map(np.asarray, tensors)):
                if i < centre:
                    M = A.reshape(-1, A.shape[2])
                    np.testing.assert_allclose(M.T @ M, np.eye(M.shape[1]), atol=1e-10)
                elif i > centre:
                    M = A.reshape(A.shape[0], -1)
                    np.testing.assert_allclose(M @ M.T, np.eye(M.shape[0]), atol=1e-10)
                for a, n, b in np.argwhere(np.abs(A) > 1e-12):
                    assert charges[i][a] + n == charges[i + 1][b]


@pytest.mark.parametrize("spin_batch", [True, False])
def test_truncated_batched_conversion_matches_the_original(model, spin_batch):
    """With truncation the batched conversion keeps the same Schmidt subspaces as
    the original: the states agree up to the gauge, for both spins and with and
    without spin batching. channel_mps_host (NumPy) equals channel_mps."""
    plan, bond = _plans(model, CHI_TRUNC)
    converter = g.make_converter(plan, plan, bond, bond, linalg="batched", spin_batch=spin_batch)
    assert converter.spin_batched == spin_batch
    convert = jax.jit(converter.convert)
    for ca, cb in model.walkers:
        alpha, beta, gauges = convert(jnp.asarray(ca), jnp.asarray(cb))
        for C, dev, g_dev in ((ca, alpha, gauges[0]), (cb, beta, gauges[1])):
            host, _, g_host = g.channel_mps_host(C, plan, bond)
            original, _, g_original = ref.channel_mps(jnp.asarray(C), plan, bond)
            reference = g_host * _overlap(host, host)
            assert abs(float(g_dev) * _overlap(host, dev) - reference) <= 1e-9 * abs(reference)
            assert abs(float(g_original) * _overlap(host, original) - reference) <= 1e-9 * abs(reference)


def test_spin_batching_keeps_each_channels_own_truncation(model):
    """Alpha and beta with different kept counts (hence different bond labels)
    still convert in one batch, each exactly as the original converts it alone."""
    plan = ref.make_orbital_plan(model.C)
    bond_a, bond_b = ref.plan_bonds(model.C, plan, CHI_TRUNC), ref.plan_bonds(model.C, plan, CHI_TRUNC - 1)
    assert any(len(x) != len(y) for x, y in zip(bond_a.charges, bond_b.charges))
    converter = g.make_converter(plan, plan, bond_a, bond_b, linalg="batched", spin_batch=True)
    assert converter.spin_batched
    convert = jax.jit(converter.convert)
    for ca, cb in model.walkers:
        alpha, beta, gauges = convert(jnp.asarray(ca), jnp.asarray(cb))
        for C, bond, dev, g_dev, labels in ((ca, bond_a, alpha, gauges[0], converter.charges[0]),
                                           (cb, bond_b, beta, gauges[1], converter.charges[1])):
            assert [A.shape[0] for A in dev] == [len(q) for q in labels[:-1]]
            original, _, g_original = ref.channel_mps(jnp.asarray(C), plan, bond)
            reference = float(g_original) * _overlap(original, original)
            assert abs(float(g_dev) * _overlap(original, dev) - reference) <= 1e-9 * abs(reference)


@pytest.fixture(scope="module")
def allocation():
    """allocation_study.py, whose NumPy gmps defines the frozen, union and padded schemes."""
    return pytest.importorskip("trot.gmps.allocation_study")


def _perturbed(C, n, seed, scale=0.3):
    rng = np.random.default_rng(seed)
    return [np.linalg.qr(C + scale * rng.standard_normal(C.shape))[0] for _ in range(n)]


def test_counts_bond_plan_matches_the_study_allocation(model, allocation):
    """counts_bond_plan turns a per-gate allocation (here a padding learned on a few
    walkers) into the BondPlan whose conversion is the study's gmps(counts=...)."""
    plan = ref.make_orbital_plan(model.C, "adaptive", 1e-10)
    train, test = _perturbed(model.C, 5, 5), _perturbed(model.C, 4, 6)
    pad = allocation.padding([allocation.gmps(Q, plan, chi=3)[2] for Q in train])
    bond = g.counts_bond_plan(plan, pad)
    for Q in train + test:
        want, g_want, *_ = allocation.gmps(Q, plan, counts=pad)
        got, _, g_got = ref.channel_mps(jnp.asarray(Q), plan, bond)
        np.testing.assert_allclose(float(g_got) * allocation.mps_to_dense(got),
                                   g_want * allocation.mps_to_dense(want), atol=1e-10)


@pytest.mark.parametrize("chi", [2, 3, 5])
def test_dynamic_allocation_matches_the_studys_padded_scheme(model, allocation, chi):
    """The GPU padded scheme (each walker's own chi largest values inside the
    training padding, dynamic_chi) reproduces allocation_study.gmps(chi=chi,
    caps=pad) state for state, capped walkers included, for both spins."""
    plan = ref.make_orbital_plan(model.C, "adaptive", 1e-10)
    train, test = _perturbed(model.C, 5, 7), _perturbed(model.C, 6, 8, scale=0.6)
    pad = allocation.padding([allocation.gmps(Q, plan, chi=chi)[2] for Q in train])
    bond = g.counts_bond_plan(plan, pad)
    converter = g.make_converter(plan, plan, bond, bond, linalg="batched", spin_batch=True, dynamic_chi=chi)
    assert g.circuit_stats(converter.circuits[0])["dynamic_chi"] == chi
    convert = jax.jit(converter.convert)
    for Qa, Qb in zip(train + test, (test + train)[::-1]):
        alpha, beta, gauges = convert(jnp.asarray(Qa), jnp.asarray(Qb))
        for Q, tensors, gauge in ((Qa, alpha, gauges[0]), (Qb, beta, gauges[1])):
            want, g_want, *_ = allocation.gmps(Q, plan, chi=chi, caps=pad)
            np.testing.assert_allclose(float(gauge) * allocation.mps_to_dense(tensors),
                                       g_want * allocation.mps_to_dense(want), atol=1e-10)


def test_cholesky_qr2_matches_trot_qr():
    """CholeskyQR2 gives trot's Q (diag R > 0) and det R; a batch with a walker it
    cannot factor falls back to Householder QR as a whole."""
    rng = np.random.default_rng(4)
    C = jnp.asarray(rng.standard_normal((5, L, N)))
    Q, d = jax.vmap(g.cholesky_qr2)(C)
    Qr, dr = jax.vmap(qr_with_det)(C)
    np.testing.assert_allclose(np.asarray(Q), np.asarray(Qr), atol=1e-12)
    np.testing.assert_allclose(np.asarray(d), np.asarray(dr), rtol=1e-12)

    bad = np.asarray(C).copy()
    bad[0, :, 1] = 0.0  # exactly singular Gram matrix: the Cholesky breaks down
    Qb, db = jax.jit(g.make_batch_qr("cholesky"))(jnp.asarray(bad))
    Qn, dn = jax.vmap(qr_with_det)(jnp.asarray(bad))
    np.testing.assert_allclose(np.asarray(Qb), np.asarray(Qn), atol=1e-12)
    np.testing.assert_allclose(np.asarray(db), np.asarray(dn), atol=1e-12)


def test_probe_and_self_check(model):
    """init_state converts one walker and broadcasts; the start-up self-check
    against the NumPy conversion passes."""
    ops, plan, bond = _ops(model, CHI_TRUNC)
    system = System(norb=L, nelec=(N, N), walker_kind="unrestricted")
    trial_data = UhfTrial(mo_coeff_a=jnp.asarray(model.C), mo_coeff_b=jnp.asarray(model.C))
    params = QmcParams(dt=DT, n_walkers=4, n_prop_steps=1, n_blocks=1, n_eql_blocks=0, seed=0)
    state, probe = g.init_state(ops, system, trial_data, params)
    assert max(g.conversion_self_check(ops, (plan, plan), (bond, bond), probe)) < 1e-9
    ref_ops = ref.make_walker_ops(model.C, model.C, plan, plan, bond, bond, model.trial_np, model.trial_charges)
    want = float(jax.jit(ref_ops.overlap)((state.walkers[0][0], state.walkers[1][0])))
    np.testing.assert_allclose(np.asarray(state.overlaps), want, rtol=1e-9)
    assert state.e_estimate is not state.pop_control_ene_shift  # donated separately


# --------------------------------------------------------------------------
# Contractions and energy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("linalg", ["batched", "native"])
@pytest.mark.parametrize("chi", [None, CHI_TRUNC])
def test_overlap_matches_the_original_and_enumeration(model, chi, linalg):
    """Factorized contraction (never forming the d=4 walker) against the
    original blocked overlap, on non-orthonormal walkers (det R gauge)."""
    ops, plan, bond = _ops(model, chi, linalg=linalg)
    ref_ops = ref.make_walker_ops(model.C, model.C, plan, plan, bond, bond, model.trial_np, model.trial_charges)
    ca, cb = _walker_batch(model, mix=0.3)
    got = np.asarray(jax.jit(ops.overlaps)(jnp.asarray(ca), jnp.asarray(cb), ops.data))
    overlap = jax.jit(ref_ops.overlap)
    want = np.asarray([float(overlap((jnp.asarray(a), jnp.asarray(b)))) for a, b in zip(ca, cb)])
    np.testing.assert_allclose(got, want, rtol=1e-9)
    if chi is None:
        exact = [_exact_overlap(model, a, b) for a, b in zip(ca, cb)]
        np.testing.assert_allclose(got, exact, rtol=1e-9)


def test_compress_mps_qn_keeps_labels_and_state(model):
    tensors, labels = model.H_qn
    for i, A in enumerate(tensors):
        for a, p, b in np.argwhere(np.abs(A) > 1e-12):
            assert np.array_equal(labels[i][a] + ref.PHYSICAL_CHARGE[p], labels[i + 1][b])
    uncompressed, _ = g.apply_mpo_qn(model.mpo, model.trial_np, model.trial_charges)
    assert _overlap(tensors, model.trial_np) == pytest.approx(_overlap(uncompressed, model.trial_np), rel=1e-11)
    assert _overlap(tensors, tensors) == pytest.approx(_overlap(uncompressed, uncompressed), rel=1e-11)
    assert max(A.shape[0] for A in tensors) <= max(A.shape[0] for A in uncompressed)


@pytest.mark.parametrize("energy", ["blocked", "dense"])
def test_local_energy_matches_enumeration(model, energy):
    ops, _, _ = _ops(model, energy=energy)
    ca, cb = _walker_batch(model, mix=0.3)
    got = np.asarray(jax.jit(ops.energies)(jnp.asarray(ca), jnp.asarray(cb), ops.data))
    exact = [_exact_energy(model, a, b) for a, b in zip(ca, cb)]
    np.testing.assert_allclose(got, exact, atol=1e-9)


# --------------------------------------------------------------------------
# Sweep, step, block
# --------------------------------------------------------------------------

@pytest.mark.parametrize("chi", [None, CHI_TRUNC])
def test_sweep_matches_the_original(model, chi):
    """step_batch with the field sweep against mps_cpmc_new's fast sweep: same
    fields, overlaps before and after, weight factor and node count."""
    ops, plan, bond = _ops(model, chi)
    ref_ops = ref.make_walker_ops(model.C, model.C, plan, plan, bond, bond, model.trial_np, model.trial_charges)
    hs = model.prop_ctx.hs_constant
    ca, cb = _walker_batch(model)
    randoms = jnp.asarray(np.random.default_rng(2).random((len(ca), L)))
    step = jax.jit(lambda a, b, r, data: ops.step_batch(a, b, r, True, data, 1e-8))
    ca2, cb2, before, after, factor, nodes = step(jnp.asarray(ca), jnp.asarray(cb), randoms, ops.data)
    sweep = jax.jit(ref_ops.sweep)
    for k in range(len(ca)):
        want = sweep(jnp.asarray(ca[k]), jnp.asarray(cb[k]), randoms[k], hs, 1e-8)
        np.testing.assert_allclose(np.asarray(ca2[k]), np.asarray(want[0]), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(cb2[k]), np.asarray(want[1]), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(float(before[k]), float(want[2]), rtol=1e-9)
        np.testing.assert_allclose(float(after[k]), float(want[3]), rtol=1e-9)
        np.testing.assert_allclose(float(factor[k]), float(want[4]), rtol=1e-9)
        assert int(nodes[k]) == int(want[5])


@pytest.mark.parametrize("weight_floor", [1e-8, 0.5])
def test_half_steps_match_the_original_step(model, weight_floor):
    """Two half steps (one scan, one conversion call site, lax.cond sweep) equal
    one step of mps_cpmc_new.make_fast_prop_ops, three times in a row."""
    ops, plan, bond = _ops(model, CHI_TRUNC)
    ref_ops = ref.make_walker_ops(model.C, model.C, plan, plan, bond, bond, model.trial_np, model.trial_charges)
    params = QmcParams(dt=DT, n_walkers=6, n_prop_steps=1, n_blocks=1, n_eql_blocks=0,
                       weight_floor=weight_floor, seed=0)
    fast = ref.make_fast_prop_ops(model.ham, "unrestricted", ref_ops.overlap, ref_ops.sweep)
    ref_step = jax.jit(lambda s: fast.step(s, params=params, ham_data=model.ham, trial_data=None, trial_ops=None,
                                           meas_ops=None, meas_ctx=None, prop_ctx=model.prop_ctx))
    gpu_step = _propagate(ops, params, 1, 2)
    got = want = _state(ops, *_walker_batch(model))
    for _ in range(3):
        want, got = ref_step(want), gpu_step(got, ops.data)
    for x, y in zip(got.walkers, want.walkers):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.weights), np.asarray(want.weights), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(np.asarray(got.overlaps), np.asarray(want.overlaps), rtol=1e-9)
    np.testing.assert_allclose(float(got.pop_control_ene_shift), float(want.pop_control_ene_shift), rtol=1e-9)
    assert int(got.node_encounters) == int(want.node_encounters)
    assert np.array_equal(np.asarray(got.rng_key), np.asarray(want.rng_key))


def test_block_matches_trot_block(model):
    """make_block (det-rescaled and gathered overlaps, blocked energy) against
    trot.prop.blocks.block driving the original step, overlap and dense energy."""
    ops, plan, bond = _ops(model, CHI_TRUNC)
    Htrial = tuple(jnp.asarray(A) for A in model.Htrial_np)
    ref_ops = ref.make_walker_ops(model.C, model.C, plan, plan, bond, bond, model.trial_np, model.trial_charges,
                                  Htrial)
    system = System(norb=L, nelec=(N, N), walker_kind="unrestricted")
    params = QmcParams(dt=DT, n_walkers=6, n_prop_steps=2, n_blocks=1, n_eql_blocks=0, weight_floor=1e-8, seed=0)
    fast = ref.make_fast_prop_ops(model.ham, "unrestricted", ref_ops.overlap, ref_ops.sweep)
    trial_ops = make_auto_trial_ops(system, overlap_u=ref_ops.overlap, get_rdm1=uhf_get_rdm1)
    meas_ops = MeasOps(overlap=ref_ops.overlap, kernels={k_energy: ref_ops.energy})
    state0 = _state(ops, *_walker_batch(model, mix=0.1))

    want_state, want_obs = jax.jit(lambda s: blocks.block(
        s, sys=system, params=params, ham_data=model.ham, trial_data=None, trial_ops=trial_ops,
        meas_ops=meas_ops, meas_ctx=None, prop_ops=fast, prop_ctx=model.prop_ctx))(state0)
    got_state, got_obs = jax.jit(g.make_block(ops, params, 1, 1))(state0, ops.data)

    np.testing.assert_allclose(float(got_obs["energy"]), float(want_obs.scalars["energy"]), rtol=1e-9)
    np.testing.assert_allclose(float(got_obs["weight"]), float(want_obs.scalars["weight"]), rtol=1e-9)
    for x, y in zip(got_state.walkers, want_state.walkers):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-10)
    np.testing.assert_allclose(np.asarray(got_state.weights), np.asarray(want_state.weights), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(got_state.overlaps), np.asarray(want_state.overlaps), rtol=1e-8)
    np.testing.assert_allclose(float(got_state.e_estimate), float(want_state.e_estimate), rtol=1e-9)
    assert int(got_state.node_encounters) == int(want_state.node_encounters)
    assert np.array_equal(np.asarray(got_state.rng_key), np.asarray(want_state.rng_key))


def test_walker_chunking_is_exact(model):
    ops, _, _ = _ops(model, CHI_TRUNC)
    params = QmcParams(dt=DT, n_walkers=6, n_prop_steps=2, n_blocks=1, n_eql_blocks=0, seed=0)
    state0 = _state(ops, *_walker_batch(model))
    one = _propagate(ops, params, 1, 4)(state0, ops.data)
    three = _propagate(ops, params, 3, 4)(state0, ops.data)
    for x, y in zip(jax.tree_util.tree_leaves(one), jax.tree_util.tree_leaves(three)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-10, atol=1e-14)


def test_choose_chunks_divides_the_population():
    assert g.choose_chunks(1000, 1.0, None) == 1
    assert g.choose_chunks(1000, 1e6, 3e8) == 4  # needs 3.3 -> the next divisor of 1000
    assert g.choose_chunks(7, 1e9, 1e9) == 7


# --------------------------------------------------------------------------
# GPU usage (skipped without a GPU)
# --------------------------------------------------------------------------

@gpu
def test_gpu_step_stays_on_the_device(model):
    """The compiled half steps run with host transfers disallowed and leave every
    array, in float64, on the GPU."""
    ops, _, _ = _ops(model, CHI_TRUNC)
    params = QmcParams(dt=DT, n_walkers=256, n_prop_steps=1, n_blocks=1, n_eql_blocks=0, seed=0)
    state0 = _state(ops, *_walker_batch(model, 256))
    compiled = _propagate(ops, params, 1, 4).lower(state0, ops.data).compile()
    jax.block_until_ready(compiled(state0, ops.data))
    with jax.transfer_guard("disallow"):
        out = compiled(state0, ops.data)
        jax.block_until_ready(out)
    for leaf in jax.tree_util.tree_leaves(out):
        assert {d.platform for d in leaf.devices()} == {"gpu"}
    assert out.weights.dtype == jnp.float64 and out.walkers[0].dtype == jnp.float64


@gpu
def test_gpu_walker_batching_scales(model):
    """16x the walkers must cost far less than 16x the time: the step is a batch
    over walkers, not a loop over them (at L=8 it is launch-bound, so a batched
    step is nearly flat in the walker count)."""
    ops, _, _ = _ops(model, CHI_TRUNC)
    times = {}
    for n in (64, 1024):
        params = QmcParams(dt=DT, n_walkers=n, n_prop_steps=1, n_blocks=1, n_eql_blocks=0, seed=0)
        state0 = _state(ops, *_walker_batch(model, n))
        compiled = _propagate(ops, params, 1, 4).lower(state0, ops.data).compile()
        jax.block_until_ready(compiled(state0, ops.data))
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            jax.block_until_ready(compiled(state0, ops.data))
            best = min(best, time.perf_counter() - t)
        times[n] = best
    assert times[1024] < 8.0 * times[64], times


# --------------------------------------------------------------------------
# Rotated trials (Config.trial_rotation)
# --------------------------------------------------------------------------

def test_spin_rotation_y():
    from trot.trial.mps import spin_rotation_unitary, spin_rotation_y

    np.testing.assert_allclose(spin_rotation_y(90.0), np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0), atol=1e-15)
    np.testing.assert_allclose(spin_rotation_y(0.0), np.eye(2), atol=1e-15)
    assert abs(spin_rotation_unitary(spin_rotation_y(37.0))[3, 3] - 1.0) < 1e-14  # a rotation: det R = 1


def test_rotated_trial_overlaps_and_energies_match_enumeration(model):
    """Config.trial_rotation = 90: build() rotates the DMRG trial and projects it onto (N, N) (trot's
    make_mps_trial); the engine's overlaps and blocked local energies against that trial, with an exact walker
    conversion, equal exact enumeration with the projected trial's own amplitudes."""
    cfg = g.Config(L=L, n_up=N, n_down=N, interaction=U, trial_chi=16, dmrg_sweeps=8, trial_rotation=90.0,
                   walker_channel_chi=None, orbital_plan="rank_exact", n_walkers=6, self_check=False)
    setup = g.build(cfg, verbose=False)
    assert 0.0 < setup.info["trial_sector_weight"] <= 1.0 + 1e-12  # = 1 only for an exact singlet
    trial_np, _ = setup.trial
    np.testing.assert_allclose(g.mps_overlap_host(trial_np, trial_np), 1.0, atol=1e-10)
    amp = _signed_amplitudes(trial_np, model.occ)
    hamp = _signed_amplitudes(g.compress_mps(g.apply_mpo(model.mpo, trial_np)), model.occ)
    ca, cb = _walker_batch(model)
    da = np.stack([np.linalg.det(c[model.rows]) for c in ca])
    db = np.stack([np.linalg.det(c[model.rows]) for c in cb])
    exact_overlap = np.einsum("wi,ij,wj->w", da, amp, db)
    exact_energy = np.einsum("wi,ij,wj->w", da, hamp, db) / exact_overlap
    got_overlap = jax.jit(setup.ops.overlaps)(jnp.asarray(ca), jnp.asarray(cb), setup.ops.data)
    got_energy = jax.jit(setup.ops.energies)(jnp.asarray(ca), jnp.asarray(cb), setup.ops.data)
    np.testing.assert_allclose(np.asarray(got_overlap), exact_overlap, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(got_energy), exact_energy, rtol=1e-8)
