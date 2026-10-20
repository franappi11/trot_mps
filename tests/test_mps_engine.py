"""The batched MPS-CPMC engine, trot/gmps/engine.py, on CPU or GPU.

The engine compiles the walker conversion (Fishman-White gates, charge-sector factorisations batched over walkers,
spins and sectors), contracts the walker channels with fixed MPS blocks without forming the d=4 walker, and runs the
CPMC half steps. The references are never the engine:

- exact Fock-space enumeration (tests/helpers/hubbard_fock.py): overlaps and local energies of exactly converted
  walkers, non-orthonormal ones included, and the field sweep's proposals;
- the NumPy host oracle of the conversion, gpu.channel_mps_host (the per-sector algorithm of
  trot.gmps.utils.channel_mps), wherever walkers are truncated;
- trot.gmps.utils for the static circuit: the branches of _factor_block, the bond labels of channel_mps and
  plan_bonds;
- trot.walkers._qr for CholeskyQR2;
- a NumPy CPMC step written here (trot.prop.cpmc's order, RNG use and weight-floor rule, fresh overlaps) on
  enumerated amplitudes, for the field sweep and the half steps.

Systems: an L=8 open chain and a 2x4 lattice, open and antiperiodic along y; U=4, (N_up, N_dn) = (4, 4). No trial
needs pyblock3: a random MPS with exact (N_up, N_dn) labels, the same MPS spin-rotated (particle-number labels, used as
it is) and an unrestricted determinant (mps_trial_from_sd). The walker plans are frozen on the natural orbitals of the
random trial: rank_exact without truncation ("exact") and adaptive with a per-channel bond cap ("truncated").
trot.prop.mps_cpmc.block against trot.prop.blocks.block is tested in test_mps_unified.py. The tests marked gpu only
run on a GPU backend.

    python -m pytest tests/test_mps_engine.py
"""

from trot import config

config.configure_once()

import time
from collections import Counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from tests.helpers import hubbard_fock as hf
from trot.core.system import System
from trot.gmps import dense_conversion as dc
from trot.gmps import engine
from trot.gmps import trials
from trot.gmps import utils as gmps_utils
from trot.ham.hubbard import HamHubbard, hopping_matrix, square_hopping_matrix
from trot.meas.mps import build_mps_meas_ctx, hubbard_mpo_from_h1, szsz_fns, trial_times_h
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.types import PropState, QmcParams, QmcParamsMps
from trot.trial.mps import (
    compress_mps_qn,
    make_dense_walker_plan,
    make_mps_trial,
    make_walker_plan,
    make_walker_plan_from_plans,
    mps_trial_from_sd,
    rotate_spin,
    spin_rotation_y,
)
from trot.walkers import _qr as qr_with_det

L, NELEC, U = 8, (4, 4), 4.0
DT = 0.1  # large enough that the weight floor 0.5 removes HS proposals
CHI_TRUNC = 4  # per-channel walker bond cap that truncates visibly at 8 sites
N_WALKERS = 6
ROTATION = spin_rotation_y(70.0)
CASES = {
    "L8": hopping_matrix(L, 1.0),
    "sq2x4oo": square_hopping_matrix(2, 4, 1.0),
    "sq2x4oa": square_hopping_matrix(2, 4, 1.0, "open", "antiperiodic"),
}
OCC = tuple(hf.sector_basis(L, n)[1] for n in NELEC)  # 0/1 occupations of every configuration, per spin
PLAN_MODES = [("rank_exact", None), ("rank_exact", CHI_TRUNC), ("adaptive", CHI_TRUNC)]
PLAN_IDS = ["rank_exact", "rank_exact-chi4", "adaptive-chi4"]
BUCKETS = {"2": (2,), "2-4": (2, 4), "8-16": (8, 16)}
SWEEP_CASES = [("exact", "sz"), ("exact", "number"), ("exact", "sd"), ("truncated", "sz"), ("truncated", "number")]
gpu = pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU backend")


# ---------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------


def _quiet(*args, **kwargs):
    pass


def _random_labelled_mps(rng, max_multiplicity=2):
    """A random real d=4 MPS with exact (N_up, N_dn) bond labels from (0, 0) to NELEC: every label on such a path,
    1..max_multiplicity indices each, random entries in every charge-allowed block (a generic trial, no pyblock3)."""
    labels = []
    for bond in range(L + 1):
        rest = L - bond
        allowed = np.array([(a, b) for a in range(max(0, NELEC[0] - rest), min(bond, NELEC[0]) + 1)
                            for b in range(max(0, NELEC[1] - rest), min(bond, NELEC[1]) + 1)])
        repeats = rng.integers(1, max_multiplicity + 1, len(allowed)) if 0 < bond < L else np.ones(len(allowed), int)
        labels.append(np.repeat(allowed, repeats, axis=0))
    tensors = []
    for site in range(L):
        left, right = labels[site], labels[site + 1]
        A = np.zeros((len(left), 4, len(right)))
        for p, delta in enumerate(gmps_utils.PHYSICAL_CHARGE):
            allowed = np.all(left[:, None, :] + delta == right[None, :, :], axis=-1)
            A[:, p, :] = np.where(allowed, rng.standard_normal(allowed.shape), 0.0)
        tensors.append(A / np.sqrt(len(left)))
    return tensors, tuple(labels)


def _mps_params(**overrides):
    settings = dict(dt=DT, n_walkers=N_WALKERS, n_prop_steps=1, n_blocks=1, n_eql_blocks=0, seed=0, n_chunks=1,
                    auto_n_chunks=False, plan_reference="natural")
    settings.update(overrides)
    return QmcParamsMps(**settings)


def _qmc_params(**overrides):
    settings = dict(dt=DT, n_walkers=N_WALKERS, n_prop_steps=1, n_blocks=1, n_eql_blocks=0, weight_floor=1e-8,
                    seed=0)
    settings.update(overrides)
    return QmcParams(**settings)


def _mps_overlap(a, b):
    """<a|b> of two real MPS with equal physical dimensions (d=2 channels or d=4 trials)."""
    env = np.ones((1, 1))
    for x, y in zip(a, b):
        env = np.einsum("ab,apc,bpd->cd", env, np.asarray(x), np.asarray(y))
    return float(env.reshape(()))


@pytest.fixture(scope="module", params=list(CASES))
def model(request):
    """One lattice: three trials, the exact and the truncated walker plan, walkers diffused from the plans' reference,
    the (4, 4) amplitudes <config|trial> and <config|H|trial> (hubbard_fock's alpha-block convention) and the
    charge-labelled, compressed H|trial> of the trials with small bonds."""
    h1 = CASES[request.param]
    ham = HamHubbard(h1=jnp.asarray(h1), u=U)
    system = System(norb=L, nelec=NELEC, walker_kind="unrestricted")
    sz = make_mps_trial(*_random_labelled_mps(np.random.default_rng(7)), nelec=NELEC)
    number = make_mps_trial(rotate_spin([np.asarray(A) for A in sz.tensors], ROTATION), nelec=NELEC)
    sd = mps_trial_from_sd(*hf.staggered_determinant(h1, *NELEC, field=0.7))
    trial_set = {"sz": sz, "number": number, "sd": sd}
    plans = {
        "exact": make_walker_plan(ham, sz, system, _mps_params(orbital_plan="rank_exact", walker_channel_chi=None,
                                                               sector_buckets=(), walker_qr="native")),
        "truncated": make_walker_plan(ham, sz, system, _mps_params(orbital_plan="adaptive",
                                                                   walker_channel_chi=CHI_TRUNC,
                                                                   sector_buckets=(8, 16), walker_qr="cholesky")),
    }
    reference = tuple(np.asarray(R) for R in plans["exact"].reference)
    H = hf.hubbard_sector_hamiltonian(h1, U, *NELEC)
    amps = {}
    for kind, trial in trial_set.items():
        psi = hf.mps_sector_amplitudes([np.asarray(A) for A in trial.tensors], *NELEC)
        amps[kind] = (psi, (H @ psi.ravel()).reshape(psi.shape))
    del H
    W = hubbard_mpo_from_h1(h1, U)
    htrial = {}
    for kind in ("sz", "number"):
        tensors = [np.asarray(A) for A in trial_set[kind].tensors]
        h_np, h_q = compress_mps_qn(*trial_times_h(W, tensors, trial_set[kind].charge_arrays()))
        htrial[kind] = (h_np, h_q, _mps_overlap(h_np, tensors))  # <T|H|T>: the trial is normalised
    return SimpleNamespace(
        name=request.param, h1=h1, ham=ham, system=system, trials=trial_set, plans=plans, reference=reference,
        walkers=hf.random_field_walkers(h1, U, 0.1, *reference, n=N_WALKERS, steps=20, seed=3),
        amps=amps, htrial=htrial, contexts={}, props={}, jitted={})


def _prop(model, dt=DT):
    """trot's HubbardCpmcCtx: exp(-dt K/2) and the HS factors."""
    if dt not in model.props:
        model.props[dt] = _build_prop_ctx(model.ham, dt)
    return model.props[dt]


def _ctx(model, plan_kind, trial_kind="sz", kernel="blocked"):
    """build_mps_meas_ctx (trot.meas.mps) for a plan and a trial, made once per model: the model's labelled,
    compressed H|trial> ("blocked") or H|trial> compressed densely ("dense")."""
    key = (plan_kind, trial_kind, kernel)
    if key not in model.contexts:
        htrial = model.htrial[trial_kind] if kernel == "blocked" else None
        model.contexts[key] = build_mps_meas_ctx(model.ham, model.trials[trial_kind], plan=model.plans[plan_kind],
                                                 kernel=kernel, htrial=htrial)
    return model.contexts[key]


def _sweep_engine(model, plan_kind, trial_kind, dt=DT):
    """kernels_for(plan, labels, energy=None) and its DeviceData: the trial blocks (plan_blocks) and the propagator."""
    plan, trial = model.plans[plan_kind], model.trials[trial_kind]
    kernels = engine.kernels_for(plan, trial.charges, energy=None)
    prop = _prop(model, dt)
    data = engine.DeviceData(engine.plan_blocks(trial.tensors, kernels.overlap_plan), (), (), prop.exp_h1_half,
                             prop.hs_constant)
    return kernels, data


def _step_batch(model, plan_kind, trial_kind):
    """kernels.step_batch jitted once per model, plan and trial, with the sweep flag (lax.cond, as make_half_step
    uses it) and the weight floor traced."""
    key = ("step_batch", plan_kind, trial_kind)
    if key not in model.jitted:
        kernels, _ = _sweep_engine(model, plan_kind, trial_kind)
        model.jitted[key] = jax.jit(lambda a, b, r, d, sweep, floor: kernels.step_batch(a, b, r, sweep, d, floor))
    return model.jitted[key]


def _batch(model, n=None, mix=True, seed=1):
    """The model's walkers stacked (repeated to n); with mix every other one is made non-orthonormal (the same
    determinant times a known scalar, hubbard_fock.nonorthonormal)."""
    rng = np.random.default_rng(seed)
    pairs = [(hf.nonorthonormal(a, rng), hf.nonorthonormal(b, rng)) if mix and k % 2 else (a, b)
             for k, (a, b) in enumerate(model.walkers)]
    ca, cb = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
    if n is not None:
        reps = -(-n // len(ca))
        ca, cb = np.concatenate([ca] * reps)[:n], np.concatenate([cb] * reps)[:n]
    return ca, cb


def _state(ca, cb, overlaps, weights=None, e0=-6.0, seed=0):
    n = len(ca)
    return PropState(walkers=(jnp.asarray(ca), jnp.asarray(cb)),
                     weights=jnp.asarray(np.ones(n) if weights is None else weights, jnp.float64),
                     overlaps=jnp.asarray(overlaps, jnp.float64), rng_key=jax.random.PRNGKey(seed),
                     pop_control_ene_shift=jnp.asarray(e0, jnp.float64), e_estimate=jnp.asarray(e0, jnp.float64),
                     node_encounters=jnp.zeros((), jnp.int64))


def _half_steps(kernels, params, n_chunks, n_half):
    """n_half half steps of make_half_step in one jitted scan, as trot.prop.mps_cpmc runs them."""
    half = engine.make_half_step(kernels, params, n_chunks)
    return jax.jit(lambda state, data: lax.scan(lambda s, i: (half(s, i, data), None), state, jnp.arange(n_half))[0])


def _assert_states_close(got, want, rtol=1e-9):
    for x, y in zip(got.walkers, want.walkers):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.weights), np.asarray(want.weights), rtol=rtol, atol=1e-14)
    np.testing.assert_allclose(np.asarray(got.overlaps), np.asarray(want.overlaps), rtol=rtol, atol=0)
    np.testing.assert_allclose(float(got.pop_control_ene_shift), float(want.pop_control_ene_shift), rtol=rtol)
    assert float(got.e_estimate) == float(want.e_estimate)
    assert int(got.node_encounters) == int(want.node_encounters)
    np.testing.assert_array_equal(np.asarray(got.rng_key), np.asarray(want.rng_key))


# ---------------------------------------------------------------------------------------------
# References: amplitudes, the CPMC sweep and step in NumPy, the circuit's sectors
# ---------------------------------------------------------------------------------------------


def _amplitude(tensors, occupation):
    """<occupation|MPS> of a d=2 channel MPS."""
    v = np.ones((1, 1))
    for A, n in zip(tensors, occupation):
        v = v @ np.asarray(A)[:, int(n), :]
    return float(v[0, 0])


def _channel_amplitudes(C, spin, plan=None, bond=None):
    """A spin channel's amplitudes over its configurations (hubbard_fock.sector_basis order): det C[A] for an exact
    conversion (plan None), else det R times the host oracle's (channel_mps_host) truncated channel MPS and gauge."""
    if plan is None:
        rows, _ = hf.sector_basis(L, NELEC[spin])
        return np.array([np.linalg.det(C[r]) for r in rows])
    Q, R = np.linalg.qr(C)
    tensors, _, gauge = engine.channel_mps_host(Q, plan, bond)
    return np.linalg.det(R) * gauge * np.array([_amplitude(tensors, o) for o in OCC[spin]])


def _walker_amplitudes(model, plan_kind, ca, cb):
    """Both channels' amplitudes as the plan converts the walker: exactly (the determinants) for the exact plan,
    truncated (the host oracle) for the truncated one."""
    if plan_kind == "exact":
        return _channel_amplitudes(ca, 0), _channel_amplitudes(cb, 1)
    plan = model.plans[plan_kind]
    return tuple(_channel_amplitudes(C, s, plan.orbital_plans[s], plan.bond_plans[s]) for s, C in enumerate((ca, cb)))


def _reference_values(model, plan_kind, trial_kind, ca, cb):
    """<trial|walker> and <trial|H|walker>/<trial|walker> of each walker as the plan converts it, with the trial's
    enumerated (4, 4) amplitudes."""
    psi, hpsi = model.amps[trial_kind]
    overlaps, energies = [], []
    for wa, wb in zip(ca, cb):
        a, b = _walker_amplitudes(model, plan_kind, wa, wb)
        overlaps.append(a @ psi @ b)
        energies.append(a @ hpsi @ b / overlaps[-1])
    return np.array(overlaps), np.array(energies)


def _reference_sweep(a, b, psi, hs, randoms, floor):
    """The diagonal HS sweep of one walker, site by site as trot's CPMC (fresh overlaps), on amplitudes: a and b over
    the channels' configurations, psi the trial's. The proposal of field f at a site is the exact overlap of the
    trial with the walker after the earlier sites' chosen factors and this site's factor of f, which multiplies a
    configuration by hs[f, spin] per particle of that spin on the site. Returns the fields, the overlap before and
    after, the weight factor, the node count and how many positive ratios the floor removed."""
    overlap_in = overlap = a @ psi @ b
    fields, nodes, floored, log_weight = [], 0, 0, 0.0
    for site in range(L):
        factors = [(hs[f, 0] ** OCC[0][:, site], hs[f, 1] ** OCC[1][:, site]) for f in (0, 1)]
        proposed = np.array([(a * fa) @ psi @ (b * fb) for fa, fb in factors])
        ratios = proposed / overlap
        floored += int(np.sum((ratios > 0.0) & (ratios <= floor)))
        ratios = np.where(ratios <= floor, 0.0, ratios)  # the floor, and the constraint
        nodes += int(np.sum(ratios <= 0.0))
        probabilities = 0.5 * ratios
        norm = probabilities.sum() + 1.0e-13
        field = 0 if randoms[site] < probabilities[0] / norm else 1
        a, b = a * factors[field][0], b * factors[field][1]
        overlap = proposed[field]
        log_weight += np.log(norm)
        fields.append(field)
    return np.array(fields), overlap_in, overlap, np.exp(log_weight), nodes, floored


def _reference_half_steps(model, plan_kind, trial_kind, state, n_half, params):
    """n_half half steps of trot's CPMC step in NumPy: exp(-dt K/2) and the overlap ratio (weight floor, cap) on every
    half step, the HS sweep on even ones (one RNG split per step), the population-control shift on odd ones. Walkers
    are converted as _walker_amplitudes does. Returns the state and how many positive ratios the floor removed."""
    psi = model.amps[trial_kind][0]
    prop = _prop(model, params.dt)
    hs, expk = np.asarray(prop.hs_constant), np.asarray(prop.exp_h1_half)
    floor, cap = float(params.weight_floor), float(params.weight_cap)
    up, dn = (np.array(w) for w in state.walkers)
    weights, overlaps = np.array(state.weights), np.array(state.overlaps)
    key, shift, e_estimate = state.rng_key, float(state.pop_control_ene_shift), float(state.e_estimate)
    nodes, floored = int(state.node_encounters), 0
    for index in range(n_half):
        even = index % 2 == 0
        up, dn = expk @ up, expk @ dn
        now, after, factor = np.empty(len(up)), np.empty(len(up)), np.ones(len(up))
        if even:
            key, subkey = jax.random.split(key)
            randoms = np.asarray(jax.random.uniform(subkey, up.shape[:2]))
        for w in range(len(up)):
            a, b = _walker_amplitudes(model, plan_kind, up[w], dn[w])
            if even:
                fields, now[w], after[w], factor[w], n, f = _reference_sweep(a, b, psi, hs, randoms[w], floor)
                up[w], dn[w] = up[w] * hs[fields, 0][:, None], dn[w] * hs[fields, 1][:, None]
                nodes, floored = nodes + n, floored + f
            else:
                now[w] = after[w] = a @ psi @ b
        ratio = now / overlaps
        floored += int(np.sum((ratio > 0.0) & (ratio <= floor)))
        ratio = np.where(ratio <= floor, 0.0, ratio)
        nodes += int(np.sum(ratio <= 0.0))
        weights = weights * ratio
        weights = np.where(weights > cap, 0.0, weights) * factor
        if not even:
            weights = weights * np.exp(params.dt * shift)
            weights = np.where(weights > cap, 0.0, weights)
            shift = e_estimate - params.pop_control_damping * np.log(max(np.mean(weights), 1.0e-300)) / params.dt
        overlaps = after
    return PropState((up, dn), weights, overlaps, np.asarray(key), shift, e_estimate, nodes), floored


def _channel_plans(R, mode="rank_exact", chi=None):
    """trot.gmps.utils' orbital plan of one channel frozen on R and, with a cap, its bond plan (plan_bonds)."""
    plan = gmps_utils.make_orbital_plan(R, mode)
    return plan, None if chi is None else gmps_utils.plan_bonds(R, plan, chi)


def _last_centre(C, plan):
    """Where channel_mps leaves the orthogonality centre: right of the last gate it applies (the first planned)."""
    angles, _ = gmps_utils.channel_angles(np.asarray(C), plan, xp=np)
    return angles[0][0] + 1


def _jit_channel_mps(plan, bond):
    """trot.gmps.utils.channel_mps (the reference conversion in jnp) jitted: tensors and gauge."""
    def convert(C):
        tensors, _, gauge = gmps_utils.channel_mps(C, plan, bond)
        return tensors, gauge
    return jax.jit(convert)


def _host_factorisations(monkeypatch, C, plan, bond):
    """channel_mps_host on C with every factorisation recorded: per centre move or gate split, in order, how many
    charge sectors take each branch of trot.gmps.utils._factor_block (closed form for a single row or column, QR when
    exact, the eigh of the smaller Gram matrix when truncating)."""
    ops = []
    factor_block, shift_centre, gate_pair = gmps_utils._factor_block, gmps_utils._shift_centre, engine.gate_pair

    def factor(block, rank, truncate, xp=jnp):
        rows, cols = block.shape
        if min(rows, cols) == 1:
            branch = "row1" if rows == 1 else "col1"
        else:
            branch = "eigh_rows" if truncate and rows <= cols else "eigh_cols" if truncate else "qr"
        ops[-1][1][branch] += 1
        return factor_block(block, rank, truncate, xp)

    def move(tensors, charges, site, step, xp=jnp):
        ops.append(("move", Counter()))
        return shift_centre(tensors, charges, site, step, xp)

    def gate(A, B, theta, xp=jnp):
        ops.append(("gate", Counter()))
        return gate_pair(A, B, theta, xp)

    monkeypatch.setattr(gmps_utils, "_factor_block", factor)
    monkeypatch.setattr(gmps_utils, "_shift_centre", move)
    monkeypatch.setattr(engine, "_factor_block", factor)
    monkeypatch.setattr(engine, "gate_pair", gate)
    try:
        engine.channel_mps_host(C, plan, bond)
    finally:
        monkeypatch.undo()
    return ops


def _class_sectors(op, cls, spin=0):
    """(kind, rows, columns) of a channel's real charge sectors in one class of a factorisation (padding dropped)."""
    real = cls.gather[spin] != op.n_rows * op.n_cols  # (S, r, c)
    rows, cols = real.any(axis=-1).sum(axis=-1), real.any(axis=-2).sum(axis=-1)
    return [(cls.kind, int(r), int(c)) for r, c in zip(rows, cols) if r]


def _sector_shapes(op, spin=0):
    return [sector for cls in op.classes for sector in _class_sectors(op, cls, spin)]


def _circuit_factorisations(circuit, spin=0):
    """Per factorisation of a compiled circuit: its kind and how many of the channel's sectors each method takes."""
    return [(op.kind, Counter(kind for kind, _, _ in _sector_shapes(op, spin))) for op in circuit.ops]


def _n_classes(converter):
    return sum(len(op.classes) for circuit in converter.circuits for op in circuit.ops)


def _bucket_of(size, buckets):
    return next((i for i, bound in enumerate(buckets) if size <= bound), len(buckets))


def _bucket_size(kind, rows, cols):
    """The size a sector is bucketed by: its Gram dimension for an eigh, its larger side for a QR."""
    return rows if kind == "eigh_rows" else cols if kind == "eigh_cols" else max(rows, cols)


def _assert_layout_blocks(plan, tensors, labels):
    """A factorized plan has one transition per walker label pair and charge-conserving physical state, and each
    transition holds the fixed MPS's (left label, p, right label) block, zero-padded."""
    labels = [gmps_utils.label_array(q) for q in labels]
    for site, (sp, B) in enumerate(zip(plan.sites, plan.blocks)):
        shared, following = plan.shared[site], set(plan.shared[site + 1])
        steps = {(c, p) for c in shared for p, (na, nb) in enumerate(gmps_utils.PHYSICAL_CHARGE.tolist())
                 if (c[0] + na, c[1] + nb) in following}
        assert len(sp.src) == len(steps)
        assert {(shared[s], int(p)) for s, p in zip(sp.src, sp.physical)} == steps
        A = np.asarray(tensors[site])
        for t, (left, p, right) in enumerate(sp.fixed_keys):
            rows = np.flatnonzero(np.all(labels[site] == np.asarray(left), axis=1))
            cols = np.flatnonzero(np.all(labels[site + 1] == np.asarray(right), axis=1))
            want = np.zeros(B.shape[1:])
            want[:len(rows), :len(cols)] = A[np.ix_(rows, [p], cols)][:, 0]
            np.testing.assert_array_equal(B[t], want)


def _jit_fixed_blocks(layout):
    return jax.jit(lambda tensors: engine.fixed_blocks(tensors, layout))


def _jit_chunked_overlaps(kernels, n_chunks):
    return jax.jit(lambda a, b, d: engine.chunked(lambda x, y: kernels.overlaps(x, y, d), n_chunks, a, b))


# ---------------------------------------------------------------------------------------------
# The compiled conversion
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode, chi", PLAN_MODES, ids=PLAN_IDS)
def test_compiled_circuit_labels_are_the_conversions_labels(model, mode, chi):
    """compile_circuit replays channel_mps on labels alone: its gate order, final bond labels and pads are those of
    trot.gmps.utils.channel_mps and plan_bonds; make_walker_plan's channel labels are those of its converter."""
    for R in model.reference:
        plan, bond = _channel_plans(R, mode, chi)
        circuit = engine.compile_circuit((plan,), (bond,))
        _, labels, _ = gmps_utils.channel_mps(jnp.asarray(R), plan, bond)
        assert len(circuit.charges[0]) == len(labels) == L + 1
        assert all(np.array_equal(a, b) for a, b in zip(circuit.charges[0], labels))
        assert circuit.pads == tuple(len(q) for q in labels)
        angles, _ = gmps_utils.channel_angles(R, plan, xp=np)
        assert circuit.gate_sites == tuple(int(site) for site, _ in reversed(angles))
        if bond is not None:
            assert all(np.array_equal(a, b) for a, b in zip(circuit.charges[0], bond.charges))
    for plan in model.plans.values():
        for got, want in zip(engine.converter_for(plan).charges, plan.channel_charges):
            assert all(np.array_equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize("mode, chi", PLAN_MODES, ids=PLAN_IDS)
def test_sector_methods_follow_factor_block(model, monkeypatch, mode, chi):
    """Every factorisation of the compiled circuit factors each charge sector by the branch trot.gmps.utils.
    _factor_block takes for it in the host oracle, op by op, alone or with both channels in one batch: centre moves
    never truncate, and only sectors cut below their rank use an eigh. circuit_stats counts the same ops."""
    plans = [_channel_plans(R, mode, chi) for R in model.reference]
    recorded = [_host_factorisations(monkeypatch, R, plan, bond) for R, (plan, bond) in zip(model.reference, plans)]
    for (plan, bond), ops in zip(plans, recorded):
        circuit = engine.compile_circuit((plan,), (bond,))
        assert _circuit_factorisations(circuit) == ops
        stats = engine.circuit_stats(circuit)
        assert (stats["spins"], stats["ops"]) == (1, len(ops))
        assert stats["gates"] == len(circuit.gate_sites) == sum(kind == "gate" for kind, _ in ops)
        assert stats["moves"] == sum(kind == "move" for kind, _ in ops)
        assert stats["closed_form_batches"] > 0
        assert (stats["eigh_calls"] > 0) == any(k.startswith("eigh") for _, counts in ops for k in counts)
    if mode == "rank_exact":  # one gate sequence: both channels in one batch, each with its own sectors and methods
        both = engine.compile_circuit(tuple(p for p, _ in plans), tuple(b for _, b in plans))
        for spin, ops in enumerate(recorded):
            assert _circuit_factorisations(both, spin) == ops
    used = {"move": set(), "gate": set()}
    for ops in recorded:
        for kind, counts in ops:
            used[kind] |= set(counts)
    assert not used["move"] & {"eigh_rows", "eigh_cols"}
    assert bool(used["gate"] & {"eigh_rows", "eigh_cols"}) == (chi is not None)  # the cap truncates some sector


@pytest.mark.parametrize("mode", ["rank_exact", "maximal", "adaptive"])
def test_batched_conversion_is_exact(model, mode):
    """Without truncation the batched conversion times its gauge gives every determinant of both channels: for any
    walker with the rank_exact and maximal plans, for the reference it was planned on with the adaptive one."""
    plans = [gmps_utils.make_orbital_plan(R, mode) for R in model.reference]
    convert = jax.jit(engine.make_converter(*plans).convert)
    walkers = [model.reference] if mode == "adaptive" else model.walkers
    tol = 1e-8 if mode == "adaptive" else 1e-12
    for ca, cb in walkers:
        alpha, beta, gauges = convert(jnp.asarray(ca), jnp.asarray(cb))
        for spin, (C, tensors, gauge) in enumerate(zip((ca, cb), (alpha, beta), gauges)):
            got = float(gauge) * np.array([_amplitude(tensors, o) for o in OCC[spin]])
            np.testing.assert_allclose(got, _channel_amplitudes(C, spin), rtol=0, atol=tol)


@pytest.mark.parametrize("chi", [None, CHI_TRUNC], ids=["exact", "chi4"])
def test_batched_conversion_ends_in_mixed_canonical_form(model, chi):
    """Every batched factor is an isometry and conserves the charge, so each channel ends left-isometric before the
    orthogonality centre, right-isometric after it, and every nonzero entry respects the circuit's labels."""
    plans = [_channel_plans(R, "rank_exact", chi) for R in model.reference]
    converter = engine.make_converter(plans[0][0], plans[1][0], plans[0][1], plans[1][1])
    convert = jax.jit(converter.convert)
    for ca, cb in model.walkers:
        alpha, beta, _ = convert(jnp.asarray(ca), jnp.asarray(cb))
        for C, (plan, _), tensors, labels in zip((ca, cb), plans, (alpha, beta), converter.charges):
            centre = _last_centre(C, plan)
            for i, A in enumerate(map(np.asarray, tensors)):
                if i < centre:
                    M = A.reshape(-1, A.shape[2])
                    np.testing.assert_allclose(M.T @ M, np.eye(M.shape[1]), atol=1e-10)
                elif i > centre:
                    M = A.reshape(A.shape[0], -1)
                    np.testing.assert_allclose(M @ M.T, np.eye(M.shape[0]), atol=1e-10)
                for a, n, b in np.argwhere(np.abs(A) > 1e-12):
                    assert labels[i][a] + n == labels[i + 1][b]


@pytest.mark.parametrize("spin_batch", [True, False])
def test_truncated_batched_conversion_matches_the_host_oracle(model, spin_batch):
    """With truncation the batched conversion keeps the host oracle's Schmidt subspaces: g_dev <host|dev> equals
    g_host <host|host> per spin, with and without spin batching; so does trot.gmps.utils.channel_mps."""
    plans = [_channel_plans(R, "rank_exact", CHI_TRUNC) for R in model.reference]
    assert all(bond.reference_discarded_weight > 0 for _, bond in plans)
    converter = engine.make_converter(plans[0][0], plans[1][0], plans[0][1], plans[1][1], spin_batch=spin_batch)
    assert converter.spin_batched == spin_batch and len(converter.circuits) == (1 if spin_batch else 2)
    convert = jax.jit(converter.convert)
    originals = [_jit_channel_mps(plan, bond) for plan, bond in plans]
    for ca, cb in model.walkers:
        alpha, beta, gauges = convert(jnp.asarray(ca), jnp.asarray(cb))
        for C, (plan, bond), dev, g_dev, original in zip((ca, cb), plans, (alpha, beta), gauges, originals):
            host, _, g_host = engine.channel_mps_host(C, plan, bond)
            reference = g_host * _mps_overlap(host, host)
            assert abs(float(g_dev) * _mps_overlap(host, dev) - reference) <= 1e-10 * abs(reference)
            tensors, g_original = original(jnp.asarray(C))
            assert abs(float(g_original) * _mps_overlap(host, tensors) - reference) <= 1e-10 * abs(reference)


def test_plan_bonds_with_given_counts_rebuild_and_transfer_an_allocation(model):
    """plan_bonds(..., counts=...) keeps the given number of states in each charge sector. A walker's own top-chi
    counts (recorded with spectra=, whose dropped weights add up to the discarded weight) rebuild its own bond plan;
    the bond structure depends on the counts only, so the same counts planned on the reference give the walker's bond
    labels and, in the host oracle, its own truncation; make_walker_plan_from_plans runs such a trained bond plan on
    the device, and the device conversion keeps the host oracle's Schmidt subspaces."""
    plans, trained = [], []
    for R, C in zip(model.reference, model.walkers[1]):
        plan = gmps_utils.make_orbital_plan(R, "adaptive")
        Q = np.linalg.qr(np.asarray(C))[0]
        spectra = []
        own = gmps_utils.plan_bonds(Q, plan, CHI_TRUNC, spectra=spectra)
        assert own.reference_discarded_weight > 1e-6 and len(spectra) == len(own.kept_per_sector)
        dropped = sum(float(np.sum(w[k:])) for gate, kept in zip(spectra, own.kept_per_sector)
                      for w, k in zip(gate.values(), kept))
        np.testing.assert_allclose(dropped, own.reference_discarded_weight, rtol=1e-12)
        counts = [dict(zip(gate.keys(), kept)) for gate, kept in zip(spectra, own.kept_per_sector)]
        again = gmps_utils.plan_bonds(Q, plan, counts=counts)
        assert again.kept_per_sector == own.kept_per_sector
        assert all(np.array_equal(a, b) for a, b in zip(again.charges, own.charges))
        np.testing.assert_allclose(again.reference_discarded_weight, own.reference_discarded_weight, rtol=1e-12)
        moved = gmps_utils.plan_bonds(R, plan, counts=counts)  # the walker's counts, planned on the reference
        assert all(np.array_equal(a, b) for a, b in zip(moved.charges, own.charges))
        tensors, _, _ = engine.channel_mps_host(Q, plan, moved)
        np.testing.assert_allclose(1.0 - _mps_overlap(tensors, tensors), own.reference_discarded_weight,
                                   rtol=1e-6, atol=1e-12)
        plans.append(plan)
        trained.append(moved)
    walker_plan = make_walker_plan_from_plans(*model.reference, plans, trained)
    convert = jax.jit(engine.converter_for(walker_plan).convert)
    for ca, cb in model.walkers:
        alpha, beta, gauges = convert(jnp.asarray(ca), jnp.asarray(cb))
        for C, plan, bond, dev, g_dev in zip((ca, cb), plans, trained, (alpha, beta), gauges):
            host, _, g_host = engine.channel_mps_host(C, plan, bond)
            reference = g_host * _mps_overlap(host, host)
            assert abs(float(g_dev) * _mps_overlap(host, dev) - reference) <= 1e-10 * abs(reference)


def test_trained_counts_follow_their_rules_and_run_on_the_device(model):
    """trot.gmps.allocation: trained on one walker, every rule at its bond gives that walker's own top-chi counts;
    trained on several, the rules keep at most chi states per gate, the union holds every walker's own counts and is
    the 100th percentile; trained_walker_plan keeps the plan's gates and reference, and the device conversion with
    its trained counts keeps the host oracle's Schmidt subspaces."""
    from trot.gmps import allocation

    plan = model.plans["truncated"]
    for s in range(2):
        channel = [w[s] for w in model.walkers]
        own = allocation.allocations_of(channel, plan.orbital_plans[s], CHI_TRUNC)
        alone = [{q: k for q, k in gate.items() if k} for gate in own[0][0]]
        for rule in allocation.RULES:
            assert allocation.train_counts(own[:1], CHI_TRUNC, rule) == alone
            counts = allocation.train_counts(own, CHI_TRUNC, rule)
            assert len(counts) == len(own[0][0]) and all(0 < sum(gate.values()) <= CHI_TRUNC for gate in counts)
        union = allocation.train_counts(own, rule="union")
        assert union == allocation.train_counts(own, rule="p100")
        assert all(union[g].get(q, 0) >= k for counts, _, _ in own for g, gate in enumerate(counts)
                   for q, k in gate.items())
        with pytest.raises(ValueError):
            allocation.train_counts(own, CHI_TRUNC, rule="majority")
    trained = allocation.trained_walker_plan(plan, tuple(np.stack([w[s] for w in model.walkers]) for s in range(2)),
                                             CHI_TRUNC, "consensus")
    assert all(a is b for a, b in zip(trained.orbital_plans, plan.orbital_plans))
    assert all(np.array_equal(a, b) for a, b in zip(trained.reference, plan.reference))
    assert all(b.max_bond <= CHI_TRUNC for b in trained.bond_plans)
    convert = jax.jit(engine.converter_for(trained).convert)
    for ca, cb in model.walkers:
        alpha, beta, gauges = convert(jnp.asarray(ca), jnp.asarray(cb))
        for C, orbital_plan, bond, dev, g_dev in zip((ca, cb), trained.orbital_plans, trained.bond_plans,
                                                     (alpha, beta), gauges):
            host, _, g_host = engine.channel_mps_host(C, orbital_plan, bond)
            reference = g_host * _mps_overlap(host, host)
            assert abs(float(g_dev) * _mps_overlap(host, dev) - reference) <= 1e-10 * abs(reference)


def test_drift_check_scores_walkers_with_the_host_oracle(model):
    """allocation.discarded_under is the norm channel_mps_host's conversion loses, per walker and spin;
    slots_changed is 0 between an allocation and itself; drift_check takes the trained candidate exactly when it
    lowers the mean discarded weight by more than the gain."""
    from trot.gmps import allocation

    plan = model.plans["truncated"]
    walkers = tuple(np.stack([w[s] for w in model.walkers]) for s in range(2))
    lost = allocation.discarded_under(walkers, plan.orbital_plans, plan.bond_plans)
    assert lost.shape == (len(model.walkers), 2) and np.max(lost) > 1e-6
    for k, walker in enumerate(model.walkers):
        for s in range(2):
            Q = np.linalg.qr(np.asarray(walker[s]))[0]
            tensors, _, _ = engine.channel_mps_host(Q, plan.orbital_plans[s], plan.bond_plans[s])
            np.testing.assert_allclose(lost[k, s], 1.0 - _mps_overlap(tensors, tensors), rtol=1e-12, atol=1e-14)
    assert allocation.slots_changed(plan.bond_plans, plan.bond_plans) == 0.0
    taken, info = allocation.drift_check(plan, walkers, CHI_TRUNC, gain=-1.0)
    assert taken is not None and info["accepted"] and 0.0 <= info["slots_changed"] <= 1.0
    kept, info = allocation.drift_check(plan, walkers, CHI_TRUNC, gain=1.0)
    assert kept is None and not info["accepted"] and info["discarded_current"] > 0.0


def test_spin_batching_keeps_each_channels_own_truncation(model):
    """Alpha and beta with different kept counts (hence different bond labels) still convert in one batch, each
    with its own labels and bond dimensions, exactly as the host oracle converts it alone."""
    (plan_a, bond_a), (plan_b, bond_b) = (_channel_plans(model.reference[0], "rank_exact", CHI_TRUNC),
                                          _channel_plans(model.reference[1], "rank_exact", CHI_TRUNC - 1))
    assert any(len(x) != len(y) for x, y in zip(bond_a.charges, bond_b.charges))
    converter = engine.make_converter(plan_a, plan_b, bond_a, bond_b, spin_batch=True)
    assert converter.spin_batched and len(converter.circuits) == 1
    convert = jax.jit(converter.convert)
    for ca, cb in model.walkers:
        alpha, beta, gauges = convert(jnp.asarray(ca), jnp.asarray(cb))
        for C, plan, bond, dev, g_dev, labels in zip((ca, cb), (plan_a, plan_b), (bond_a, bond_b), (alpha, beta),
                                                     gauges, converter.charges):
            assert all(np.array_equal(x, y) for x, y in zip(labels, bond.charges))
            assert [A.shape[0] for A in dev] == [len(q) for q in labels[:-1]]
            host, _, g_host = engine.channel_mps_host(C, plan, bond)
            reference = g_host * _mps_overlap(host, host)
            assert abs(float(g_dev) * _mps_overlap(host, dev) - reference) <= 1e-10 * abs(reference)


@pytest.mark.parametrize("chi", [None, CHI_TRUNC], ids=["exact", "chi4"])
@pytest.mark.parametrize("buckets", list(BUCKETS.values()), ids=list(BUCKETS))
def test_buckets_regroup_sectors_without_changing_the_circuit(model, buckets, chi):
    """Size buckets only regroup each factorisation's sectors: the same ops, labels, pads and sector factorisations;
    every class lies in one bucket and is padded to its own largest member, so there are as many classes as
    (class, bucket) pairs and no more padding (padding_stats, circuit_stats)."""
    plan, bond = _channel_plans(model.reference[0], "rank_exact", chi)
    plain = engine.compile_circuit((plan,), (bond,))
    bucketed = engine.compile_circuit((plan,), (bond,), buckets=buckets)
    shape = lambda c: [(op.kind, op.site, op.step, op.gate, op.n_rows, op.n_cols, op.K) for op in c.ops]
    assert shape(bucketed) == shape(plain) and bucketed.pads == plain.pads
    assert all(np.array_equal(a, b) for a, b in zip(bucketed.charges[0], plain.charges[0]))
    expected = 0
    for op, op_b in zip(plain.ops, bucketed.ops):
        assert sorted(_sector_shapes(op_b)) == sorted(_sector_shapes(op))
        for cls in op.classes:
            closed = cls.kind in ("row1", "col1")
            expected += 1 if closed else len({_bucket_of(_bucket_size(*s), buckets) for s in _class_sectors(op, cls)})
        for cls in op_b.classes:
            members = _class_sectors(op_b, cls)
            assert cls.gather.shape[-2:] == (max(r for _, r, _ in members), max(c for _, _, c in members))
            if cls.kind not in ("row1", "col1"):
                assert len({_bucket_of(_bucket_size(*s), buckets) for s in members}) == 1
    n_classes = lambda c: sum(len(op.classes) for op in c.ops)
    assert n_classes(bucketed) == expected >= n_classes(plain)
    if buckets == (2, 4) and chi is None:
        assert expected > n_classes(plain)  # these bounds split some class at 8 sites
    a, b = engine.padding_stats(plain), engine.padding_stats(bucketed)
    for key in ("eigh_waste", "qr_waste"):
        assert 0.0 <= b[key] <= a[key] < 1.0
    for key in ("eigh_n_median", "eigh_n_p90", "eigh_n_max"):
        assert b[key] == a[key]
    sa, sb = engine.circuit_stats(plain), engine.circuit_stats(bucketed)
    for key in ("spins", "ops", "gates", "moves", "max_bond", "max_eigh"):
        assert sb[key] == sa[key]
    assert sb["max_qr"] <= sa["max_qr"] and sb["qr_calls"] >= sa["qr_calls"] and sb["eigh_calls"] >= sa["eigh_calls"]
    if chi is None:
        assert sa["eigh_calls"] == 0  # nothing truncates


@pytest.mark.parametrize("buckets", [(2, 4), (8, 16)], ids=["2-4", "8-16"])
def test_bucketed_engine_gives_the_same_overlaps_energies_and_steps(model, buckets):
    """QmcParamsMps.sector_buckets reaches the plan's converter; the same factorisations on smaller padded matrices
    give the one-class-per-kind engine's overlaps, local energies and three CPMC steps to rounding."""
    trial, prop, params = model.trials["sz"], _prop(model), _qmc_params()
    ca, cb = _batch(model)
    out = []
    for b in ((), buckets):
        plan = make_walker_plan(model.ham, trial, model.system,
                                _mps_params(orbital_plan="adaptive", walker_channel_chi=CHI_TRUNC, sector_buckets=b,
                                            walker_qr="native"))
        converter = engine.converter_for(plan)
        assert _n_classes(converter) == _n_classes(engine.make_converter(*plan.orbital_plans, *plan.bond_plans,
                                                                         buckets=b or None))
        ctx = build_mps_meas_ctx(model.ham, trial, plan=plan, kernel="blocked", htrial=model.htrial["sz"])
        kernels, data = ctx.kernels, ctx.data(prop)
        overlaps = np.asarray(jax.jit(kernels.overlaps)(ca, cb, data))
        energies = np.asarray(jax.jit(kernels.energies)(ca, cb, data))
        start = _state(ca, cb, out[0][0] if out else overlaps)
        out.append((overlaps, energies, _half_steps(kernels, params, 1, 6)(start, data), _n_classes(converter)))
    (o0, e0, s0, n0), (o1, e1, s1, n1) = out
    assert n1 >= n0
    np.testing.assert_allclose(o1, o0, rtol=1e-11, atol=0)
    np.testing.assert_allclose(e1, e0, rtol=1e-11, atol=1e-12)
    _assert_states_close(s1, s0, rtol=1e-10)


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
def test_null_mode_handles_rank_deficient_blocks(xp):
    """The empty mode of an exact plan's step lies in null(block^T) also for a block with B <= N whose leading
    columns are dependent, where the last vector of a complete QR need not: Householder QR without pivoting can
    leave the rank deficiency above the last row. The "maximal" plan's late blocks of a walker equal to the reference
    can be such blocks (the SD capstone's start walker on CPU, jobs 7179317 and 7180352). Batched over spins too."""
    rng = np.random.default_rng(4)
    B, N, rank = 3, 4, 2
    basis = np.linalg.qr(rng.standard_normal((B, rank)))[0]
    block = np.zeros((B, N))  # leading column zero, rank 2 < B <= N
    block[:, 1:] = basis @ rng.standard_normal((rank, N - 1))
    for M in (block, block[:, ::-1], np.concatenate([block[:, 1:2], block], axis=1)):
        v = np.asarray(gmps_utils.null_mode(xp.asarray(M), xp))
        assert abs(np.linalg.norm(v) - 1.0) < 1e-12 and np.abs(v @ M).max() < 1e-12
    pair = np.stack([block, block[::-1, ::-1]])
    v = np.asarray(gmps_utils.null_mode(jnp.asarray(pair)))
    assert np.abs(np.einsum("sb,sbn->sn", v, pair)).max() < 1e-12


def test_cholesky_qr2_matches_trots_qr():
    """CholeskyQR2 gives trot's Q (diag R > 0) and det R; a batch with a walker it cannot factor falls back to
    Householder QR as a whole."""
    rng = np.random.default_rng(4)
    C = jnp.asarray(rng.standard_normal((5, L, NELEC[0])))
    Q, d = jax.vmap(engine.cholesky_qr2)(C)
    Qr, dr = jax.vmap(qr_with_det)(C)
    np.testing.assert_allclose(np.asarray(Q), np.asarray(Qr), atol=1e-12)
    np.testing.assert_allclose(np.asarray(d), np.asarray(dr), rtol=1e-12)

    bad = np.asarray(C).copy()
    bad[0, :, 1] = 0.0  # an exactly singular Gram matrix: the Cholesky breaks down
    Qb, db = jax.jit(engine.make_batch_qr("cholesky"))(jnp.asarray(bad))
    Qn, dn = jax.vmap(qr_with_det)(jnp.asarray(bad))
    np.testing.assert_allclose(np.asarray(Qb), np.asarray(Qn), atol=1e-12)
    np.testing.assert_allclose(np.asarray(db), np.asarray(dn), atol=1e-12)
    assert engine.resolve_walker_qr("auto") == ("native" if jax.default_backend() == "cpu" else "cholesky")
    with pytest.raises(ValueError):
        engine.resolve_walker_qr("eigh")
    with pytest.raises(ValueError):
        engine.make_batch_qr("eigh")


# ---------------------------------------------------------------------------------------------
# Kernels: overlaps, local energies, the factorized layouts
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("forward", [True, False])
def test_legs_give_the_same_block_in_every_contraction_order(forward):
    """_legs writes each einsum in dot_general's output order and transposes at most once at the end: every one of
    the six leg orders gives the canonical (t, A, B, K) (left) or (t, a, b, k) (right) block of the three operands."""
    import itertools

    rng = np.random.default_rng(7)
    T, lower, upper = 3, (2, 3, 4), (5, 2, 3)
    start, end = (lower, upper) if forward else (upper, lower)
    X = rng.normal(size=(T,) + start)
    operands = tuple(rng.normal(size=(T, n, m)) for n, m in zip(lower, upper))  # (t, lower, upper) per leg
    spec = "tabk,taA,tbB,tkK->tABK" if forward else "tABK,taA,tbB,tkK->tabk"
    want = np.einsum(spec, X, *operands)
    assert want.shape == (T,) + end
    for order in itertools.permutations(range(3)):
        got = engine._legs(jnp.asarray(X), tuple(jnp.asarray(W) for W in operands), order, forward)
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=1e-12)


def test_leg_order_prefers_the_transpose_free_order_unless_it_costs_much():
    """engine._leg_order keeps the order a, b, k within `tolerance` of the cheapest and otherwise takes the cheapest;
    prefer=None is the plain cheapest order."""
    T = 4
    assert engine._leg_order(T, (7, 6, 1500), (7, 6, 1500))[0] == (0, 1, 2)
    start, end = (2, 2, 1000), (40, 40, 1000)  # contracting k first is far cheaper than last
    forced = engine._leg_order(T, start, end, tolerance=float("inf"))
    cheapest = engine._leg_order(T, start, end, prefer=None)
    default = engine._leg_order(T, start, end)
    assert forced[0] == (0, 1, 2) and cheapest[1] < forced[1] / 1.25
    assert default == cheapest and default[0] != (0, 1, 2)
    # a near tie keeps the preferred order even though another is a little cheaper
    near = engine._leg_order(T, (6, 6, 800), (7, 6, 800))
    plain = engine._leg_order(T, (6, 6, 800), (7, 6, 800), prefer=None)
    assert near[0] == (0, 1, 2) and near[1] <= 1.25 * plain[1]


def test_measure_chunks_cap_the_walkers_per_chunk():
    """The energy and the observables are measured in chunks of at most MEASURE_CHUNK walkers (and never in fewer
    chunks than the steps use); small runs are unchanged."""
    cap = engine.MEASURE_CHUNK
    assert engine.measure_chunks(cap * 4, 1) == 4  # `cap` walkers per chunk
    assert engine.measure_chunks(cap * 4, 8) == 8
    assert engine.measure_chunks(6, 1) == 1 and engine.measure_chunks(cap, 1) == 1
    assert engine.measure_chunks(cap * 3 // 2, 1) == 2  # the smallest divisor with at most `cap` walkers per chunk
    assert engine.measure_chunks(cap * 5 + 1, 1) >= 5 and (cap * 5 + 1) % engine.measure_chunks(cap * 5 + 1, 1) == 0


@pytest.mark.parametrize("trial_kind", ["sz", "number"])
def test_size_bucketed_energy_blocks_give_the_uniform_numerators(model, trial_kind):
    """engine.bucket_plan regroups the transitions of every site by block size: <H trial|walker> through the size
    classes equals the uniformly padded contraction (the same sums), for any class split, (N_up, N_dn) and N labels,
    and the padded entries lie between the blocks' own entries and the uniform padding."""
    from trot.trial.mps import _hashable_charges

    plan = model.plans["truncated"]
    h_np, h_q, _ = model.htrial[trial_kind]
    layout = engine.layout_for(plan, _hashable_charges(h_q))
    uniform = tuple(jnp.asarray(b) for b in engine.fixed_blocks(h_np, layout, xp=np))
    kernels = engine.kernels_for(plan, model.trials[trial_kind].charges, energy=None)
    ca, cb = _batch(model, n=4)
    qa, _ = kernels.batch_qr(jnp.asarray(ca))
    qb, _ = kernels.batch_qr(jnp.asarray(cb))

    def numerators(energy_plan, blocks):
        def one(a, b):
            alpha, beta, _ = engine.converter_for(plan).convert(a, b)
            return engine.energy_numerator(alpha, beta, blocks, energy_plan)
        return np.asarray(jax.vmap(one)(qa, qb))

    want = numerators(layout, uniform)
    for settings in (dict(), dict(ratio=1.2, minimum=2), dict(bounds=[2, 4, 8])):
        bucketed = engine.bucket_plan(layout, **settings)
        blocks = tuple(tuple(jnp.asarray(b) for b in site) for site in engine.bucketed_blocks(h_np, bucketed))
        np.testing.assert_allclose(numerators(bucketed, blocks), want, rtol=1e-12, atol=1e-14)
        stats = bucketed.stats
        assert stats["fixed_exact_entries"] <= stats["fixed_bucketed_entries"] <= stats["fixed_uniform_entries"]
        assert sum(c.rows * c.cols * len(c.idx) for site in bucketed.sites for c in site) == \
            stats["fixed_bucketed_entries"]


def _block_form(tensors, labels):
    """A dense charge-labelled MPS in block form {(left label, p, right label): block} with the engine's fixed keys:
    trials.to_blocks for (N_up, N_dn) labels, the same per particle number N for N labels (the rotated trial's
    H|trial>, whose cached block form trials.to_blocks does not make)."""
    labels = [gmps_utils.label_array(q) for q in labels]
    if labels[0].shape[1] == 2:
        return trials.to_blocks(tensors, labels)
    out = []
    for k, A in enumerate(tensors):
        A = np.asarray(A)
        left, right = engine._charge_index(labels[k]), engine._charge_index(labels[k + 1])
        site = {}
        for ql, rows in left.items():
            for p in range(4):
                qr = (ql[0] + int(engine.N_PHYSICAL[p]),)
                if qr in right:
                    block = A[np.ix_(rows, [p], right[qr])][:, 0]
                    if block.any():
                        site[(ql, p, qr)] = block
        out.append(site)
    return out


@pytest.mark.parametrize("trial_kind", ["sz", "number"])
def test_deduplicated_energy_blocks_give_the_same_numerators(model, trial_kind):
    """bucket_plan(dedup=True) stores each distinct H|trial> block of a class once and left_contract gathers it per
    transition: <H trial|walker> equals the per-transition blocks', from dense and block-form H|trial> sites alike.
    With particle-number labels several walker label pairs meet the same block, so fewer entries are stored."""
    from trot.trial.mps import _hashable_charges

    plan = model.plans["truncated"]
    h_np, h_q, _ = model.htrial[trial_kind]
    layout = engine.layout_for(plan, _hashable_charges(h_q))
    kernels = engine.kernels_for(plan, model.trials[trial_kind].charges, energy=None)
    ca, cb = _batch(model, n=4)
    qa, _ = kernels.batch_qr(jnp.asarray(ca))
    qb, _ = kernels.batch_qr(jnp.asarray(cb))

    def numerators(energy_plan, blocks):
        def one(a, b):
            alpha, beta, _ = engine.converter_for(plan).convert(a, b)
            return engine.energy_numerator(alpha, beta, blocks, energy_plan)
        return np.asarray(jax.jit(jax.vmap(one))(qa, qb))

    plain = engine.bucket_plan(layout)
    want = numerators(plain, engine.bucketed_blocks(h_np, plain))
    dedup = engine.bucket_plan(layout, dedup=True)
    assert dedup.dedup and not plain.dedup
    block_form = _block_form(h_np, h_q)
    np.testing.assert_allclose(numerators(plain, engine.bucketed_blocks(block_form, plain)), want, rtol=1e-12,
                               atol=1e-14)  # the block form itself is right
    for sites in (h_np, block_form):
        blocks = engine.bucketed_blocks(sites, dedup)
        assert all(B.shape[0] == len(c.unique_first) for site, classes in zip(blocks, dedup.sites)
                   for B, c in zip(site, classes))
        np.testing.assert_allclose(numerators(dedup, blocks), want, rtol=1e-12, atol=1e-14)
    stats = dedup.stats
    assert stats["fixed_unique_entries"] <= stats["fixed_bucketed_entries"]
    if trial_kind == "number":
        assert stats["fixed_unique_entries"] < stats["fixed_bucketed_entries"]
    with pytest.raises(ValueError, match="left_contract only"):
        engine.right_environments((), (), dedup)


def test_energy_dedup_flag_reaches_the_context_and_the_kernels(model, monkeypatch):
    """TROT_ENERGY_DEDUP (engine.ENERGY_DEDUP): the measurement context stores the distinct H|trial> blocks
    (energy_plan_for, bucketed_blocks) and its kernels contract them, with the local energies of the per-transition
    blocks; the flag is part of the plan's and the kernels' cache keys."""
    trial = model.trials["number"]
    ca, cb = _batch(model)
    energies = []
    for flag in (False, True):
        monkeypatch.setattr(engine, "ENERGY_DEDUP", flag)
        plan = make_walker_plan(model.ham, model.trials["sz"], model.system,
                                _mps_params(orbital_plan="adaptive", walker_channel_chi=CHI_TRUNC,
                                            sector_buckets=(8, 16), walker_qr="cholesky"))
        ctx = build_mps_meas_ctx(model.ham, trial, plan=plan, kernel="blocked", htrial=model.htrial["number"])
        assert ctx.kernels.energy_plan.dedup == flag
        energies.append(np.asarray(jax.jit(ctx.kernels.energies)(ca, cb, ctx.data())))
    np.testing.assert_allclose(energies[1], energies[0], rtol=1e-11, atol=1e-12)


def _assert_trees_close(got, want, rtol=1e-10):
    """Leaf by leaf, each to rtol of its own largest entry (environments are not normalised)."""
    got, want = jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)
    assert len(got) == len(want)
    for x, y in zip(got, want):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape
        np.testing.assert_allclose(x, y, rtol=rtol, atol=rtol * np.max(np.abs(y), initial=0.0))


@pytest.mark.parametrize("plan_kind, trial_kind", SWEEP_CASES, ids=[f"{p}-{t}" for p, t in SWEEP_CASES])
def test_size_bucketed_overlap_plan_gives_the_uniform_contractions(model, plan_kind, trial_kind):
    """bucket_plan on the trial's layout (the overlap plan with TROT_OVERLAP_BUCKET_RATIO): every contraction through
    the size classes, each class's walker and trial blocks cut to its largest and its sums padded back to the bond,
    equals the uniform one for any class split: the right environments at every bond, szsz, the overlaps and the
    step with its field sweep. The host blocks equal the traced gathers (plan_blocks)."""
    plan, trial = model.plans[plan_kind], model.trials[trial_kind]
    layout, prop = engine.layout_for(plan, trial.charges), _prop(model)
    ca, cb = _batch(model)
    randoms = np.random.default_rng(2).random((len(ca), L))

    def contractions(op, blocks):
        kernels = engine.make_kernels(engine.converter_for(plan), op, None, None,
                                      engine.resolve_walker_qr(plan.walker_qr))
        data = engine.DeviceData(blocks, (), (), prop.exp_h1_half, prop.hs_constant)

        def per_walker(qa, qb):
            alpha, beta, _ = kernels.converter.convert(qa, qb)
            wb = engine.walker_blocks(alpha, beta, op)
            return engine.right_environments(wb, blocks, op), engine.szsz_contract(wb, blocks, op)

        def run(a, b, r):
            qa, _ = kernels.batch_qr(a)
            qb, _ = kernels.batch_qr(b)
            return jax.vmap(per_walker)(qa, qb), kernels.overlaps(a, b, data), kernels.step_batch(a, b, r, True, data,
                                                                                                    1e-8)
        return jax.jit(run)(ca, cb, randoms)

    want = contractions(layout, engine.plan_blocks(trial.tensors, layout))
    for settings in (dict(), dict(ratio=1.2, minimum=2), dict(bounds=[2, 4, 8])):
        bucketed = engine.bucket_plan(layout, **settings)
        stats = bucketed.stats
        assert stats["fixed_exact_entries"] <= stats["fixed_bucketed_entries"] <= stats["fixed_uniform_entries"]
        host = engine.plan_blocks([np.asarray(A) for A in trial.tensors], bucketed, xp=np)
        traced = jax.jit(lambda tensors: engine.plan_blocks(tensors, bucketed))(trial.tensors)
        _assert_trees_close(traced, host, rtol=0)
        _assert_trees_close(contractions(bucketed, traced), want)


def test_overlap_bucket_ratio_buckets_the_contexts_trial_blocks(model, monkeypatch):
    """TROT_OVERLAP_BUCKET_RATIO > 0 (engine.OVERLAP_BUCKET_RATIO): the kernels and the measurement context both take
    the size-bucketed overlap plan (overlap_plan_for) and its trial blocks per class, and the overlaps, local energies,
    szsz and three CPMC steps equal those of the uniform plan; the ratio is part of the kernels' cache key."""
    trial, prop, params = model.trials["number"], _prop(model), _qmc_params()
    ca, cb = _batch(model)
    out = []
    for ratio in (0.0, 1.2):
        monkeypatch.setattr(engine, "OVERLAP_BUCKET_RATIO", ratio)
        plan = make_walker_plan(model.ham, model.trials["sz"], model.system,
                                _mps_params(orbital_plan="adaptive", walker_channel_chi=CHI_TRUNC,
                                            sector_buckets=(8, 16), walker_qr="cholesky"))
        ctx = build_mps_meas_ctx(model.ham, trial, plan=plan, kernel="blocked", htrial=model.htrial["number"])
        kernels, data = ctx.kernels, ctx.data(prop)
        assert isinstance(kernels.overlap_plan, engine.BucketedPlan) == (ratio > 0)
        assert kernels.overlap_plan is engine.overlap_plan_for(plan, trial.charges)
        overlaps = np.asarray(jax.jit(kernels.overlaps)(ca, cb, data))
        energies = np.asarray(jax.jit(kernels.energies)(ca, cb, data))
        szsz = np.asarray(jax.jit(szsz_fns(ctx).batch)(ca, cb, data))
        start = _state(ca, cb, out[0][0] if out else overlaps)
        out.append((overlaps, energies, szsz, _half_steps(kernels, params, 1, 6)(start, data)))
    (o0, e0, z0, s0), (o1, e1, z1, s1) = out
    np.testing.assert_allclose(o1, o0, rtol=1e-11, atol=0)
    np.testing.assert_allclose(e1, e0, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(z1, z0, rtol=1e-10, atol=1e-12)
    _assert_states_close(s1, s0, rtol=1e-10)


@pytest.mark.parametrize("spin_batch", [True, False])
@pytest.mark.parametrize("plan_kind", ["exact", "truncated"])
def test_overlaps_match_the_references(model, plan_kind, spin_batch):
    """make_kernels on make_converter + make_factorized_layout (the d=4 walker never formed), for each trial (exact
    labels, particle-number labels, a determinant): exact enumeration with the exact plan, the host oracle's
    truncated channels with the truncated one; non-orthonormal walkers included (det R); both spins in one batch
    or one after the other."""
    plan = model.plans[plan_kind]
    converter = engine.make_converter(*plan.orbital_plans, *plan.bond_plans, spin_batch=spin_batch)
    ca, cb = _batch(model)
    got = {}
    for trial_kind, trial in model.trials.items():
        layout = engine.make_factorized_layout(*converter.charges, trial.charge_arrays())
        kernels = engine.make_kernels(converter, layout, None, None, engine.resolve_walker_qr(plan.walker_qr))
        data = engine.DeviceData(engine.fixed_blocks(trial.tensors, layout), (), (), None, None)
        got[trial_kind] = np.asarray(jax.jit(kernels.overlaps)(ca, cb, data))
        want, _ = _reference_values(model, plan_kind, trial_kind, ca, cb)
        np.testing.assert_allclose(got[trial_kind], want, rtol=1e-10, atol=0)
    if plan_kind == "truncated":  # the cap truncates these walkers
        exact, _ = _reference_values(model, "exact", "sz", ca, cb)
        assert np.max(np.abs(got["sz"] / exact - 1.0)) > 1e-6


@pytest.mark.parametrize("kernel", ["blocked", "blocked-uncompressed", "dense"])
@pytest.mark.parametrize("plan_kind", ["exact", "truncated"])
def test_local_energies_match_the_references(model, plan_kind, kernel):
    """<H trial|walker>/<trial|walker> through the measurement context's kernels (kernels_for on the plan), for the
    trial with exact labels and the one with particle-number labels: the blocked kernel on the compressed or the
    uncompressed labelled H|trial>, the dense kernel on the d=4 walker; against enumeration with the exact
    amplitudes (exact plan) or the host oracle's (truncated plan)."""
    ca, cb = _batch(model)
    for trial_kind in ("sz", "number"):
        if kernel == "blocked-uncompressed":
            trial = model.trials[trial_kind]
            h_np, h_q = trial_times_h(hubbard_mpo_from_h1(model.h1, U), [np.asarray(A) for A in trial.tensors],
                                      trial.charge_arrays())
            ctx = build_mps_meas_ctx(model.ham, trial, plan=model.plans[plan_kind], kernel="blocked",
                                     htrial=(h_np, h_q, model.htrial[trial_kind][2]))
        else:
            ctx = _ctx(model, plan_kind, trial_kind, kernel)
        got = np.asarray(jax.jit(ctx.kernels.energies)(ca, cb, ctx.data()))
        _, want = _reference_values(model, plan_kind, trial_kind, ca, cb)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("plan_kind", ["exact", "truncated"])
def test_one_walker_kernels_and_the_probe_equal_the_batched_ones(model, plan_kind):
    """overlap_one, energy_one and the probe (trot's single-walker QR) give the batched kernels' values; the kernels
    are cached on the plan; the probe's conversion passes conversion_self_check against the host oracle."""
    plan, trial = model.plans[plan_kind], model.trials["sz"]
    ctx = _ctx(model, plan_kind)
    kernels, data = ctx.kernels, ctx.data()
    assert kernels is engine.kernels_for(plan, trial.charges, ctx.h_charges, "blocked")
    assert kernels.converter is engine.converter_for(plan)
    assert kernels.overlap_plan is engine.overlap_plan_for(plan, trial.charges)
    ca, cb = _batch(model)
    overlaps = np.asarray(jax.jit(kernels.overlaps)(ca, cb, data))
    energies = np.asarray(jax.jit(kernels.energies)(ca, cb, data))
    overlap_one, energy_one = jax.jit(kernels.overlap_one), jax.jit(kernels.energy_one)
    for k in range(len(ca)):
        probe = kernels.jit_probe(ca[k], cb[k], data)
        for value, want in ((overlap_one(ca[k], cb[k], data), overlaps[k]), (probe["overlap"], overlaps[k]),
                            (energy_one(ca[k], cb[k], data), energies[k]), (probe["energy"], energies[k])):
            # truncated walkers amplify the rounding of the two QR paths (single-walker vs batched) to ~4e-11
            np.testing.assert_allclose(float(value), want, rtol=1e-11 if plan_kind == "exact" else 1e-9)
        assert max(engine.conversion_self_check(kernels, plan.orbital_plans, plan.bond_plans, probe)) < 1e-10


@pytest.mark.parametrize("fixed", ["trial", "htrial"])
@pytest.mark.parametrize("trial_kind", ["sz", "number"])
def test_factorized_plan_host_blocks_equal_the_device_gathers(model, trial_kind, fixed):
    """make_factorized_plan's host blocks (the fixed MPS's label blocks, zero-padded, one per allowed transition)
    equal fixed_blocks gathered on the device from the dense tensors, eagerly and traced inside jit, for (N_up, N_dn)
    and particle-number labels, for the trial and for H|trial>."""
    trial = model.trials[trial_kind]
    if fixed == "trial":
        tensors, labels = [np.asarray(A) for A in trial.tensors], trial.charge_arrays()
    else:
        tensors, labels, _ = model.htrial[trial_kind]
    on_device = tuple(jnp.asarray(A) for A in tensors)
    for plan in model.plans.values():
        qa, qb = engine.converter_for(plan).charges
        layout = engine.make_factorized_layout(qa, qb, labels)
        host = engine.make_factorized_plan(qa, qb, tensors, labels)
        assert layout.blocks is None and host.stats == layout.stats and host.shared == layout.shared
        _assert_layout_blocks(host, tensors, labels)
        gathered, traced = engine.fixed_blocks(on_device, layout), _jit_fixed_blocks(layout)(on_device)
        for h, d, t in zip(host.blocks, gathered, traced):
            np.testing.assert_array_equal(np.asarray(d), h)
            np.testing.assert_array_equal(np.asarray(t), h)


@pytest.mark.parametrize("plan_kind", ["exact", "truncated"])
def test_block_form_sites_give_the_same_blocks_and_energies(model, plan_kind):
    """Sites in block form ({(left label, p, right label): block}, trot.gmps.trials.to_blocks) give the dense
    tensors' plan blocks; the block-form H|trial> made without dense tensors (trials.load_or_make_htrial) gives the
    dense path's trial energy and local energies through the engine."""
    plan, trial = model.plans[plan_kind], model.trials["sz"]
    trial_np, labels = [np.asarray(A) for A in trial.tensors], trial.charge_arrays()
    h_np, h_q, _ = model.htrial["sz"]
    qa, qb = engine.converter_for(plan).charges
    for tensors, q in ((trial_np, labels), (h_np, h_q)):
        blocks = trials.to_blocks(tensors, q)
        for a, b in zip(trials.from_blocks(blocks, q), tensors):
            np.testing.assert_array_equal(a, b)
        dense = engine.make_factorized_plan(qa, qb, tensors, q)
        blocked = engine.make_factorized_plan(qa, qb, blocks, q)
        assert blocked.stats == dense.stats
        for a, b, c in zip(blocked.blocks, dense.blocks, engine.fixed_blocks(blocks, dense)):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(np.asarray(c), b)
        # device=True: each site goes to the device as it is assembled (block-form and dense sites alike)
        for streamed in (engine.fixed_blocks(blocks, dense, xp=np, device=True),
                         engine.fixed_blocks(tensors, dense, xp=np, device=True)):
            assert len(streamed) == len(dense.blocks)
            for s, b in zip(streamed, dense.blocks):
                assert isinstance(s, jax.Array)
                np.testing.assert_array_equal(np.asarray(s), b)

    (h_blocks, h_labels), info = trials.load_or_make_htrial(None, model.h1, U, trial_np, labels, None, say=_quiet)
    for a, b in zip(h_labels, h_q):
        np.testing.assert_array_equal(a, b)
    ctx = build_mps_meas_ctx(model.ham, trial, plan=plan, kernel="blocked",
                             htrial=(h_blocks, h_labels, info["trial_energy"]))
    reference = _ctx(model, plan_kind)
    assert ctx.trial_energy == pytest.approx(reference.trial_energy, rel=1e-12)
    ca, cb = _batch(model)
    np.testing.assert_allclose(np.asarray(jax.jit(ctx.kernels.energies)(ca, cb, ctx.data())),
                               np.asarray(jax.jit(reference.kernels.energies)(ca, cb, reference.data())),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("trial_kind", ["sz", "number"])
def test_layouts_pair_each_walker_label_with_its_trial_sector(model, trial_kind):
    """A walker label pair (N_alpha, N_beta) meets the trial's own pair, or with particle-number labels (a trial used
    as it is, read off its nonzeros by number_labels) the trial's N_alpha + N_beta sector; the layout is cached on
    the plan."""
    trial = model.trials[trial_kind]
    labels = trial.charge_arrays()
    if trial_kind == "number":
        assert trial.label_width == 1 and trial.sector_weight is None
        for a, b in zip(gmps_utils.number_labels([np.asarray(A) for A in trial.tensors]), labels):
            np.testing.assert_array_equal(a, b)
        key = lambda a, b: (a + b,)
    else:
        assert trial.label_width == 2 and trial.sector_weight == 1.0
        key = lambda a, b: (a, b)
    for plan in model.plans.values():
        qa, qb = engine.converter_for(plan).charges
        layout = engine.layout_for(plan, trial.charges)
        assert layout is engine.layout_for(plan, trial.charges)
        for i, shared in enumerate(layout.shared):
            sectors = {tuple(int(x) for x in row) for row in labels[i]}
            want = sorted((a, b) for a in set(qa[i].tolist()) for b in set(qb[i].tolist()) if key(a, b) in sectors)
            assert [tuple(c) for c in shared] == want


# ---------------------------------------------------------------------------------------------
# The field sweep and the half steps
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("floor", [1e-8, 0.5])
@pytest.mark.parametrize("plan_kind, trial_kind", SWEEP_CASES, ids=[f"{p}-{t}" for p, t in SWEEP_CASES])
def test_field_sweep_matches_the_reference_sweep(model, plan_kind, trial_kind, floor):
    """step_batch's field sweep (one conversion, cached right environments, both proposals per site from the
    marginals) against the NumPy sweep whose proposals are exact overlaps of the walker with that site's diagonal HS
    factor applied (enumeration, or the host oracle's truncated amplitudes): the fields, the overlap before and
    after, the weight factor and the node count, under the floor rule. Without the sweep (a traced False, the odd
    half steps) the walkers come back unchanged with their overlap, weight factor 1 and no nodes."""
    kernels, data = _sweep_engine(model, plan_kind, trial_kind)
    step = _step_batch(model, plan_kind, trial_kind)
    ca, cb = _batch(model)
    randoms = np.random.default_rng(2).random((len(ca), L))
    on = [np.asarray(x) for x in step(ca, cb, randoms, data, jnp.asarray(True), floor)]
    ca2, cb2, before, after, factor, nodes = on
    hs, psi = np.asarray(data.hs), model.amps[trial_kind][0]
    floored = 0
    for k in range(len(ca)):
        a, b = _walker_amplitudes(model, plan_kind, ca[k], cb[k])
        fields, want_before, want_after, want_factor, want_nodes, f = _reference_sweep(a, b, psi, hs, randoms[k], floor)
        floored += f
        np.testing.assert_allclose(ca2[k], ca[k] * hs[fields, 0][:, None], rtol=1e-12, atol=0)
        np.testing.assert_allclose(cb2[k], cb[k] * hs[fields, 1][:, None], rtol=1e-12, atol=0)
        np.testing.assert_allclose(before[k], want_before, rtol=1e-10)
        np.testing.assert_allclose(after[k], want_after, rtol=1e-10)
        np.testing.assert_allclose(factor[k], want_factor, rtol=1e-10)
        assert int(nodes[k]) == want_nodes
    if floor == 0.5:
        assert floored > 0  # the floor removes proposals here

    off = [np.asarray(x) for x in step(ca, cb, randoms, data, jnp.asarray(False), floor)]
    np.testing.assert_array_equal(off[0], ca)
    np.testing.assert_array_equal(off[1], cb)
    np.testing.assert_allclose(off[2], before, rtol=1e-12)
    np.testing.assert_array_equal(off[3], off[2])
    np.testing.assert_array_equal(off[4], np.ones(len(ca)))
    np.testing.assert_array_equal(off[5], np.zeros(len(ca)))
    if floor == 1e-8:  # a Python bool skips the lax.cond: the same sweep
        static = jax.jit(lambda a, b, r, d, fl: kernels.step_batch(a, b, r, True, d, fl))
        for x, y in zip(static(ca, cb, randoms, data, floor), on):
            np.testing.assert_allclose(np.asarray(x), y, rtol=1e-12, atol=0)


@pytest.mark.parametrize("floor", [1e-8, 0.5])
@pytest.mark.parametrize("plan_kind", ["exact", "truncated"])
def test_half_steps_match_the_reference_step(model, plan_kind, floor):
    """Three CPMC steps as six half steps of make_half_step in one jitted scan (one conversion call site, the sweep
    behind lax.cond) against the NumPy step: walkers, weights, overlaps, shift, nodes and RNG key."""
    kernels, data = _sweep_engine(model, plan_kind, "sz")
    params = _qmc_params(weight_floor=floor)
    ca, cb = _batch(model)
    overlaps, _ = _reference_values(model, plan_kind, "sz", ca, cb)
    state = _state(ca, cb, overlaps, np.random.default_rng(5).uniform(0.5, 1.5, len(ca)))
    got = _half_steps(kernels, params, 1, 6)(state, data)
    want, floored = _reference_half_steps(model, plan_kind, "sz", state, 6, params)
    if floor == 0.5:
        assert floored > 0
    _assert_states_close(got, want)


def test_walker_chunking_is_exact(model):
    """Walkers in 1, 2 or 3 sequential chunks (lax.map) give the same half steps and overlaps; divisor_at_least
    picks the chunk counts that divide the population."""
    kernels, data = _sweep_engine(model, "truncated", "sz")
    params = _qmc_params(weight_floor=1e-3)
    ca, cb = _batch(model)
    whole = jax.jit(kernels.overlaps)(ca, cb, data)
    state = _state(ca, cb, whole)
    runs = [_half_steps(kernels, params, n, 4)(state, data) for n in (1, 2, 3)]
    for other in runs[1:]:
        for x, y in zip(jax.tree_util.tree_leaves(runs[0]), jax.tree_util.tree_leaves(other)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-10, atol=1e-14)
    for n in (2, 3):
        np.testing.assert_allclose(np.asarray(_jit_chunked_overlaps(kernels, n)(ca, cb, data)), np.asarray(whole),
                                   rtol=1e-12, atol=0)
    assert [engine.divisor_at_least(6, k) for k in (0, 1, 2, 4)] == [1, 1, 2, 6]
    assert engine.divisor_at_least(1000, 3) == 4 and engine.divisor_at_least(7, 2) == 7


# ---------------------------------------------------------------------------------------------
# Dense walker legs (trot.gmps.dense_engine)
# ---------------------------------------------------------------------------------------------


def _dense_kernels(model, chi, trial_kind, move="gram", gap=None, trial="sector"):
    """(dense walker plan, kernels_for it, its DeviceData with the trial's blocks and the propagator): the exact plan's
    gates at bond chi, the trial in particle-number sector blocks or as full matrices."""
    plan = make_dense_walker_plan(*model.reference, model.plans["exact"].orbital_plans,
                                  dc.DenseSpec(chi, move, gap, trial=trial), walker_qr="native")
    trial, prop = model.trials[trial_kind], _prop(model)
    kernels = engine.kernels_for(plan, trial.charges, energy=None)
    assert isinstance(kernels.overlap_plan, engine.DensePlan) and kernels.converter is None
    data = engine.DeviceData(engine.plan_blocks(trial.tensors, kernels.overlap_plan), (), (), prop.exp_h1_half,
                             prop.hs_constant)
    return plan, kernels, data


@pytest.mark.parametrize("trial_kind", ["sz", "number", "sd"])
def test_dense_overlaps_equal_the_exact_references_when_nothing_is_truncated(model, trial_kind):
    """A dense bond far above what 8 sites need converts exactly: the overlaps (non-orthonormal walkers included, the
    trial with (N_up, N_dn) labels, particle-number labels or a determinant) are the enumerated amplitudes."""
    _, kernels, data = _dense_kernels(model, 64, trial_kind)
    ca, cb = _batch(model)
    got = np.asarray(jax.jit(kernels.overlaps)(ca, cb, data))
    want, _ = _reference_values(model, "exact", trial_kind, ca, cb)
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=0)
    overlap_one = jax.jit(kernels.overlap_one)
    one = np.array([float(overlap_one(a, b, data)) for a, b in zip(ca, cb)])
    np.testing.assert_allclose(one, got, rtol=1e-9, atol=0)  # one walker vs the batch: rounding (1.1e-10 seen)


@pytest.mark.parametrize("trial_kind", ["sz", "number"])
def test_dense_truncated_overlaps_equal_the_dense_contraction_of_the_converted_walkers(model, trial_kind):
    """With the bond capped (each walker keeping its own largest states) the dense kernels give det R g_a g_b times the
    plain contraction of the converted walkers' d=4 MPS (combine_channels with the converter's labels) against the
    trial: the left contraction, environments and sign of dense_engine, against trot.gmps.utils."""
    plan, kernels, data = _dense_kernels(model, CHI_TRUNC, trial_kind)
    converters = [jax.jit(dc.make_dense_converter(p, CHI_TRUNC, "gram", charge_gap=plan.dense.gap()).convert)
                  for p in plan.orbital_plans]
    ca, cb = _batch(model)
    got = np.asarray(jax.jit(kernels.overlaps)(ca, cb, data))
    trial = [jnp.asarray(A) for A in model.trials[trial_kind].tensors]
    for w, (a, b) in enumerate(zip(ca, cb)):
        qa, da = qr_with_det(jnp.asarray(a))
        qb, db = qr_with_det(jnp.asarray(b))
        alpha, charges_a, ga = converters[0](qa)
        beta, charges_b, gb = converters[1](qb)
        label = lambda charges: [np.round(np.asarray(q)).astype(int) for q in charges]
        tensors, _ = gmps_utils.combine_channels(alpha, label(charges_a), beta, label(charges_b))
        want = float(da * db * ga * gb * gmps_utils.contract_real(tensors, trial))
        np.testing.assert_allclose(got[w], want, rtol=1e-9, atol=0)
    exact, _ = _reference_values(model, "exact", trial_kind, ca, cb)
    assert np.max(np.abs(got / exact - 1.0)) > 1e-6  # the cap truncates these walkers


@pytest.mark.parametrize("floor", [1e-8, 0.5])
@pytest.mark.parametrize("trial_kind", ["sz", "number"])
def test_dense_step_equals_the_sector_step_when_nothing_is_truncated(model, trial_kind, floor):
    """step_batch of the dense kernels (conversion, right environments, overlap, the HS field sweep with the sampled
    fields, the weights and the node count) equals the sector engine's on the exact plan for the same randoms, with and
    without the sweep; the high floor makes the constraint remove proposals."""
    ca, cb = _batch(model)
    randoms = np.random.default_rng(2).random((len(ca), L))
    sector, sector_data = _sweep_engine(model, "exact", trial_kind)
    _, dense, dense_data = _dense_kernels(model, 64, trial_kind)
    for sweep in (True, False):
        want = jax.jit(lambda a, b, r, d: sector.step_batch(a, b, r, sweep, d, floor))(ca, cb, randoms, sector_data)
        got = jax.jit(lambda a, b, r, d: dense.step_batch(a, b, r, sweep, d, floor))(ca, cb, randoms, dense_data)
        _assert_trees_close(got, want, rtol=1e-8)


@pytest.mark.parametrize("trial_kind", ["sz", "number", "sd"])
def test_dense_sector_trial_blocks_equal_the_full_trial_matrices(model, trial_kind):
    """The trial applied block by block between its particle-number sectors (each bond's reached states sorted by N,
    unreached ones dropped) gives the overlaps and the step (right environments, field sweep) of the full trial
    matrices, for (N_up, N_dn) labels, particle-number labels and a determinant trial; the layout's sectors hold one
    N each, and its multiply-adds are fewer."""
    ca, cb = _batch(model)
    randoms = np.random.default_rng(2).random((len(ca), L))
    results = []
    for trial in ("sector", "dense"):
        _, kernels, data = _dense_kernels(model, CHI_TRUNC, trial_kind, trial=trial)
        assert (kernels.overlap_plan.layout is None) == (trial == "dense")
        step = jax.jit(lambda a, b, r, d, k=kernels: k.step_batch(a, b, r, True, d, 1e-8))
        results.append((jax.jit(kernels.overlaps)(ca, cb, data), step(ca, cb, randoms, data)))
    _assert_trees_close(results[0], results[1], rtol=1e-11)
    _, kernels, _ = _dense_kernels(model, CHI_TRUNC, trial_kind)
    layout, charges = kernels.overlap_plan.layout, model.trials[trial_kind].charges
    for order, sectors, q in zip(layout.orders, layout.sectors, charges):
        n = gmps_utils.label_array(q).sum(axis=1)[order]
        assert [N for N, _, _ in sectors] == sorted({int(x) for x in n})
        assert all(np.all(n[start:stop] == N) for N, start, stop in sectors)
    stats = kernels.overlap_plan.stats
    assert stats["trial_mac_sector"] <= stats["trial_mac_dense"]


@pytest.mark.parametrize("trial_kind", ["sz", "number", "sd"])
def test_mpo_local_energies_equal_the_exact_references(model, trial_kind):
    """energy_kernel="mpo": <T|H|W>/<T|W> through the MPO on the fly (no H|trial>), dense walker legs at a bond far above
    8 sites' needs and the trial in particle-number sector blocks or as full matrices, equals the enumerated local
    energies (non-orthonormal walkers included), through the context, the one-walker kernel and the probe; the context's
    <T|H|T>/<T|T> is the trial energy of the stored H|trial>."""
    ca, cb = _batch(model)
    _, want = _reference_values(model, "exact", trial_kind, ca, cb)
    trial = model.trials[trial_kind]
    for layout in ("sector", "dense"):
        plan = make_dense_walker_plan(*model.reference, model.plans["exact"].orbital_plans,
                                      dc.DenseSpec(64, trial=layout), walker_qr="native")
        ctx = build_mps_meas_ctx(model.ham, trial, plan=plan, kernel="mpo")
        assert ctx.kernel == "mpo" and ctx.kernels.energy_kind == "mpo" and ctx.h_charges is None
        if trial_kind != "sd":
            assert ctx.trial_energy == pytest.approx(model.htrial[trial_kind][2], rel=1e-10)
        got = np.asarray(jax.jit(ctx.kernels.energies)(ca, cb, ctx.data()))
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)
        one = jax.jit(ctx.kernels.energy_one)
        np.testing.assert_allclose([float(one(a, b, ctx.data())) for a, b in zip(ca, cb)], got, rtol=1e-9)
        probe = jax.jit(ctx.kernels.probe)(ca[0], cb[0], ctx.data())
        np.testing.assert_allclose(float(probe["energy"]), got[0], rtol=1e-9)


@pytest.mark.parametrize("trial_kind", ["sz", "number"])
def test_mpo_truncated_energies_equal_the_contraction_with_the_stored_h_trial(model, trial_kind):
    """With the bond capped, the MPO energy is that of the converted walkers: the plain contraction of their d=4 MPS
    (combine_channels, the converter's labels) with the uncompressed H|trial> (trial_times_h) and with the trial,
    independent of the MPO contraction and of the sector layout."""
    plan = make_dense_walker_plan(*model.reference, model.plans["exact"].orbital_plans, dc.DenseSpec(CHI_TRUNC),
                                  walker_qr="native")
    trial = model.trials[trial_kind]
    ctx = build_mps_meas_ctx(model.ham, trial, plan=plan, kernel="mpo")
    ca, cb = _batch(model)
    got = np.asarray(jax.jit(ctx.kernels.energies)(ca, cb, ctx.data()))
    h_np, _ = trial_times_h(hubbard_mpo_from_h1(model.h1, U), [np.asarray(A) for A in trial.tensors],
                            trial.charge_arrays())
    h_np, trial_np = [jnp.asarray(A) for A in h_np], [jnp.asarray(A) for A in trial.tensors]
    converters = [jax.jit(dc.make_dense_converter(p, CHI_TRUNC, "gram", snap=True).convert)
                  for p in plan.orbital_plans]
    label = lambda charges: [np.round(np.asarray(q)).astype(int) for q in charges]
    for w, (a, b) in enumerate(zip(ca, cb)):
        qa, _ = qr_with_det(jnp.asarray(a))
        qb, _ = qr_with_det(jnp.asarray(b))
        alpha, charges_a, _ = converters[0](qa)
        beta, charges_b, _ = converters[1](qb)
        tensors, _ = gmps_utils.combine_channels(alpha, label(charges_a), beta, label(charges_b))
        want = float(gmps_utils.contract_real(tensors, h_np) / gmps_utils.contract_real(tensors, trial_np))
        np.testing.assert_allclose(got[w], want, rtol=1e-9)
    exact = _reference_values(model, "exact", trial_kind, ca, cb)[1]
    assert np.max(np.abs(got / exact - 1.0)) > 1e-8  # the cap truncates these walkers


def test_mpo_kernel_needs_a_dense_plan_and_no_szsz(model):
    """energy_kernel="mpo" on a sector plan has nothing to contract with; on a dense plan the szsz observable is not
    available; the stored-H|trial> kernels are not offered by a dense plan."""
    trial = model.trials["sz"]
    with pytest.raises(ValueError, match="dense walker plan"):
        build_mps_meas_ctx(model.ham, trial, plan=model.plans["exact"], kernel="mpo")
    plan = make_dense_walker_plan(*model.reference, model.plans["exact"].orbital_plans, dc.DenseSpec(8),
                                  walker_qr="native")
    ctx = build_mps_meas_ctx(model.ham, trial, plan=plan, kernel="mpo")
    with pytest.raises(NotImplementedError, match="sector measurement plan"):
        szsz_fns(ctx)
    with pytest.raises(ValueError, match="measures its energy through the MPO"):
        engine.kernels_for(plan, trial.charges, energy="blocked")


def test_dense_half_steps_run_in_a_scan(model):
    """The dense kernels in trot's half-step scan (make_half_step): finite weights and overlaps, the walkers keep
    their shapes, and the state's overlaps stay those of the walkers (recomputed fresh by the next step's right
    environments)."""
    _, kernels, data = _dense_kernels(model, CHI_TRUNC, "sz")
    ca, cb = _batch(model)
    state = _state(ca, cb, jax.jit(kernels.overlaps)(ca, cb, data))
    out = _half_steps(kernels, _qmc_params(n_walkers=len(ca)), 1, 4)(state, data)
    assert np.all(np.isfinite(np.asarray(out.weights))) and np.all(np.isfinite(np.asarray(out.overlaps)))
    assert out.walkers[0].shape == state.walkers[0].shape
    fresh = jax.jit(kernels.overlaps)(out.walkers[0], out.walkers[1], data)
    np.testing.assert_allclose(np.asarray(out.overlaps), np.asarray(fresh), rtol=1e-7)


# ---------------------------------------------------------------------------------------------
# GPU usage (skipped without a GPU)
# ---------------------------------------------------------------------------------------------


@gpu
def test_gpu_step_stays_on_the_device(model):
    """The compiled half steps run with host transfers disallowed and leave every array, in float64, on the GPU."""
    kernels, data = _sweep_engine(model, "truncated", "sz")
    ca, cb = _batch(model, n=256)
    state = _state(ca, cb, jax.jit(kernels.overlaps)(ca, cb, data))
    compiled = _half_steps(kernels, _qmc_params(n_walkers=256), 1, 4).lower(state, data).compile()
    jax.block_until_ready(compiled(state, data))
    with jax.transfer_guard("disallow"):
        out = compiled(state, data)
        jax.block_until_ready(out)
    for leaf in jax.tree_util.tree_leaves(out):
        assert {d.platform for d in leaf.devices()} == {"gpu"}
    assert out.weights.dtype == jnp.float64 and out.walkers[0].dtype == jnp.float64


@gpu
def test_gpu_walker_batching_scales(model):
    """16x the walkers must cost far less than 16x the time: the step is a batch over walkers, not a loop over them
    (at 8 sites it is launch-bound, so a batched step is nearly flat in the walker count)."""
    kernels, data = _sweep_engine(model, "truncated", "sz")
    times = {}
    for n in (64, 1024):
        ca, cb = _batch(model, n=n)
        state = _state(ca, cb, jax.jit(kernels.overlaps)(ca, cb, data))
        compiled = _half_steps(kernels, _qmc_params(n_walkers=n), 1, 4).lower(state, data).compile()
        jax.block_until_ready(compiled(state, data))
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            jax.block_until_ready(compiled(state, data))
            best = min(best, time.perf_counter() - start)
        times[n] = best
    assert times[1024] < 8.0 * times[64], times
