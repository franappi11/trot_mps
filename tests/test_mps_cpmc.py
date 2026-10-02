"""MPS-CPMC through trot's driver: propagation, run_qmc / run_qmc_mps and legacy compatibility.

Capstones. Plain-SD trot CPMC (GhfTrial) and the same run with the SD trial and every walker turned
into an MPS (maximal orbital plan, no truncation) make identical decisions, so every block energy and
weight agrees to rounding. The same holds for a spin-rotated GHF trial against the rotated SD-as-MPS
trial, which the MPS code projects onto the walkers' (N_up, N_dn) sector.
"""

from trot import config

config.configure_once()

import dataclasses
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from tests.helpers import hubbard_fock as hf
from trot.core.ops import k_energy
from trot.core.system import System
from trot.driver import make_run_blocks, run_qmc
from trot.gmps.driver import make_mps_cpmc_ops, make_walker_ops, run_qmc_chunk_sizes, run_qmc_mps
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix, square_hopping_matrix
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.prop import blocks, cpmc, cpmc_slow
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.mps_cpmc import make_fast_prop_ops
from trot.prop.types import PropState, QmcParams, QmcParamsMps
from trot.trial.ghf import GhfTrial, get_rdm1_block_diag, make_ghf_trial_ops
from trot.trial.mps import make_mps_trial, mps_trial_from_sd, natural_orbitals, rotate_spin

REPO = Path(__file__).resolve().parents[1]
# HEAD's mps_cpmc_new.py is self-contained and AST-identical to the pre-refactor code.
FROZEN_LEGACY_COMMIT = "78db00b"
U = 4.0
E_L4 = -1.953145308685  # exact, L=4 chain, (2,2), U=4


# ---------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------


def _lattice23():
    h1 = square_hopping_matrix(2, 3, 1.0, "periodic", "open")
    h1[0, 0], h1[4, 4] = 0.3, -0.2
    return h1


def _params(**overrides: Any) -> QmcParamsMps:
    values: dict[str, Any] = dict(
        dt=0.05,
        n_walkers=10,
        n_prop_steps=5,
        n_eql_blocks=5,
        n_blocks=10,
        seed=11,
        auto_n_chunks=False,
        n_chunks=1,
        orbital_plan="maximal",
        walker_channel_chi=None,
    )
    values.update(overrides)
    return QmcParamsMps(**values)


def _system(L, nelec):
    return System(norb=L, nelec=tuple(nelec), walker_kind="unrestricted")


def _determinant(kind, h1, nelec):
    nup, ndn = nelec
    vectors = np.linalg.eigh(h1)[1]
    if kind == "rhf":
        return vectors[:, :nup].copy(), vectors[:, :ndn].copy()
    if kind == "uhf":
        return hf.staggered_determinant(h1, nup, ndn)
    if kind == "nodal":  # excited up channel: the walkers cross its nodes
        assert tuple(nelec) == (3, 2)
        return vectors[:, [0, 2, 3]].copy(), vectors[:, [0, 1]].copy()
    raise ValueError(kind)


def _spin_rotation(kind, theta=0.7):
    c, s = np.cos(theta), np.sin(theta)
    if kind == "rotation":
        return np.array([[c, -s], [s, c]])
    return np.array([[c, s], [s, -c]])  # reflection, det = -1


def _exact_trial(h1, nelec):
    L = len(h1)
    e0, psi = hf.ground_state(h1, U, *nelec)
    tensors, charges = hf.exact_mps_from_sector_state(psi, L, *nelec)
    return e0, make_mps_trial(tensors, charges, nelec=nelec)


def _sd_side(ham, sys_, params, C_ghf, propagator):
    """The user's plain trot CPMC template: GhfTrial + GHF ops + a trot CPMC propagator."""
    trial = GhfTrial(mo_coeff=jnp.asarray(C_ghf))
    trial_ops = make_ghf_trial_ops(sys_)
    meas_ops = make_ghf_meas_ops_hubbard(sys_)
    if propagator == "fast":
        prop_ops = cpmc.make_prop_ops(ham, "unrestricted", trial_ops)
    else:
        prop_ops = cpmc_slow.make_prop_ops(ham, "unrestricted")
    prop_ctx = prop_ops.build_prop_ctx(ham, None, params)  # pyright: ignore[reportArgumentType]
    return trial, trial_ops, meas_ops, prop_ops, prop_ctx


def _mps_side(ham, sys_, params, trial):
    ops = make_mps_cpmc_ops(ham, trial, sys_, params)
    prop_ctx = ops.prop_ops.build_prop_ctx(ham, ops.trial_ops.get_rdm1(trial), params)
    return trial, ops.trial_ops, ops.meas_ops, ops.prop_ops, prop_ctx


def _run(sys_, params, ham, side):
    trial, trial_ops, meas_ops, prop_ops, prop_ctx = side
    return run_qmc(
        sys=sys_,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        prop_ctx=prop_ctx,
        block_fn=blocks.block,
    )


def _blocks_twin(sys_, params, ham, side, n_blocks=2):
    """Initial state plus n_blocks jitted blocks, returning the full final state."""
    trial, trial_ops, meas_ops, prop_ops, prop_ctx = side
    meas_ctx = meas_ops.build_meas_ctx(ham, trial)
    state = prop_ops.init_prop_state(
        sys=sys_,
        ham_data=ham,
        trial_ops=trial_ops,
        trial_data=trial,
        meas_ops=meas_ops,
        params=params,
        meas_ctx=meas_ctx,
    )
    run_blocks = make_run_blocks(
        block_fn=blocks.block,
        sys=sys_,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )
    final, scalars, _ = run_blocks(
        state,
        ham_data=ham,
        trial_data=trial,
        meas_ctx=meas_ctx,
        prop_ctx=prop_ctx,
        n_blocks=n_blocks,
    )
    return state, final, scalars


def _assert_identical_runs(sd, mps):
    e_sd, e_mps = np.asarray(sd.block_energies), np.asarray(mps.block_energies)
    w_sd, w_mps = np.asarray(sd.block_weights), np.asarray(mps.block_weights)
    assert e_sd.shape == e_mps.shape and w_sd.shape == w_mps.shape
    np.testing.assert_allclose(e_mps, e_sd, rtol=1e-9, atol=0)
    np.testing.assert_allclose(w_mps, w_sd, rtol=1e-9, atol=0)
    assert np.ptp(e_sd[1:]) > 1e-3, "the comparison needs a genuinely stochastic run"
    np.testing.assert_allclose(float(mps.mean_energy), float(sd.mean_energy), rtol=1e-9, atol=0)
    if np.isfinite(sd.stderr_energy):
        np.testing.assert_allclose(
            float(mps.stderr_energy), float(sd.stderr_energy), rtol=1e-9, atol=1e-12
        )
    else:
        assert not np.isfinite(mps.stderr_energy)


def _assert_identical_states(sd, mps, *, nodes_positive):
    (sd_initial, sd_final, sd_scalars), (mps_initial, mps_final, mps_scalars) = sd, mps
    for a, b in zip(mps_initial.walkers, sd_initial.walkers):
        assert np.array_equal(np.asarray(a), np.asarray(b)), "starting walkers must be identical"
    for a, b in zip(mps_final.walkers, sd_final.walkers):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
    w_sd, w_mps = np.asarray(sd_final.weights), np.asarray(mps_final.weights)
    assert np.all(np.isfinite(w_mps)) and np.all(np.isfinite(w_sd))
    assert np.array_equal(w_sd == 0.0, w_mps == 0.0), "a walker died on one side only"
    np.testing.assert_allclose(w_mps, w_sd, rtol=1e-10, atol=0)
    ratio = np.asarray(mps_final.overlaps) / np.asarray(sd_final.overlaps)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)
    assert int(mps_final.node_encounters) == int(sd_final.node_encounters)
    if nodes_positive:
        assert int(mps_final.node_encounters) > 0, "the constraint never acted"
    assert np.array_equal(np.asarray(mps_final.rng_key), np.asarray(sd_final.rng_key))
    for name in ("energy", "weight"):
        assert np.all(np.isfinite(np.asarray(mps_scalars[name])))
        np.testing.assert_allclose(
            np.asarray(mps_scalars[name]), np.asarray(sd_scalars[name]), rtol=1e-10, atol=0
        )


# ---------------------------------------------------------------------------------------------
# Capstones
# ---------------------------------------------------------------------------------------------

# (determinant, nelec, SD propagator, MPS propagator, weight_floor, dt, energy kernel)
CAPSTONE_A = [
    pytest.param("uhf", (3, 2), "fast", "fast", 1e-3, 0.05, "blocked", id="plain-uhf-32"),
    pytest.param("uhf", (3, 2), "fast", "fast", 0.5, 0.1, "blocked", id="floor-uhf-32"),
    pytest.param("nodal", (3, 2), "slow", "fast", 0.0, 0.1, "blocked", id="nodal-slow-vs-fast"),
    pytest.param("uhf", (3, 2), "slow", "slow", 0.25, 0.1, "blocked", id="floor-uhf-32-slow"),
    pytest.param("uhf", (3, 2), "fast", "fast", 1e-3, 0.05, "dense", id="plain-uhf-32-dense"),
    pytest.param("rhf", (3, 3), "fast", "fast", 1e-3, 0.05, "blocked", id="plain-rhf-33"),
    pytest.param("uhf", (3, 3), "fast", "fast", 1e-3, 0.05, "blocked", id="plain-uhf-33"),
    pytest.param("rhf", (3, 2), "fast", "fast", 1e-3, 0.05, "blocked", id="plain-rhf-32"),
    pytest.param("rhf", (3, 3), "fast", "fast", 0.5, 0.1, "blocked", id="floor-rhf-33"),
    pytest.param("uhf", (3, 3), "fast", "fast", 0.5, 0.1, "blocked", id="floor-uhf-33"),
    pytest.param("nodal", (3, 2), "fast", "fast", 1e-3, 0.1, "blocked", id="nodal-fast"),
    pytest.param("nodal", (3, 2), "slow", "slow", 1e-3, 0.1, "blocked", id="nodal-slow"),
    pytest.param("uhf", (3, 3), "slow", "slow", 1e-3, 0.05, "dense", id="plain-uhf-33-slow"),
]


@pytest.mark.parametrize(
    "kind, nelec, sd_propagator, mps_propagator, floor, dt, kernel", CAPSTONE_A
)
def test_sd_and_sd_as_mps_cpmc_runs_are_identical(
    kind, nelec, sd_propagator, mps_propagator, floor, dt, kernel
):
    """Plain-SD trot CPMC and the exact SD->MPS run agree block for block (same seed)."""
    if sd_propagator == "fast":
        # trot's fast cpmc kills a walker when both proposals at a site are floored (zero stored
        # overlap) while the MPS step keeps it; floor <= 0.5 with dt*U <= 0.4 avoids that event.
        assert floor <= 0.5 and dt * U <= 0.4 + 1e-12
    elif mps_propagator == "fast":
        assert floor == 0.0  # cpmc_slow floors half the ratio: the rules coincide only at 0
    else:
        # cpmc_slow zeroes ratios below 2*floor; at floor 0.5 it keeps only the overlap-increasing
        # field at almost every site, the walkers follow one deterministic path and energies freeze.
        assert floor <= 0.25
    L = 6
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(dt=dt, weight_floor=floor, energy_kernel=kernel, propagator=mps_propagator)
    Ca, Cb = _determinant(kind, h1, nelec)
    sd = _sd_side(ham, sys_, params, scipy.linalg.block_diag(Ca, Cb), sd_propagator)
    trial = mps_trial_from_sd(Ca, Cb, rdm1=sd[1].get_rdm1(sd[0]))
    mps = _mps_side(ham, sys_, params, trial)

    _assert_identical_runs(_run(sys_, params, ham, sd), _run(sys_, params, ham, mps))
    if floor > 0.1 or kind == "nodal":
        # Floor events are frequent at floor 0.5; genuine node crossings need the whole run.
        n_blocks = params.n_eql_blocks + params.n_blocks if kind == "nodal" else 2
        _assert_identical_states(
            _blocks_twin(sys_, params, ham, sd, n_blocks),
            _blocks_twin(sys_, params, ham, mps, n_blocks),
            nodes_positive=True,
        )


# (nelec, spin rotation, weight_floor, dt, propagator on both sides)
CAPSTONE_B = [
    pytest.param((3, 2), "rotation", 1e-3, 0.05, "fast", id="rotation-32-plain"),
    pytest.param((3, 3), "reflection", 0.5, 0.1, "fast", id="reflection-33-floor"),
    pytest.param((3, 2), "rotation", 0.25, 0.1, "slow", id="rotation-32-floor-slow"),
    pytest.param((3, 3), "rotation", 1e-3, 0.05, "fast", id="rotation-33-plain"),
    pytest.param((3, 2), "reflection", 0.5, 0.1, "fast", id="reflection-32-floor"),
    pytest.param((3, 2), "rotation", 0.5, 0.1, "fast", id="rotation-32-floor"),
    pytest.param((3, 3), "reflection", 1e-3, 0.05, "fast", id="reflection-33-plain"),
]


@pytest.mark.parametrize("nelec, rotation, floor, dt, propagator", CAPSTONE_B)
def test_rotated_ghf_and_rotated_sd_mps_runs_are_identical(nelec, rotation, floor, dt, propagator):
    """A spin-rotated GhfTrial and the rotated SD-as-MPS trial (projected) agree block for block."""
    L = 6
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(dt=dt, weight_floor=floor, propagator=propagator)
    R = _spin_rotation(rotation)
    Ca, Cb = hf.staggered_determinant(h1, *nelec)  # a singlet RHF determinant would not rotate
    C_ghf = np.kron(R, np.eye(L)) @ cast(np.ndarray, scipy.linalg.block_diag(Ca, Cb))
    sd = _sd_side(ham, sys_, params, C_ghf, propagator)
    rotated = rotate_spin(sd_to_gmps(Ca, Cb, mode="maximal").tensors, R)
    trial = make_mps_trial(rotated, nelec=nelec, rdm1=get_rdm1_block_diag(sd[0]))
    assert trial.sector_weight < 0.9, "the rotation must move weight out of the walkers' sector"
    mps = _mps_side(ham, sys_, params, trial)

    _assert_identical_runs(_run(sys_, params, ham, sd), _run(sys_, params, ham, mps))
    if floor > 0.1:
        _assert_identical_states(
            _blocks_twin(sys_, params, ham, sd),
            _blocks_twin(sys_, params, ham, mps),
            nodes_positive=True,
        )


# ---------------------------------------------------------------------------------------------
# Params, initialisation and compilation
# ---------------------------------------------------------------------------------------------


def test_qmc_params_mps_defaults_and_validation():
    """QmcParamsMps keeps mps_cpmc_new's Config defaults, validates choices, survives replace."""
    p = QmcParamsMps(seed=1)
    assert isinstance(p, QmcParams)
    assert (p.trial_chi, p.dmrg_sweeps, p.dmrg_seed) == (64, 14, 0)
    assert (p.orbital_plan, p.occupation_tolerance) == ("adaptive", 1.0e-10)
    assert (p.walker_channel_chi, p.walker_cutoff) == (4, 0.0)
    assert (p.plan_reference, p.walker_start, p.energy_kernel) == ("natural", "natural", "blocked")
    assert p.propagator == "fast"
    for bad in (
        dict(orbital_plan="greedy"),
        dict(plan_reference="uhf"),
        dict(walker_start="random"),
        dict(energy_kernel="sparse"),
        dict(propagator="medium"),
        dict(trial_chi=0),
        dict(dmrg_sweeps=1),
        dict(walker_channel_chi=0),
        dict(walker_cutoff=-1.0),
    ):
        with pytest.raises(ValueError):
            QmcParamsMps(seed=1, **bad)  # pyright: ignore[reportArgumentType]
    q = dataclasses.replace(p, n_chunks=3)
    assert type(q) is QmcParamsMps and q.n_chunks == 3 and q.trial_chi == 64
    hash(q)


@pytest.fixture(scope="module")
def lattice_model():
    h1 = _lattice23()
    nelec = (3, 2)
    e0, trial = _exact_trial(h1, nelec)
    return types.SimpleNamespace(
        h1=h1,
        nelec=nelec,
        e0=e0,
        trial=trial,
        ham=HamHubbard(h1=jnp.asarray(h1), u=U),
        sys=_system(6, nelec),
    )


def _projector(C):
    Q = np.linalg.qr(np.asarray(C))[0]
    return Q @ Q.T


@pytest.mark.parametrize("walker_start", ["natural", "rhf"])
def test_init_prop_state_start_overlaps_and_strong_types(lattice_model, walker_start):
    """Walkers start on the natural-orbital or RHF determinant; overlaps/energy consistent; strong dtypes."""
    m = lattice_model
    params = _params(walker_start=walker_start, orbital_plan="adaptive", walker_channel_chi=2)
    ops = make_mps_cpmc_ops(m.ham, m.trial, m.sys, params)
    meas_ctx = ops.meas_ops.build_meas_ctx(m.ham, m.trial)
    kwargs: dict[str, Any] = dict(
        sys=m.sys,
        ham_data=m.ham,
        trial_ops=ops.trial_ops,
        trial_data=m.trial,
        meas_ops=ops.meas_ops,
        params=params,
        meas_ctx=meas_ctx,
    )
    state = ops.prop_ops.init_prop_state(**kwargs)
    if walker_start == "natural":
        rdm1 = np.asarray(m.trial.rdm1)
        refs = [natural_orbitals(rdm1[s], m.nelec[s])[0] for s in range(2)]
    else:
        vectors = np.linalg.eigh(m.h1)[1]
        refs = [vectors[:, : m.nelec[0]], vectors[:, : m.nelec[1]]]
    for walkers, ref in zip(state.walkers, refs):
        assert walkers.shape[0] == params.n_walkers
        for w in np.asarray(walkers):
            np.testing.assert_allclose(_projector(w), _projector(ref), rtol=0, atol=1e-12)
    overlaps = jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))(state.walkers, m.trial)
    assert np.array_equal(np.asarray(state.overlaps), np.asarray(overlaps))
    energies = jax.vmap(ops.meas_ops.kernels[k_energy], in_axes=(0, None, None, None))(
        state.walkers, m.ham, meas_ctx, m.trial
    )
    np.testing.assert_allclose(float(state.e_estimate), float(np.mean(energies)), rtol=1e-12)
    for leaf, dtype in (
        (state.node_encounters, jnp.int64),
        (state.e_estimate, jnp.float64),
        (state.pop_control_ene_shift, jnp.float64),
    ):
        assert leaf.dtype == dtype and not leaf.weak_type
    pinned = ops.prop_ops.init_prop_state(
        **kwargs, initial_e_estimate=-2.0  # pyright: ignore[reportArgumentType]
    )
    assert not pinned.e_estimate.weak_type and not pinned.pop_control_ene_shift.weak_type
    assert float(pinned.e_estimate) == -2.0


def test_run_blocks_compiles_once(lattice_model):
    """The jitted block scan compiles once: the initial state already has the carry's dtypes."""
    m = lattice_model
    params = _params(orbital_plan="adaptive", walker_channel_chi=2, n_prop_steps=1)
    ops = make_mps_cpmc_ops(m.ham, m.trial, m.sys, params)
    meas_ctx = ops.meas_ops.build_meas_ctx(m.ham, m.trial)
    prop_ctx = ops.prop_ops.build_prop_ctx(m.ham, ops.trial_ops.get_rdm1(m.trial), params)
    state = ops.prop_ops.init_prop_state(
        sys=m.sys,
        ham_data=m.ham,
        trial_ops=ops.trial_ops,
        trial_data=m.trial,
        meas_ops=ops.meas_ops,
        params=params,
        meas_ctx=meas_ctx,
    )
    run_blocks = make_run_blocks(
        block_fn=blocks.block,
        sys=m.sys,
        params=params,
        trial_ops=ops.trial_ops,
        meas_ops=ops.meas_ops,
        prop_ops=ops.prop_ops,
    )
    if not hasattr(run_blocks, "_cache_size"):
        pytest.skip("jax private _cache_size unavailable")
    for _ in range(2):
        state, _, _ = run_blocks(
            state,
            ham_data=m.ham,
            trial_data=m.trial,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=1,
        )
    assert cast(Any, run_blocks)._cache_size() == 1


# ---------------------------------------------------------------------------------------------
# The step: against trot's slow CPMC and against the legacy closure API
# ---------------------------------------------------------------------------------------------


def _three_steps(prop_ops, state, *, params, ham, trial, trial_ops, meas_ops, meas_ctx, prop_ctx):
    step = jax.jit(
        lambda s: prop_ops.step(
            s,
            params=params,
            ham_data=ham,
            trial_data=trial,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
        )
    )
    for _ in range(3):
        state = step(state)
    return state


def _compare_states(got, want, *, walkers_atol, rtol):
    for g, w in zip(got.walkers, want.walkers):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=0, atol=walkers_atol)
    np.testing.assert_allclose(np.asarray(got.weights), np.asarray(want.weights), rtol=rtol, atol=0)
    np.testing.assert_allclose(
        np.asarray(got.overlaps), np.asarray(want.overlaps), rtol=rtol, atol=0
    )
    np.testing.assert_allclose(
        float(got.pop_control_ene_shift), float(want.pop_control_ene_shift), rtol=rtol
    )
    assert int(got.node_encounters) == int(want.node_encounters)
    assert np.array_equal(np.asarray(got.rng_key), np.asarray(want.rng_key))


def _initial_state(ham, trial, ops, params, meas_ctx, sys_):
    return ops.prop_ops.init_prop_state(
        sys=sys_,
        ham_data=ham,
        trial_ops=ops.trial_ops,
        trial_data=trial,
        meas_ops=ops.meas_ops,
        params=params,
        meas_ctx=meas_ctx,
    )


def _diffused_state(trial, ops, h1, Ca, Cb, n_walkers, seed):
    """Walkers diffused away from the trial by unguided HS fields, so nodes get crossed."""
    walkers = hf.random_field_walkers(h1, U, 0.1, Ca, Cb, n=n_walkers, steps=40, seed=seed)
    stack = tuple(jnp.asarray(np.stack([w[s] for w in walkers])) for s in range(2))
    return PropState(
        walkers=stack,
        weights=jnp.ones(n_walkers),
        overlaps=jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))(stack, trial),
        rng_key=jax.random.PRNGKey(seed),
        pop_control_ene_shift=jnp.asarray(-3.0),
        e_estimate=jnp.asarray(-3.0),
        node_encounters=jnp.zeros((), jnp.int64),
    )


def _compare_with_slow_cpmc(ham, trial, ops, params, state, n_rounds):
    """n_rounds x 3 steps of the MPS step and of trot's cpmc_slow on the MPS overlap; node count."""
    meas_ctx = ops.meas_ops.build_meas_ctx(ham, trial)
    common = dict(
        params=params,
        ham=ham,
        trial=trial,
        trial_ops=ops.trial_ops,
        meas_ops=ops.meas_ops,
        meas_ctx=meas_ctx,
        prop_ctx=ops.prop_ops.build_prop_ctx(ham, None, params),
    )
    nodes = 0
    for _ in range(n_rounds):
        got = _three_steps(ops.prop_ops, state, **common)
        want = _three_steps(cpmc_slow.make_prop_ops(ham, "unrestricted"), state, **common)
        _compare_states(got, want, walkers_atol=1e-12, rtol=1e-10)
        nodes += int(got.node_encounters) - int(state.node_encounters)
        state = got
    return nodes


def test_step_matches_trot_slow_cpmc_on_a_nodal_trial():
    """The cached-environment sweep equals trot's cpmc_slow (fresh overlaps) at floor 0, with nodes."""
    L, nelec = 6, (3, 2)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(dt=0.1, weight_floor=0.0, n_walkers=20, n_prop_steps=1)
    Ca, Cb = _determinant("nodal", h1, nelec)
    trial = mps_trial_from_sd(Ca, Cb)
    ops = make_mps_cpmc_ops(ham, trial, sys_, params)
    state = _diffused_state(trial, ops, h1, Ca, Cb, params.n_walkers, seed=7)
    assert _compare_with_slow_cpmc(ham, trial, ops, params, state, n_rounds=4) > 0


def test_step_matches_trot_slow_cpmc_with_a_dmrg_trial():
    """With a DMRG trial and exact walker conversion the sweep equals cpmc_slow at floor 0.

    With truncated walkers the two differ by design: the sweep converts a walker once and applies
    the HS fields to that MPS, cpmc_slow re-converts (and re-truncates) after every field.
    """
    pytest.importorskip("pyblock3")
    from trot.gmps.dmrg import make_dmrg_trial

    L, nelec = 6, (3, 3)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    trial = make_dmrg_trial(ham, sys_, chi=8, n_sweeps=6, seed=0).trial
    params = _params(dt=0.1, weight_floor=0.0, n_walkers=20, n_prop_steps=1)
    ops = make_mps_cpmc_ops(ham, trial, sys_, params)
    Ca, Cb = ops.plan.reference
    state = _diffused_state(trial, ops, h1, Ca, Cb, params.n_walkers, seed=8)
    _compare_with_slow_cpmc(ham, trial, ops, params, state, n_rounds=2)


@pytest.mark.parametrize("walker_channel_chi", [None, 2])
def test_new_step_matches_the_legacy_closure_step(lattice_model, walker_channel_chi):
    """make_prop_ops (trial from trial_data, blocks from meas_ctx) = legacy make_fast_prop_ops."""
    m = lattice_model
    params = _params(orbital_plan="adaptive", walker_channel_chi=walker_channel_chi)
    ops = make_mps_cpmc_ops(m.ham, m.trial, m.sys, params)
    meas_ctx = ops.meas_ops.build_meas_ctx(m.ham, m.trial)
    prop_ctx = ops.prop_ops.build_prop_ctx(m.ham, ops.trial_ops.get_rdm1(m.trial), params)
    plan = ops.plan
    legacy_ops = make_walker_ops(
        plan.reference[0],
        plan.reference[1],
        plan.orbital_plans[0],
        plan.orbital_plans[1],
        plan.bond_plans[0],
        plan.bond_plans[1],
        [np.asarray(A) for A in m.trial.tensors],
        m.trial.charge_arrays(),
    )
    legacy = make_fast_prop_ops(m.ham, "unrestricted", legacy_ops.overlap, legacy_ops.sweep)
    state = _initial_state(m.ham, m.trial, ops, params, meas_ctx, m.sys)
    common = dict(
        params=params,
        ham=m.ham,
        trial_ops=ops.trial_ops,
        meas_ops=ops.meas_ops,
        meas_ctx=meas_ctx,
        prop_ctx=prop_ctx,
    )
    got = _three_steps(ops.prop_ops, state, trial=m.trial, **common)
    want = _three_steps(legacy, state, trial=None, **common)  # legacy ignores trial_data
    _compare_states(got, want, walkers_atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# run_qmc and run_qmc_mps
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("energy_kernel", ["blocked", "dense"])
@pytest.mark.parametrize("walker_channel_chi", [None, 2])
def test_zero_variance_with_the_exact_trial(lattice_model, walker_channel_chi, energy_kernel):
    """An exact trial gives E_loc = E0 for every walker, so every run_qmc block equals E0."""
    m = lattice_model
    params = _params(
        n_walkers=6,
        n_prop_steps=3,
        n_eql_blocks=2,
        n_blocks=4,
        seed=3,
        orbital_plan="adaptive",
        walker_channel_chi=walker_channel_chi,
        energy_kernel=energy_kernel,
    )
    result = _run(m.sys, params, m.ham, _mps_side(m.ham, m.sys, params, m.trial))
    np.testing.assert_allclose(np.asarray(result.block_energies), m.e0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(result.mean_energy), m.e0, rtol=0, atol=1e-8)


def _small_model():
    h1 = hopping_matrix(4, 1.0)
    nelec = (2, 2)
    e0, trial = _exact_trial(h1, nelec)
    return h1, nelec, e0, trial, HamHubbard(h1=jnp.asarray(h1), u=U), _system(4, nelec)


def test_run_qmc_mps_builds_the_dmrg_trial(monkeypatch):
    """With no trial, run_qmc_mps runs DMRG once with the params and gets the exact L=4 energy."""
    pytest.importorskip("pyblock3")
    import trot.gmps.dmrg as dmrg

    h1, nelec, _, _, ham, sys_ = _small_model()
    calls = []
    original = dmrg.make_dmrg_trial

    def spy(ham_data, sys_arg, **kwargs):
        before = np.random.get_state()[1].copy()
        out = original(ham_data, sys_arg, **kwargs)
        calls.append((kwargs, np.array_equal(before, np.random.get_state()[1])))
        return out

    monkeypatch.setattr(dmrg, "make_dmrg_trial", spy)
    params = _params(
        n_walkers=6,
        n_prop_steps=3,
        n_eql_blocks=2,
        n_blocks=4,
        seed=3,
        trial_chi=16,
        dmrg_sweeps=8,
        orbital_plan="adaptive",
        walker_channel_chi=2,
    )
    result = run_qmc_mps(sys=sys_, params=params, ham_data=ham)
    assert len(calls) == 1
    kwargs, rng_restored = calls[0]
    assert kwargs == dict(chi=16, n_sweeps=8, seed=0)
    assert rng_restored, "make_dmrg_trial must restore numpy's global RNG"
    np.testing.assert_allclose(np.asarray(result.block_energies), E_L4, rtol=0, atol=1e-8)


def test_run_qmc_mps_with_a_ready_trial_equals_manual_run_qmc(monkeypatch):
    """A ready trial skips DMRG; run_qmc_mps gives exactly the manually assembled run_qmc result."""

    def no_dmrg(*args, **kwargs):
        raise AssertionError("DMRG must not run when a trial is given")

    monkeypatch.setitem(
        sys.modules, "trot.gmps.dmrg", types.SimpleNamespace(make_dmrg_trial=no_dmrg)
    )
    h1, nelec, _, trial, ham, sys_ = _small_model()
    rng = np.random.default_rng(4)
    noisy = make_mps_trial(  # a non-exact trial, so the energies fluctuate
        [
            np.asarray(A) + 0.05 * rng.standard_normal(np.shape(A)) * (np.asarray(A) != 0)
            for A in trial.tensors
        ],
        trial.charge_arrays(),
        nelec=nelec,
    )
    params = _params(n_walkers=6, n_prop_steps=3, n_eql_blocks=2, n_blocks=4, seed=3)
    via_driver = run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=noisy)
    manual = _run(sys_, params, ham, _mps_side(ham, sys_, params, noisy))
    assert np.array_equal(np.asarray(via_driver.block_energies), np.asarray(manual.block_energies))
    assert np.array_equal(np.asarray(via_driver.block_weights), np.asarray(manual.block_weights))
    assert np.ptp(np.asarray(manual.block_energies)) > 1e-4


def test_run_qmc_chunk_sizes_and_compile_warning():
    """run_qmc's block batches are computed exactly; run_qmc_mps warns when several would compile."""
    sizes = lambda n_eql, n_blocks: run_qmc_chunk_sizes(
        QmcParamsMps(seed=1, n_eql_blocks=n_eql, n_blocks=n_blocks)
    )
    assert sizes(13, 20) == {1, 2}
    assert sizes(4, 10) == {1}
    assert sizes(100, 500) == {20, 50}
    assert sizes(100, 200) == {20}
    assert sizes(0, 7) == {1}
    h1, nelec, _, trial, ham, sys_ = _small_model()
    params = _params(n_walkers=4, n_prop_steps=1, n_eql_blocks=10, n_blocks=4, seed=3)
    with pytest.warns(UserWarning, match="once per block-batch size"):
        run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=trial)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_run_qmc_mps_reports_population_collapse():
    """A killed population (every weight above weight_cap) raises a clear RuntimeError."""
    h1, nelec, _, trial, ham, sys_ = _small_model()
    params = _params(
        n_walkers=4, n_prop_steps=1, n_eql_blocks=1, n_blocks=4, seed=3, weight_cap=1e-3
    )
    with pytest.raises(RuntimeError, match="collapsed"):
        run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=trial)


# ---------------------------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------------------------

PRE_REFACTOR_NAMES = {
    "mps_cpmc_new": (
        "BondPlan CFG Callable Config FCIDUMP HamHubbard Hamiltonian MPE MeasOps NamedTuple "
        "OrbitalPlan PHYSICAL_CHARGE Path PropOps PropState QmcParams SectorPlan System UhfTrial "
        "WalkerOps _assemble _block_plan _build_prop_ctx _charge_index _factor_block _key "
        "_move_centre _rotate_mode_to_front _shift_centre _vector_qr annotations apply_mpo asdict "
        "blocked_contract_from_blocks blocking_analysis_ratio blocks build_dmrg_hamiltonian "
        "channel_angles channel_mps combine_channels combined_charges compress_mps constrain_ratio "
        "contract_real contraction_report dataclass densify_with_charges extract_channel_blocks "
        "extract_fixed_blocks flat_blocks gate_pair hopping_matrix hubbard_dmrg_mpo hubbard_mpo "
        "init_prop_state init_prop_state_typed jax jnp json k_energy main make_auto_trial_ops "
        "make_block_logger make_channel_block_maps make_contraction_plan make_fast_prop_ops "
        "make_fast_sweep make_hubbard_cpmc_ops make_orbital_plan make_run_blocks make_walker_ops "
        "math natural_orbitals np one_rdm plan_bonds qr_with_det reject_outliers "
        "right_environments run_dmrg run_qmc_fixed_chunks save_result sector_plan "
        "spin_occupations split_pair time uhf_get_rdm1 wk"
    ).split(),
    "mps_cpmc_2d": (
        "CHANNEL_CHARGE Config FCIDUMP HamHubbard Hamiltonian MPE MeasOps QmcParams System "
        "UhfTrial annotations apply_mpo asdict ast blocks build_dmrg_hamiltonian config_from_args "
        "dataclass densified_trial describe dmrg_schedule hubbard_mpo_from_h1 jnp k_energy "
        "lattice_hopping m main make_auto_trial_ops make_blocked_energy np reference run_dmrg "
        "square_hopping_matrix sys time trial_times_h uhf_get_rdm1"
    ).split(),
}

MOVED = {
    "trot.trial.mps": "PHYSICAL_CHARGE one_rdm natural_orbitals _charge_index make_contraction_plan "
    "extract_fixed_blocks make_channel_block_maps extract_channel_blocks "
    "blocked_contract_from_blocks contraction_report",
    "trot.meas.mps": "hubbard_mpo apply_mpo compress_mps",
    "trot.prop.mps_cpmc": "right_environments constrain_ratio make_fast_sweep "
    "init_prop_state_typed make_fast_prop_ops",
    "trot.gmps.dmrg": "build_dmrg_hamiltonian hubbard_dmrg_mpo run_dmrg",
    "trot.gmps.driver": "WalkerOps make_walker_ops make_block_logger run_qmc_fixed_chunks save_result",
    "trot.ham.hubbard": "hopping_matrix",
}


def test_legacy_scripts_reexport_every_pre_refactor_name():
    """mps_cpmc_new / mps_cpmc_2d keep every pre-refactor name; moved objects are the trot ones."""
    pytest.importorskip("pyblock3")
    import importlib

    from trot.gmps import mps_cpmc_2d, mps_cpmc_new

    for module, names in ((mps_cpmc_new, "mps_cpmc_new"), (mps_cpmc_2d, "mps_cpmc_2d")):
        missing = [n for n in PRE_REFACTOR_NAMES[names] if not hasattr(module, n)]
        assert not missing, f"{names} lost {missing}"
        assert set(module.__all__) == set(PRE_REFACTOR_NAMES[names]) - {"annotations"}
    for module_name, names in MOVED.items():
        home = importlib.import_module(module_name)
        for name in names.split():
            assert getattr(mps_cpmc_new, name) is getattr(home, name), name
    meas = importlib.import_module("trot.meas.mps")
    for name in ("hubbard_mpo_from_h1", "apply_mpo", "trial_times_h", "CHANNEL_CHARGE"):
        assert getattr(mps_cpmc_2d, name) is getattr(meas, name), name


@pytest.fixture(scope="module")
def frozen_legacy():
    """HEAD's self-contained mps_cpmc_new.py, executed as a separate module."""
    pytest.importorskip("pyblock3")
    try:
        source = subprocess.run(
            ["git", "show", f"{FROZEN_LEGACY_COMMIT}:trot/gmps/mps_cpmc_new.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {FROZEN_LEGACY_COMMIT} unavailable")
    module = types.ModuleType("mps_cpmc_new_frozen")
    module.__file__ = f"{FROZEN_LEGACY_COMMIT}:trot/gmps/mps_cpmc_new.py"
    sys.modules[module.__name__] = module  # dataclasses look the module up
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        yield module
    finally:
        sys.modules.pop(module.__name__, None)


def _assert_same(a, b):
    a, b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_legacy_api_is_bitwise_identical_to_the_frozen_head(frozen_legacy):
    """The moved legacy API reproduces HEAD's self-contained mps_cpmc_new bit for bit."""
    from trot.core.ops import MeasOps as Meas
    from trot.gmps import mps_cpmc_new as m
    from trot.trial.auto import make_auto_trial_ops
    from trot.trial.uhf import UhfTrial, get_rdm1

    old = frozen_legacy
    L, N = 6, 3
    h1 = hopping_matrix(L, 1.0)
    kwargs: dict[str, Any] = dict(L=L, n_up=N, n_down=N, interaction=U, trial_chi=16, dmrg_sweeps=8)
    cfg_old, cfg_new = old.Config(**kwargs), m.Config(**kwargs)
    mps_old, e_old = old.run_dmrg(old.build_dmrg_hamiltonian(cfg_old), cfg_old)
    rng_old = np.random.get_state()[1].copy()
    mps_new, e_new = m.run_dmrg(m.build_dmrg_hamiltonian(cfg_new), cfg_new)
    assert e_old == e_new
    assert np.array_equal(rng_old, np.random.get_state()[1]), "run_dmrg's RNG side effect changed"
    trial_old, charges_old = old.densify_with_charges(mps_old, L)
    trial_new, charges_new = m.densify_with_charges(mps_new, L)
    _assert_same(trial_old, trial_new)
    _assert_same(charges_old, charges_new)
    W = old.hubbard_mpo(L, 1.0, U)
    _assert_same(W, m.hubbard_mpo(L, 1.0, U))
    H_old = old.compress_mps(old.apply_mpo(W, trial_old))
    H_new = m.compress_mps(m.apply_mpo(W, trial_new))
    _assert_same(H_old, H_new)
    _assert_same(old.one_rdm(trial_old), m.one_rdm(trial_new))

    C = np.linalg.eigh(h1)[1][:, :N]
    plan_old, plan_new = old.make_orbital_plan(C), m.make_orbital_plan(C)
    bond_old, bond_new = old.plan_bonds(C, plan_old, 4), m.plan_bonds(C, plan_new, 4)
    H_old_j = tuple(jnp.asarray(A) for A in H_old)
    H_new_j = tuple(jnp.asarray(A) for A in H_new)
    ops_old = old.make_walker_ops(
        C, C, plan_old, plan_old, bond_old, bond_old, trial_old, charges_old, H_old_j
    )
    ops_new = m.make_walker_ops(
        C, C, plan_new, plan_new, bond_new, bond_new, trial_new, charges_new, H_new_j
    )
    _assert_same(ops_old.walker_charges, ops_new.walker_charges)
    assert ops_new.energy is not None
    hs = _build_prop_ctx(HamHubbard(h1=jnp.asarray(h1), u=U), 0.01).hs_constant
    randoms = jnp.asarray(np.random.default_rng(2).random(L))
    walkers = hf.random_field_walkers(h1, U, 0.01, C, C, n=3, steps=50, seed=0)
    for ca, cb in walkers:
        w = (jnp.asarray(ca), jnp.asarray(cb))
        _assert_same(ops_old.overlap(w), ops_new.overlap(w))
        _assert_same(ops_old.energy(w), ops_new.energy(w))
        _assert_same(ops_old.sweep(*w, randoms, hs, 1e-3), ops_new.sweep(*w, randoms, hs, 1e-3))

    ham = HamHubbard(h1=jnp.asarray(h1), u=U)
    sys_ = _system(L, (N, N))
    params = QmcParams(
        dt=0.01, n_walkers=3, n_prop_steps=2, n_blocks=3, n_eql_blocks=0, weight_floor=1e-3, seed=5
    )
    prop_ctx = _build_prop_ctx(ham, params.dt)
    stack = tuple(jnp.asarray(np.stack([w[s] for w in walkers])) for s in range(2))

    def state(ops):
        return PropState(
            walkers=stack,
            weights=jnp.ones(3),
            overlaps=jax.vmap(ops.overlap)(stack),
            rng_key=jax.random.PRNGKey(5),
            pop_control_ene_shift=jnp.asarray(-4.0),
            e_estimate=jnp.asarray(-4.0),
            node_encounters=jnp.zeros((), jnp.int64),
        )

    results = []
    for module, ops in ((old, ops_old), (m, ops_new)):
        prop = module.make_fast_prop_ops(ham, "unrestricted", ops.overlap, ops.sweep)
        step = jax.jit(
            lambda s, prop=prop: prop.step(
                s,
                params=params,
                ham_data=ham,
                trial_data=None,
                trial_ops=None,
                meas_ops=None,
                meas_ctx=None,
                prop_ctx=prop_ctx,
            )
        )
        trial_ops = make_auto_trial_ops(sys_, overlap_u=ops.overlap, get_rdm1=get_rdm1)
        assert ops.energy is not None
        meas_ops = Meas(overlap=ops.overlap, kernels={k_energy: ops.energy})
        run_blocks = make_run_blocks(
            block_fn=blocks.block,
            sys=sys_,
            params=params,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop,
        )
        final, scalars, _ = run_blocks(
            state(ops),
            ham_data=ham,
            trial_data=UhfTrial(jnp.asarray(C), jnp.asarray(C)),
            meas_ctx=None,
            prop_ctx=prop_ctx,
            n_blocks=3,
        )
        results.append((step(state(ops)), final, scalars))
    _assert_same(results[0], results[1])


# ---------------------------------------------------------------------------------------------
# Batched engine (QmcParamsMps.engine = "batched", trot/gmps/gpu.py) against the reference engine
# ---------------------------------------------------------------------------------------------

BATCHED_LINALG = [
    pytest.param("batched", "native", id="batched-householder"),
    pytest.param("batched", "cholesky", id="batched-choleskyqr2"),
    pytest.param("native", "native", id="loop-householder"),
]


def _both_engines(m, **overrides):
    """The reference and the batched ops (and their meas_ctx) for the same trial and settings."""
    out = {}
    for engine in ("reference", "batched"):
        params = _params(engine=engine, **overrides)
        ops = make_mps_cpmc_ops(m.ham, m.trial, m.sys, params)
        out[engine] = types.SimpleNamespace(
            params=params, ops=ops, ctx=ops.meas_ops.build_meas_ctx(m.ham, m.trial)
        )
    assert out["batched"].ops.engine is not None and out["reference"].ops.engine is None
    return out


def test_engine_auto_is_batched_exactly_on_a_gpu(lattice_model):
    m = lattice_model
    ops = make_mps_cpmc_ops(m.ham, m.trial, m.sys, _params())
    assert (ops.engine is not None) == (jax.default_backend() == "gpu")  # "fast" propagator (the default)
    with pytest.raises(ValueError):
        _params(engine="fastest")


@pytest.mark.parametrize("linalg, walker_qr", BATCHED_LINALG)
@pytest.mark.parametrize("walker_channel_chi", [None, 2])
@pytest.mark.parametrize("energy_kernel", ["blocked", "dense"])
def test_batched_overlaps_and_energies_match_the_reference(
    lattice_model, linalg, walker_qr, walker_channel_chi, energy_kernel
):
    """Diffused walkers (nodes crossed): overlaps and local energies of both engines agree."""
    m = lattice_model
    e = _both_engines(
        m,
        orbital_plan="adaptive",
        walker_channel_chi=walker_channel_chi,
        energy_kernel=energy_kernel,
        linalg=linalg,
        walker_qr=walker_qr,
    )
    Ca, Cb = _determinant("uhf", m.h1, m.nelec)
    walkers = _diffused_state(m.trial, e["reference"].ops, m.h1, Ca, Cb, n_walkers=8, seed=3).walkers
    results = {}
    for name, side in e.items():
        overlap = jax.vmap(side.ops.trial_ops.overlap, in_axes=(0, None))(walkers, m.trial)
        kernel = side.ops.meas_ops.require_kernel(k_energy)
        energy = jax.vmap(kernel, in_axes=(0, None, None, None))(walkers, m.ham, side.ctx, m.trial)
        results[name] = (np.asarray(overlap), np.asarray(energy))
    np.testing.assert_allclose(results["batched"][0], results["reference"][0], rtol=1e-9)
    np.testing.assert_allclose(results["batched"][1], results["reference"][1], rtol=1e-9)
    assert e["batched"].ctx.trial_energy == pytest.approx(e["reference"].ctx.trial_energy, rel=1e-12)
    assert e["batched"].ctx.kernel == energy_kernel


@pytest.mark.parametrize("walker_qr", ["native", "cholesky"])
@pytest.mark.parametrize("walker_channel_chi", [None, 2])
def test_batched_step_matches_the_reference_step(lattice_model, walker_qr, walker_channel_chi):
    """Three steps of each engine from the same state: same walkers, weights, overlaps, shift, nodes, RNG."""
    m = lattice_model
    e = _both_engines(
        m, orbital_plan="adaptive", walker_channel_chi=walker_channel_chi, linalg="batched", walker_qr=walker_qr
    )
    ref = e["reference"]
    state = _initial_state(m.ham, m.trial, ref.ops, ref.params, ref.ctx, m.sys)
    out = {}
    for name, side in e.items():
        out[name] = _three_steps(
            side.ops.prop_ops,
            state,
            params=side.params,
            ham=m.ham,
            trial=m.trial,
            trial_ops=side.ops.trial_ops,
            meas_ops=side.ops.meas_ops,
            meas_ctx=side.ctx,
            prop_ctx=side.ops.prop_ops.build_prop_ctx(m.ham, None, side.params),
        )
    _compare_states(out["batched"], out["reference"], walkers_atol=1e-12, rtol=1e-9)


def test_batched_step_chunking_is_exact(lattice_model):
    """n_chunks that does not divide the walkers (3 of 10) is rounded up to a divisor; same result as one chunk."""
    m = lattice_model
    results = []
    for n_chunks in (1, 3):
        params = _params(engine="batched", orbital_plan="adaptive", walker_channel_chi=2, n_chunks=n_chunks)
        ops = make_mps_cpmc_ops(m.ham, m.trial, m.sys, params)
        ctx = ops.meas_ops.build_meas_ctx(m.ham, m.trial)
        state = _initial_state(m.ham, m.trial, ops, params, ctx, m.sys)
        results.append(
            _three_steps(
                ops.prop_ops,
                state,
                params=params,
                ham=m.ham,
                trial=m.trial,
                trial_ops=ops.trial_ops,
                meas_ops=ops.meas_ops,
                meas_ctx=ctx,
                prop_ctx=ops.prop_ops.build_prop_ctx(m.ham, None, params),
            )
        )
    _compare_states(results[1], results[0], walkers_atol=1e-13, rtol=1e-11)


def test_run_qmc_mps_engines_agree(lattice_model):
    """run_qmc_mps end to end (init, trot's block and measurement, statistics) with either engine."""
    m = lattice_model
    runs = {}
    for engine in ("reference", "batched"):
        params = _params(
            engine=engine,
            orbital_plan="adaptive",
            walker_channel_chi=2,
            linalg="batched",
            walker_qr="native",
            n_eql_blocks=2,
            n_blocks=4,
        )
        runs[engine] = run_qmc_mps(sys=m.sys, params=params, ham_data=m.ham, trial_data=m.trial)
    for key in ("block_energies", "block_weights"):
        np.testing.assert_allclose(
            np.asarray(getattr(runs["batched"], key)), np.asarray(getattr(runs["reference"], key)), rtol=1e-9
        )
    assert float(runs["batched"].mean_energy) == pytest.approx(float(runs["reference"].mean_energy), rel=1e-9)
