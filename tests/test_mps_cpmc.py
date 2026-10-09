"""MPS-CPMC through trot's driver: propagation, the MPS block, run_qmc and run_qmc_mps.

Capstones. Plain-SD trot CPMC (GhfTrial) and the same run with the SD trial and every walker turned
into an MPS (maximal orbital plan, no truncation) make identical decisions, so every block energy and
weight agrees to rounding. The same holds for a spin-rotated GHF trial against the rotated SD-as-MPS
trial, which the MPS code uses as it is (particle-number labels): the walkers pick their sector.
"""

from trot import config

config.configure_once()

import dataclasses
import sys
import types
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
from trot.gmps.driver import make_mps_cpmc_ops, run_qmc_chunk_sizes, run_qmc_mps
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix, square_hopping_matrix
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.prop import blocks, cpmc, cpmc_slow, mps_cpmc
from trot.prop.types import PropState, QmcParams, QmcParamsMps
from trot.trial.ghf import GhfTrial, get_rdm1_block_diag, make_ghf_trial_ops
from trot.trial.mps import make_mps_trial, mps_trial_from_sd, natural_orbitals, rotate_spin

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


def _determinant_energy(h1, Ca, Cb):
    """<SD|H|SD>/<SD|SD> of SD(Ca, Cb) by exact enumeration."""
    H = hf.hubbard_sector_hamiltonian(h1, U, Ca.shape[1], Cb.shape[1])
    return hf.local_energy(hf.sd_amplitudes(Ca, Cb), H, Ca, Cb)


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


def _noisy_trial(trial, nelec, seed):
    """trial with every nonzero entry perturbed: same bond labels, no longer an eigenstate."""
    rng = np.random.default_rng(seed)
    return make_mps_trial(
        [
            np.asarray(A) + 0.05 * rng.standard_normal(np.shape(A)) * (np.asarray(A) != 0)
            for A in trial.tensors
        ],
        trial.charge_arrays(),
        nelec=nelec,
    )


def _sd_side(ham, sys_, params, C_ghf, propagator="fast"):
    """The user's plain trot CPMC template: GhfTrial + GHF ops + trot's fast (cpmc) or slow (cpmc_slow) step."""
    trial = GhfTrial(mo_coeff=jnp.asarray(C_ghf))
    trial_ops = make_ghf_trial_ops(sys_)
    meas_ops = make_ghf_meas_ops_hubbard(sys_)
    if propagator == "fast":
        prop_ops = cpmc.make_prop_ops(ham, "unrestricted", trial_ops)
    elif propagator == "slow":
        prop_ops = cpmc_slow.make_prop_ops(ham, "unrestricted")
    else:
        raise ValueError(propagator)
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

# (determinant, nelec, SD propagator, weight_floor, dt, energy kernel); the MPS side runs the MPS step
CAPSTONE_A = [
    pytest.param("uhf", (3, 2), "fast", 1e-3, 0.05, "blocked", id="plain-uhf-32"),
    pytest.param("uhf", (3, 2), "fast", 0.5, 0.1, "blocked", id="floor-uhf-32"),
    pytest.param("nodal", (3, 2), "slow", 0.0, 0.1, "blocked", id="nodal-slow-vs-fast"),
    pytest.param("uhf", (3, 2), "fast", 1e-3, 0.05, "dense", id="plain-uhf-32-dense"),
    pytest.param("rhf", (3, 3), "fast", 1e-3, 0.05, "blocked", id="plain-rhf-33"),
    pytest.param("uhf", (3, 3), "fast", 1e-3, 0.05, "blocked", id="plain-uhf-33"),
    pytest.param("rhf", (3, 2), "fast", 1e-3, 0.05, "blocked", id="plain-rhf-32"),
    pytest.param("rhf", (3, 3), "fast", 0.5, 0.1, "blocked", id="floor-rhf-33"),
    pytest.param("uhf", (3, 3), "fast", 0.5, 0.1, "blocked", id="floor-uhf-33"),
    pytest.param("nodal", (3, 2), "fast", 1e-3, 0.1, "blocked", id="nodal-fast"),
]


@pytest.mark.parametrize("kind, nelec, sd_propagator, floor, dt, kernel", CAPSTONE_A)
def test_sd_and_sd_as_mps_cpmc_runs_are_identical(kind, nelec, sd_propagator, floor, dt, kernel):
    """Plain-SD trot CPMC and the exact SD->MPS run agree block for block (same seed)."""
    if sd_propagator == "fast":
        # trot's fast cpmc kills a walker when both proposals at a site are floored (zero stored
        # overlap) while the MPS step keeps it; floor <= 0.5 with dt*U <= 0.4 avoids that event.
        assert floor <= 0.5 and dt * U <= 0.4 + 1e-12
    else:
        assert floor == 0.0  # cpmc_slow floors half the ratio: the rules coincide only at 0
    L = 6
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(dt=dt, weight_floor=floor, energy_kernel=kernel)
    Ca, Cb = _determinant(kind, h1, nelec)
    sd = _sd_side(ham, sys_, params, scipy.linalg.block_diag(Ca, Cb), sd_propagator)
    trial = mps_trial_from_sd(Ca, Cb, rdm1=sd[1].get_rdm1(sd[0]))
    mps = _mps_side(ham, sys_, params, trial)

    sd_run, mps_run = _run(sys_, params, ham, sd), _run(sys_, params, ham, mps)
    # Block entry 0 is the initial e_estimate. The walkers start on the natural orbitals of the trial's
    # projector rdm1, i.e. on the trial determinant itself, so both sides start at its variational energy.
    e_trial = _determinant_energy(h1, Ca, Cb)
    for name, run in (("SD", sd_run), ("MPS", mps_run)):
        e0 = float(np.asarray(run.block_energies)[0])
        assert e0 == pytest.approx(e_trial, rel=1e-10), f"{name} side starts at {e0}, not {e_trial}"
    _assert_identical_runs(sd_run, mps_run)
    if floor > 0.1 or kind == "nodal":
        # Floor events are frequent at floor 0.5; genuine node crossings need the whole run.
        n_blocks = params.n_eql_blocks + params.n_blocks if kind == "nodal" else 2
        _assert_identical_states(
            _blocks_twin(sys_, params, ham, sd, n_blocks),
            _blocks_twin(sys_, params, ham, mps, n_blocks),
            nodes_positive=True,
        )


# (nelec, spin rotation, weight_floor, dt); trot's fast cpmc on the GHF side
CAPSTONE_B = [
    pytest.param((3, 2), "rotation", 1e-3, 0.05, id="rotation-32-plain"),
    pytest.param((3, 3), "reflection", 0.5, 0.1, id="reflection-33-floor"),
    pytest.param((3, 3), "rotation", 1e-3, 0.05, id="rotation-33-plain"),
    pytest.param((3, 2), "reflection", 0.5, 0.1, id="reflection-32-floor"),
    pytest.param((3, 2), "rotation", 0.5, 0.1, id="rotation-32-floor"),
    pytest.param((3, 3), "reflection", 1e-3, 0.05, id="reflection-33-plain"),
]


@pytest.mark.parametrize("nelec, rotation, floor, dt", CAPSTONE_B)
def test_rotated_ghf_and_rotated_sd_mps_runs_are_identical(nelec, rotation, floor, dt):
    """A spin-rotated GhfTrial and the rotated SD-as-MPS trial (used as it is) agree block for block."""
    L = 6
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(dt=dt, weight_floor=floor)
    R = _spin_rotation(rotation)
    Ca, Cb = hf.staggered_determinant(h1, *nelec)  # a singlet RHF determinant would not rotate
    C_ghf = np.kron(R, np.eye(L)) @ cast(np.ndarray, scipy.linalg.block_diag(Ca, Cb))
    sd = _sd_side(ham, sys_, params, C_ghf)
    rotated = rotate_spin(sd_to_gmps(Ca, Cb, mode="maximal").tensors, R)
    trial = make_mps_trial(rotated, nelec=nelec, rdm1=get_rdm1_block_diag(sd[0]))
    assert trial.label_width == 1 and trial.sector_weight is None  # no definite (N_up, N_dn)
    weight = float(np.sum(hf.mps_sector_amplitudes(trial.tensors, *nelec) ** 2))
    assert weight < 0.9, "the rotation must move weight out of the walkers' sector"
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
    """QmcParamsMps defaults and validated choices; engine, linalg and propagator are gone; survives replace."""
    p = QmcParamsMps(seed=1)
    assert isinstance(p, QmcParams)
    assert (p.trial_chi, p.dmrg_sweeps, p.dmrg_seed) == (64, 14, 0)
    assert (p.orbital_plan, p.occupation_tolerance) == ("adaptive", 1.0e-10)
    assert (p.walker_channel_chi, p.walker_cutoff) == (4, 0.0)
    assert (p.plan_reference, p.walker_start, p.energy_kernel) == ("natural", "natural", "blocked")
    assert (p.walker_qr, p.sector_buckets) == ("auto", (8, 16))
    for bad in (
        dict(orbital_plan="greedy"),
        dict(plan_reference="uhf"),
        dict(walker_start="random"),
        dict(energy_kernel="sparse"),
        dict(walker_qr="householder"),
        dict(sector_buckets=(16, 8)),
        dict(sector_buckets=(8, 8)),
        dict(sector_buckets=(0, 8)),
        dict(trial_chi=0),
        dict(dmrg_sweeps=1),
        dict(walker_channel_chi=0),
        dict(walker_cutoff=-1.0),
    ):
        with pytest.raises(ValueError):
            QmcParamsMps(seed=1, **bad)  # pyright: ignore[reportArgumentType]
    for removed in (dict(engine="batched"), dict(linalg="native"), dict(propagator="fast")):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            QmcParamsMps(seed=1, **removed)  # pyright: ignore[reportCallIssue]
    listed = [4, 12]
    buckets = QmcParamsMps(seed=1, sector_buckets=listed).sector_buckets  # pyright: ignore[reportArgumentType]
    assert buckets == (4, 12) and type(buckets) is tuple
    assert QmcParamsMps(seed=1, sector_buckets=()).sector_buckets == ()
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
    """Walkers start on the natural-orbital or RHF determinant; overlaps/energy consistent (one probed walker
    broadcast, or a given batch); strong dtypes."""
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
    overlap = jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))
    energy = jax.vmap(ops.meas_ops.kernels[k_energy], in_axes=(0, None, None, None))
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
    np.testing.assert_allclose(
        np.asarray(state.overlaps), np.asarray(overlap(state.walkers, m.trial)), rtol=1e-12
    )
    energies = energy(state.walkers, m.ham, meas_ctx, m.trial)
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

    walkers = hf.random_field_walkers(m.h1, U, 0.1, *refs, n=params.n_walkers, steps=10, seed=2)
    stack = tuple(jnp.asarray(np.stack([w[s] for w in walkers])) for s in range(2))
    given = ops.prop_ops.init_prop_state(**kwargs, initial_walkers=stack)
    want = np.asarray(overlap(stack, m.trial))
    assert np.ptp(want / want[0]) > 1e-3  # the given walkers differ
    np.testing.assert_allclose(np.asarray(given.overlaps), want, rtol=1e-10)
    energies = energy(stack, m.ham, meas_ctx, m.trial)
    np.testing.assert_allclose(float(given.e_estimate), float(np.mean(energies)), rtol=1e-10)


@pytest.mark.parametrize("block_fn", [blocks.block, mps_cpmc.block], ids=["trot-block", "mps-block"])
def test_run_blocks_compiles_once(lattice_model, block_fn):
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
        block_fn=block_fn,
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


@pytest.mark.parametrize("energy_kernel", ["blocked", "dense"])
@pytest.mark.parametrize("walker_qr", ["native", "cholesky"])
def test_overlaps_and_energies_match_enumeration(lattice_model, walker_qr, energy_kernel):
    """Exact walker conversion: the trot ops (one walker) and the engine's batch kernels (walker_qr) give the
    enumerated <T|W> and <T|H|W>/<T|W> of diffused, non-orthonormal walkers for a non-eigenstate trial."""
    m = lattice_model
    trial = _noisy_trial(m.trial, m.nelec, seed=5)
    params = _params(walker_qr=walker_qr, energy_kernel=energy_kernel)  # maximal plan, exact walkers
    ops = make_mps_cpmc_ops(m.ham, trial, m.sys, params)
    meas_ctx = ops.meas_ops.build_meas_ctx(m.ham, trial)
    rng = np.random.default_rng(3)
    walkers = [
        (hf.nonorthonormal(a, rng), hf.nonorthonormal(b, rng))
        for a, b in hf.random_field_walkers(
            m.h1, U, 0.1, *_determinant("uhf", m.h1, m.nelec), n=8, steps=40, seed=3
        )
    ]
    amplitudes = hf.mps_sector_amplitudes(trial.tensors, *m.nelec)
    H = hf.hubbard_sector_hamiltonian(m.h1, U, *m.nelec)
    want_overlaps = [hf.overlap_with_sd(amplitudes, a, b) for a, b in walkers]
    want_energies = [hf.local_energy(amplitudes, H, a, b) for a, b in walkers]
    stack = tuple(jnp.asarray(np.stack([w[s] for w in walkers])) for s in range(2))
    per_walker = (
        jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))(stack, trial),
        jax.vmap(ops.meas_ops.require_kernel(k_energy), in_axes=(0, None, None, None))(
            stack, m.ham, meas_ctx, trial
        ),
    )
    kernels, data = meas_ctx.kernels, meas_ctx.data()
    batch = (jax.jit(kernels.overlaps)(*stack, data), jax.jit(kernels.energies)(*stack, data))
    for overlaps, energies in (per_walker, batch):
        np.testing.assert_allclose(np.asarray(overlaps), want_overlaps, rtol=1e-9)
        np.testing.assert_allclose(np.asarray(energies), want_energies, rtol=1e-9)
    assert meas_ctx.kernel == kernels.energy_kind == energy_kernel


# ---------------------------------------------------------------------------------------------
# The step: against trot's CPMC (fast on the SD trial, slow on the MPS overlap) and in chunks
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


@pytest.mark.parametrize("floor, dt", [(1e-3, 0.05), (0.5, 0.1)])
def test_step_matches_trot_fast_cpmc_on_sd_trials(floor, dt):
    """Three MPS steps with an SD trial as an MPS equal trot's fast cpmc step with the GhfTrial: same walkers,
    weights, shift, nodes and RNG, overlaps up to the trial's constant factor. The walkers stay near the trial,
    so no site has both proposals floored (where trot's cpmc kills the walker and the MPS step keeps it);
    floor 0.5 separates the floor on the ratio (both) from cpmc_slow's floor on half the ratio."""
    L, nelec = 6, (3, 2)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(dt=dt, weight_floor=floor, n_walkers=8, n_prop_steps=1)
    Ca, Cb = _determinant("uhf", h1, nelec)
    rng = np.random.default_rng(6)
    near = lambda C: np.linalg.qr(C + 0.15 * rng.standard_normal(C.shape))[0]
    walkers = tuple(
        jnp.asarray(np.stack([near(C) for _ in range(params.n_walkers)])) for C in (Ca, Cb)
    )
    ends = []
    for side in (
        _sd_side(ham, sys_, params, scipy.linalg.block_diag(Ca, Cb)),
        _mps_side(ham, sys_, params, mps_trial_from_sd(Ca, Cb)),
    ):
        trial, trial_ops, meas_ops, prop_ops, prop_ctx = side
        start = PropState(
            walkers=walkers,
            weights=jnp.ones(params.n_walkers),
            overlaps=jax.vmap(trial_ops.overlap, in_axes=(0, None))(walkers, trial),
            rng_key=jax.random.PRNGKey(6),
            pop_control_ene_shift=jnp.asarray(-3.0),
            e_estimate=jnp.asarray(-3.0),
            node_encounters=jnp.zeros((), jnp.int64),
        )
        ends.append(
            _three_steps(
                prop_ops,
                start,
                params=params,
                ham=ham,
                trial=trial,
                trial_ops=trial_ops,
                meas_ops=meas_ops,
                meas_ctx=meas_ops.build_meas_ctx(ham, trial),
                prop_ctx=prop_ctx,
            )
        )
    sd_end, mps_end = ends
    ratio = np.asarray(mps_end.overlaps) / np.asarray(sd_end.overlaps)  # MPS trial = SD x constant
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)
    scaled = sd_end._replace(overlaps=ratio[0] * sd_end.overlaps)
    _compare_states(mps_end, scaled, walkers_atol=1e-12, rtol=1e-10)


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


@pytest.mark.parametrize("walker_qr", ["native", "cholesky"])
def test_step_chunking_is_exact(lattice_model, walker_qr):
    """n_chunks that does not divide the walkers (3 of 10) is rounded up to a divisor; same result as one chunk."""
    m = lattice_model
    results = []
    for n_chunks in (1, 3):
        params = _params(
            orbital_plan="adaptive", walker_channel_chi=2, n_chunks=n_chunks, walker_qr=walker_qr
        )
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


# ---------------------------------------------------------------------------------------------
# The MPS block, run_qmc and run_qmc_mps
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "walker_channel_chi, energy_kernel", [(None, "blocked"), (2, "blocked"), (None, "dense")]
)
def test_mps_block_matches_trots_block(walker_channel_chi, energy_kernel):
    """trot.prop.mps_cpmc.block (det-R rescaled and gathered overlaps, 2 n + 1 conversions) against
    trot.prop.blocks.block on the same ops, from diffused, non-orthonormal walkers with unequal weights."""
    L, nelec = 6, (3, 2)
    h1 = hopping_matrix(L, 1.0)
    ham, sys_ = HamHubbard(h1=jnp.asarray(h1), u=U), _system(L, nelec)
    params = _params(
        orbital_plan="adaptive", walker_channel_chi=walker_channel_chi, energy_kernel=energy_kernel
    )
    Ca, Cb = _determinant("uhf", h1, nelec)
    trial = mps_trial_from_sd(Ca, Cb)
    ops = make_mps_cpmc_ops(ham, trial, sys_, params)
    state = _diffused_state(trial, ops, h1, Ca, Cb, params.n_walkers, seed=4)
    rng = np.random.default_rng(4)
    walkers = tuple(
        jnp.asarray(np.stack([hf.nonorthonormal(np.asarray(w), rng) for w in spin]))
        for spin in state.walkers
    )
    state = state._replace(
        walkers=walkers,
        weights=jnp.asarray(rng.uniform(0.5, 1.5, params.n_walkers)),
        overlaps=jax.vmap(ops.trial_ops.overlap, in_axes=(0, None))(walkers, trial),
    )
    kwargs: dict[str, Any] = dict(
        sys=sys_,
        params=params,
        ham_data=ham,
        trial_data=trial,
        trial_ops=ops.trial_ops,
        meas_ops=ops.meas_ops,
        meas_ctx=ops.meas_ops.build_meas_ctx(ham, trial),
        prop_ops=ops.prop_ops,
        prop_ctx=ops.prop_ops.build_prop_ctx(ham, None, params),
    )
    got, got_obs = jax.jit(lambda s: mps_cpmc.block(s, **kwargs))(state)
    want, want_obs = jax.jit(lambda s: blocks.block(s, **kwargs))(state)

    for key in ("energy", "weight"):
        np.testing.assert_allclose(
            float(got_obs.scalars[key]), float(want_obs.scalars[key]), rtol=1e-9
        )
    for x, y in zip(got.walkers, want.walkers):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(got.weights), np.asarray(want.weights), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(got.overlaps), np.asarray(want.overlaps), rtol=1e-8)
    np.testing.assert_allclose(float(got.e_estimate), float(want.e_estimate), rtol=1e-9)
    np.testing.assert_allclose(
        float(got.pop_control_ene_shift), float(want.pop_control_ene_shift), rtol=1e-9
    )
    assert int(got.node_encounters) == int(want.node_encounters)
    assert np.array_equal(np.asarray(got.rng_key), np.asarray(want.rng_key))


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
    assert kwargs == dict(chi=16, n_sweeps=8, seed=0, init="auto")  # init: params.dmrg_init
    assert rng_restored, "make_dmrg_trial must restore numpy's global RNG"
    # the default Neel start leaves the 8-sweep trial ~3e-8 above E0 in the first blocks
    np.testing.assert_allclose(np.asarray(result.block_energies), E_L4, rtol=0, atol=1e-7)


def test_run_qmc_mps_with_a_ready_trial_equals_manual_run_qmc(monkeypatch):
    """A ready trial skips DMRG; with trot's block run_qmc_mps gives exactly the manually assembled run_qmc
    result, and with its default block (trot.prop.mps_cpmc.block) the same blocks to rounding."""

    def no_dmrg(*args, **kwargs):
        raise AssertionError("DMRG must not run when a trial is given")

    monkeypatch.setitem(
        sys.modules, "trot.gmps.dmrg", types.SimpleNamespace(make_dmrg_trial=no_dmrg)
    )
    h1, nelec, _, trial, ham, sys_ = _small_model()
    noisy = _noisy_trial(trial, nelec, seed=4)  # a non-exact trial, so the energies fluctuate
    params = _params(n_walkers=6, n_prop_steps=3, n_eql_blocks=2, n_blocks=4, seed=3)
    via_driver = run_qmc_mps(
        sys=sys_, params=params, ham_data=ham, trial_data=noisy, block_fn=blocks.block
    )
    manual = _run(sys_, params, ham, _mps_side(ham, sys_, params, noisy))
    assert np.array_equal(np.asarray(via_driver.block_energies), np.asarray(manual.block_energies))
    assert np.array_equal(np.asarray(via_driver.block_weights), np.asarray(manual.block_weights))
    assert np.ptp(np.asarray(manual.block_energies)) > 1e-4
    default = run_qmc_mps(sys=sys_, params=params, ham_data=ham, trial_data=noisy)
    for key in ("block_energies", "block_weights"):
        np.testing.assert_allclose(
            np.asarray(getattr(default, key)), np.asarray(getattr(manual, key)), rtol=1e-9
        )
    assert float(default.mean_energy) == pytest.approx(float(manual.mean_energy), rel=1e-9)


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
