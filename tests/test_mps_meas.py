"""Tests for trot/meas/mps.py: Hubbard MPOs, H|trial> and the MPS-CPMC local-energy kernels.

Every reference comes from tests/helpers/hubbard_fock.py (exact Fock-space enumeration), from
trot's GHF Hubbard kernel or from pyblock3, never from the MPS code under test. The MPO identity
uses random MPS on five h1 (U=3.7); the trials live on an L=6 open chain and on a 2x3 lattice
periodic in x (doubled rungs) with two on-site terms, U=4, (N_up, N_dn) = (3, 2). The spin-rotated
determinant has no definite (N_up, N_dn), so make_mps_trial uses it as it is (particle-number
labels): the walkers pick their sector, and the local energies are those of the projected trial.
"""

from trot import config

config.configure_once()

from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from tests.helpers import hubbard_fock as hf
from trot.core.system import System
from trot.gmps import engine
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix, square_hopping_matrix
from trot.meas.ghf import energy_kernel_hubbard_u
from trot.meas.mps import (
    MpsMeasCtx,
    apply_mpo,
    hubbard_h1,
    hubbard_mpo,
    hubbard_mpo_from_h1,
    make_mps_meas_ops_hubbard,
    trial_times_h,
)
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.trial.ghf import GhfTrial
from trot.trial.mps import (
    compress_mps_qn,
    make_mps_trial,
    make_walker_plan_from_reference,
    mps_trial_from_sd,
    natural_orbitals,
    physical_charge,
    rotate_spin,
)

U = 4.0
NELEC = (3, 2)
E_CHAIN = -3.984358962762  # exact, L=6 open chain, U=4, (3, 2)
KERNELS = ("blocked", "dense")
ENERGY_CASES = ("ed_2x3", "sd_chain", "rotated_sd_chain")
THETA = 0.4
R = np.array([[np.cos(THETA), -np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]])


# ---------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------


def _lattice_h1():
    h1 = square_hopping_matrix(2, 3, 1.0, "periodic", "open")
    h1[0, 0] = 0.3
    h1[4, 4] = -0.2
    return h1


def _ring_h1():
    h1 = hopping_matrix(5, 1.0)
    h1[0, 4] = h1[4, 0] = -1.0
    return h1


def _dense_h1():
    a = np.random.default_rng(3).standard_normal((4, 4))
    return a + a.T  # exactly symmetric, nonzero diagonal


MPO_H1 = {
    "chain5": hopping_matrix(5, 1.0),
    "ring5": _ring_h1(),
    "2x3_open": square_hopping_matrix(2, 3, 1.0),
    "2x3_px_diag": _lattice_h1(),
    "dense4": _dense_h1(),
}


def _random_mps(L, rng):
    """Random real d=4 MPS with small bonds, no charge structure (all particle sectors)."""
    bonds = [1, *rng.integers(2, 4, L - 1), 1]
    return [rng.standard_normal((bonds[i], 4, bonds[i + 1])) for i in range(L)]


def _rotated_sd_trial(Ca, Cb):
    """The spin-rotated determinant: no definite (N_up, N_dn), so make_mps_trial uses it as it is."""
    tensors = [np.asarray(t) for t in sd_to_gmps(Ca, Cb, mode="maximal").tensors]
    trial = make_mps_trial(rotate_spin(tensors, R), nelec=NELEC)
    assert trial.label_width == 1 and trial.sector_weight is None
    return trial


def _natural_reference(trial):
    rdm1 = np.asarray(trial.rdm1)
    return tuple(natural_orbitals(rdm1[s], n)[0] for s, n in enumerate(NELEC))


def _case(h1, trial, reference, *, seed):
    """Trial, walkers, exact and truncated walker plans, and the four (plan, kernel) contexts."""
    nup, ndn = NELEC
    ham = HamHubbard(h1=jnp.asarray(h1), u=U)
    rng = np.random.default_rng(seed)
    start = hf.staggered_determinant(h1, nup, ndn, field=0.3)
    walkers = []
    for i, (wa, wb) in enumerate(
        hf.random_field_walkers(h1, U, 0.1, *start, n=6, steps=10, seed=seed)
    ):
        if i % 2:  # the same determinants up to a scalar, no longer orthonormal
            wa, wb = hf.nonorthonormal(wa, rng), hf.nonorthonormal(wb, rng)
        walkers.append((jnp.asarray(wa), jnp.asarray(wb)))
    plans = {
        "exact": make_walker_plan_from_reference(
            *reference, orbital_plan="maximal", walker_channel_chi=None
        ),
        "truncated": make_walker_plan_from_reference(
            *reference, orbital_plan="adaptive", walker_channel_chi=2
        ),
    }
    ops = {
        (p, k): make_mps_meas_ops_hubbard(plan, energy_kernel=k)
        for p, plan in plans.items()
        for k in KERNELS
    }
    return SimpleNamespace(
        h1=h1,
        ham=ham,
        trial=trial,
        walkers=walkers,
        plans=plans,
        ops=ops,
        ctx={key: op.build_meas_ctx(ham, trial) for key, op in ops.items()},
        amplitudes=hf.mps_sector_amplitudes(trial.tensors, nup, ndn),
        H=hf.hubbard_sector_hamiltonian(h1, U, nup, ndn),
        C=None,
        E0=None,
        trial_energy=None,
    )


def _mps_energies(case, plan_kind, kernel):
    energy = case.ops[plan_kind, kernel].kernels["energy"]
    ctx = case.ctx[plan_kind, kernel]
    return np.array([float(energy(w, case.ham, ctx, case.trial)) for w in case.walkers])


def _assert_labels_respected(tensors, charges):
    """Bond labels from zero to NELEC, (N_up, N_dn) pairs or particle numbers N, that every nonzero entry respects."""
    width = np.shape(charges[0])[1]
    assert len(charges) == len(tensors) + 1
    np.testing.assert_array_equal(charges[0], [[0] * width])
    np.testing.assert_array_equal(charges[-1], [list(NELEC) if width == 2 else [sum(NELEC)]])
    for s, A in enumerate(tensors):
        A = np.asarray(A)
        assert A.shape[0] == len(charges[s]) and A.shape[2] == len(charges[s + 1])
        for p, delta in enumerate(physical_charge(width)):
            left, right = np.nonzero(A[:, p, :])
            reached = np.asarray(charges[s])[left][:, 0] >= 0  # -1: an index nothing reaches (its block is zero)
            np.testing.assert_array_equal(np.asarray(charges[s + 1])[right[reached]],
                                          np.asarray(charges[s])[left[reached]] + delta)


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ed_2x3():
    """2x3 lattice: the exact ground state as a charge-labelled MPS trial."""
    h1 = _lattice_h1()
    E0, psi = hf.ground_state(h1, U, *NELEC)
    trial = make_mps_trial(*hf.exact_mps_from_sector_state(psi, 6, *NELEC), nelec=NELEC)
    case = _case(h1, trial, _natural_reference(trial), seed=1)
    case.E0 = case.trial_energy = E0
    return case


@pytest.fixture(scope="module")
def sd_chain():
    """L=6 chain: an unrestricted determinant trial, walker plan frozen on its orbitals."""
    h1 = hopping_matrix(6, 1.0)
    assert abs(hf.ground_state(h1, U, *NELEC)[0] - E_CHAIN) < 1e-10  # the oracle's H is right
    Ca, Cb = hf.staggered_determinant(h1, *NELEC)
    case = _case(h1, mps_trial_from_sd(Ca, Cb), (Ca, Cb), seed=2)
    case.C = scipy.linalg.block_diag(Ca, Cb)
    case.trial_energy = hf.local_energy(hf.sd_amplitudes(Ca, Cb), case.H, Ca, Cb)
    return case


@pytest.fixture(scope="module")
def rotated_sd_chain():
    """L=6 chain: the spin-rotated determinant, used as it is (particle-number labels, every sector)."""
    h1 = hopping_matrix(6, 1.0)
    Ca, Cb = hf.staggered_determinant(h1, *NELEC)
    trial = _rotated_sd_trial(Ca, Cb)
    case = _case(h1, trial, _natural_reference(trial), seed=3)
    case.C = np.kron(R, np.eye(6)) @ cast(np.ndarray, scipy.linalg.block_diag(Ca, Cb))
    v = hf.ghf_full_vector(case.C)  # the trial used as it is spans every sector with N = 5
    case.trial_energy = float(v @ hf.apply_hubbard_full(h1, U, v)) / float(v @ v)
    return case


@pytest.fixture(scope="module")
def rotated_sd_2x3():
    """2x3 lattice: the spin-rotated determinant trial (only h1 and the trial)."""
    h1 = _lattice_h1()
    return SimpleNamespace(h1=h1, trial=_rotated_sd_trial(*hf.staggered_determinant(h1, *NELEC)))


@pytest.fixture(scope="module")
def dmrg_2x3():
    """2x3 lattice: a truncated pyblock3 DMRG trial (chi=6)."""
    pytest.importorskip("pyblock3")
    from trot.gmps.dmrg import make_dmrg_trial

    ham = HamHubbard(h1=jnp.asarray(_lattice_h1()), u=U)
    sys_ = System(norb=6, nelec=NELEC, walker_kind="unrestricted")
    return ham, make_dmrg_trial(ham, sys_, chi=6, n_sweeps=8, seed=0)


@pytest.fixture
def case(request):
    """The module-scoped case fixture named by the (indirect) parameter."""
    return request.getfixturevalue(request.param)


# ---------------------------------------------------------------------------------------------
# 1-2: MPOs
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MPO_H1))
def test_mpo_from_h1_acts_as_hubbard_hamiltonian(name):
    """apply_mpo(hubbard_mpo_from_h1(h1, U)) is H on random MPS vectors in every sector."""
    h1, u = MPO_H1[name], 3.7
    W = hubbard_mpo_from_h1(h1, u)
    rng = np.random.default_rng(11)
    for _ in range(3):
        tensors = _random_mps(len(h1), rng)
        want = hf.apply_hubbard_full(h1, u, hf.mps_full_vector(tensors))
        got = hf.mps_full_vector(apply_mpo(W, tensors))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("t", [1.0, 0.7])
@pytest.mark.parametrize("L", [4, 6])
def test_mpo_from_h1_chain_limit_is_hubbard_mpo(L, t):
    """For an open chain hubbard_mpo_from_h1 reproduces hubbard_mpo exactly."""
    W = hubbard_mpo_from_h1(hopping_matrix(L, t), U)
    reference = hubbard_mpo(L, t, U)
    assert W.shape == reference.shape and np.array_equal(W, reference)


# ---------------------------------------------------------------------------------------------
# 3: H|trial> with bond labels
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["ed_2x3", "rotated_sd_2x3"], indirect=True)
def test_trial_times_h_is_labelled_h_trial(case):
    """trial_times_h is exactly H|T> with valid labels ((N_up, N_dn) for the ED trial, N for the rotated one);
    compress_mps_qn keeps vector and labels."""
    W = hubbard_mpo_from_h1(case.h1, U)
    tensors, charges = trial_times_h(W, case.trial.tensors, case.trial.charge_arrays())
    assert np.shape(charges[0])[1] == case.trial.label_width
    _assert_labels_respected(tensors, charges)  # (a)

    got = hf.mps_full_vector(tensors)
    want = hf.apply_hubbard_full(case.h1, U, hf.mps_full_vector(case.trial.tensors))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-11 * np.abs(want).max())  # (b)

    small, small_charges = compress_mps_qn(tensors, charges)  # (c)
    np.testing.assert_allclose(
        hf.mps_full_vector(small), got, rtol=0, atol=1e-11 * np.abs(got).max()
    )
    assert all(len(qs) <= len(q) for qs, q in zip(small_charges, charges))
    _assert_labels_respected(small, small_charges)


# ---------------------------------------------------------------------------------------------
# 4-6: local energies
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_energy_matches_enumeration(case, kernel):
    """With exact walkers the kernel is <T|H|W>/<T|W> of the enumerated sector Hamiltonian."""
    want = [
        hf.local_energy(case.amplitudes, case.H, np.asarray(wa), np.asarray(wb))
        for wa, wb in case.walkers
    ]
    np.testing.assert_allclose(_mps_energies(case, "exact", kernel), want, rtol=0, atol=1e-10)


@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_truncated_walkers_blocked_equals_dense(case):
    """With chi=2 adaptive walker channels the blocked and dense kernels give the same energy."""
    blocked = _mps_energies(case, "truncated", "blocked")
    np.testing.assert_allclose(
        blocked, _mps_energies(case, "truncated", "dense"), rtol=0, atol=1e-11
    )
    if case.E0 is None:  # sanity: the truncation is real (the eigenstate trial cannot show it)
        assert np.abs(blocked - _mps_energies(case, "exact", "blocked")).max() > 1e-6


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("plan_kind", ["exact", "truncated"])
def test_zero_variance_for_exact_eigenstate_trial(ed_2x3, plan_kind, kernel):
    """For the exact ground-state trial E_loc = E0 for every walker, truncated or not."""
    np.testing.assert_allclose(
        _mps_energies(ed_2x3, plan_kind, kernel), ed_2x3.E0, rtol=0, atol=1e-9
    )
    if plan_kind == "truncated":  # sanity: the chi=2 walkers really differ from the exact ones
        exact = ed_2x3.ops["exact", kernel].overlap
        truncated = ed_2x3.ops["truncated", kernel].overlap
        ratios = [
            float(truncated(w, ed_2x3.trial) / exact(w, ed_2x3.trial)) for w in ed_2x3.walkers
        ]
        assert np.abs(np.array(ratios) - 1.0).max() > 1e-6


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", ["sd_chain", "rotated_sd_chain"], indirect=True)
def test_energy_matches_ghf_hubbard_kernel(case, kernel):
    """For (spin-rotated) determinant trials the kernel equals trot's GHF Hubbard energy."""
    ghf = GhfTrial(mo_coeff=jnp.asarray(case.C))
    want = [float(energy_kernel_hubbard_u(w, case.ham, None, ghf)) for w in case.walkers]
    np.testing.assert_allclose(_mps_energies(case, "exact", kernel), want, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------------------------
# 7: the measurement context
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
def test_meas_ctx_is_pytree_holding_one_kernel(ed_2x3, kernel):
    """MpsMeasCtx flattens to its padded blocks and back; only the chosen kernel's H|T> is stored. Its kernels
    are the plan's cached engine and data(prop_ctx) its DeviceData."""
    ctx, L, plan = ed_2x3.ctx["exact", kernel], ed_2x3.trial.norb, ed_2x3.plans["exact"]
    assert isinstance(ctx, MpsMeasCtx) and ctx.kernel == kernel
    assert ctx.plan is plan and ctx.trial_charges == ed_2x3.trial.charges
    assert len(ctx.trial_blocks) == len(ctx.h_blocks) == L
    if kernel == "blocked":
        assert ctx.h_charges is not None and ctx.dense_trial == ()
    else:  # the dense H|T> tensors and the trial tensors
        assert ctx.h_charges is None and len(ctx.dense_trial) == L
    leaves, treedef = jax.tree_util.tree_flatten(ctx)
    n_leaves = (2 if kernel == "blocked" else 3) * L
    assert len(leaves) == n_leaves and all(isinstance(x, jax.Array) for x in leaves)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, MpsMeasCtx) and rebuilt.key == ctx.key
    assert all(a is b for a, b in zip(jax.tree_util.tree_leaves(rebuilt), leaves))
    passed = jax.jit(lambda c: c)(ctx)
    assert isinstance(passed, MpsMeasCtx) and passed.key == ctx.key
    for a, b in zip(jax.tree_util.tree_leaves(passed), leaves):
        np.testing.assert_array_equal(a, b)

    kernels = ctx.kernels
    assert kernels is engine.kernels_for(plan, ctx.trial_charges, ctx.h_charges, kernel)
    assert kernels.energy_kind == kernel
    assert kernels.overlap_plan is engine.layout_for(plan, ctx.trial_charges)
    assert (kernels.energy_plan is None) == (kernel == "dense")
    prop_ctx = _build_prop_ctx(ed_2x3.ham, 0.05)
    for data, propagation in ((ctx.data(), None), (ctx.data(prop_ctx), prop_ctx)):
        assert isinstance(data, engine.DeviceData)
        assert data.trial is ctx.trial_blocks and data.htrial is ctx.h_blocks
        assert data.dense_trial is ctx.dense_trial
        if propagation is None:
            assert data.exp_h1_half is None and data.hs is None
        else:
            assert data.exp_h1_half is prop_ctx.exp_h1_half and data.hs is prop_ctx.hs_constant


@pytest.mark.parametrize("kernel", KERNELS)
def test_energy_kernel_rejects_mismatched_ctx(ed_2x3, kernel):
    """The energy kernel raises ValueError for no ctx, another plan's ctx or another trial's ctx."""
    ops = ed_2x3.ops["exact", kernel]
    other = mps_trial_from_sd(*hf.staggered_determinant(ed_2x3.h1, *NELEC))
    assert other.charges != ed_2x3.trial.charges
    bad = [
        (None, "build_meas_ctx"),
        (ed_2x3.ctx["truncated", kernel], "different walker plan"),
        (ops.build_meas_ctx(ed_2x3.ham, other), "different trial"),
    ]
    for ctx, message in bad:
        with pytest.raises(ValueError, match=message):
            ops.kernels["energy"](ed_2x3.walkers[0], ed_2x3.ham, ctx, ed_2x3.trial)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_ctx_trial_energy_is_trial_expectation(case, kernel):
    """ctx.trial_energy is <T|H|T>/<T|T>: E0 for the ED trial, the determinant energy for SDs (over every
    sector for the rotated determinant used as it is)."""
    assert abs(case.ctx["exact", kernel].trial_energy - case.trial_energy) < 1e-10


# ---------------------------------------------------------------------------------------------
# 8: hubbard_h1
# ---------------------------------------------------------------------------------------------


def _complex_h1():
    h1 = _lattice_h1().astype(complex)
    h1[0, 1] += 0.1j
    h1[1, 0] -= 0.1j  # Hermitian, but complex
    return h1


def _asymmetric_h1(delta):
    h1 = _lattice_h1()
    h1[0, 1] += delta
    return h1


@pytest.mark.parametrize(
    "h1",
    [
        pytest.param(_complex_h1(), id="complex"),
        pytest.param(_lattice_h1()[:, :5], id="non_square"),
        pytest.param(np.ones(6), id="one_dimensional"),
        pytest.param(_asymmetric_h1(3e-12), id="asymmetric_3e-12"),
        pytest.param(_asymmetric_h1(1e-9), id="asymmetric_1e-9"),
    ],
)
def test_hubbard_h1_rejects_invalid(h1):
    """hubbard_h1 raises ValueError for complex, non-square or (beyond 1e-12) asymmetric h1."""
    with pytest.raises(ValueError):
        hubbard_h1(HamHubbard(h1=h1, u=U))


def test_hubbard_h1_symmetrises_tiny_asymmetry_exactly():
    """A 1e-14 asymmetry is accepted and removed exactly, as 0.5 (h1 + h1^T)."""
    h1 = _asymmetric_h1(1e-14)
    assert not np.array_equal(h1, h1.T)
    out = hubbard_h1(HamHubbard(h1=h1, u=U))  # pyright: ignore[reportArgumentType]
    assert out.dtype == np.float64 and out.shape == h1.shape
    assert np.array_equal(out, out.T)
    np.testing.assert_array_equal(out, 0.5 * (h1 + h1.T))
    np.testing.assert_allclose(out, h1, rtol=0, atol=1e-14)
    hubbard_mpo_from_h1(out, U)  # the exact-symmetry check of the MPO builder passes now
    with pytest.raises(ValueError):
        hubbard_mpo_from_h1(h1, U)


# ---------------------------------------------------------------------------------------------
# 9: pyblock3 DMRG trial
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
def test_dmrg_variational_energy_is_ctx_trial_energy(dmrg_2x3, kernel):
    """ctx.trial_energy equals pyblock3's variational (not Davidson) energy of a DMRG trial."""
    ham, dmrg = dmrg_2x3
    assert max(dmrg.trial.bond_dims) <= 6  # a truncated trial
    plan = make_walker_plan_from_reference(
        *_natural_reference(dmrg.trial), orbital_plan="maximal", walker_channel_chi=None
    )
    ctx = make_mps_meas_ops_hubbard(plan, energy_kernel=kernel).build_meas_ctx(ham, dmrg.trial)
    assert abs(ctx.trial_energy - dmrg.variational_energy) < 1e-11
    assert abs(dmrg.davidson_energy - dmrg.variational_energy) > 1e-6
