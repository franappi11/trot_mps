"""Tests for trot/meas/mps.py: Hubbard MPOs, H|trial> and the MPS-CPMC local-energy kernels.

Every reference comes from tests/helpers/hubbard_fock.py (exact Fock-space enumeration), from
trot's GHF Hubbard kernel or from pyblock3, never from the MPS code under test. The MPO identity
uses random MPS on five h1 (U=3.7); the trials live on an L=6 open chain and on a 2x3 lattice
periodic in x (doubled rungs) with two on-site terms, U=4, (N_up, N_dn) = (3, 2).
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
from trot.gmps.utils import sd_to_gmps
from trot.ham.hubbard import HamHubbard, hopping_matrix, square_hopping_matrix
from trot.meas import mps as meas_mps
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
from trot.trial.ghf import GhfTrial
from trot.trial.mps import (
    PHYSICAL_CHARGE,
    compress_mps_qn,
    make_mps_trial,
    make_walker_plan_from_reference,
    mps_trial_from_sd,
    natural_orbitals,
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
    tensors = [np.asarray(t) for t in sd_to_gmps(Ca, Cb, mode="maximal").tensors]
    return make_mps_trial(rotate_spin(tensors, R), nelec=NELEC)


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
    """(N_up, N_dn) bond labels from (0, 0) to NELEC that every nonzero entry respects."""
    assert len(charges) == len(tensors) + 1
    np.testing.assert_array_equal(charges[0], [[0, 0]])
    np.testing.assert_array_equal(charges[-1], [list(NELEC)])
    for s, A in enumerate(tensors):
        A = np.asarray(A)
        assert A.shape[0] == len(charges[s]) and A.shape[2] == len(charges[s + 1])
        for p, delta in enumerate(PHYSICAL_CHARGE):
            left, right = np.nonzero(A[:, p, :])
            np.testing.assert_array_equal(charges[s + 1][right], charges[s][left] + delta)


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
    """L=6 chain: the spin-rotated determinant, projected onto (3, 2) by make_mps_trial."""
    h1 = hopping_matrix(6, 1.0)
    Ca, Cb = hf.staggered_determinant(h1, *NELEC)
    trial = _rotated_sd_trial(Ca, Cb)
    case = _case(h1, trial, _natural_reference(trial), seed=3)
    case.C = np.kron(R, np.eye(6)) @ cast(np.ndarray, scipy.linalg.block_diag(Ca, Cb))
    a = hf.ghf_amplitudes(case.C, *NELEC).ravel()
    case.trial_energy = float(a @ case.H @ a) / float(a @ a)
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
# 1-3: MPOs
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


def test_scripts_reexport_meas_objects():
    """mps_cpmc_2d and mps_cpmc_new use the very objects of trot.meas.mps / trot.ham.hubbard."""
    pytest.importorskip("pyblock3")
    from trot.gmps import mps_cpmc_2d, mps_cpmc_new

    for name in ("hubbard_mpo_from_h1", "apply_mpo", "trial_times_h", "CHANNEL_CHARGE"):
        assert getattr(mps_cpmc_2d, name) is getattr(meas_mps, name), name
    assert mps_cpmc_2d.square_hopping_matrix is square_hopping_matrix
    assert mps_cpmc_new.hubbard_mpo is meas_mps.hubbard_mpo


def _apply_mpo_old(W, tensors):
    """apply_mpo of trot/gmps/mps_cpmc_new.py before the refactor (last site: channel 5:6)."""
    out = []
    for i, (operator, A) in enumerate(zip(W, tensors)):
        if i == 0:
            operator = operator[:1]
        if i == len(tensors) - 1:
            operator = operator[:, :, :, 5:6]
        T = np.einsum("apqb,cqd->acpbd", operator, np.asarray(A))
        dl, cl, d, dr, cr = T.shape
        out.append(T.reshape(dl * cl, d, dr * cr))
    return out


def test_apply_mpo_bitwise_equals_old_code_on_chain_mpo():
    """On the six-channel chain MPO apply_mpo is bitwise the pre-refactor 5:6 version."""
    W = hubbard_mpo(6, 1.0, 4.0)
    tensors = _random_mps(6, np.random.default_rng(7))
    new, old = apply_mpo(W, tensors), _apply_mpo_old(W, tensors)
    assert len(new) == len(old) == 6
    for a, b in zip(new, old):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------------------------
# 4: H|trial> with bond labels
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["ed_2x3", "rotated_sd_2x3"], indirect=True)
def test_trial_times_h_is_labelled_h_trial(case):
    """trial_times_h is exactly H|T> with valid labels; compress_mps_qn keeps vector and labels."""
    W = hubbard_mpo_from_h1(case.h1, U)
    tensors, charges = trial_times_h(W, case.trial.tensors, case.trial.charge_arrays())
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
# 5-7: local energies
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
# 8: the measurement context
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
def test_meas_ctx_is_pytree_holding_one_kernel(ed_2x3, kernel):
    """MpsMeasCtx flattens to its blocks and back; only the chosen kernel's H|T> is stored."""
    ctx, L = ed_2x3.ctx["exact", kernel], ed_2x3.trial.norb
    assert isinstance(ctx, MpsMeasCtx) and ctx.kernel == kernel
    assert ctx.plan is ed_2x3.plans["exact"]
    if kernel == "blocked":
        assert ctx.h_tensors is None and ctx.h_blocks is not None and len(ctx.h_blocks) == L
    else:
        assert ctx.h_blocks is None and ctx.h_tensors is not None and len(ctx.h_tensors) == L
    leaves, treedef = jax.tree_util.tree_flatten(ctx)
    assert len(leaves) == 2 * L and all(isinstance(x, jax.Array) for x in leaves)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, MpsMeasCtx) and rebuilt.key == ctx.key
    assert all(a is b for a, b in zip(jax.tree_util.tree_leaves(rebuilt), leaves))
    passed = jax.jit(lambda c: c)(ctx)
    assert isinstance(passed, MpsMeasCtx) and passed.key == ctx.key
    for a, b in zip(jax.tree_util.tree_leaves(passed), leaves):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kernel", KERNELS)
def test_energy_kernel_rejects_mismatched_ctx(ed_2x3, kernel):
    """The energy kernel raises ValueError for no ctx, another plan's ctx or the other kernel's."""
    other = "dense" if kernel == "blocked" else "blocked"
    energy = ed_2x3.ops["exact", kernel].kernels["energy"]
    walker = ed_2x3.walkers[0]
    bad = [
        (None, "build_meas_ctx"),
        (ed_2x3.ctx["truncated", kernel], "different walker plan"),
        (ed_2x3.ctx["exact", other], f"holds the '{other}' kernel"),
    ]
    for ctx, message in bad:
        with pytest.raises(ValueError, match=message):
            energy(walker, ed_2x3.ham, ctx, ed_2x3.trial)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", ENERGY_CASES, indirect=True)
def test_ctx_trial_energy_is_trial_expectation(case, kernel):
    """ctx.trial_energy is <T|H|T>/<T|T>: E0 for the ED trial, the determinant energy for SDs."""
    assert abs(case.ctx["exact", kernel].trial_energy - case.trial_energy) < 1e-10


# ---------------------------------------------------------------------------------------------
# 9: hubbard_h1
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
# 10: pyblock3 DMRG trial
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
