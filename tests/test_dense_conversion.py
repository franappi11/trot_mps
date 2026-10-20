"""The dense-bond conversion, trot/gmps/dense_conversion.py, against the host oracle of the own-counts truncation.

plan_bonds(Q, plan, chi) followed by engine.channel_mps_host is the oracle: each walker keeps its own chi largest
Schmidt states (global across charge sectors) at every gate, which is what the dense bonds do. The dense MPS has other
bond bases and labels than the oracle's, so the comparison is gauge invariant: g <host|dense> against g <host|host>
(engine.conversion_self_check's) and the discarded weight 1 - <MPS|MPS>. The charge labels the conversion carries must
be usable: no weighted state mixes sectors, and the final tensors conserve the electron number under the labels.

    python -m pytest tests/test_dense_conversion.py
"""

from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.gmps import dense_conversion as dc
from trot.gmps import dense_engine
from trot.gmps import engine
from trot.gmps import utils

L, N, CHI = 8, 3, 4  # chi 4 truncates visibly at 8 sites
TOLERANCE = {"qr": 1.0e-9, "gram": 1.0e-7, "cans": 1.0e-7}


def _orthonormal(seed):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(L, N)))[0]


@pytest.fixture(scope="module")
def plan():
    return utils.make_orbital_plan(_orthonormal(100), "rank_exact")  # exact for every walker: no plan loss


def test_polar_coefficients_send_the_interval_to_one():
    a, c = dc.polar_coefficients(1.0e-9)
    assert len(a) <= 30  # 26: a cubic multiplies small values by at most ~2.6 per step once the top end overshoots
    x = np.logspace(-9, 0, 300)
    for ak, ck in zip(a, c):
        x = ak * x - ck * x ** 3
    np.testing.assert_allclose(x, 1.0, atol=1.0e-12)


@pytest.mark.parametrize("method", ["cans", "gram"])
def test_polar_factors_of_full_rank_and_rank_deficient_matrices(method):
    a, c = dc.polar_coefficients(1.0e-9)
    polar = (lambda M: dc.polar_cans(jnp.asarray(M), a, c)) if method == "cans" else \
        (lambda M: dc.polar_gram(jnp.asarray(M), 1.0e-10))
    rng = np.random.default_rng(0)
    M = rng.normal(size=(9, 5))
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    np.testing.assert_allclose(np.asarray(polar(M)), U @ Vt, atol=1.0e-9)
    M = rng.normal(size=(9, 3)) @ rng.normal(size=(3, 5))  # rank 3, two zero singular values
    W = np.asarray(polar(M))
    np.testing.assert_allclose(W @ (W.T @ M), M, atol=1.0e-9)  # W spans M's range
    P = W.T @ W
    np.testing.assert_allclose(P @ P, P, atol=1.0e-9)  # a partial isometry ...
    assert abs(np.trace(P) - 3.0) < 1.0e-9  # ... of rank 3: the zero modes stay zero


def test_dense_ops_follow_the_engines_circuit(plan):
    circuit = engine.compile_circuit([plan], [None])
    assert [(op.kind, op.site, op.step) for op in circuit.ops] == [(k, s, st) for k, s, st, _ in dc.dense_ops(plan)]


@pytest.mark.parametrize("move, gap", [("qr", 1.0e-7), ("gram", 1.0e-7), ("cans", 1.0e-7), ("gram", 2.0), ("cans", 2.0)])
def test_dense_conversion_keeps_each_walkers_own_states(plan, move, gap):
    """gap 2: the disjoint-band shift of the gate's eigh (see dense_conversion's docstring) changes neither the
    states nor the labels on a small system."""
    converter = dc.make_dense_converter(plan, CHI, move, charge_gap=gap)
    run = jax.jit(lambda C: converter.convert(C, stats=True))
    tolerance, truncated = TOLERANCE[move], 0
    for seed in range(4):
        Q = _orthonormal(seed)
        tensors, charges, gauge, stats = run(jnp.asarray(Q))
        assert all(max(A.shape[0], A.shape[2]) <= CHI for A in tensors)
        bond = utils.plan_bonds(Q, plan, CHI)
        host, _, g_host = engine.channel_mps_host(Q, plan, bond)
        reference = g_host * engine.mps_overlap_host(host, host)
        dense = [np.asarray(A) for A in tensors]
        assert abs(float(gauge) * engine.mps_overlap_host(host, dense) - reference) <= tolerance * abs(reference)
        discarded = 1.0 - float(dc.mps_norm2(tensors))
        assert abs(discarded - bond.reference_discarded_weight) <= tolerance
        truncated += bond.reference_discarded_weight > 1.0e-6
        if move != "qr":  # dense Householder QR gives arbitrary completions to dependent columns: its labels are not usable
            assert float(jnp.max(stats["impurity"])) < 1.0e-9 and float(jnp.max(stats["mislabeled_weight"])) < 1.0e-15
            assert float(stats["leakage"]) < 1.0e-10
    assert truncated, "chi 4 should discard weight of these walkers"


def test_dense_conversion_is_exact_when_the_bond_is_large_enough(plan):
    converter = dc.make_dense_converter(plan, 64, "gram")
    Q = _orthonormal(7)
    tensors, _, gauge = jax.jit(converter.convert)(jnp.asarray(Q))
    host, _, g_host = engine.channel_mps_host(Q, plan, None)
    assert abs(1.0 - float(dc.mps_norm2(tensors))) < 1.0e-9
    reference = g_host * engine.mps_overlap_host(host, host)
    dense = [np.asarray(A) for A in tensors]
    assert abs(float(gauge) * engine.mps_overlap_host(host, dense) - reference) <= 1.0e-8 * abs(reference)


# ---------------------------------------------------------------------------------------------
# Beyond 32 x 32: the bond-32 label failure and snapping
# ---------------------------------------------------------------------------------------------
# Bond 32 on 16 sites factors 64 x 64 Gram matrices. On GPUs the eigh there is not the batched Jacobi solver (which never
# rotates a pair coupled by an exact zero, so it keeps the sector blocks' zeros); on CPUs it is LAPACK's, which keeps
# them at no size. The H100 benchmarks lost the labels at bond 32 and kept them exactly at bond 16.

L32, N32, CHI32 = 16, 8, 32
TOLERANCE32 = {"gram": 1.0e-5, "cans": 1.0e-6}  # gram moves lose ~1e-8 each (squared singular values)


def _orthonormal32(seed):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(L32, N32)))[0]


@pytest.fixture(scope="module")
def plan32():
    return utils.make_orbital_plan(_orthonormal32(300), "rank_exact")


def _gate_eigh_size(converter):
    return max(min(shape) for (kind, *_), shape in zip(converter.ops, converter.shapes) if kind == "gate")


def _overlap_with(blocks, alpha, beta, beta_charges):
    """<trial|walker channels> through dense_engine (the fermionic sign from the beta bond labels)."""
    dressed = [dense_engine._dressed(jnp.asarray(B), dense_engine._parity_signs(jnp.asarray(q, dtype=float)))
               for B, q in zip(beta, beta_charges[:-1])]
    return float(dense_engine.left_contract([jnp.asarray(A) for A in alpha], dressed, blocks))


def test_without_snapping_a_non_jacobi_eigh_seeds_off_sector_entries(plan32):
    """The premise of snapping: a 64 x 64 eigh that is not Jacobi (LAPACK here) returns eigenvectors with off-sector
    entries (offlabel > 0), so the sector blocks' exact zeros are lost from the first such gate on."""
    converter = dc.make_dense_converter(plan32, CHI32, "cans")
    assert _gate_eigh_size(converter) > 32
    _, _, _, stats = jax.jit(lambda C: converter.convert(C, stats=True))(jnp.asarray(_orthonormal32(0)))
    assert float(jnp.max(stats["offlabel"])) > 0.0


@pytest.mark.parametrize("move", ["cans", "gram"])
def test_snapping_keeps_the_sector_blocks_exact_beyond_32x32(plan32, move):
    """With snap every factorisation's input is exactly sector-structured (input_leak exactly 0), the final tensors
    conserve the carried charges exactly, and the state is still the own-counts oracle's."""
    converter = dc.make_dense_converter(plan32, CHI32, move, snap=True)
    assert _gate_eigh_size(converter) > 32
    run = jax.jit(lambda C: converter.convert(C, stats=True))
    for seed in range(2):
        Q = _orthonormal32(seed)
        tensors, _, gauge, stats = run(jnp.asarray(Q))
        assert float(jnp.max(stats["input_leak"])) == 0.0 and float(stats["leakage"]) == 0.0
        bond = utils.plan_bonds(Q, plan32, CHI32)
        assert bond.reference_discarded_weight > 1.0e-6  # bond 32 truncates these walkers
        host, _, g_host = engine.channel_mps_host(Q, plan32, bond)
        reference = g_host * engine.mps_overlap_host(host, host)
        dense = [np.asarray(A) for A in tensors]
        assert abs(float(gauge) * engine.mps_overlap_host(host, dense) - reference) <= TOLERANCE32[move] * abs(reference)


@pytest.mark.parametrize("move", ["cans", "gram"])
def test_snapped_labels_give_the_oracles_overlap_with_a_random_trial(plan32, move):
    """The labels in use: both channels of a walker at bond 32, snapped, contracted with a random d=4 MPS (every
    component weighed with a random amplitude) give the own-counts oracle's overlap, whose labels are exact. A wrong
    sign on a state of weight w moves the overlap by about sqrt(w) of itself; the state errors (near-ties of the
    truncation, the moves' precision) by about their own size."""
    rng = np.random.default_rng(5)
    dims = [1] + [6] * (L32 - 1) + [1]
    trial = [rng.normal(size=(dims[i], 4, dims[i + 1])) for i in range(L32)]
    blocks = engine.dense_trial_blocks(trial)
    scale = float(np.sqrt(engine.mps_overlap_host(trial, trial)))
    convert = jax.jit(dc.make_dense_converter(plan32, CHI32, move, snap=True).convert)
    for seed in range(2):
        Qa, Qb = _orthonormal32(10 + seed), _orthonormal32(20 + seed)
        alpha, _, ga = convert(jnp.asarray(Qa))
        beta, qb, gb = convert(jnp.asarray(Qb))
        (ha, _, gha), (hb, qhb, ghb) = (engine.channel_mps_host(Q, plan32, utils.plan_bonds(Q, plan32, CHI32))
                                        for Q in (Qa, Qb))
        reference = gha * ghb * _overlap_with(blocks, ha, hb, qhb)
        value = float(ga * gb) * _overlap_with(blocks, alpha, beta, qb)
        assert abs(reference) > 1.0e-9 * scale  # not an accidental near-zero overlap
        assert abs(value - reference) <= 10.0 * TOLERANCE32[move] * abs(reference)
        # the check sees wrong labels: one bond's labels shifted by 1 flip the sign of every alpha-occupied term there
        shifted = list(qb)
        shifted[L32 // 2] = qb[L32 // 2] + 1.0
        wrong = float(ga * gb) * _overlap_with(blocks, alpha, beta, shifted)
        assert abs(wrong - reference) > 100.0 * abs(value - reference)
