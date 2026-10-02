"""Regression tests for trot/gmps/mps_cpmc_2d_gpu.py (CPU or GPU backend).

The script is mps_cpmc_gpu's device code with mps_cpmc_2d's lattice, DMRG trial and
general-h1 H|trial> (copied, then compressed per charge sector), so the checks are:
the copies still equal mps_cpmc_2d, the trial cache and the charge-labelled
compression are right, and the GPU path set up by build() reproduces, on 8-site
lattices with open, periodic and antiperiodic sides, exact enumeration and the CPU
path of mps_cpmc_2d (mps_cpmc_new's walker ops with its blocked energy) for the
overlap, local energy, field sweep, half steps and the measurement block. main()
runs end to end once. The test marked gpu only runs on a GPU backend.

    python -m pytest tests/test_gmps_mps_cpmc_2d_gpu.py
    sbatch --export=ALL,TARGET=pytest run_mps_gpu.sh -x -q /mnt/home/fnappi/trot_mps/tests/test_gmps_mps_cpmc_2d_gpu.py
"""
import itertools
import json
import shutil
from dataclasses import replace
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
from trot.gmps import mps_cpmc_2d as s
from trot.gmps import mps_cpmc_2d_gpu as sg
from trot.gmps import mps_cpmc_gpu as g
from trot.gmps import mps_cpmc_new as ref
from trot.ham.hubbard import HamHubbard
from trot.prop import blocks
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.types import PropState, QmcParams
from trot.trial.auto import make_auto_trial_ops
from trot.trial.uhf import get_rdm1 as uhf_get_rdm1

L, N, U, DT = 8, 4, 4.0, 0.01  # every lattice below has 8 sites
CHI_TRUNC = 4  # per-channel bond cap that truncates visibly at 8 sites
LATTICES = {  # Lx, Ly, boundary_x, boundary_y
    "4x2-open": (4, 2, "open", "open"),
    "4x2-periodic": (4, 2, "periodic", "periodic"),  # degenerate free-fermion shell, doubled y bonds
    "2x4-periodic-antiperiodic": (2, 4, "periodic", "antiperiodic"),
}
BOUNDARIES = ("open", "periodic", "antiperiodic")
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


def _random_field_walkers(h1, Ca, Cb, n, steps, seed):
    """The start determinants propagated by the CPMC propagator with unguided random fields."""
    rng = np.random.default_rng(seed)
    half = scipy.linalg.expm(-0.5 * DT * h1)
    gamma = np.arccosh(np.exp(0.5 * DT * U))
    out = []
    for _ in range(n):
        ca, cb = Ca.copy(), Cb.copy()
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


def _make_model(name, cache_dir):
    """One lattice: its DMRG trial (written to the trial cache, which build() then
    reads), H|trial> exact and compressed, and enumeration data at 8 sites."""
    Lx, Ly, bx, by = LATTICES[name]
    cfg = sg.Config(Lx=Lx, Ly=Ly, boundary_x=bx, boundary_y=by, n_up=N, n_down=N, interaction=U,
                    trial_chi=16, dmrg_sweeps=8, dt=DT, n_walkers=6, linalg="batched", walker_qr="cholesky",
                    trial_cache=cache_dir)
    h1 = sg.lattice_hopping(cfg)
    trial_np, trial_charges, dmrg_energy, mps_energy = sg.load_or_run_trial(cfg, h1)
    H_exact = sg.trial_times_h(sg.hubbard_mpo_from_h1(h1, U), trial_np, trial_charges)
    H_dense = g.compress_mps(H_exact[0])
    gamma_a, gamma_b = g.one_rdm(trial_np)
    Ra, Rb = g.natural_orbitals(gamma_a, N)[0], g.natural_orbitals(gamma_b, N)[0]  # build()'s plan reference
    ham = HamHubbard(h1=jnp.asarray(h1), u=U)
    rows, occ = _all_occupations()
    return SimpleNamespace(
        name=name, cfg=cfg, h1=h1, trial_np=trial_np, trial_charges=trial_charges, dmrg_energy=dmrg_energy,
        mps_energy=mps_energy, H_exact=H_exact, H_qn=g.compress_mps_qn(*H_exact), H_dense=H_dense,
        ham=ham, prop_ctx=_build_prop_ctx(ham, DT), rows=rows, occ=occ,
        amp=_signed_amplitudes(trial_np, occ), hamp=_signed_amplitudes(H_dense, occ),
        walkers=_random_field_walkers(h1, Ra, Rb, n=6, steps=150, seed=0))


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    cache_dir = str(tmp_path_factory.mktemp("trial_cache"))
    built = {}

    def get(name):
        if name not in built:
            built[name] = _make_model(name, cache_dir)
        return built[name]
    return get


@pytest.fixture(scope="module", params=list(LATTICES))
def model(models, request):
    return models(request.param)


@pytest.fixture(scope="module")
def periodic(models):
    return models("4x2-periodic")


def _setup(model, chi=None, **options):
    """build() on the model's lattice; the trial comes from the model's cache file."""
    return sg.build(replace(model.cfg, walker_channel_chi=chi, **options), verbose=False)


def _ref_ops(model, setup):
    """mps_cpmc_2d's CPU walker ops for the same plans and truncation."""
    return ref.make_walker_ops(*setup.references, *setup.plans, *setup.bonds, model.trial_np, model.trial_charges)


def _ref_energy(model, setup, ref_ops):
    """mps_cpmc_2d's local energy (charge-blocked, uncompressed H|trial>)."""
    return s.make_blocked_energy(ref_ops, *setup.references, model.trial_np, *model.H_exact)


def _exact_overlap(model, ca, cb):
    return np.linalg.det(ca[model.rows]) @ model.amp @ np.linalg.det(cb[model.rows])


def _exact_energy(model, ca, cb):
    da, db = np.linalg.det(ca[model.rows]), np.linalg.det(cb[model.rows])
    return (da @ model.hamp @ db) / (da @ model.amp @ db)


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


def _state(ops, ca, cb, e0, key=0):
    overlaps = jax.jit(ops.overlaps)(jnp.asarray(ca), jnp.asarray(cb), ops.data)
    return PropState(walkers=(jnp.asarray(ca), jnp.asarray(cb)), weights=jnp.ones(len(ca), jnp.float64),
                     overlaps=overlaps, rng_key=jax.random.PRNGKey(key),
                     pop_control_ene_shift=jnp.asarray(e0, jnp.float64),
                     e_estimate=jnp.asarray(e0, jnp.float64), node_encounters=jnp.zeros((), jnp.int64))


def _propagate(ops, params, n_chunks, n_half):
    half = g.make_half_step(ops, params, n_chunks)
    return jax.jit(lambda st, data: lax.scan(lambda x, i: (half(x, i, data), None), st, jnp.arange(n_half))[0])


# --------------------------------------------------------------------------
# Host code: the copies of mps_cpmc_2d, the trial cache, H|trial>
# --------------------------------------------------------------------------

@pytest.mark.parametrize("Lx,Ly", [(4, 2), (2, 4), (3, 3), (5, 1), (1, 4)])
def test_lattice_code_matches_mps_cpmc_2d(Lx, Ly):
    """The copied hopping matrix and general-h1 MPO equal mps_cpmc_2d's for every
    boundary combination, and the MPO also for a dense random h1."""
    for bx, by in itertools.product(BOUNDARIES, repeat=2):
        h1 = sg.square_hopping_matrix(Lx, Ly, 0.7, bx, by)
        np.testing.assert_array_equal(h1, s.square_hopping_matrix(Lx, Ly, 0.7, bx, by))
        np.testing.assert_array_equal(sg.hubbard_mpo_from_h1(h1, U), s.hubbard_mpo_from_h1(h1, U))
    h1 = np.random.default_rng(Lx * Ly).normal(size=(Lx * Ly, Lx * Ly))
    np.testing.assert_array_equal(sg.hubbard_mpo_from_h1(h1 + h1.T, U), s.hubbard_mpo_from_h1(h1 + h1.T, U))
    with pytest.raises(ValueError):
        sg.square_hopping_matrix(Lx, Ly, 1.0, "twisted")


def test_dmrg_schedule_matches_mps_cpmc_2d():
    for cfg in (sg.Config(), sg.Config(trial_chi=8, dmrg_sweeps=3), sg.Config(dmrg_bdims=(8, 16), dmrg_sweeps=10),
                sg.Config(dmrg_bdims=tuple(range(8, 80, 8)), dmrg_sweeps=12)):
        assert sg.dmrg_schedule(cfg) == s.dmrg_schedule(cfg)
    with pytest.raises(ValueError):
        sg.dmrg_schedule(sg.Config(dmrg_bdims=(8, 16), dmrg_sweeps=2))


def test_trial_times_h_matches_mps_cpmc_2d(model):
    tensors, charges = s.trial_times_h(s.hubbard_mpo_from_h1(model.h1, U), model.trial_np, model.trial_charges)
    assert len(tensors) == len(model.H_exact[0]) and len(charges) == len(model.H_exact[1])
    for a, b in zip(model.H_exact[0], tensors):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(model.H_exact[1], charges):
        np.testing.assert_array_equal(a, b)


def test_dmrg_trial_matches_mps_cpmc_2d(models):
    """The copied DMRG gives mps_cpmc_2d's trial (same schedule, MPO and seed)."""
    model = models("4x2-open")  # non-degenerate: the trial is unique
    fields = ("Lx", "Ly", "boundary_x", "boundary_y", "n_up", "n_down", "hopping", "interaction", "trial_chi",
              "dmrg_bdims", "dmrg_sweeps", "dmrg_tol", "dmrg_seed")
    cfg = s.Config(**{f: getattr(model.cfg, f) for f in fields})
    trial_np, _, dmrg_energy, _, mps_energy = s.densified_trial(cfg, model.h1)
    assert dmrg_energy == pytest.approx(model.dmrg_energy, abs=1e-8)
    assert mps_energy == pytest.approx(model.mps_energy, abs=1e-8)
    assert abs(_overlap(trial_np, model.trial_np)) == pytest.approx(1.0, abs=1e-6)


def test_trial_cache_round_trip_and_names(periodic, tmp_path):
    """The cache file is named by lattice, boundaries, filling, t, U and DMRG schedule,
    loads back bit for bit, and a file whose h1 differs from the run's is refused."""
    cfg = periodic.cfg
    path = sg._trial_cache_file(cfg)
    assert path.name == "sq4x2pp_n4-4_t1_U4_chi16_sw8_tol1e-06_seed0.npz" and path.exists()
    tensors, charges, dmrg_energy, mps_energy = sg.load_or_run_trial(cfg, periodic.h1)
    for a, b in zip(tensors, periodic.trial_np):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(charges, periodic.trial_charges):
        np.testing.assert_array_equal(a, b)
    assert (dmrg_energy, mps_energy) == (periodic.dmrg_energy, periodic.mps_energy)

    with np.load(path) as data:
        stored = dict(data)
    assert len(stored["sweep_energies"]) > 0
    np.testing.assert_array_equal(stored["h1"], periodic.h1)
    other = replace(cfg, trial_cache=str(tmp_path))
    stored["h1"] = 2.0 * stored["h1"]
    np.savez(sg._trial_cache_file(other), **stored)
    with pytest.raises(ValueError, match="different h1"):
        sg.load_or_run_trial(other, periodic.h1)

    variants = (dict(), dict(boundary_x="antiperiodic"), dict(boundary_y="open"), dict(Lx=2, Ly=4),
                dict(trial_chi=32), dict(dmrg_bdims=(8, 16)), dict(dmrg_tol=1e-8), dict(n_up=3))
    names = [sg._trial_cache_file(replace(cfg, **kw)).name for kw in variants]
    assert len(set(names)) == len(names)
    assert sg._trial_cache_file(replace(cfg, dmrg_bdims=(8, 16), trial_chi=32)).name == names[5]  # ramp wins


def test_compressed_htrial_keeps_labels_and_state(model):
    """trial_times_h's labels are exact and compress_mps_qn keeps them, the state
    and its overlap with the trial, which is the variational energy of pyblock3's MPO."""
    for tensors, labels in (model.H_exact, model.H_qn):
        for i, A in enumerate(tensors):
            nz = np.argwhere(np.abs(A) > 1e-12)
            np.testing.assert_array_equal(labels[i][nz[:, 0]] + g.PHYSICAL_CHARGE[nz[:, 1]], labels[i + 1][nz[:, 2]])
    exact, compressed = model.H_exact[0], model.H_qn[0]
    assert _overlap(compressed, model.trial_np) == pytest.approx(_overlap(exact, model.trial_np), rel=1e-11)
    assert _overlap(compressed, compressed) == pytest.approx(_overlap(exact, exact), rel=1e-11)
    assert max(A.shape[0] for A in compressed) <= max(A.shape[0] for A in exact)
    assert _overlap(exact, model.trial_np) == pytest.approx(model.mps_energy, abs=1e-8)


def _compress_mps_qn_dense_products(tensors, charges, relative_tolerance=1.0e-13):
    """mps_cpmc_gpu.compress_mps_qn before 2026-09-29, verbatim: the same per-sector
    factorisations, with the R and U S factors multiplied in as dense block-diagonal matrices."""
    A = [np.array(t, dtype=float, copy=True) for t in tensors]
    Q = [np.asarray(q, int).reshape(-1, 2) for q in charges]
    for i in range(len(A) - 1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl * d, Dr)
        rows = (Q[i][:, None, :] + g.PHYSICAL_CHARGE[None]).reshape(-1, 2)
        left, right, labels = [], [], []
        for c, r, k in g._label_sectors(rows, Q[i + 1]):
            q, rr = np.linalg.qr(M[np.ix_(r, k)])
            lq = np.zeros((Dl * d, q.shape[1]))
            lq[r] = q
            rq = np.zeros((q.shape[1], Dr))
            rq[:, k] = rr
            left.append(lq)
            right.append(rq)
            labels += [c] * q.shape[1]
        A[i] = np.concatenate(left, axis=1).reshape(Dl, d, -1)
        A[i + 1] = np.tensordot(np.concatenate(right, axis=0), A[i + 1], axes=1)
        Q[i + 1] = np.asarray(labels, int).reshape(-1, 2)
    for i in range(len(A) - 1, 0, -1):
        Dl, d, Dr = A[i].shape
        M = A[i].reshape(Dl, d * Dr)
        columns = (Q[i + 1][None, :, :] - g.PHYSICAL_CHARGE[:, None, :]).reshape(-1, 2)
        factors = []
        for c, r, k in g._label_sectors(Q[i], columns):
            u, s, vh = np.linalg.svd(M[np.ix_(r, k)], full_matrices=False)
            factors.append((c, r, k, u, s, vh))
        cut = relative_tolerance * max(max(f[4][0] for f in factors), 1e-300)
        left, right, labels = [], [], []
        for c, r, k, u, s, vh in factors:
            n = int(np.sum(s > cut))
            if n == 0:
                continue
            lu = np.zeros((Dl, n))
            lu[r] = u[:, :n] * s[:n]
            rv = np.zeros((n, d * Dr))
            rv[:, k] = vh[:n]
            left.append(lu)
            right.append(rv)
            labels += [c] * n
        A[i] = np.concatenate(right, axis=0).reshape(-1, d, Dr)
        A[i - 1] = np.tensordot(A[i - 1], np.concatenate(left, axis=1), axes=(2, 0))
        Q[i] = np.asarray(labels, int).reshape(-1, 2)
    return A, tuple(Q)


def test_sector_products_match_dense_products(model):
    """compress_mps_qn multiplies each sector's factor into its neighbour on that sector's
    rows or columns only. It must reproduce the dense-product version: the same bonds and
    labels, and the same state to roundoff. Tensors are not compared one by one: rank-
    deficient QR columns and degenerate singular vectors may differ, the state may not."""
    new, new_labels = model.H_qn
    old, old_labels = _compress_mps_qn_dense_products(*model.H_exact)
    assert [A.shape for A in new] == [A.shape for A in old]
    for a, b in zip(new_labels, old_labels):
        np.testing.assert_array_equal(a, b)
    nn, oo, no = _overlap(new, new), _overlap(old, old), _overlap(new, old)
    assert nn == pytest.approx(oo, rel=1e-12)
    assert 1.0 - no ** 2 / (nn * oo) < 1e-12
    assert _overlap(new, model.trial_np) == pytest.approx(_overlap(old, model.trial_np), rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------
# Block-form H|trial> (cache_htrial, prepare_sq_trial.py), term-built DMRG MPO
# --------------------------------------------------------------------------

def test_block_form_round_trip(model):
    blocks = sg.to_blocks(model.trial_np, model.trial_charges)
    for a, b in zip(sg.from_blocks(blocks, model.trial_charges), model.trial_np):
        np.testing.assert_array_equal(a, b)
    assert sg.overlap_blocks(blocks, blocks) == pytest.approx(_overlap(model.trial_np, model.trial_np), rel=1e-13)


def test_block_product_matches_trial_times_h(model):
    """trial_times_h_blocks has trial_times_h's labels and entries, and htrial_block_bytes
    is the size it allocates."""
    W = sg.hubbard_mpo_from_h1(model.h1, U)
    trial = sg.to_blocks(model.trial_np, model.trial_charges)
    tensors, charges = sg.trial_times_h_blocks(W, trial, model.trial_charges)
    for a, b in zip(charges, model.H_exact[1]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(sg.from_blocks(tensors, charges), model.H_exact[0]):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)
    assert sg.htrial_block_bytes(W, trial, model.trial_charges) == sum(
        X.nbytes for site in tensors for X in site.values())


def test_block_compression_matches_compress_mps_qn(model):
    """compress_blocks_qn gives compress_mps_qn's bonds and labels and the same state (tensors
    are not compared one by one, as in test_sector_products_match_dense_products), and
    overlap_blocks gives the trial energy."""
    W = sg.hubbard_mpo_from_h1(model.h1, U)
    trial = sg.to_blocks(model.trial_np, model.trial_charges)
    tensors, charges = sg.compress_blocks_qn(*sg.trial_times_h_blocks(W, trial, model.trial_charges), consume=True)
    new = sg.from_blocks(tensors, charges)
    old, old_labels = model.H_qn
    assert [A.shape for A in new] == [A.shape for A in old]
    for a, b in zip(charges, old_labels):
        np.testing.assert_array_equal(a, b)
    nn, oo, no = _overlap(new, new), _overlap(old, old), _overlap(new, old)
    assert nn == pytest.approx(oo, rel=1e-12)
    assert 1.0 - no ** 2 / (nn * oo) < 1e-12
    energy = sg.overlap_blocks(trial, tensors) / sg.overlap_blocks(trial, trial)
    assert energy == pytest.approx(_overlap(old, model.trial_np), rel=1e-12)
    assert energy == pytest.approx(model.mps_energy, abs=1e-8)


def test_htrial_cache_round_trip_and_names(periodic, tmp_path):
    """load_or_make_htrial writes <trial>_htrial.npz, reads it back bit for bit with the
    trial's rdm1, and refuses it for another trial; term-MPO trials have their own names."""
    cfg = replace(periodic.cfg, cache_htrial=True, trial_cache=str(tmp_path))
    args = (cfg, periodic.h1, periodic.trial_np, periodic.trial_charges, periodic.mps_energy)
    made, made_info = sg.load_or_make_htrial(*args)
    path = sg._htrial_cache_file(cfg)
    assert path.name == "sq4x2pp_n4-4_t1_U4_chi16_sw8_tol1e-06_seed0_htrial.npz" and path.exists()
    loaded, info = sg.load_or_make_htrial(*args)
    for a, b in zip(made[0], loaded[0]):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    for a, b in zip(made[1], loaded[1]):
        np.testing.assert_array_equal(a, b)
    assert (info["trial_energy"], info["uncompressed_bonds"]) == (made_info["trial_energy"],
                                                                  made_info["uncompressed_bonds"])
    for a, b in zip(info["gamma"], g.one_rdm(periodic.trial_np)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="another trial"):
        sg.load_or_make_htrial(*args[:-1], periodic.mps_energy + 1e-3)
    assert sg._trial_cache_file(replace(cfg, dmrg_mpo="terms")).name == (
        "sq4x2pp_n4-4_t1_U4_chi16_sw8_tol1e-06_seed0_mpoterms.npz")


def test_factorized_plan_takes_block_tensors(model):
    """make_factorized_plan gathers the same device blocks from the block form as from the
    dense tensors."""
    qa, qb = _setup(model, CHI_TRUNC).ops.converter.charges
    dense = g.make_factorized_plan(qa, qb, *model.H_qn)
    blocked = g.make_factorized_plan(qa, qb, sg.to_blocks(*model.H_qn), model.H_qn[1])
    assert blocked.stats == dense.stats
    for a, b in zip(blocked.blocks, dense.blocks):
        np.testing.assert_array_equal(a, b)


def test_build_with_cached_htrial(periodic, tmp_path):
    """build() with cache_htrial makes the block-form H|trial> on the first call and loads
    it on the second; both give the dense path's trial energy and local energies."""
    shutil.copy(sg._trial_cache_file(periodic.cfg), tmp_path)
    options = dict(cache_htrial=True, trial_cache=str(tmp_path))
    made = _setup(periodic, CHI_TRUNC, **options)
    assert sg._htrial_cache_file(replace(periodic.cfg, **options)).exists()
    loaded = _setup(periodic, CHI_TRUNC, **options)
    dense = _setup(periodic, CHI_TRUNC)
    ca, cb = _walker_batch(periodic, mix=0.3)
    want = np.asarray(jax.jit(dense.ops.energies)(jnp.asarray(ca), jnp.asarray(cb), dense.ops.data))
    for setup in (made, loaded):
        assert setup.info["trial_energy"] == pytest.approx(dense.info["trial_energy"], rel=1e-12)
        assert setup.info["htrial_bond"] == dense.info["htrial_bond"] and setup.htrial is None
        got = np.asarray(jax.jit(setup.ops.energies)(jnp.asarray(ca), jnp.asarray(cb), setup.ops.data))
        np.testing.assert_allclose(got, want, rtol=1e-10)


def test_terms_mpo_is_the_qc_mpo(models):
    """The term-built DMRG MPO is pyblock3's quantum-chemistry MPO (the same <H> on one MPS),
    and its trial's pyblock3 energy is the nearest-neighbour MPO's <trial|H|trial>."""
    model = models("4x2-open")
    terms_cfg = replace(model.cfg, dmrg_mpo="terms")
    hamiltonian = sg.build_dmrg_hamiltonian(model.cfg, model.h1)
    mps, *_ = sg.run_dmrg(hamiltonian, model.cfg)
    norm = float(mps @ mps)
    energies = [float(g.MPE(mps, sg.build_dmrg_mpo(hamiltonian, cfg), mps)[0:2].expectation) / norm
                for cfg in (model.cfg, terms_cfg)]
    assert energies[1] == pytest.approx(energies[0], abs=1e-10)
    trial_np, charges, _, _, mps_energy = sg.densified_trial(terms_cfg, model.h1)
    H = sg.trial_times_h(sg.hubbard_mpo_from_h1(model.h1, U), trial_np, charges)[0]
    assert _overlap(H, trial_np) == pytest.approx(mps_energy, abs=1e-9)
    assert mps_energy == pytest.approx(model.mps_energy, abs=1e-4)  # the same ground-state approximation


def test_config_from_args():
    cfg = sg.config_from_args(["Lx=4", "Ly=3", "boundary_x=periodic", "dmrg_bdims=64", "walker_channel_chi=None",
                               "compress_htrial=False", "energy=dense", "interaction=8", "n_chunks=4"])
    assert cfg == replace(sg.Config(), Lx=4, Ly=3, boundary_x="periodic", dmrg_bdims=(64,), walker_channel_chi=None,
                          compress_htrial=False, energy="dense", interaction=8.0, n_chunks=4)
    assert sg.config_from_args(["dmrg_bdims=500,1000"]).dmrg_bdims == (500, 1000)
    with pytest.raises(ValueError, match="not a number"):
        sg.config_from_args(["n_up=four"])


# --------------------------------------------------------------------------
# Device path set up by build()
# --------------------------------------------------------------------------

def test_build_matches_the_script_setup(periodic):
    """build() plans on the trial's natural orbitals, batches both spins and reports
    the dense-MPO trial energy, which agrees with pyblock3's."""
    setup = _setup(periodic, CHI_TRUNC)
    info = setup.info
    assert info["spin_batched"] and info["linalg"] == "batched" and info["walker_qr"] == "cholesky"
    assert info["trial_energy"] == pytest.approx(info["mps_energy"], abs=1e-8)
    assert info["mpo_bond"] == sg.hubbard_mpo_from_h1(periodic.h1, U).shape[1]
    assert info["htrial_bond"] == max(A.shape[0] for A in periodic.H_qn[0])
    np.testing.assert_allclose(np.asarray(setup.ops.data.exp_h1_half), np.asarray(periodic.prop_ctx.exp_h1_half))
    assert setup.bonds[0].reference_discarded_weight > 0  # CHI_TRUNC truncates
    for C in setup.references:
        np.testing.assert_allclose(C.T @ C, np.eye(N), atol=1e-12)


@pytest.mark.parametrize("chi,linalg", [(None, "batched"), (CHI_TRUNC, "batched"), (CHI_TRUNC, "native")])
def test_overlap_matches_the_original_and_enumeration(model, chi, linalg):
    """The factorized contraction against mps_cpmc_new's blocked overlap on the
    lattice, on non-orthonormal walkers (det R gauge), and exact enumeration."""
    setup = _setup(model, chi, linalg=linalg)
    ops, ref_ops = setup.ops, _ref_ops(model, setup)
    ca, cb = _walker_batch(model, mix=0.3)
    got = np.asarray(jax.jit(ops.overlaps)(jnp.asarray(ca), jnp.asarray(cb), ops.data))
    overlap = jax.jit(ref_ops.overlap)
    want = np.asarray([float(overlap((jnp.asarray(a), jnp.asarray(b)))) for a, b in zip(ca, cb)])
    np.testing.assert_allclose(got, want, rtol=1e-9)
    if chi is None:
        exact = [_exact_overlap(model, a, b) for a, b in zip(ca, cb)]
        np.testing.assert_allclose(got, exact, rtol=1e-9)


@pytest.mark.parametrize("energy,compress", [("blocked", True), ("blocked", False), ("dense", True)],
                         ids=["blocked-compressed", "blocked-uncompressed", "dense"])
def test_local_energy_matches_enumeration(model, energy, compress):
    """Each H|trial> variant gives the exact local energy of untruncated walkers."""
    setup = _setup(model, energy=energy, compress_htrial=compress)
    ops = setup.ops
    ca, cb = _walker_batch(model, mix=0.3)
    got = np.asarray(jax.jit(ops.energies)(jnp.asarray(ca), jnp.asarray(cb), ops.data))
    exact = [_exact_energy(model, a, b) for a, b in zip(ca, cb)]
    np.testing.assert_allclose(got, exact, atol=1e-9)


def test_truncated_local_energy_matches_mps_cpmc_2d(model):
    """With walker truncation, the energy of the compressed charge-labelled H|trial>
    equals mps_cpmc_2d's blocked energy (uncompressed H|trial>) walker by walker."""
    setup = _setup(model, CHI_TRUNC)
    ops = setup.ops
    energy = jax.jit(_ref_energy(model, setup, _ref_ops(model, setup)))
    ca, cb = _walker_batch(model, mix=0.3)
    got = np.asarray(jax.jit(ops.energies)(jnp.asarray(ca), jnp.asarray(cb), ops.data))
    want = [float(energy((jnp.asarray(a), jnp.asarray(b)))) for a, b in zip(ca, cb)]
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_initial_state_and_self_check(periodic):
    """init_state's overlap and energy are mps_cpmc_2d's for the starting walker, and
    the start-up self-check against the NumPy conversion passes."""
    setup = _setup(periodic, CHI_TRUNC, n_walkers=4)
    state, probe = g.init_state(setup.ops, setup.system, setup.trial_data, setup.params)
    assert max(g.conversion_self_check(setup.ops, setup.plans, setup.bonds, probe)) < 1e-9
    ref_ops = _ref_ops(periodic, setup)
    walker = (state.walkers[0][0], state.walkers[1][0])
    np.testing.assert_allclose(np.asarray(state.overlaps), float(jax.jit(ref_ops.overlap)(walker)), rtol=1e-9)
    energy = jax.jit(_ref_energy(periodic, setup, ref_ops))
    np.testing.assert_allclose(float(state.e_estimate), float(energy(walker)), rtol=1e-9)


# --------------------------------------------------------------------------
# Sweep, step, block
# --------------------------------------------------------------------------

def test_sweep_matches_the_original(model):
    """step_batch with the field sweep against mps_cpmc_new's fast sweep: same
    fields, overlaps before and after, weight factor and node count."""
    setup = _setup(model, CHI_TRUNC)
    ops, ref_ops = setup.ops, _ref_ops(model, setup)
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
def test_half_steps_match_the_original_step(periodic, weight_floor):
    """Two half steps equal one step of mps_cpmc_new.make_fast_prop_ops with the
    lattice's one-body propagator, three times in a row."""
    setup = _setup(periodic, CHI_TRUNC)
    ops, ref_ops = setup.ops, _ref_ops(periodic, setup)
    params = QmcParams(dt=DT, n_walkers=6, n_prop_steps=1, n_blocks=1, n_eql_blocks=0,
                       weight_floor=weight_floor, seed=0)
    fast = ref.make_fast_prop_ops(periodic.ham, "unrestricted", ref_ops.overlap, ref_ops.sweep)
    ref_step = jax.jit(lambda st: fast.step(st, params=params, ham_data=periodic.ham, trial_data=None,
                                            trial_ops=None, meas_ops=None, meas_ctx=None,
                                            prop_ctx=periodic.prop_ctx))
    gpu_step = _propagate(ops, params, 1, 2)
    got = want = _state(ops, *_walker_batch(periodic), periodic.mps_energy)
    for _ in range(3):
        want, got = ref_step(want), gpu_step(got, ops.data)
    for x, y in zip(got.walkers, want.walkers):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got.weights), np.asarray(want.weights), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(np.asarray(got.overlaps), np.asarray(want.overlaps), rtol=1e-9)
    np.testing.assert_allclose(float(got.pop_control_ene_shift), float(want.pop_control_ene_shift), rtol=1e-9)
    assert int(got.node_encounters) == int(want.node_encounters)
    assert np.array_equal(np.asarray(got.rng_key), np.asarray(want.rng_key))


def test_block_matches_trot_block(periodic):
    """make_block against trot.prop.blocks.block driving mps_cpmc_2d's CPU path:
    mps_cpmc_new's step and overlap with mps_cpmc_2d's blocked energy."""
    setup = _setup(periodic, CHI_TRUNC)
    ops, ref_ops = setup.ops, _ref_ops(periodic, setup)
    system = setup.system
    params = QmcParams(dt=DT, n_walkers=6, n_prop_steps=2, n_blocks=1, n_eql_blocks=0, weight_floor=1e-8, seed=0)
    fast = ref.make_fast_prop_ops(periodic.ham, "unrestricted", ref_ops.overlap, ref_ops.sweep)
    trial_ops = make_auto_trial_ops(system, overlap_u=ref_ops.overlap, get_rdm1=uhf_get_rdm1)
    meas_ops = MeasOps(overlap=ref_ops.overlap, kernels={k_energy: _ref_energy(periodic, setup, ref_ops)})
    state0 = _state(ops, *_walker_batch(periodic, mix=0.1), periodic.mps_energy)

    want_state, want_obs = jax.jit(lambda st: blocks.block(
        st, sys=system, params=params, ham_data=periodic.ham, trial_data=None, trial_ops=trial_ops,
        meas_ops=meas_ops, meas_ctx=None, prop_ops=fast, prop_ctx=periodic.prop_ctx))(state0)
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


def test_main_writes_every_output(periodic, tmp_path):
    """main() end to end on a few blocks: the result record, the block log, the
    walker snapshots and the trial export, each with the lattice's h1."""
    out = lambda name: str(tmp_path / name)
    cfg = replace(periodic.cfg, walker_channel_chi=CHI_TRUNC, n_walkers=8, n_equilibration=2, n_blocks=4,
                  n_steps=2, result_json=out("results.jsonl"), block_log=out("blocks.jsonl"),
                  walker_snapshots=out("walkers.npz"), trial_export=out("trial.npz"), tag="smoke")
    record = sg.main(cfg)
    assert record["kind"] == "mps_2d_gpu" and record["n_sites"] == L and record["spin_batched"]
    assert np.isfinite(record["cpmc_energy"]) and record["collapsed_after_block"] is None
    assert record["trial_energy"] == pytest.approx(record["mps_energy"], abs=1e-8)
    assert set(record["setup_seconds"]) == {"trial", "htrial", "natural_orbitals", "bond_plans", "ops", "init", "export"}
    assert record["host_cpus"] >= 1

    with open(cfg.result_json) as stream:
        [saved] = [json.loads(line) for line in stream]
    assert saved["cpmc_energy"] == record["cpmc_energy"] and saved["dmrg_bdims"] == []
    with open(cfg.block_log) as stream:
        logged = [json.loads(line) for line in stream]
    assert [b["phase"] for b in logged] == ["equilibration"] * 2 + ["sampling"] * 4
    assert all(b["tag"] == "smoke" for b in logged)

    with np.load(cfg.walker_snapshots) as walkers:
        assert walkers["up"].shape == (7, 8, L, N) and walkers["dn"].shape == (7, 8, L, N)
        assert len(walkers["energies"]) == 6 and walkers["comb_index"].shape == (6, 8)
        np.testing.assert_array_equal(walkers["h1"], periodic.h1)
        config = json.loads(str(walkers["config"]))
    assert (config["L"], config["LX"], config["LY"], config["BOUNDARY_X"]) == (L, 4, 2, "periodic")
    assert config["module"] == "mps_cpmc_2d_gpu" and config["DMRG_CHI_T"] == 16

    with np.load(cfg.trial_export) as trial:
        tensors = [trial[f"T{i}"] for i in range(L)]
        htrial = [trial[f"H{i}"] for i in range(L)]
        np.testing.assert_array_equal(trial["h1"], periodic.h1)
        assert trial["gamma"].shape == (2, L, L) and len([k for k in trial.files if k.startswith("q")]) == L + 1
    for a, b in zip(tensors, periodic.trial_np):
        np.testing.assert_array_equal(a, b)
    assert _overlap(htrial, tensors) == pytest.approx(record["trial_energy"], abs=1e-9)


# --------------------------------------------------------------------------
# GPU usage (skipped without a GPU)
# --------------------------------------------------------------------------

@gpu
def test_gpu_build_runs_batched_on_the_device(periodic):
    """On a GPU the auto options pick the batched conversion and CholeskyQR2, and the
    compiled half steps on the lattice run with host transfers disallowed."""
    setup = _setup(periodic, CHI_TRUNC, linalg="auto", walker_qr="auto", n_walkers=256)
    assert (setup.info["linalg"], setup.info["walker_qr"]) == ("batched", "cholesky")
    assert setup.info["spin_batched"]
    ops = setup.ops
    params = QmcParams(dt=DT, n_walkers=256, n_prop_steps=1, n_blocks=1, n_eql_blocks=0, seed=0)
    state0 = _state(ops, *_walker_batch(periodic, 256), periodic.mps_energy)
    compiled = _propagate(ops, params, 1, 4).lower(state0, ops.data).compile()
    jax.block_until_ready(compiled(state0, ops.data))
    with jax.transfer_guard("disallow"):
        out = compiled(state0, ops.data)
        jax.block_until_ready(out)
    for leaf in jax.tree_util.tree_leaves(out):
        assert {d.platform for d in leaf.devices()} == {"gpu"}
    assert out.weights.dtype == jnp.float64 and out.walkers[0].dtype == jnp.float64
