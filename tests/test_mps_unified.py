"""The one MPS-CPMC path for every lattice, on CPU or GPU.

trot.gmps.trials reads the geometry off h1, caches the DMRG trial and builds the (rotated) trial; trot.gmps.driver
runs trot's MPS ops (trot.trial.mps, trot.meas.mps, trot.prop.mps_cpmc) through trot's driver with the MPS
measurement block, block log and walker snapshots;
trot/gmps/run_mps_cpmc.py is the command line. Chains and square lattices go through the same code, so every test
runs on both: an 8-site chain (open, antiperiodic) and a 2 x 4 lattice (open, periodic along y). All four are
closed shells, so the natural orbitals and the walkers' start are unique.

References are exact enumeration (tests/helpers/hubbard_fock.py) and trot's generic block (trot.prop.blocks.block)
driving the same ops. test_mps_unified_legacy.py compares this path with the scripts it replaces.
"""

from trot import config

config.configure_once()

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers import hubbard_fock as hf
from trot.core.system import System
from trot.gmps import driver, trials
from trot.ham.hubbard import HamHubbard, hopping_matrix, square_hopping_matrix
from trot.prop import blocks
from trot.prop.types import QmcParamsMps
from trot.meas.mps import _dense_overlap, hubbard_mpo_from_h1, trial_times_h
from trot.trial.mps import MpsTrial, compress_mps_qn, label_array, natural_orbitals, one_rdm

U, NELEC, CHI, SWEEPS = 4.0, (4, 4), 16, 8
CASES = {
    "L8": hopping_matrix(8, 1.0),
    "L8a": square_hopping_matrix(8, 1, 1.0, "antiperiodic", "open"),
    "sq2x4oo": square_hopping_matrix(2, 4, 1.0),
    "sq2x4op": square_hopping_matrix(2, 4, 1.0, "open", "periodic"),
}


def _quiet(*args, **kwargs):
    pass


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    pytest.importorskip("pyblock3")
    return tmp_path_factory.mktemp("trial_cache")


@pytest.fixture(scope="module")
def dmrg(cache):
    """The DMRG trial of each case (chi 16, 8 sweeps, the lattice's default DMRG options), made once."""
    made = {}

    def get(name):
        if name not in made:
            made[name] = trials.load_or_make_dmrg_trial(CASES[name], U, NELEC, chi=CHI, sweeps=SWEEPS,
                                                        cache_dir=cache, say=_quiet)
        return made[name]

    return get


def _params(**overrides):
    settings = dict(dt=0.01, n_walkers=6, n_prop_steps=2, n_eql_blocks=1, n_blocks=2, seed=5, n_chunks=1,
                    auto_n_chunks=False, orbital_plan="rank_exact", walker_channel_chi=None)
    settings.update(overrides)
    return QmcParamsMps(**settings)


def _prepare(h1, trial, params, htrial=None):
    system = System(norb=len(h1), nelec=NELEC, walker_kind="unrestricted")
    ham = HamHubbard(h1=jnp.asarray(h1), u=U)
    return driver.prepare_mps_cpmc(sys=system, params=params, ham_data=ham, trial=trial, htrial=htrial,
                                   verbose=False)


def _walkers(h1, trial, n=6, seed=3):
    """Non-orthonormal determinants diffused from the natural orbitals of the trial's rdm1."""
    rdm1 = np.asarray(trial.rdm1)
    Ra, Rb = (natural_orbitals(rdm1[s], NELEC[s])[0] for s in range(2))
    rng = np.random.default_rng(seed)
    return [(hf.nonorthonormal(a, rng), hf.nonorthonormal(b, rng))
            for a, b in hf.random_field_walkers(h1, U, 0.1, Ra, Rb, n=n, steps=20, seed=seed)]


def _stack(walkers):
    return tuple(jnp.asarray(np.stack([w[s] for w in walkers])) for s in range(2))


def _exact(h1, tensors, walkers):
    """<T|SD> and <T|H|SD>/<T|SD> of each walker in the NELEC sector, and <T|H|T>/<T|T> there."""
    amp = hf.mps_sector_amplitudes([np.asarray(A) for A in tensors], *NELEC).ravel()
    hamp = hf.hubbard_sector_hamiltonian(h1, U, *NELEC) @ amp
    sds = [hf.sd_amplitudes(ca, cb).ravel() for ca, cb in walkers]
    return (np.array([amp @ sd for sd in sds]), np.array([hamp @ sd / (amp @ sd) for sd in sds]),
            float(amp @ hamp / (amp @ amp)))


def _engine_values(run, walkers):
    ca, cb = _stack(walkers)
    kernels, data = run.meas_ctx.kernels, run.meas_ctx.data()
    return (np.asarray(jax.jit(kernels.overlaps)(ca, cb, data)),
            np.asarray(jax.jit(kernels.energies)(ca, cb, data)))


# ---------------------------------------------------------------------------------------------
# Geometry from h1, cache names
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h1,kind,name,shape",
    [
        (hopping_matrix(100, 1.0), "chain", "L100", (100, 1, "open")),
        (square_hopping_matrix(16, 1, 1.0, "periodic", "open"), "chain", "L16p", (16, 1, "periodic")),
        (square_hopping_matrix(16, 1, 0.5, "antiperiodic", "open"), "chain", "L16a", (16, 1, "antiperiodic")),
        (square_hopping_matrix(2, 4, 1.0), "square", "sq2x4oo", (2, 4, "open")),
        (square_hopping_matrix(4, 4, 1.0), "square", "sq4x4oo", (4, 4, "open")),
        (square_hopping_matrix(4, 4, 1.0, "periodic", "periodic"), "square", "sq4x4pp", (4, 4, "periodic")),
        (square_hopping_matrix(6, 4, 1.0, "open", "antiperiodic"), "square", "sq6x4oa", (6, 4, "open")),
    ],
)
def test_describe_h1_reads_the_geometry(h1, kind, name, shape):
    lattice = trials.describe_h1(h1)
    assert (lattice.kind, lattice.name, lattice.n_sites) == (kind, name, len(h1))
    assert (lattice.Lx, lattice.Ly, lattice.boundary_x) == shape
    assert lattice.hopping == pytest.approx(np.abs(h1).max())


def test_describe_h1_names_any_other_h1_by_its_hash():
    chain = hopping_matrix(8, 1.0)
    onsite = chain + np.diag(np.linspace(-0.5, 0.5, 8))
    longer = chain.copy()
    longer[0, 2] = longer[2, 0] = -1.0
    names = set()
    for h1 in (onsite, longer, 2.0 * chain - np.diag(np.ones(8))):
        lattice = trials.describe_h1(h1)
        assert lattice.kind == "general" and lattice.name.startswith("h1-") and len(lattice.name) == 13
        assert trials.describe_h1(h1.copy()).name == lattice.name
        names.add(lattice.name)
    assert len(names) == 3


def test_lattice_record_has_the_plot_keys():
    """plot_cpmc_runs reads L (or n_sites) and tells a square lattice by its Lx key."""
    chain = trials.describe_h1(hopping_matrix(8, 1.0)).record()
    square = trials.describe_h1(square_hopping_matrix(2, 4, 1.0, "open", "periodic")).record()
    assert chain["L"] == 8 and "Lx" not in chain
    assert (square["Lx"], square["Ly"], square["boundary_y"], square["n_sites"]) == (2, 4, "periodic", 8)


def test_trial_cache_names_keep_the_production_names():
    """The names the production caches have (trial_cache_warm/L100_..., the 2D caches); no two options share one."""
    chain = trials.describe_h1(hopping_matrix(100, 1.0))
    sq = trials.describe_h1(square_hopping_matrix(4, 4, 1.0))
    big = trials.describe_h1(square_hopping_matrix(8, 8, 1.0))
    name = lambda lattice, nelec, u, **kw: trials.trial_cache_file("c", lattice, nelec, u, **kw).name
    assert name(chain, (50, 50), 8.0, chi=16, sweeps=30, mpo="terms", schedule="warmup") \
        == "L100_n50-50_t1_U8_chi16_sw30_seed0.npz"
    assert name(sq, (8, 8), 8.0, chi=256, sweeps=14, mpo="qc", schedule="plain", tol=1e-6) \
        == "sq4x4oo_n8-8_t1_U8_chi256_sw14_tol1e-06_seed0.npz"
    assert name(big, (32, 32), 8.0, chi=512, sweeps=20, mpo="terms", schedule="plain", bdims=(128, 256, 512),
                tol=1e-6) == "sq8x8oo_n32-32_t1_U8_chi128-256-512_sw20_tol1e-06_seed0_mpoterms.npz"
    chain8 = trials.describe_h1(hopping_matrix(8, 1.0))
    options = [dict(mpo=m, schedule=s, tol=t) for m in ("terms", "qc") for s in ("warmup", "plain")
               for t in (None, 1e-6)]
    names = {name(chain8, (4, 4), 4.0, chi=16, sweeps=8, **o) for o in options}
    assert len(names) == len(options)


# ---------------------------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------------------------


def test_dmrg_trial_cache_round_trip_and_h1_check(tmp_path):
    pytest.importorskip("pyblock3")
    h1 = hopping_matrix(6, 1.0)
    messages = []
    say = lambda message, **kw: messages.append(message)
    made = trials.load_or_make_dmrg_trial(h1, U, (3, 3), chi=8, sweeps=4, cache_dir=tmp_path, say=say)
    assert made.path == tmp_path / "L6_n3-3_t1_U4_chi8_sw4_seed0.npz" and made.path.exists()
    loaded = trials.load_or_make_dmrg_trial(h1, U, (3, 3), chi=8, sweeps=4, cache_dir=tmp_path, say=say)
    assert messages[-1].startswith("trial loaded from")
    for a, b in zip(made.tensors, loaded.tensors):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(made.charges, loaded.charges):
        np.testing.assert_array_equal(a, b)
    assert (loaded.davidson_energy, loaded.variational_energy) == (made.davidson_energy, made.variational_energy)
    exact = hf.ground_state(h1, U, 3, 3)[0]
    assert exact - 1e-9 < made.variational_energy < exact + 0.05

    with np.load(made.path) as data:  # a file written for another h1 under this name is refused
        arrays = dict(data)
    np.savez(made.path, **{**arrays, "h1": 2.0 * h1})
    with pytest.raises(ValueError, match="different h1"):
        trials.load_or_make_dmrg_trial(h1, U, (3, 3), chi=8, sweeps=4, cache_dir=tmp_path, say=say)
    old = {k: v for k, v in arrays.items() if k not in ("h1", "mps_energy", "sweep_energies", "dmrg_mpo")}
    np.savez(made.path, **old)  # the chain cache files of mps_cpmc_gpu: no h1, no variational energy
    loaded = trials.load_or_make_dmrg_trial(h1, U, (3, 3), chi=8, sweeps=4, cache_dir=tmp_path, say=say)
    assert loaded.variational_energy is None and loaded.davidson_energy == made.davidson_energy


@pytest.mark.parametrize("name", ["L8", "sq2x4oo"])
def test_make_trial_options(dmrg, name):
    d = dmrg(name)
    gamma = np.stack(one_rdm(d.tensors))
    trial, info = trials.make_trial(d.tensors, d.charges, NELEC)
    assert isinstance(trial, MpsTrial) and trial.label_width == 2 and info["sector_weight"] == 1.0
    assert info["rotated_trial"] is None
    np.testing.assert_allclose(np.asarray(trial.rdm1), gamma, atol=1e-12)

    projected, info = trials.make_trial(d.tensors, d.charges, NELEC, rotation=90.0)
    assert projected.label_width == 2 and 0.0 < info["sector_weight"] < 1.0
    assert info["sector_weight"] == pytest.approx(projected.sector_weight)
    rdm1 = np.asarray(projected.rdm1)  # the rotated trial's, before the projection: spin averaged at 90 degrees
    np.testing.assert_allclose(rdm1[0], rdm1[1], atol=1e-10)
    np.testing.assert_allclose(rdm1[0], gamma.mean(axis=0), atol=1e-10)

    as_is, info = trials.make_trial(d.tensors, d.charges, NELEC, rotation=90.0, rotated_trial="as_is",
                                    natural_rdm1="before")
    assert as_is.label_width == 1 and info["sector_weight"] is None  # N labels: the rotated MPS as it is
    assert (info["rotated_trial"], info["natural_rdm1"]) == ("as_is", "before")
    np.testing.assert_allclose(np.asarray(as_is.rdm1), gamma, atol=1e-12)
    with pytest.raises(ValueError):
        trials.make_trial(d.tensors, d.charges, NELEC, rotation=90.0, rotated_trial="other")
    with pytest.raises(ValueError):
        trials.make_trial(d.tensors, d.charges, NELEC, rotation=90.0, natural_rdm1="other")


# ---------------------------------------------------------------------------------------------
# The engine and the measurement block
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rotated", [None, "projected", "as_is"])
@pytest.mark.parametrize("name", CASES)
def test_engine_matches_exact_enumeration(dmrg, name, rotated):
    """Overlaps, local energies, <T|H|T> and the walkers' start, exact plan and walker bonds."""
    h1, d = CASES[name], dmrg(name)
    trial, _ = trials.make_trial(d.tensors, d.charges, NELEC, rotation=90.0 if rotated else 0.0,
                                 rotated_trial=rotated or "projected")
    run = _prepare(h1, trial, _params())
    walkers = _walkers(h1, trial)
    overlaps, energies = _engine_values(run, walkers)
    want_overlaps, want_energies, trial_energy = _exact(h1, trial.tensors, walkers)
    np.testing.assert_allclose(overlaps, want_overlaps, rtol=1e-9, atol=0)
    np.testing.assert_allclose(energies, want_energies, rtol=1e-9, atol=0)
    if rotated == "as_is":  # every sector of the rotated trial; H commutes with the rotation
        assert run.meas_ctx.trial_energy == pytest.approx(d.variational_energy, rel=1e-8)
    else:
        assert run.meas_ctx.trial_energy == pytest.approx(trial_energy, rel=1e-10)

    start = [(np.asarray(run.state.walkers[0][0]), np.asarray(run.state.walkers[1][0]))]
    start_overlap, start_energy, _ = _exact(h1, trial.tensors, start)
    np.testing.assert_allclose(np.asarray(run.state.overlaps), start_overlap[0], rtol=1e-9)
    assert float(run.state.e_estimate) == pytest.approx(start_energy[0], rel=1e-9)
    rdm1 = np.asarray(trial.rdm1)
    for s, w in enumerate(run.state.walkers):  # the natural orbitals of the trial's rdm1
        N = natural_orbitals(rdm1[s], NELEC[s])[0]
        np.testing.assert_allclose(np.asarray(w[0]) @ np.asarray(w[0]).T, N @ N.T, atol=1e-10)


@pytest.mark.parametrize("name,chi_w,rotated", [("L8", 4, None), ("L8a", None, "as_is"), ("sq2x4oo", 4, None),
                                                ("sq2x4op", 4, "projected")])
def test_mps_block_matches_trots_block(dmrg, name, chi_w, rotated):
    """driver.make_mps_block (det-R rescaled and gathered overlaps) against trot.prop.blocks.block on the same ops:
    the same energy, weights, walkers, overlaps, shift, nodes and RNG."""
    h1, d = CASES[name], dmrg(name)
    trial, _ = trials.make_trial(d.tensors, d.charges, NELEC, rotation=90.0 if rotated else 0.0,
                                 rotated_trial=rotated or "projected")
    params = _params(orbital_plan="adaptive", walker_channel_chi=chi_w, weight_floor=1e-8)
    run = _prepare(h1, trial, params)
    ca, cb = _stack(_walkers(h1, trial))
    weights = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, len(ca)))
    state = run.state._replace(walkers=(ca, cb), weights=weights,
                               overlaps=jax.jit(run.meas_ctx.kernels.overlaps)(ca, cb, run.meas_ctx.data()))
    kwargs = dict(sys=run.sys, params=params, ham_data=run.ham_data, trial_data=trial, trial_ops=run.ops.trial_ops,
                  meas_ops=run.ops.meas_ops, meas_ctx=run.meas_ctx, prop_ops=run.ops.prop_ops,
                  prop_ctx=run.prop_ctx)
    mps_block = driver.make_mps_block()
    got, got_obs = jax.jit(lambda s: mps_block(s, **kwargs))(state)
    want, want_obs = jax.jit(lambda s: blocks.block(s, **kwargs))(state)

    for key in ("energy", "weight"):
        np.testing.assert_allclose(float(got_obs.scalars[key]), float(want_obs.scalars[key]), rtol=1e-9)
    for x, y in zip(got.walkers, want.walkers):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-10)
    np.testing.assert_allclose(np.asarray(got.weights), np.asarray(want.weights), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(got.overlaps), np.asarray(want.overlaps), rtol=1e-8)
    np.testing.assert_allclose(float(got.e_estimate), float(want.e_estimate), rtol=1e-9)
    assert int(got.node_encounters) == int(want.node_encounters)
    np.testing.assert_array_equal(np.asarray(got.rng_key), np.asarray(want.rng_key))


@pytest.mark.parametrize("name", ["L8", "sq2x4oo"])
def test_block_form_htrial_gives_the_dense_energies(dmrg, name):
    """The block-form H|trial> (6x6 and larger), made once and cached next to the trial, through the htrial argument
    of make_mps_meas_ops_hubbard."""
    h1, d = CASES[name], dmrg(name)
    messages = []
    say = lambda message, **kw: messages.append(message)
    (tensors, labels), info = trials.load_or_make_htrial(d.path, h1, U, d.tensors, d.charges, d.variational_energy,
                                                        say=say)
    assert trials.htrial_cache_file(d.path).exists()
    _, again = trials.load_or_make_htrial(d.path, h1, U, d.tensors, d.charges, d.variational_energy, say=say)
    assert messages[-1].startswith("H|trial> loaded from") and again["trial_energy"] == info["trial_energy"]
    np.testing.assert_allclose(info["gamma"], np.stack(one_rdm(d.tensors)), atol=1e-12)

    trial, _ = trials.make_trial(d.tensors, d.charges, NELEC, rdm1=info["gamma"])
    params = _params(orbital_plan="adaptive", walker_channel_chi=4)
    dense = _prepare(h1, trial, params)
    blocked = _prepare(h1, trial, params, htrial=(tensors, labels, info["trial_energy"]))
    walkers = _walkers(h1, trial)
    np.testing.assert_allclose(_engine_values(blocked, walkers)[1], _engine_values(dense, walkers)[1], rtol=1e-10)
    assert blocked.meas_ctx.trial_energy == pytest.approx(dense.meas_ctx.trial_energy, rel=1e-12)


@pytest.mark.parametrize("name", ["L8", "sq2x4oo"])
def test_block_form_htrial_matches_the_dense_product(dmrg, name):
    """The block form of trot.gmps.trials: to_blocks / from_blocks round trip; trial_times_h_blocks has trial_times_h's
    labels and entries, and htrial_block_bytes is what it allocates; compress_blocks_qn gives compress_mps_qn's bonds,
    labels and state, and overlap_blocks the trial energy."""
    h1, d = CASES[name], dmrg(name)
    tensors, charges = [np.asarray(A) for A in d.tensors], tuple(label_array(q) for q in d.charges)
    blocks = trials.to_blocks(tensors, charges)
    for a, b in zip(trials.from_blocks(blocks, charges), tensors):
        np.testing.assert_array_equal(a, b)
    assert trials.overlap_blocks(blocks, blocks) == pytest.approx(_dense_overlap(tensors, tensors), rel=1e-13)

    W = hubbard_mpo_from_h1(h1, U)
    exact, exact_labels = trial_times_h(W, tensors, charges)
    product, product_labels = trials.trial_times_h_blocks(W, blocks, charges)
    for a, b in zip(product_labels, exact_labels):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(trials.from_blocks(product, product_labels), exact):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)
    assert trials.htrial_block_bytes(W, blocks, charges) == sum(X.nbytes for site in product for X in site.values())

    compressed, labels = trials.compress_blocks_qn(product, product_labels, consume=True)
    dense, dense_labels = compress_mps_qn(exact, exact_labels)
    new = trials.from_blocks(compressed, labels)
    assert [A.shape for A in new] == [A.shape for A in dense]
    for a, b in zip(labels, dense_labels):
        np.testing.assert_array_equal(a, b)
    nn, dd, nd = _dense_overlap(new, new), _dense_overlap(dense, dense), _dense_overlap(new, dense)
    assert nn == pytest.approx(dd, rel=1e-12) and 1.0 - nd ** 2 / (nn * dd) < 1e-12
    energy = trials.overlap_blocks(blocks, compressed) / trials.overlap_blocks(blocks, blocks)
    assert energy == pytest.approx(_dense_overlap(dense, tensors) / _dense_overlap(tensors, tensors), rel=1e-12)
    assert energy == pytest.approx(d.variational_energy, abs=1e-8)


# ---------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------


def _cli(out, cache, *extra):
    from trot.gmps import run_mps_cpmc

    argv = ["--U", "4", "--trial-chi", str(CHI), "--dmrg-sweeps", str(SWEEPS), "--trial-cache", str(cache),
            "--chi-w", "4", "--walkers", "8", "--eql", "5", "--blocks", "10", "--steps", "3", "--dt", "0.01",
            "--seed", "3", "--n-chunks", "1", "--compile-cache", "", "--out", str(out), *extra]
    return run_mps_cpmc.main(argv)


def _block_log(out):
    return [json.loads(line) for line in (out / "blocks.jsonl").read_text().splitlines()]


def test_lattice_flags_and_an_h1_file_are_the_same_run(tmp_path):
    from trot.gmps import run_mps_cpmc as cli

    np.save(tmp_path / "h1.npy", square_hopping_matrix(2, 4, 1.0))
    runs = [cli.parse_args(["--h1", str(tmp_path / "h1.npy"), "--trial-chi", "16"]),
            cli.parse_args(["--Lx", "2", "--Ly", "4", "--trial-chi", "16"])]
    lattices = [trials.describe_h1(cli.make_h1(args)) for args in runs]
    assert lattices[0] == lattices[1] and lattices[0].name == "sq2x4oo"
    assert cli.run_tag(runs[0], lattices[0]) == cli.run_tag(runs[1], lattices[1])
    with pytest.raises(SystemExit):
        cli.parse_args(["--L", "8", "--Lx", "2", "--Ly", "4", "--trial-chi", "16"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--trial-chi", "16"])


@pytest.mark.parametrize(
    "lattice,tag",
    [
        (["--L", "8"], "L8_T16_w4_NOplan_NOstart_s3"),
        (["--Lx", "2", "--Ly", "4"], "sq2x4oo_U4_T16_w4_NOplan_NOstart_s3"),
        (["--L", "8", "--trial-rotation", "90", "--rotated-trial", "as_is", "--natural-rdm1", "before"],
         "L8_T16_w4_NOplan_NOstart_rot90_asis_NObefore_s3"),
    ],
)
def test_cli_run_writes_the_production_outputs(tmp_path, cache, lattice, tag):
    """Log, block log, result record (as plot_cpmc_runs reads it), walker snapshots and the trial export."""
    record = _cli(tmp_path, cache, *lattice, "--save-walkers")
    assert record["tag"] == tag and np.isfinite(record["cpmc_energy"])
    rows = _block_log(tmp_path)
    assert [b["block"] for b in rows] == list(range(15)) and {b["tag"] for b in rows} == {tag}
    assert [b["phase"] for b in rows] == ["equilibration"] * 5 + ["sampling"] * 10
    energies = np.array([b["energy"] for b in rows])
    assert "CPMC energy" in (tmp_path / f"{tag}.log").read_text()

    from trot.gmps import plot_cpmc_runs

    (run,) = plot_cpmc_runs.load_runs(tmp_path / "results.jsonl")
    np.testing.assert_array_equal(run["energies"], energies)
    assert (run["config"]["L"], run["config"]["N_PROP"], run["config"]["DT"]) == (8, 3, 0.01)
    assert run["config"].get("LX") == (2 if "--Lx" in lattice else None)
    assert run["config"]["TRIAL_ROTATION"] == (90.0 if "--trial-rotation" in lattice else 0.0)

    with np.load(tmp_path / f"{tag}_walkers.npz") as z:
        assert z["up"].shape == z["dn"].shape == (16, 8, 8, 4)
        assert z["comb_index"].shape == z["pre_comb_weights"].shape == (15, 8)
        np.testing.assert_array_equal(z["energies"], energies)
        np.testing.assert_allclose(z["tau_blocks"], 0.03 * np.arange(1, 16))
        np.testing.assert_array_equal(z["start_up"], z["up"][0, 0])
        gram = np.einsum("kwia,kwic->kwac", z["up"][1:], z["up"][1:])  # orthonormal after every block's QR
        np.testing.assert_allclose(gram, np.broadcast_to(np.eye(4), gram.shape), atol=1e-10)
        assert json.loads(str(z["config"]))["tag"] == tag
    with np.load(record["trial_export"]) as t:
        assert {"e_dmrg", "gamma", "h1", "reference_up", "reference_dn", "start_up", "start_dn"} <= set(t.files)
        assert sum(k[0] == "T" and k[1:].isdigit() for k in t.files) == 8
        assert sum(k[0] == "q" and k[1:].isdigit() for k in t.files) == 9


def test_prepare_only_and_the_cached_htrial_run(tmp_path):
    """--prepare-only makes the trial and the block-form H|trial> and stops; a run with --cache-htrial then loads
    them and gives the blocks of a run with the dense H|trial>."""
    pytest.importorskip("pyblock3")
    cache = tmp_path / "cache"
    assert _cli(tmp_path / "prepared", cache, "--Lx", "2", "--Ly", "4", "--cache-htrial", "--prepare-only") is None
    trial_file = cache / f"sq2x4oo_n4-4_t1_U4_chi{CHI}_sw{SWEEPS}_tol1e-06_seed0.npz"
    assert trial_file.exists() and trials.htrial_cache_file(trial_file).exists()
    assert not (tmp_path / "prepared" / "results.jsonl").exists()

    _cli(tmp_path / "htrial", cache, "--Lx", "2", "--Ly", "4", "--cache-htrial")
    _cli(tmp_path / "dense", cache, "--Lx", "2", "--Ly", "4")
    assert "H|trial> loaded from" in (tmp_path / "htrial" / "sq2x4oo_U4_T16_w4_NOplan_NOstart_s3.log").read_text()
    for key in ("energy", "weight", "e_estimate"):
        np.testing.assert_allclose([b[key] for b in _block_log(tmp_path / "htrial")],
                                   [b[key] for b in _block_log(tmp_path / "dense")], rtol=1e-9)


def test_dmrg_reference_mode_writes_what_the_2d_plots_read(tmp_path):
    """--dmrg-reference (the former mps_cpmc_2d.py dmrg mode) runs trot.gmps.dmrg.dmrg_h1 with the lattice's defaults
    (qc MPO, plain schedule, tol 1e-6) and writes a kind="dmrg" record in results.jsonl and a dmrg_*.log that
    plot_cpmc_2d_runs.py reads back with the lattice's model. e_mps is variational, so above the exact energy; it is
    not close to it here: the 2x4 order x * Ly + y cuts all four rungs in the middle (exact bond up to 70^2), so chi 64
    still sits 0.06 above (job 7180801)."""
    pytest.importorskip("pyblock3")
    from trot.gmps import plot_cpmc_2d_runs, run_mps_cpmc
    from trot.gmps.dmrg import dmrg_h1

    record = run_mps_cpmc.main(["--Lx", "2", "--Ly", "4", "--U", "4", "--trial-chi", str(CHI), "--dmrg-sweeps",
                                str(SWEEPS), "--dmrg-reference", "--compile-cache", "", "--out", str(tmp_path)])
    assert record["kind"] == "dmrg" and record["tag"] == f"dmrg_sq2x4oo_U4_chi{CHI}_sw{SWEEPS}_seed0"
    assert (record["dmrg_mpo"], record["dmrg_schedule"], record["dmrg_tol"]) == ("qc", "plain", 1e-6)
    direct = dmrg_h1(CASES["sq2x4oo"], U, NELEC, chi=CHI, n_sweeps=SWEEPS, mpo="qc", schedule="plain", tol=1e-6)
    assert record["e_davidson"] == pytest.approx(direct[2], rel=1e-10)
    assert record["e_mps"] == pytest.approx(direct[4], rel=1e-10)
    assert record["e_mps"] > hf.ground_state(CASES["sq2x4oo"], U, *NELEC)[0] - 1e-9
    (from_results,) = plot_cpmc_2d_runs.references_from_results(tmp_path / "results.jsonl")
    from_log, reason = plot_cpmc_2d_runs.reference_from_log(tmp_path / f"{record['tag']}.log")
    assert reason == ""
    for ref in (from_results, from_log):
        model = tuple(ref[k] for k in ("LX", "LY", "BOUNDARY_X", "BOUNDARY_Y", "N_UP", "N_DN", "U"))
        assert model == (2, 4, "open", "open", 4, 4, U)
        assert ref["energy"] == pytest.approx(record["e_mps"], rel=1e-12) and ref["chi"] == max(record["bond_dims"])
