"""Tests for trot/gmps/uhf_cpmc.py, the UHF trial's fast-update ops for trot's CPMC (CPU or GPU backend).

The ops must equal trot's GHF ops on the spin-block-diagonal trial (whose formulas they restrict),
follow a recomputed overlap and Green's function through a whole field sweep, and make
trot.prop.cpmc (fast updates) reproduce trot.prop.cpmc_slow (the propagator of
uhf_trial_cpmc.ipynb) block by block: both draw the same random numbers and use them the same way,
so they pick the same fields and agree to rounding. The runs are the notebook's chain (L = 100,
U = 8, dt = 0.005, 50 steps per block, weight floor 1e-8) with 16 walkers, and a 12-site chain at
U = 4, starting from the UHF determinant as uhf_trial_cpmc_gpu.py does. The largest deviations are
printed (pytest -s); the bound is 1e-9, relative.

The last test checks hf_trial_cpmc_gpu.py's trot-native path (trot's GHF trial ops and the HamHubbard
energy of make_ghf_meas_ops_hubbard, no code of ours) against that validated path, block by block, with
the UHF trial on the notebook's chain and the RHF trial on the 12-site one.

    python -m pytest -s tests/test_gmps_uhf_fast_update.py
    sbatch --export=ALL,TARGET=pytest run_mps_gpu.sh -x -q -s /mnt/home/fnappi/trot_mps/tests/test_gmps_uhf_fast_update.py
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

jax.config.update("jax_enable_x64", True)

from trot.core.system import System
from trot.driver import make_run_blocks
from trot.gmps import uhf_cpmc as uc
from trot.ham.chol import HamChol
from trot.ham.hubbard import HamHubbard
from trot.meas.ghf import make_ghf_meas_ops_hubbard
from trot.meas.uhf import make_uhf_meas_ops
from trot.prop import blocks, cpmc, cpmc_slow
from trot.prop.types import QmcParams
from trot.trial import ghf
from trot.trial.uhf import UhfTrial, make_uhf_trial_ops, overlap_u

TOL = 1e-9
CHAINS = {  # L, electrons per spin, U, dt, blocks
    "notebook-L100-U8": (100, 50, 8.0, 0.005, 3),
    "L12-U4": (12, 6, 4.0, 0.01, 6),
}
N_WALKERS, N_PROP, WEIGHT_FLOOR, SEED = 16, 50, 1.0e-8, 7


def rel(a, b):
    """Largest deviation of a from b relative to the largest entry of b."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _chain(L):
    h1 = np.zeros((L, L))
    i = np.arange(L - 1)
    h1[i, i + 1] = h1[i + 1, i] = -1.0
    return h1


def _scf(h1, u, n, trial_kind):
    """The trial orbitals (up, down) of uhf_trial_cpmc_gpu.py and hf_trial_cpmc_gpu.py, and the SCF energy."""
    if trial_kind == "rhf":
        C, e, _, _ = uc.rhf_scf(h1, u, n)
        return C, C, e
    Ca, Cb, e, _, _ = uc.uhf_scf(h1, u, n, n)
    return Ca, Cb, e


def _setup(L, n, u, dt, n_blocks, prop, trial_kind="uhf"):
    # the objects of uhf_trial_cpmc_gpu.py for an open chain at t = 1
    h1 = _chain(L)
    Ca, Cb, _ = _scf(h1, u, n, trial_kind)
    ham = HamHubbard(h1=jnp.asarray(h1), u=u)
    onsite = np.zeros((L, L, L))
    onsite[np.arange(L), np.arange(L), np.arange(L)] = np.sqrt(u)
    ham_meas = HamChol(h0=jnp.zeros(()), h1=jnp.asarray(h1), chol=jnp.asarray(onsite))
    system = System(norb=L, nelec=(n, n), walker_kind="unrestricted")
    trial = UhfTrial(mo_coeff_a=jnp.asarray(Ca), mo_coeff_b=jnp.asarray(Cb))
    meas_ops = make_uhf_meas_ops(system)
    if prop == "fast":
        trial_ops = uc.make_uhf_cpmc_trial_ops(system)
        prop_ops = cpmc.make_prop_ops(ham, system.walker_kind, trial_ops)
    else:
        trial_ops = make_uhf_trial_ops(system)
        prop_ops = cpmc_slow.make_prop_ops(ham, system.walker_kind)
    params = QmcParams(dt=dt, n_walkers=N_WALKERS, n_prop_steps=N_PROP, n_blocks=n_blocks, n_eql_blocks=1,
                       weight_floor=WEIGHT_FLOOR, seed=SEED)
    run_blocks = make_run_blocks(block_fn=blocks.block, sys=system, params=params,
                                 trial_ops=trial_ops, meas_ops=meas_ops, prop_ops=prop_ops)
    ctx = dict(ham_data=ham_meas, trial_data=trial, meas_ctx=meas_ops.build_meas_ctx(ham_meas, trial),
               prop_ctx=prop_ops.build_prop_ctx(ham, trial_ops.get_rdm1(trial), params))
    state = prop_ops.init_prop_state(sys=system, ham_data=ham_meas, trial_ops=trial_ops, trial_data=trial,
                                     meas_ops=meas_ops, params=params)
    return SimpleNamespace(run_blocks=run_blocks, ctx=ctx, state=state, trial=trial, L=L)


def _setup_trot(L, n, u, dt, n_blocks, trial_kind):
    # the objects of hf_trial_cpmc_gpu.py: trot's GHF trial [[C_up, 0], [0, C_dn]] with its own fast-update ops,
    # the energy of make_ghf_meas_ops_hubbard straight from the HamHubbard, trot.prop.cpmc
    h1 = _chain(L)
    Ca, Cb, e_scf = _scf(h1, u, n, trial_kind)
    ham = HamHubbard(h1=jnp.asarray(h1), u=u)
    system = System(norb=L, nelec=(n, n), walker_kind="unrestricted")
    trial = ghf.GhfTrial(mo_coeff=jnp.asarray(np.block([[Ca, np.zeros_like(Cb)], [np.zeros_like(Ca), Cb]])))
    trial_ops = ghf.make_ghf_trial_ops(system)
    meas_ops = make_ghf_meas_ops_hubbard(system)
    prop_ops = cpmc.make_prop_ops(ham, system.walker_kind, trial_ops)
    params = QmcParams(dt=dt, n_walkers=N_WALKERS, n_prop_steps=N_PROP, n_blocks=n_blocks, n_eql_blocks=1,
                       weight_floor=WEIGHT_FLOOR, seed=SEED)
    run_blocks = make_run_blocks(block_fn=blocks.block, sys=system, params=params,
                                 trial_ops=trial_ops, meas_ops=meas_ops, prop_ops=prop_ops)
    ctx = dict(ham_data=ham, trial_data=trial, meas_ctx=meas_ops.build_meas_ctx(ham, trial),
               prop_ctx=prop_ops.build_prop_ctx(ham, trial_ops.get_rdm1(trial), params))
    state = prop_ops.init_prop_state(sys=system, ham_data=ham, trial_ops=trial_ops, trial_data=trial,
                                     meas_ops=meas_ops, params=params)
    return SimpleNamespace(run_blocks=run_blocks, ctx=ctx, state=state, trial=trial, L=L, e_scf=e_scf)


def _projectors(walkers):
    """W W^T per walker and spin: the walkers are orthonormal after each block, and the two paths may start
    from different orthonormal bases of the same occupied space (eigh of a projector), so compare these."""
    return [np.einsum("wia,wja->wij", np.asarray(w), np.asarray(w)) for w in walkers]


@pytest.fixture(scope="module", params=list(CHAINS))
def runs(request):
    """The same blocks with the slow and the fast propagator, from the same state and random key."""
    L, n, u, dt, n_blocks = CHAINS[request.param]
    out = {}
    for prop in ("slow", "fast"):
        s = _setup(L, n, u, dt, n_blocks, prop)
        state, scalars, _ = s.run_blocks(s.state, **s.ctx, n_blocks=n_blocks)
        out[prop] = SimpleNamespace(setup=s, state=state, scalars={k: np.asarray(v) for k, v in scalars.items()})
    print(f"\n{request.param} on {jax.default_backend()}: {n_blocks} blocks x {N_PROP} steps, {N_WALKERS} walkers")
    return request.param, out


def test_fast_blocks_match_slow_blocks(runs):
    name, out = runs
    slow, fast = out["slow"], out["fast"]
    errs = dict(
        block_energy=rel(fast.scalars["energy"], slow.scalars["energy"]),
        block_weight=rel(fast.scalars["weight"], slow.scalars["weight"]),
        walkers=max(rel(f, s) for f, s in zip(fast.state.walkers, slow.state.walkers)),
        weights=rel(fast.state.weights, slow.state.weights),
        overlaps=rel(fast.state.overlaps, slow.state.overlaps),
    )
    print(f"{name}: fast vs slow, largest relative deviations " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
          + f"; node encounters {int(fast.state.node_encounters)} vs {int(slow.state.node_encounters)}")
    assert int(fast.state.node_encounters) == int(slow.state.node_encounters)
    assert max(errs.values()) <= TOL, errs


def test_ops_equal_ghf_ops_on_the_block_diagonal_trial(runs):
    # on the fast run's final (propagated, reconfigured) walkers
    name, out = runs
    s, walkers = out["fast"].setup, out["fast"].state.walkers
    L, trial = s.L, s.trial
    Ca, Cb = np.asarray(trial.mo_coeff_a), np.asarray(trial.mo_coeff_b)
    gtrial = ghf.GhfTrial(mo_coeff=jnp.asarray(np.block([[Ca, np.zeros_like(Cb)], [np.zeros_like(Ca), Cb]])))
    hs = s.ctx["prop_ctx"].hs_constant

    ov = jax.vmap(overlap_u, in_axes=(0, None))(walkers, trial)
    ov_g = jax.vmap(ghf.overlap_u, in_axes=(0, None))(walkers, gtrial)
    G = jax.vmap(uc.calc_green, in_axes=(0, None))(walkers, trial)                # (nw, 2, L, L)
    G_g = jax.vmap(ghf.calc_green_u, in_axes=(0, None))(walkers, gtrial)          # (nw, 2L, 2L)
    errs = dict(overlap=rel(ov, ov_g), green=max(rel(G[:, 0], G_g[:, :L, :L]), rel(G[:, 1], G_g[:, L:, L:])),
                ratio=0.0, update=0.0)
    off_diagonal = float(np.max(np.abs(np.asarray(G_g)[:, :L, L:]))) + float(np.max(np.abs(np.asarray(G_g)[:, L:, :L])))
    for x in sorted({0, 1, L // 2, L - 2, L - 1}):
        idx = jnp.array([[0, x], [1, x]], dtype=jnp.int32)
        for field in (0, 1):
            upd = hs[field] - 1.0
            r = jax.vmap(uc.calc_overlap_ratio, in_axes=(0, None, None))(G, idx, upd)
            r_g = jax.vmap(ghf.calc_overlap_ratio, in_axes=(0, None, None))(G_g, idx, upd)
            Gn = np.asarray(jax.vmap(uc.update_green, in_axes=(0, None, None))(G, idx, upd))
            Gn_g = np.asarray(jax.vmap(ghf.update_green, in_axes=(0, None, None))(G_g, idx, upd))
            errs["ratio"] = max(errs["ratio"], rel(r, r_g))
            errs["update"] = max(errs["update"], rel(Gn[:, 0], Gn_g[:, :L, :L]), rel(Gn[:, 1], Gn_g[:, L:, L:]))
    print(f"{name}: UHF ops vs GHF ops, largest relative deviations " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
          + f"; GHF off-diagonal spin blocks {off_diagonal:.1e}")
    assert off_diagonal <= TOL * float(np.max(np.abs(np.asarray(G_g))))
    assert max(errs.values()) <= TOL, errs


def test_field_sweep_follows_recomputed_overlaps(runs):
    # one field sweep over every site, as in a CPMC step: at each site the chosen ratio against the
    # recomputed overlap ratio, and at the end the updated G against a recomputed one
    name, out = runs
    s, walkers = out["fast"].setup, out["fast"].state.walkers
    L, trial = s.L, s.trial
    hs = s.ctx["prop_ctx"].hs_constant
    rns = jax.random.uniform(jax.random.PRNGKey(0), (N_WALKERS, L))

    def sweep(walker, rn):
        def site(carry, x):
            (wu, wd), g, ov, worst = carry
            idx = jnp.stack([jnp.array([0, x]), jnp.array([1, x])]).astype(jnp.int32)
            r0 = uc.calc_overlap_ratio(g, idx, hs[0] - 1.0)
            r1 = uc.calc_overlap_ratio(g, idx, hs[1] - 1.0)
            pick0 = rn[x] < jnp.abs(r0) / (jnp.abs(r0) + jnp.abs(r1))
            c = jnp.where(pick0, hs[0], hs[1])
            wu, wd = wu.at[x].mul(c[0]), wd.at[x].mul(c[1])
            ov_new = overlap_u((wu, wd), trial)
            exact = ov_new / ov
            worst = jnp.maximum(worst, jnp.abs(jnp.where(pick0, r0, r1) - exact) / jnp.abs(exact))
            return ((wu, wd), uc.update_green(g, idx, c - 1.0), ov_new, worst), None

        carry = (walker, uc.calc_green(walker, trial), overlap_u(walker, trial), jnp.asarray(0.0))
        (w_end, g_end, _, worst), _ = lax.scan(site, carry, jnp.arange(L))
        g_exact = uc.calc_green(w_end, trial)
        return worst, jnp.max(jnp.abs(g_end - g_exact)) / jnp.max(jnp.abs(g_exact))

    ratio_err, green_err = jax.vmap(sweep)(walkers, rns)
    ratio_err, green_err = float(jnp.max(ratio_err)), float(jnp.max(green_err))
    print(f"{name}: {L}-site field sweep with updates vs recomputation, largest relative deviations"
          f" ratio {ratio_err:.1e}, G at the end {green_err:.1e}")
    assert ratio_err <= TOL and green_err <= TOL


TROT_CASES = {"notebook-L100-U8-uhf": ("notebook-L100-U8", "uhf"), "L12-U4-rhf": ("L12-U4", "rhf")}


@pytest.fixture(scope="module", params=list(TROT_CASES))
def trot_runs(request):
    """The same blocks through the validated path (UhfTrial, uhf_cpmc ops, HamChol energy) and the trot-native one."""
    chain, trial_kind = TROT_CASES[request.param]
    L, n, u, dt, n_blocks = CHAINS[chain]
    ours = _setup(L, n, u, dt, n_blocks, "fast", trial_kind)
    native = _setup_trot(L, n, u, dt, n_blocks, trial_kind)
    out = {}
    for name, s in (("ours", ours), ("trot", native)):
        state, scalars, _ = s.run_blocks(s.state, **s.ctx, n_blocks=n_blocks)
        out[name] = SimpleNamespace(setup=s, state0=s.state, state=state,
                                    scalars={k: np.asarray(v) for k, v in scalars.items()})
    print(f"\n{request.param} on {jax.default_backend()}: {n_blocks} blocks x {N_PROP} steps, {N_WALKERS} walkers")
    return request.param, out


def test_trot_native_path_matches_validated_path(trot_runs):
    name, out = trot_runs
    ours, trot = out["ours"], out["trot"]
    e0 = float(trot.state0.e_estimate)
    errs = dict(
        tau0_energy_vs_scf=abs(e0 - trot.setup.e_scf) / abs(trot.setup.e_scf),
        tau0_energy_vs_ours=abs(e0 - float(ours.state0.e_estimate)) / abs(e0),
        block_energy=rel(trot.scalars["energy"], ours.scalars["energy"]),
        block_weight=rel(trot.scalars["weight"], ours.scalars["weight"]),
        walker_projectors=max(float(np.max(np.abs(p - q))) for p, q in
                              zip(_projectors(trot.state.walkers), _projectors(ours.state.walkers))),
        weights=rel(trot.state.weights, ours.state.weights),
        abs_overlaps=rel(np.abs(trot.state.overlaps), np.abs(ours.state.overlaps)),
    )
    print(f"{name}: trot-native vs validated path, largest relative deviations "
          + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
          + f"; node encounters {int(trot.state.node_encounters)} vs {int(ours.state.node_encounters)}")
    assert int(trot.state.node_encounters) == int(ours.state.node_encounters)
    assert max(errs.values()) <= TOL, errs
