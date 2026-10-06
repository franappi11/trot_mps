"""CPMC propagation for MPS trials (trot.trial.mps) with locally cached HS updates.

One step: a one-body half step, a diagonal HS sweep over the sites, a second one-body half step, and population
control, exactly as trot's CPMC. The sweep converts each walker once and caches its right environments, so both
field proposals at every site cost one environment contraction (trot.gmps.engine.field_sweep, batched over walkers).

Semantics: the overlap after each site is the recomputed overlap of the chosen proposal (fresh-overlap semantics,
as in trot.prop.cpmc_slow), and ratios at or below weight_floor are zeroed as in trot.prop.cpmc. trot.prop.cpmc
instead accumulates overlaps *= floored ratio, so when *both* proposals at a site are floored it stores a zero
overlap and kills the walker, while this step (and cpmc_slow) keeps it with weight times 1e-13. The trajectories
are identical whenever that event does not occur.

make_prop_ops(ham_data, sys, plan) is the factory, as trot.prop.cpmc.make_prop_ops: prop_ctx is trot's
HubbardCpmcCtx (exp(-dt K/2) and the HS factors) and the trial blocks come from meas_ctx (trot.meas.mps), which
also names the plan's engine. block is the MPS measurement block for trot.driver.run_qmc: trot.prop.blocks.block
with 2 n + 1 walker conversions per block of n steps (blocks.block, which also works, needs 2 n + 4).
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from jax import lax

from trot import walkers as wk
from trot.gmps import engine
from trot.meas.mps import MpsMeasCtx, check_meas_ctx
from trot.prop.blocks import BlockObs
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx
from trot.prop.types import PropOps, PropState
from trot.sharding import shard_prop_state
from trot.trial.mps import MpsWalkerPlan, rhf_orbitals
from trot.walkers import init_walkers


def _start_values(meas_ctx: MpsMeasCtx):
    """Jitted overlaps and local energies of a batch of given starting walkers (cached on the plan)."""
    key = ("start_values",) + tuple(meas_ctx.key[1:4])
    fn = meas_ctx.plan.caches.get(key)
    if fn is None:
        kernels = meas_ctx.kernels

        @partial(jax.jit, static_argnames=("n_chunks",))
        def fn(ca, cb, data, n_chunks):
            overlaps = engine.chunked(lambda a, b: kernels.overlaps(a, b, data), n_chunks, ca, cb)
            energies = engine.chunked(lambda a, b: kernels.energies(a, b, data), n_chunks, ca, cb)
            return overlaps, energies

        meas_ctx.plan.caches[key] = fn
    return fn


def init_prop_state(*, sys, ham_data, trial_ops, trial_data, meas_ops, params, meas_ctx=None, initial_walkers=None,
                    initial_e_estimate=None, rdm1=None, mesh=None) -> PropState:
    """trot.prop.cpmc.init_prop_state for MPS trials.

    Walkers start from the natural orbitals of trial_ops.get_rdm1(trial_data), or of the free-fermion determinant
    of ham_data.h1 when params.walker_start == "rhf" (unless rdm1 or initial_walkers are given). init_walkers
    repeats one determinant, so it is converted once (the engine's jitted probe) and its overlap and local energy
    are broadcast: a batched program over the walkers costs as much compile time as the block itself (~10 min at
    L=100). Given initial_walkers are converted as a batch. The scalar leaves have strong dtypes, so the jitted
    block compiles once.
    """
    start = getattr(params, "walker_start", "natural")
    if start not in ("natural", "rhf"):
        raise ValueError(f"walker_start must be 'natural' or 'rhf', got {start!r}")
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    given = initial_walkers is not None
    if not given:
        if rdm1 is None and start == "rhf":
            Ra, Rb = rhf_orbitals(ham_data, sys.nelec)
            rdm1 = jnp.asarray(np.stack([Ra @ Ra.T, Rb @ Rb.T]))
        if rdm1 is None:
            rdm1 = trial_ops.get_rdm1(trial_data)
        initial_walkers = init_walkers(sys=sys, rdm1=rdm1, n_walkers=params.n_walkers)
    ca, cb = (jnp.real(w) for w in initial_walkers)
    n = ca.shape[0]
    data = meas_ctx.data()
    if given:
        overlaps, energies = _start_values(meas_ctx)(ca, cb, data, engine.divisor_at_least(n, params.n_chunks))
    else:
        values = meas_ctx.kernels.jit_probe(ca[0], cb[0], data)
        overlaps = jnp.full((n,), values["overlap"], dtype=jnp.result_type(float))
        energies = values["energy"]
    e_est = jnp.mean(energies) if initial_e_estimate is None else jnp.asarray(initial_e_estimate)
    e_est = e_est.astype(jnp.result_type(float))
    state = PropState((ca, cb), jnp.ones((n,)), overlaps, jax.random.PRNGKey(int(params.seed)), e_est, e_est,
                      jnp.zeros((), jnp.int64))
    return shard_prop_state(state, mesh)


def make_prop_ops(ham_data, sys, plan: MpsWalkerPlan) -> PropOps:
    """PropOps for MPS-CPMC with trial_data = MpsTrial and meas_ctx = MpsMeasCtx (make_mps_meas_ops_hubbard on the
    same plan). step is two half steps of the plan's engine over the whole walker batch (params.n_chunks chunks)."""
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError("MPS-CPMC needs walker_kind='unrestricted'")

    def build_prop_ctx(ham, _rdm1, params):
        return _build_prop_ctx(ham, params.dt)

    def step(state, *, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ctx):
        check_meas_ctx(meas_ctx, plan, trial_data)
        n_chunks = engine.divisor_at_least(wk.n_walkers(state.walkers), params.n_chunks)
        half = engine.make_half_step(meas_ctx.kernels, params, n_chunks)
        data = meas_ctx.data(prop_ctx)
        return half(half(state, 0, data), 1, data)

    return PropOps(init_prop_state=init_prop_state, build_prop_ctx=build_prop_ctx, step=step)


# ---------------------------------------------------------------------------------------------
# The MPS measurement block
# ---------------------------------------------------------------------------------------------


def make_block(record: Callable | None = None):
    """trot.prop.blocks.block for MPS trials, with 2 n + 1 walker conversions per block of n steps.

    The n steps are a scan over 2 n half steps with one conversion call site (engine.make_half_step). The walkers are
    then orthonormalised (the engine's batch_qr) and their overlaps rescaled by det R instead of reconverted; the
    energy is one more conversion; after the comb the overlaps are gathered. Outlier clipping, the e_estimate EMA,
    the comb and the RNG use are those of blocks.block, so both blocks agree to rounding (the conversion QRs its
    input, so the rescaled and gathered overlaps equal reconverted ones).

    record: optional host callable(up, dn, pre_comb_weights, comb_index, energy, weight, e_estimate, node_encounters)
      called after every block with the new walkers and the comb's input (trot.gmps.driver.WalkerSnapshots).
    """
    record_spec = jax.ShapeDtypeStruct((), jnp.int32)

    def mps_block(state: PropState, *, sys, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ops,
                  prop_ctx, sr_fn=wk.stochastic_reconfiguration, observable_names=()):
        if observable_names:
            raise ValueError("the MPS block measures the energy only (observable_names must be empty)")
        if not isinstance(meas_ctx, MpsMeasCtx):
            raise ValueError("the MPS block needs meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)")
        kernels, data = meas_ctx.kernels, meas_ctx.data(prop_ctx)
        n = wk.n_walkers(state.walkers)
        n_chunks = engine.divisor_at_least(n, params.n_chunks)
        half_step = engine.make_half_step(kernels, params, n_chunks)
        state, _ = lax.scan(lambda s, i: (half_step(s, i, data), None), state, jnp.arange(2 * params.n_prop_steps))
        qu, du = kernels.batch_qr(state.walkers[0])
        qd, dd = kernels.batch_qr(state.walkers[1])
        overlaps = state.overlaps / (du * dd)

        e_samples = engine.chunked(lambda a, b: kernels.energies(a, b, data), n_chunks, qu, qd)
        thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
        e_ref = state.e_estimate
        is_nan = ~jnp.isfinite(e_samples)
        e_samples = jnp.where(is_nan | (jnp.abs(e_samples - e_ref) > thresh), e_ref, e_samples)
        weights = jnp.where(is_nan, 0.0, state.weights)
        w_sum = jnp.sum(weights)
        w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
        e_block = jnp.sum(weights * e_samples) / w_sum_safe
        e_block = jnp.where(w_sum == 0, e_ref, e_block)
        alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
        e_estimate = (1.0 - alpha) * state.e_estimate + alpha * e_block

        key, subkey = jax.random.split(state.rng_key)
        zeta = jax.random.uniform(subkey)
        if sr_fn is wk.stochastic_reconfiguration:
            idx = wk._sr_indices(weights, zeta, n)
            average = jnp.cumsum(jnp.abs(weights))[-1] / n
            walkers, new_overlaps = (qu[idx], qd[idx]), overlaps[idx]
            new_weights = jnp.full((n,), average, weights.dtype)
        else:  # a sharded comb (trot.driver with a mesh): no indices, so reconvert for the overlaps
            idx = jnp.full((n,), -1, jnp.int32)
            walkers, new_weights = sr_fn((qu, qd), weights, zeta, sys.walker_kind)
            new_overlaps = engine.chunked(lambda a, b: kernels.overlaps(a, b, data), n_chunks, *walkers)
        state = PropState(walkers, new_weights, new_overlaps, key, state.pop_control_ene_shift, e_estimate,
                          state.node_encounters)
        if record is not None:
            jax.experimental.io_callback(record, record_spec, walkers[0], walkers[1], weights, idx.astype(jnp.int32),
                                         e_block, w_sum, e_estimate, state.node_encounters, ordered=True)
        return state, BlockObs(scalars={"energy": e_block, "weight": w_sum}, observables={})

    return mps_block


block = make_block()
