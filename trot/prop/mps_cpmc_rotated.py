"""CPMC propagation for a RotatedMpsTrial (an MPS without definite (N_up, N_dn)).

The step is trot.prop.mps_cpmc's (mps_cpmc_step: one-body half steps, a diagonal HS sweep and
population control) with dense contractions in place of the charge-blocked ones. dense_sweep is
fast_sweep with the walker combined into one d=4 MPS and dense left/right environments; the
field choice, floors, node count and weights follow fast_sweep line by line. propagator="slow"
runs trot.prop.cpmc_slow on the dense overlap.
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp

from trot.meas.mps_rotated import check_rotated_meas_ctx
from trot.prop.cpmc_slow import cpmc_step as cpmc_slow_step
from trot.prop.hubbard_cpmc_ops import make_hubbard_cpmc_ops
from trot.prop.mps_cpmc import constrain_ratio, make_prop_ops, mps_cpmc_step
from trot.prop.types import PropOps
from trot.trial.mps import MpsWalkerPlan
from trot.trial.mps_rotation import dense_walker_mps, rotated_overlap


def dense_sweep(ca, cb, randoms, hs, weight_floor, *, plan, trial_tensors):
    """One-conversion diagonal HS sweep of one walker against a trial without bond labels.

    Returns (ca, cb, overlap_in, overlap_out, weight_factor, node_count), as fast_sweep.
    """
    walker, prefactor = dense_walker_mps((ca, cb), plan)
    right = [jnp.ones((1, 1))]  # right[s][i, a]: walker index i, trial index a at bond s
    for W, T in zip(reversed(walker), reversed(trial_tensors)):
        right.append(jnp.einsum("ipj,apb,jb->ia", W, T, right[-1]))
    right = right[::-1]
    overlap_in = prefactor * right[0][0, 0]

    left = jnp.ones((1, 1))
    overlap, log_weight = overlap_in, jnp.zeros(())
    nodes = jnp.zeros((), jnp.int64)
    diagonal = jnp.stack((jnp.ones(2), hs[:, 0], hs[:, 1], hs[:, 0] * hs[:, 1]), axis=1)

    for site, (W, T) in enumerate(zip(walker, trial_tensors)):
        local = jnp.einsum("ia,ipj,apb->pjb", left, W, T)  # left environment past the site, per p
        marginal = jnp.einsum("pjb,jb->p", local, right[site + 1])

        proposed = prefactor * (diagonal @ marginal)
        ratios = constrain_ratio(proposed / overlap, weight_floor)
        nodes += jnp.sum(ratios <= 0.0, dtype=jnp.int64)
        probabilities = 0.5 * ratios
        norm = probabilities.sum() + 1.0e-13
        field = jnp.where(randoms[site] < probabilities[0] / norm, 0, 1)
        overlap = proposed[field]
        log_weight += jnp.log(norm)

        ca = ca.at[site].multiply(hs[field, 0])
        cb = cb.at[site].multiply(hs[field, 1])
        left = jnp.einsum("p,pjb->jb", diagonal[field], local)
    return ca, cb, overlap_in, overlap, jnp.exp(log_weight), nodes


def make_rotated_prop_ops(
    ham_data, sys, plan: MpsWalkerPlan, *, propagator: str = "fast"
) -> PropOps:
    """PropOps for trial_data = RotatedMpsTrial and meas_ctx = RotatedMeasCtx.

    init_prop_state and build_prop_ctx are trot.prop.mps_cpmc.make_prop_ops's (walker start from
    trial_ops.get_rdm1 or params.walker_start == "rhf", strong-typed scalars). The fast step is
    mps_cpmc_step driven by dense_sweep; the slow step is trot's cpmc_slow on the dense overlap.
    """
    base = make_prop_ops(ham_data, sys, plan, propagator=propagator)
    cpmc_ops = make_hubbard_cpmc_ops(ham_data, sys.walker_kind)

    def step(state, *, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ctx):
        check_rotated_meas_ctx(meas_ctx, plan, trial_data)
        return mps_cpmc_step(
            state,
            params=params,
            prop_ctx=prop_ctx,
            cpmc_ops=cpmc_ops,
            sweep_fn=partial(dense_sweep, plan=plan, trial_tensors=tuple(trial_data.tensors)),
            overlap_fn=partial(rotated_overlap, plan=plan),
            overlap_arg=trial_data,
        )

    def slow_step(state, *, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ctx):
        check_rotated_meas_ctx(meas_ctx, plan, trial_data)
        return cpmc_slow_step(
            state,
            params=params,
            trial_data=trial_data,
            meas_ops=meas_ops,
            cpmc_ops=cpmc_ops,
            prop_ctx=prop_ctx,
        )

    return PropOps(
        init_prop_state=base.init_prop_state,
        build_prop_ctx=base.build_prop_ctx,
        step=step if propagator == "fast" else slow_step,
    )
