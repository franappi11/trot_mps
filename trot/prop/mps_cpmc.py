"""CPMC propagation for MPS trials (trot.trial.mps) with locally cached HS updates.

One step: a one-body half step, a diagonal HS sweep over the sites, a second one-body half step,
and population control, exactly as trot's CPMC. The sweep converts each walker once and caches
left/right environments, so both field proposals at every site cost one environment contraction.

Semantics: the overlap after each site is the recomputed overlap of the chosen proposal
(fresh-overlap semantics, as in trot.prop.cpmc_slow), and ratios at or below weight_floor are
zeroed as in trot.prop.cpmc. trot.prop.cpmc instead accumulates overlaps *= floored ratio, so
when *both* proposals at a site are floored it stores a zero overlap and kills the walker, while
this step (and cpmc_slow) keeps it with weight times 1e-13. The trajectories are identical
whenever that event does not occur.

make_prop_ops(ham_data, sys, plan) is the trot-native factory: the trial comes from trial_data,
and the padded trial blocks from meas_ctx (built by the MPS meas_ops). make_fast_prop_ops and
make_fast_sweep are the legacy closure-based API of trot/gmps/mps_cpmc_new.py; they ignore
trial_data and run the same cores.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from trot import walkers as wk
from trot.meas.mps import check_meas_ctx
from trot.prop.cpmc import init_prop_state
from trot.prop.cpmc_slow import cpmc_step as cpmc_slow_step
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx, make_hubbard_cpmc_ops
from trot.prop.types import PropOps, PropState
from trot.trial.mps import (
    MpsWalkerPlan,
    contraction_layout,
    convert_walker,
    extract_channel_blocks,
    overlap_from_blocks,
    rhf_orbitals,
)


def right_environments(walker_blocks, trial_blocks, plan):
    right = [jnp.ones((1, plan["walker_pad"][-1], plan["trial_pad"][-1]))]
    for site in range(plan["n"] - 1, -1, -1):
        layout = plan["sites"][site]
        wb, tb = walker_blocks[site], trial_blocks[site]
        following = right[-1][layout["dst"]]
        temp = jnp.einsum("tij,tjk->tik", wb, following)
        contributions = jnp.einsum("tik,tlk->til", temp, tb)
        right.append(
            jax.ops.segment_sum(
                contributions, layout["src"], num_segments=len(plan["shared"][site])
            )
        )
    return tuple(reversed(right))


def constrain_ratio(ratio, weight_floor):
    """trot's cpmc_step rule: zero every overlap ratio at or below the floor.

    This is also the constrained-path condition (a sign change gives ratio <= 0),
    so it cannot be dropped; weight_floor=0 keeps only the constraint.
    """
    return jnp.where(ratio <= weight_floor, 0.0, ratio)


def fast_sweep(
    ca, cb, randoms, hs, weight_floor, *, convert, channel_maps, trial_blocks, contraction
):
    """One-conversion diagonal HS sweep of one walker against fixed trial blocks.

    Returns (ca, cb, overlap_in, overlap_out, weight_factor, node_count).
    """
    alpha, beta, prefactor = convert(ca, cb)
    walker_blocks = extract_channel_blocks(alpha, beta, channel_maps)
    right = right_environments(walker_blocks, trial_blocks, contraction)
    overlap_in = prefactor * right[0][0, 0, 0]

    left = jnp.ones((1, contraction["walker_pad"][0], contraction["trial_pad"][0]))
    overlap, log_weight = overlap_in, jnp.zeros(())
    nodes = jnp.zeros((), jnp.int64)
    diagonal = jnp.stack((jnp.ones(2), hs[:, 0], hs[:, 1], hs[:, 0] * hs[:, 1]), axis=1)

    for site, (wb, tb, layout) in enumerate(zip(walker_blocks, trial_blocks, contraction["sites"])):
        temp = jnp.einsum("tij,tik->tjk", wb, left[layout["src"]])
        local = jnp.einsum("tjk,tkl->tjl", temp, tb)
        by_transition = jnp.einsum("tjl,tjl->t", local, right[site + 1][layout["dst"]])
        marginal = jax.ops.segment_sum(by_transition, layout["physical"], num_segments=4)

        proposed = prefactor * (diagonal @ marginal)
        ratios = constrain_ratio(proposed / overlap, weight_floor)
        nodes += jnp.sum(ratios <= 0.0, dtype=jnp.int64)
        probabilities = 0.5 * ratios
        norm = probabilities.sum() + 1.0e-13
        field = jnp.where(randoms[site] < probabilities[0] / norm, 0, 1)
        chosen_diagonal = diagonal[field]
        overlap = proposed[field]
        log_weight += jnp.log(norm)

        ca = ca.at[site].multiply(hs[field, 0])
        cb = cb.at[site].multiply(hs[field, 1])
        left = jax.ops.segment_sum(
            chosen_diagonal[layout["physical"]][:, None, None] * local,
            layout["dst"],
            num_segments=layout["n_out"],
        )
    return ca, cb, overlap_in, overlap, jnp.exp(log_weight), nodes


def make_fast_sweep(convert_channels, channel_maps, trial_blocks, plan):
    """Build the one-conversion HS sweep for a fixed trial and static layouts (legacy API).

    The result is called as sweep(ca, cb, randoms, hs, weight_floor).
    """
    return partial(
        fast_sweep,
        convert=convert_channels,
        channel_maps=channel_maps,
        trial_blocks=trial_blocks,
        contraction=plan,
    )


def mps_cpmc_step(state, *, params, prop_ctx, cpmc_ops, sweep_fn, overlap_fn, overlap_arg):
    """One CPMC step. sweep_fn(ca, cb, randoms, hs, floor) per walker; overlap_fn(walker, overlap_arg)."""
    key, subkey = jax.random.split(state.rng_key)
    nwalkers = wk.n_walkers(state.walkers)
    randoms = jax.random.uniform(subkey, (nwalkers, cpmc_ops.n_sites()))
    floor, cap = float(params.weight_floor), float(params.weight_cap)

    walkers = cpmc_ops.apply_one_body_half(state.walkers, prop_ctx)
    sweep_many = wk.vmap_chunked(sweep_fn, params.n_chunks, in_axes=(0, 0, 0, None, None))
    ca, cb, overlap_half, overlaps, weight_factor, node_step = sweep_many(
        walkers[0], walkers[1], randoms, prop_ctx.hs_constant, floor
    )

    ratio = constrain_ratio(jnp.real(overlap_half / state.overlaps), floor)
    nodes = jnp.sum(ratio <= 0.0, dtype=jnp.int64) + jnp.sum(node_step, dtype=jnp.int64)
    weights = state.weights * ratio
    weights = jnp.where(weights > cap, 0.0, weights) * weight_factor
    walkers = (ca, cb)

    walkers = cpmc_ops.apply_one_body_half(walkers, prop_ctx)
    overlap_many = wk.vmap_chunked(overlap_fn, params.n_chunks, in_axes=(0, None))
    overlaps_new = jnp.real(overlap_many(walkers, overlap_arg))
    ratio = constrain_ratio(jnp.real(overlaps_new / overlaps), floor)
    nodes += jnp.sum(ratio <= 0.0, dtype=jnp.int64)
    weights *= ratio
    weights = jnp.where(weights > cap, 0.0, weights)

    weights *= jnp.exp(prop_ctx.dt * state.pop_control_ene_shift)
    weights = jnp.where(weights > cap, 0.0, weights)
    average = jnp.clip(jnp.mean(weights), min=1.0e-300)
    shift = state.e_estimate - params.pop_control_damping * jnp.log(average) / prop_ctx.dt
    return PropState(
        walkers, weights, overlaps_new, key, shift, state.e_estimate, state.node_encounters + nodes
    )


def init_prop_state_typed(**kwargs):
    state = init_prop_state(**kwargs)
    return state._replace(node_encounters=jnp.zeros((), dtype=jnp.int64))


def make_fast_prop_ops(ham_data, walker_kind, overlap_fn, sweep_fn):
    """Legacy PropOps around closures that hold the trial (trial_data is ignored)."""
    cpmc_ops = make_hubbard_cpmc_ops(ham_data, walker_kind)

    def step(state, *, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ctx):
        return mps_cpmc_step(
            state,
            params=params,
            prop_ctx=prop_ctx,
            cpmc_ops=cpmc_ops,
            sweep_fn=sweep_fn,
            overlap_fn=overlap_fn,
            overlap_arg=trial_data,
        )

    return PropOps(
        init_prop_state=init_prop_state_typed,
        build_prop_ctx=lambda h, _trial, p: _build_prop_ctx(h, p.dt),
        step=step,
    )


def _convert_for_sweep(ca, cb, plan):
    alpha, _, beta, _, prefactor = convert_walker((ca, cb), plan)
    return alpha, beta, prefactor


def make_prop_ops(ham_data, sys, plan: MpsWalkerPlan, *, propagator: str = "fast") -> PropOps:
    """PropOps for MPS-CPMC with trial_data = MpsTrial and meas_ctx = MpsMeasCtx.

    propagator="fast": the cached-environment sweep of this module (one walker conversion
    per sweep). propagator="slow": trot.prop.cpmc_slow's step unchanged, which converts the
    walker for every field proposal through meas_ops.overlap. The two agree for exact
    conversions whenever the weight floor never acts (always at weight_floor=0); cpmc_slow
    floors half the ratio, and re-truncates truncated walkers after every field.

    init_prop_state starts the walkers as trot does, from the natural orbitals of
    trial_ops.get_rdm1(trial_data), or from the free-fermion determinant of ham_data.h1 when
    params.walker_start == "rhf" (unless rdm1 or initial_walkers are given). The scalar state
    leaves are pinned to strong dtypes, so the jitted block compiles once.
    """
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError("MPS-CPMC needs walker_kind='unrestricted'")
    if propagator not in ("fast", "slow"):
        raise ValueError(f"propagator must be 'fast' or 'slow', got {propagator!r}")
    cpmc_ops = make_hubbard_cpmc_ops(ham_data, sys.walker_kind)
    nelec = (int(sys.nelec[0]), int(sys.nelec[1]))

    def init_prop_state_mps(**kwargs):
        params = kwargs["params"]
        start = getattr(params, "walker_start", "natural")
        if start not in ("natural", "rhf"):
            raise ValueError(f"walker_start must be 'natural' or 'rhf', got {start!r}")
        if start == "rhf" and kwargs.get("rdm1") is None and kwargs.get("initial_walkers") is None:
            Ra, Rb = rhf_orbitals(kwargs["ham_data"], nelec)
            kwargs["rdm1"] = jnp.asarray(np.stack([Ra @ Ra.T, Rb @ Rb.T]))
        state = init_prop_state(**kwargs)
        return state._replace(
            node_encounters=state.node_encounters.astype(jnp.int64),
            e_estimate=state.e_estimate.astype(jnp.result_type(float)),
            pop_control_ene_shift=state.pop_control_ene_shift.astype(jnp.result_type(float)),
        )

    def build_prop_ctx(ham, _rdm1, params):
        return _build_prop_ctx(ham, params.dt)

    def step(state, *, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ctx):
        check_meas_ctx(meas_ctx, plan, trial_data)
        layout = contraction_layout(plan, meas_ctx.trial_charges)
        sweep = partial(
            fast_sweep,
            convert=partial(_convert_for_sweep, plan=plan),
            channel_maps=layout.channel_maps,
            trial_blocks=meas_ctx.trial_blocks,
            contraction=layout.contraction,
        )
        return mps_cpmc_step(
            state,
            params=params,
            prop_ctx=prop_ctx,
            cpmc_ops=cpmc_ops,
            sweep_fn=sweep,
            overlap_fn=partial(overlap_from_blocks, layout=layout, plan=plan),
            overlap_arg=meas_ctx.trial_blocks,
        )

    def slow_step(state, *, params, ham_data, trial_data, trial_ops, meas_ops, meas_ctx, prop_ctx):
        check_meas_ctx(meas_ctx, plan, trial_data)
        return cpmc_slow_step(
            state,
            params=params,
            trial_data=trial_data,
            meas_ops=meas_ops,
            cpmc_ops=cpmc_ops,
            prop_ctx=prop_ctx,
        )

    return PropOps(
        init_prop_state=init_prop_state_mps,
        build_prop_ctx=build_prop_ctx,
        step=step if propagator == "fast" else slow_step,
    )
