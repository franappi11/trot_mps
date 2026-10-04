"""MPS-CPMC through trot's driver.

make_mps_cpmc_ops(ham_data, trial, sys, params) returns the walker plan and the trial, measurement
and propagation ops for trot.driver.run_qmc; run_qmc_mps(...) does everything in one call and,
when no trial is given, builds one by pyblock3 DMRG (trot.gmps.dmrg). make_rotated_mps_cpmc_ops
does the same for a RotatedMpsTrial (trot.trial.mps_rotation), a trial without definite S_z.

params.engine picks the implementation behind the same interface: "reference" (trot.trial.mps,
trot.meas.mps, trot.prop.mps_cpmc), "batched" (the GPU engine of trot/gmps/gpu.py: batched sector
factorisations, factorized contractions, device data as jit arguments) or "auto" (batched on a GPU
backend, reference on CPU). Both give the same trajectories up to rounding.

The second half of this module is the legacy closure-based API of trot/gmps/mps_cpmc_new.py
(make_walker_ops, make_block_logger, run_qmc_fixed_chunks, save_result), moved here unchanged.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np

from trot.core.ops import MeasOps, TrialOps
from trot.driver import make_run_blocks, run_qmc
from trot.gmps import gpu
from trot.gmps.utils import channel_angles, channel_mps, combine_channels, combined_charges
from trot.gmps.utils import contract_real
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.meas.mps_rotated import make_rotated_meas_ops_hubbard
from trot.prop import blocks
from trot.prop.mps_cpmc import make_fast_sweep, make_prop_ops
from trot.prop.mps_cpmc_rotated import make_rotated_prop_ops
from trot.prop.types import PropOps
from trot.stat_utils import blocking_analysis_ratio, reject_outliers
from trot.trial.mps import (
    MpsTrial,
    MpsWalkerPlan,
    as_mps_trial,
    blocked_contract_from_blocks,
    contraction_layout,
    contraction_report,
    extract_channel_blocks,
    extract_fixed_blocks,
    make_channel_block_maps,
    make_contraction_plan,
    make_mps_trial_ops,
    make_walker_plan,
    natural_orbitals,
    rhf_orbitals,
)
from trot.trial.mps_rotation import (
    RotatedMpsTrial,
    make_rotated_trial_ops,
    make_rotated_walker_plan,
)
from trot.walkers import _qr as qr_with_det


class MpsCpmcOps(NamedTuple):
    plan: MpsWalkerPlan
    trial_ops: TrialOps
    meas_ops: MeasOps
    prop_ops: PropOps
    engine: gpu.GpuOps | None = None  # the batched engine's ops (engine="batched"), None for the reference


def make_mps_cpmc_ops(ham_data, trial_data: MpsTrial, sys, params) -> MpsCpmcOps:
    """Walker plan and trot ops for trial_data (an MpsTrial) with QmcParamsMps settings (params.engine)."""
    if not isinstance(trial_data, MpsTrial):
        raise TypeError(
            "trial_data must be an MpsTrial; convert pyblock3/dense trials with "
            "trot.trial.mps.as_mps_trial and pass that same object to run_qmc"
        )
    plan = make_walker_plan(ham_data, trial_data, sys, params)
    if gpu.resolve_engine(params) == "batched":
        engine, trial_ops, meas_ops, prop_ops = gpu.make_batched_ops(ham_data, trial_data, plan, params)
        return MpsCpmcOps(plan, trial_ops, meas_ops, prop_ops, engine)
    return MpsCpmcOps(
        plan=plan,
        trial_ops=make_mps_trial_ops(plan),
        meas_ops=make_mps_meas_ops_hubbard(plan, energy_kernel=params.energy_kernel),
        prop_ops=make_prop_ops(ham_data, sys, plan, propagator=params.propagator),
    )


def make_rotated_mps_cpmc_ops(ham_data, trial_data: RotatedMpsTrial, sys, params) -> MpsCpmcOps:
    """Walker plan and trot ops for a RotatedMpsTrial (no S_z projection) with QmcParamsMps.

    params.engine as in make_mps_cpmc_ops. The batched engine blocks every contraction on the
    trial's particle-number labels (trot.gmps.gpu.trial_labels) and keeps the walker's spin
    channels apart; the reference engine contracts densely with the d=4 walker and ignores
    params.energy_kernel. Everything else follows make_mps_cpmc_ops.
    """
    if not isinstance(trial_data, RotatedMpsTrial):
        raise TypeError("trial_data must be a RotatedMpsTrial (see make_rotated_mps_trial)")
    plan = make_rotated_walker_plan(ham_data, trial_data, sys, params)
    if gpu.resolve_engine(params) == "batched":
        engine, trial_ops, meas_ops, prop_ops = gpu.make_batched_ops(ham_data, trial_data, plan, params)
        return MpsCpmcOps(plan, trial_ops, meas_ops, prop_ops, engine)
    return MpsCpmcOps(
        plan=plan,
        trial_ops=make_rotated_trial_ops(plan),
        meas_ops=make_rotated_meas_ops_hubbard(plan),
        prop_ops=make_rotated_prop_ops(ham_data, sys, plan, propagator=params.propagator),
    )


def run_qmc_chunk_sizes(params) -> set[int]:
    """Distinct block-batch sizes trot.driver.run_qmc compiles (one jitted block scan each)."""
    sizes = set()
    for n, divisor in ((params.n_eql_blocks, 5), (params.n_blocks, 10)):
        chunk = n // divisor if n >= divisor else 1
        sizes |= {min(chunk, n - start) for start in range(0, n, chunk)}
    return sizes


def _plan_fidelity(S, plan) -> float:
    """|gauge|^2: the fidelity the orbital plan keeps for determinant S (1 for exact plans)."""
    _, rows = channel_angles(np.asarray(S), plan, xp=np)
    return float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(plan.occupation)]))) ** 2


def _print_diagnostics(ham_data, trial, ops, params, meas_ctx, dmrg) -> None:
    plan = ops.plan
    plan_a, plan_b = plan.orbital_plans
    gates = lambda p: int(p.block_sizes.sum() - len(p.block_sizes))
    print(f"MPS trial: norb={trial.norb}, nelec={trial.nelec}, bonds {list(trial.bond_dims)}")
    if isinstance(trial, RotatedMpsTrial):
        print("  used as it is: no definite (N_up, N_dn); contractions blocked on particle number")
    elif trial.sector_weight < 1.0 - 1.0e-12:
        print(
            f"  projected onto the walkers' sector, which holds {trial.sector_weight:.6f} of the norm"
        )
    if dmrg is not None:
        print(f"  DMRG Davidson energy {dmrg.davidson_energy:.12f} (two-site, not variational)")
        print(f"  DMRG variational energy {dmrg.variational_energy:.12f}")
    print(
        f"  <T|H|T>/<T|T> = {meas_ctx.trial_energy:.12f} ({getattr(meas_ctx, 'kernel', 'dense')} kernel, "
        f"H|T> bonds max {max(meas_ctx.h_bond_dims)})"
    )
    print(
        f"walker plan: {params.orbital_plan} gates {gates(plan_a)}, {gates(plan_b)}; "
        f"reference {params.plan_reference}; walker bonds max {max(map(len, plan.walker_charges))}"
    )
    for spin, bond in zip("ab", plan.bond_plans):
        if bond is not None:
            print(
                f"  channel {spin}: reference discarded weight {bond.reference_discarded_weight:.3e}"
            )
    if params.walker_start == "rhf":
        Sa, Sb = rhf_orbitals(ham_data, trial.nelec)
    else:
        rdm1 = np.asarray(trial.rdm1)
        Sa, Sb = (natural_orbitals(rdm1[s], trial.nelec[s])[0] for s in range(2))
    print(
        f"  walkers start from the {params.walker_start} determinant; orbital-plan infidelity "
        f"{1 - _plan_fidelity(Sa, plan_a):.1e}, {1 - _plan_fidelity(Sb, plan_b):.1e}"
    )
    if ops.engine is not None:
        linalg, walker_qr = gpu.resolve_linalg(params)
        print(
            f"  batched engine: linalg={linalg}, walker_qr={walker_qr}, "
            f"spin-batched={ops.engine.converter.spin_batched}; circuit "
            f"{gpu.circuit_stats(ops.engine.converter.circuits[0])}; overlap plan {ops.engine.overlap_plan.stats}"
        )
        return
    if isinstance(trial, RotatedMpsTrial):
        return  # dense reference contractions: no charge-blocked layout to report
    report = contraction_report(contraction_layout(plan, trial.charges).contraction)
    print(
        f"  overlap environment entries {report}, dense/padded "
        f"{report['dense'] / max(report['padded_blocks'], 1):.2f}x"
    )


def run_qmc_mps(
    *,
    sys,
    params,
    ham_data,
    trial_data=None,
    block_fn=blocks.block,
    state=None,
    mesh=None,
    target_error=None,
    observable_names=(),
):
    """CPMC for a HamHubbard with an MPS trial, through trot.driver.run_qmc.

    trial_data: None (pyblock3 DMRG with params.trial_chi, dmrg_sweeps, dmrg_seed), an MpsTrial,
      a pyblock3 MPS, a DenseMps/Gmps-like object or (tensors, charges). Spin-rotated trials are
      projected onto the walkers' (N_up, N_dn) sector, except a RotatedMpsTrial, which is used as
      it is (make_rotated_mps_cpmc_ops).
    params: QmcParamsMps. block_fn: blocks.block, or e.g. make_block_logger(...) for a JSONL log.
    Returns trot's QmcResult.
    """
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError("MPS-CPMC needs walker_kind='unrestricted'")
    for name in ("orbital_plan", "walker_channel_chi", "energy_kernel", "plan_reference"):
        if not hasattr(params, name):
            raise TypeError("params must be a QmcParamsMps (trot.prop.types)")
    dmrg = None
    if trial_data is None:
        from trot.gmps.dmrg import make_dmrg_trial

        dmrg = make_dmrg_trial(
            ham_data, sys, chi=params.trial_chi, n_sweeps=params.dmrg_sweeps, seed=params.dmrg_seed
        )
        trial = dmrg.trial
    elif isinstance(trial_data, RotatedMpsTrial):
        trial = trial_data
    else:
        trial = as_mps_trial(trial_data, nelec=sys.nelec)
    make_ops = make_rotated_mps_cpmc_ops if isinstance(trial, RotatedMpsTrial) else make_mps_cpmc_ops
    ops = make_ops(ham_data, trial, sys, params)
    meas_ctx = ops.meas_ops.build_meas_ctx(ham_data, trial)
    prop_ctx = ops.prop_ops.build_prop_ctx(ham_data, ops.trial_ops.get_rdm1(trial), params)
    _print_diagnostics(ham_data, trial, ops, params, meas_ctx, dmrg)

    sizes = run_qmc_chunk_sizes(params)
    if len(sizes) > 1:
        message = (
            f"run_qmc will compile the MPS block once per block-batch size {sorted(sizes)}; "
            "choose n_eql_blocks = 5k and n_blocks = 10k (same k) to compile once"
        )
        warnings.warn(message, stacklevel=2)
        print(f"[mps] {message}", flush=True)

    try:
        result = run_qmc(
            sys=sys,
            params=params,
            ham_data=ham_data,
            trial_data=trial,
            trial_ops=ops.trial_ops,
            meas_ops=ops.meas_ops,
            prop_ops=ops.prop_ops,
            block_fn=block_fn,
            state=state,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            target_error=target_error,
            mesh=mesh,
            observable_names=observable_names,
        )
    except ValueError as exc:
        if "is zero or numerically ill-conditioned" in str(exc):
            raise RuntimeError(
                "the walker population collapsed: every sampled block has total weight 0 "
                "(all walkers were killed and reconfiguration cannot revive them)"
            ) from exc
        raise
    weights = np.asarray(result.block_weights)
    if np.any(weights == 0.0):
        first = int(np.flatnonzero(weights == 0.0)[0])
        message = (
            f"the walker population collapsed (total weight 0 from block entry {first}); "
            "trot's run_qmc does not stop, so the statistics are not meaningful"
        )
        warnings.warn(message, stacklevel=2)
        print(f"[mps] {message}", flush=True)
    return result


# ---------------------------------------------------------------------------------------------
# Legacy closure-based API (moved from trot/gmps/mps_cpmc_new.py)
# ---------------------------------------------------------------------------------------------


class WalkerOps(NamedTuple):
    convert: Callable
    overlap: Callable
    energy: Callable | None
    sweep: Callable
    walker_charges: tuple[np.ndarray, ...]
    overlap_plan: dict


def make_walker_ops(Ca, Cb, plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, Htrial=None):
    """Walker conversion, overlap, local energy and fast sweep against a fixed trial.

    The reference determinant (Ca, Cb) fixes the static walker bond labels.
    """

    def convert_channels(ca, cb):
        """Orthonormalize an SD walker and convert each spin channel to an MPS."""
        qa, det_ra = qr_with_det(ca)
        qb, det_rb = qr_with_det(cb)
        alpha, qa_charge, gauge_a = channel_mps(qa, plan_a, bond_a)
        beta, qb_charge, gauge_b = channel_mps(qb, plan_b, bond_b)
        prefactor = det_ra * det_rb * gauge_a * gauge_b
        return alpha, qa_charge, beta, qb_charge, prefactor

    _, qa_charge, _, qb_charge, _ = convert_channels(jnp.asarray(Ca), jnp.asarray(Cb))
    walker_charges = combined_charges(qa_charge, qb_charge)
    overlap_plan = make_contraction_plan(walker_charges, trial_charges)
    channel_maps = make_channel_block_maps(overlap_plan, qa_charge, qb_charge)
    trial_blocks = extract_fixed_blocks(trial_np, overlap_plan)
    trial = tuple(jnp.asarray(A) for A in trial_np)

    def overlap_fn(walker, _trial_data=None):
        ca, cb = walker
        alpha_mps, _, beta_mps, _, prefactor = convert_channels(ca, cb)
        blocks_ = extract_channel_blocks(alpha_mps, beta_mps, channel_maps)
        return prefactor * blocked_contract_from_blocks(blocks_, trial_blocks, overlap_plan)

    def energy_fn(walker, _ham=None, _ctx=None, _trial_data=None):
        ca, cb = walker
        alpha, qa, beta, qb, _ = convert_channels(ca, cb)
        tensors, _ = combine_channels(alpha, qa, beta, qb)
        return contract_real(tensors, Htrial) / contract_real(tensors, trial)

    def convert_for_sweep(ca, cb):
        alpha, _, beta, _, prefactor = convert_channels(ca, cb)
        return alpha, beta, prefactor

    sweep_fn = make_fast_sweep(convert_for_sweep, channel_maps, trial_blocks, overlap_plan)
    return WalkerOps(
        convert_channels,
        overlap_fn,
        None if Htrial is None else energy_fn,
        sweep_fn,
        walker_charges,
        overlap_plan,
    )


def make_block_logger(path, n_equilibration, tag="", base_block_fn=blocks.block):
    """Wrap a block function so each block's scalars are appended to a JSONL file
    the moment the block finishes: raw values, before trot's outlier rejection,
    so partial runs survive and every block keeps its index."""
    counter = iter(range(1 << 62))
    start = time.perf_counter()
    result_spec = jax.ShapeDtypeStruct((), jnp.int32)

    def write(energy, weight, e_estimate, nodes):
        block = next(counter)
        record = dict(
            tag=tag,
            block=block,
            phase="equilibration" if block < n_equilibration else "sampling",
            energy=float(energy),
            weight=float(weight),
            e_estimate=float(e_estimate),
            node_encounters=int(nodes),
            seconds=time.perf_counter() - start,
        )
        with Path(path).open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return np.int32(0)

    def block_fn(state, **kwargs):
        state, obs = base_block_fn(state, **kwargs)
        jax.experimental.io_callback(
            write,
            result_spec,
            obs.scalars["energy"],
            obs.scalars["weight"],
            state.e_estimate,
            state.node_encounters,
            ordered=True,
        )
        return state, obs

    return block_fn


def run_qmc_fixed_chunks(
    *, sys, params, ham_data, trial_data, meas_ops, trial_ops, prop_ops, block_fn, chunk_times=None
):
    """trot's run_qmc with one chunk size for both phases, gcd(n_eql, n_blocks),
    so the jitted block scan compiles once instead of once per chunk size.
    Same initialisation, block function, outlier rejection and blocking analysis.
    Stops early if the population collapses (total weight zero: every walker was
    killed, and reconfiguration cannot bring it back).
    Returns (mean, stderr, block energies, block weights, collapsed_after_block), all
    blocks unfiltered; mean and stderr are NaN and collapsed_after_block is set if it
    collapsed, otherwise collapsed_after_block is None.
    If chunk_times is a list, (blocks done, seconds since start) is appended after
    every chunk; the first includes the compile."""
    prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    state = prop_ops.init_prop_state(
        sys=sys,
        ham_data=ham_data,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        params=params,
    )
    run_blocks = make_run_blocks(
        block_fn=block_fn,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )
    n_eql, total = params.n_eql_blocks, params.n_eql_blocks + params.n_blocks
    chunk = math.gcd(n_eql, params.n_blocks)
    energies, weights, start, collapsed = [], [], time.perf_counter(), None
    for done in range(chunk, total + 1, chunk):
        state, scalars, _ = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=chunk,
        )
        e, w = np.asarray(scalars["energy"]), np.asarray(scalars["weight"])
        if chunk_times is not None:
            chunk_times.append((done, time.perf_counter() - start))
        energies.extend(e.tolist())
        weights.extend(w.tolist())
        print(
            f"[{'eql' if done <= n_eql else 'blk'} {done:4d}/{total}]  E_chunk {np.sum(e * w) / np.sum(w):14.10f}"
            f"  W {w.mean():12.6e}  nodes {int(state.node_encounters):10d}"
            f"  t {time.perf_counter() - start:8.1f} s",
            flush=True,
        )
        if not w[-1] > 0.0:
            collapsed = done
            print(
                f"\nPopulation collapsed: total weight is zero after block {done}. Stopping.",
                flush=True,
            )
            break

    if collapsed is not None:
        return float("nan"), float("nan"), np.asarray(energies), np.asarray(weights), collapsed
    sampled = np.column_stack((energies[n_eql:], weights[n_eql:]))
    clean, _ = reject_outliers(sampled, obs=0)
    print(f"\nRejected {len(sampled) - len(clean)} outlier blocks.\n\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(np.asarray(clean[:, 0]), np.asarray(clean[:, 1]), print_q=True)
    return stats["mu"], stats["se_star"], np.asarray(energies), np.asarray(weights), None


def save_result(path, record):
    if not path:
        return
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record) + "\n")
