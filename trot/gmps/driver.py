"""MPS-CPMC through trot's driver: one code path for CPU and GPU, chain or any lattice.

The trot ops of an MPS trial come from the trot modules, as for any trial: make_walker_plan and make_mps_trial_ops
(trot.trial.mps), make_mps_meas_ops_hubbard (trot.meas.mps) and make_prop_ops plus the MPS measurement block
(trot.prop.mps_cpmc), all on the batched engine of the walker plan (trot.gmps.engine). This module puts a run
together: make_mps_cpmc_ops builds the ops, prepare_mps_cpmc adds the contexts, the walkers' start and the
diagnostics, run_prepared runs trot.driver.run_qmc on them, and run_qmc_mps does all of it in one call, building a
DMRG trial (trot.gmps.dmrg) when none is given. make_block_logger and WalkerSnapshots record every block as it
finishes; save_result writes a run's record. The command-line runner is trot/gmps/run_mps_cpmc.py; the trials,
their cache and the spin rotation are in trot.gmps.trials.
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np

from trot.core.ops import MeasOps, TrialOps
from trot.driver import run_qmc
from trot.gmps import engine
from trot.gmps.utils import channel_angles
from trot.meas.mps import make_mps_meas_ops_hubbard
from trot.prop import blocks, mps_cpmc
from trot.prop.mps_cpmc import make_block as make_mps_block  # noqa: F401  (the MPS measurement block)
from trot.prop.mps_cpmc import make_prop_ops
from trot.prop.types import PropOps, PropState
from trot.trial.mps import (
    MpsTrial,
    MpsWalkerPlan,
    as_mps_trial,
    make_mps_trial_ops,
    make_walker_plan,
    natural_orbitals,
    rhf_orbitals,
)


class MpsCpmcOps(NamedTuple):
    plan: MpsWalkerPlan
    trial_ops: TrialOps
    meas_ops: MeasOps
    prop_ops: PropOps


def make_mps_cpmc_ops(ham_data, trial_data: MpsTrial, sys, params, htrial=None) -> MpsCpmcOps:
    """Walker plan and trot ops for trial_data (an MpsTrial) with QmcParamsMps settings.

    htrial: optional precomputed H|trial> (tensors or block form, labels, trial_energy) for energy_kernel="blocked"
    (trot.meas.mps.build_mps_meas_ctx).
    """
    if not isinstance(trial_data, MpsTrial):
        raise TypeError(
            "trial_data must be an MpsTrial; convert pyblock3/dense trials with "
            "trot.trial.mps.as_mps_trial and pass that same object to run_qmc"
        )
    plan = make_walker_plan(ham_data, trial_data, sys, params)
    return MpsCpmcOps(
        plan=plan,
        trial_ops=make_mps_trial_ops(plan),
        meas_ops=make_mps_meas_ops_hubbard(plan, energy_kernel=params.energy_kernel, htrial=htrial),
        prop_ops=make_prop_ops(ham_data, sys, plan),
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

def mps_diagnostics(ham_data, trial, ops: MpsCpmcOps, params, meas_ctx, state=None) -> dict:
    """The run's static diagnostics (trial, H|trial>, walker plan, start determinant, engine layout), as a dict for
    the result record; print_diagnostics prints it."""
    plan = ops.plan
    plan_a, plan_b = plan.orbital_plans
    gates = lambda p: int(p.block_sizes.sum() - len(p.block_sizes))
    if params.walker_start == "rhf":
        Sa, Sb = rhf_orbitals(ham_data, trial.nelec)
    else:
        rdm1 = np.asarray(trial.rdm1)
        Sa, Sb = (natural_orbitals(rdm1[s], trial.nelec[s])[0] for s in range(2))
    kernels = meas_ctx.kernels
    info: dict[str, Any] = dict(
        trial_bonds=list(trial.bond_dims),
        trial_bonds_max=int(max(trial.bond_dims)),
        trial_labels="(N_up, N_dn)" if trial.label_width == 2 else "N",
        trial_sector_weight=trial.sector_weight,
        trial_energy=float(meas_ctx.trial_energy),
        energy_kernel=meas_ctx.kernel,
        htrial_bonds_max=int(max(meas_ctx.h_bond_dims)),
        gates=gates(plan_a),
        gates_beta=gates(plan_b),
        walker_bonds_max=int(max(map(len, plan.walker_charges))),
        reference_discarded_weight=[None if b is None else float(b.reference_discarded_weight)
                                    for b in plan.bond_plans],
        plan_infidelity=[1 - _plan_fidelity(Sa, plan_a), 1 - _plan_fidelity(Sb, plan_b)],
    )
    if state is not None:
        info["initial_e_estimate"] = float(state.e_estimate)
        info["initial_overlap"] = float(np.asarray(state.overlaps)[0])
    info.update(
        walker_qr=engine.resolve_walker_qr(plan.walker_qr), spin_batched=kernels.converter.spin_batched,
        sector_buckets=list(plan.sector_buckets),
        circuits=[engine.circuit_stats(c) for c in kernels.converter.circuits],
        overlap_plan=kernels.overlap_plan.stats,
        energy_plan=None if kernels.energy_plan is None else kernels.energy_plan.stats,
    )
    return info


def print_diagnostics(info: dict, params, dmrg=None) -> None:
    print(f"MPS trial: bonds {info['trial_bonds']}")
    if info["trial_labels"] == "N":
        print("  used as it is: no definite (N_up, N_dn); contractions blocked on particle number")
    elif info["trial_sector_weight"] < 1.0 - 1.0e-12:
        print(f"  projected onto the walkers' sector, which holds {info['trial_sector_weight']:.6f} of the norm")
    if dmrg is not None:
        print(f"  DMRG Davidson energy {dmrg.davidson_energy:.12f} (two-site, not variational)")
        if dmrg.variational_energy is not None:  # None in chain cache files written before it was stored
            print(f"  DMRG variational energy {dmrg.variational_energy:.12f}")
        print(f"  DMRG initial state: {dmrg.init}")
    print(f"  <T|H|T>/<T|T> = {info['trial_energy']:.12f} ({info['energy_kernel']} kernel, "
          f"H|T> bonds max {info['htrial_bonds_max']})")
    if "setup_seconds" in info:
        print("  setup: " + ", ".join(f"{k} {v:.1f} s" for k, v in info["setup_seconds"].items()))
    print(f"walker plan: {params.orbital_plan} gates {info['gates']}, {info['gates_beta']}; reference "
          f"{params.plan_reference}; walker bonds max {info['walker_bonds_max']}")
    for spin, weight in zip("ab", info["reference_discarded_weight"]):
        if weight is not None:
            print(f"  channel {spin}: reference discarded weight {weight:.3e}")
    print(f"  walkers start from the {params.walker_start} determinant; orbital-plan infidelity "
          f"{info['plan_infidelity'][0]:.1e}, {info['plan_infidelity'][1]:.1e}")
    if "initial_e_estimate" in info:
        print(f"  start: overlap {info['initial_overlap']:.6e}, local energy {info['initial_e_estimate']:.10f}")
    print(f"  engine: walker_qr={info['walker_qr']}, spin-batched={info['spin_batched']}, sector buckets "
          f"{info['sector_buckets'] or 'none'}")
    for j, circuit in enumerate(info["circuits"]):
        print(f"  conversion circuit {j}: {circuit}")
    print(f"  overlap plan {info['overlap_plan']}")
    if info["energy_plan"] is not None:
        print(f"  energy plan {info['energy_plan']}")
    print("", flush=True)


# ---------------------------------------------------------------------------------------------
# Block records: the JSONL block log and the walker snapshots
# ---------------------------------------------------------------------------------------------


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


class WalkerSnapshots:
    """Every block's walkers, in the format fixed_block_walkers.ipynb and entanglement_vs_gmps.ipynb load.

    Snapshot 0 is the starting population and snapshot b the population after block b, after its comb (where every
    walker has the same weight): up, dn of shape (n_snap, n_walkers, L, N_sigma), float64, at imaginary time
    tau_snapshots = b * n_steps * dt. Per block b (the one ending at tau_blocks[b] and producing snapshot b + 1):
    energies, weights (the block's energy and total weight), e_estimate and node_encounters (cumulative), and the
    comb's input: pre_comb_weights[b, j] is walker j's weight at the energy measurement, and comb_index[b, i] = j says
    walker i of snapshot b + 1 is a copy of that walker j.

    Arrays are written block by block into <name>.parts/ (progress.json says how many are valid), so a crash keeps
    every finished block; finish() packs them with the config into <name>.npz and removes the parts. add_block is the
    record callback of make_mps_block.
    """

    def __init__(self, path, n_snap, n_walkers, L, n_up, n_down, config=None):
        from numpy.lib.format import open_memmap

        self.config = dict(config or {})  # stored in progress.json too, so unfinished runs can be plotted
        self.path = Path(path)
        self.parts = self.path.with_name(self.path.stem + ".parts")
        self.parts.mkdir(parents=True, exist_ok=True)

        def memmap(name, shape, dtype=np.float64):
            return open_memmap(self.parts / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)

        self.up = memmap("up", (n_snap, n_walkers, L, n_up))
        self.dn = memmap("dn", (n_snap, n_walkers, L, n_down))
        self.pre_comb_weights = memmap("pre_comb_weights", (n_snap - 1, n_walkers))
        self.comb_index = memmap("comb_index", (n_snap - 1, n_walkers), np.int32)
        self.snapshots, self.blocks = 0, []
        print(f"walker snapshots: {n_snap} x {n_walkers} walkers -> {self.path} "
              f"({(self.up.nbytes + self.dn.nbytes) / 1e9:.1f} GB)", flush=True)

    def add_snapshot(self, state):
        self.up[self.snapshots] = np.asarray(state.walkers[0])
        self.dn[self.snapshots] = np.asarray(state.walkers[1])
        self.snapshots += 1

    def add_block(self, up, dn, pre_comb_weights, comb_index, energy, weight, e_estimate, node_encounters):
        b = len(self.blocks)
        self.pre_comb_weights[b] = np.asarray(pre_comb_weights)
        self.comb_index[b] = np.asarray(comb_index)
        self.blocks.append(dict(energy=float(energy), weight=float(weight), e_estimate=float(e_estimate),
                                node_encounters=int(node_encounters)))
        self.up[self.snapshots] = np.asarray(up)
        self.dn[self.snapshots] = np.asarray(dn)
        self.snapshots += 1
        for array in (self.up, self.dn, self.pre_comb_weights, self.comb_index):
            array.flush()
        (self.parts / "progress.json").write_text(json.dumps(dict(snapshots=self.snapshots, blocks=self.blocks,
                                                                  config=self.config)))
        return np.int32(0)

    def finish(self, config, arrays=None):
        config = {**self.config, **config}
        n, nb = self.snapshots, len(self.blocks)
        step = config["N_PROP"] * config["DT"]

        def column(key, dtype=float):
            return np.array([blk[key] for blk in self.blocks], dtype=dtype)

        np.savez(self.path, up=self.up[:n], dn=self.dn[:n], energies=column("energy"), weights=column("weight"),
                 e_estimate=column("e_estimate"), node_encounters=column("node_encounters", np.int64),
                 pre_comb_weights=self.pre_comb_weights[:nb], comb_index=self.comb_index[:nb],
                 tau_snapshots=np.arange(n) * step, tau_blocks=np.arange(1, nb + 1) * step,
                 config=json.dumps(config), **(arrays or {}))
        del self.up, self.dn, self.pre_comb_weights, self.comb_index
        shutil.rmtree(self.parts)
        print(f"saved {self.path}: {n} snapshots, {nb} blocks", flush=True)


def save_result(path, record):
    """Append one run's record to a JSONL file (results.jsonl)."""
    if not path:
        return
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------------------------
# Running: prepare, run
# ---------------------------------------------------------------------------------------------


class MpsCpmcRun(NamedTuple):
    """Everything a run needs, built by prepare_mps_cpmc."""

    sys: Any
    params: Any
    ham_data: Any
    trial: Any
    ops: MpsCpmcOps
    meas_ctx: Any
    prop_ctx: Any
    state: PropState
    info: dict


def prepare_mps_cpmc(*, sys, params, ham_data, trial, htrial=None, state=None, mesh=None, dmrg=None,
                     verbose=True) -> MpsCpmcRun:
    """Ops, contexts, the walkers' start and the diagnostics for an MpsTrial.

    htrial: optional precomputed H|trial> (tensors or block form, labels, trial_energy), energy_kernel="blocked".
    dmrg: the trot.gmps.dmrg.DmrgTrial the trial came from, for the printout.
    """
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError("MPS-CPMC needs walker_kind='unrestricted'")
    for name in ("orbital_plan", "walker_channel_chi", "energy_kernel", "plan_reference"):
        if not hasattr(params, name):
            raise TypeError("params must be a QmcParamsMps (trot.prop.types)")
    clock, seconds = time.perf_counter, {}
    start = clock()
    ops = make_mps_cpmc_ops(ham_data, trial, sys, params, htrial)  # walker plan (orbital and bond plans, circuit)
    seconds["walker_plan"] = clock() - start
    start = clock()
    meas_ctx = ops.meas_ops.build_meas_ctx(ham_data, trial)  # H|trial>, layouts, padded blocks
    prop_ctx = ops.prop_ops.build_prop_ctx(ham_data, ops.trial_ops.get_rdm1(trial), params)
    seconds["meas_ctx"] = clock() - start
    if state is None:
        start = clock()
        state = ops.prop_ops.init_prop_state(sys=sys, ham_data=ham_data, trial_ops=ops.trial_ops, trial_data=trial,
                                             meas_ops=ops.meas_ops, params=params, meas_ctx=meas_ctx, mesh=mesh)
        jax.block_until_ready(state)
        seconds["walker_start"] = clock() - start  # compile included
    info = mps_diagnostics(ham_data, trial, ops, params, meas_ctx, state)
    info["setup_seconds"] = {k: round(v, 1) for k, v in seconds.items()}
    if verbose:
        print_diagnostics(info, params, dmrg)
    return MpsCpmcRun(sys, params, ham_data, trial, ops, meas_ctx, prop_ctx, state, info)


def run_prepared(run: MpsCpmcRun, *, block_fn=None, mesh=None, target_error=None, observable_names=()):
    """trot.driver.run_qmc on a prepared run. block_fn defaults to the MPS block (trot.prop.mps_cpmc.block).
    Returns trot's QmcResult."""
    if block_fn is None:
        block_fn = mps_cpmc.block
    sizes = run_qmc_chunk_sizes(run.params)
    if len(sizes) > 1:
        message = (
            f"run_qmc will compile the MPS block once per block-batch size {sorted(sizes)}; "
            "choose n_eql_blocks = 5k and n_blocks = 10k (same k) to compile once"
        )
        warnings.warn(message, stacklevel=2)
        print(f"[mps] {message}", flush=True)
    try:
        result = run_qmc(
            sys=run.sys,
            params=run.params,
            ham_data=run.ham_data,
            trial_data=run.trial,
            trial_ops=run.ops.trial_ops,
            meas_ops=run.ops.meas_ops,
            prop_ops=run.ops.prop_ops,
            block_fn=block_fn,
            state=run.state,
            meas_ctx=run.meas_ctx,
            prop_ctx=run.prop_ctx,
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

def run_qmc_mps(
    *,
    sys,
    params,
    ham_data,
    trial_data=None,
    block_fn=None,
    state=None,
    mesh=None,
    target_error=None,
    observable_names=(),
    htrial=None,
):
    """CPMC for a HamHubbard with an MPS trial, through trot.driver.run_qmc.

    trial_data: None (pyblock3 DMRG with params.trial_chi, dmrg_sweeps, dmrg_seed, dmrg_init), an MpsTrial,
      a pyblock3 MPS, a DenseMps/Gmps-like object or (tensors, charges) (trot.trial.mps.as_mps_trial).
    params: QmcParamsMps. block_fn: None (the MPS block, trot.prop.mps_cpmc.block), trot.prop.blocks.block,
      or e.g. make_block_logger(...) for a JSONL log.
    htrial: optional precomputed H|trial> (see prepare_mps_cpmc).
    Returns trot's QmcResult.
    """
    dmrg = None
    if trial_data is None:
        from trot.gmps.dmrg import make_dmrg_trial

        dmrg = make_dmrg_trial(ham_data, sys, chi=params.trial_chi, n_sweeps=params.dmrg_sweeps,
                               seed=params.dmrg_seed, init=params.dmrg_init)
        trial = dmrg.trial
    else:
        trial = as_mps_trial(trial_data, nelec=sys.nelec)
    run = prepare_mps_cpmc(sys=sys, params=params, ham_data=ham_data, trial=trial, htrial=htrial, state=state,
                           mesh=mesh, dmrg=dmrg)
    return run_prepared(run, block_fn=block_fn, mesh=mesh, target_error=target_error,
                        observable_names=observable_names)


# ---------------------------------------------------------------------------------------------
