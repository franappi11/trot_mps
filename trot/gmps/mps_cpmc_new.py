"""CPMC with a DMRG trial and locally cached Hubbard-Stratonovich updates.

The walkers remain Slater determinants.  Each spin channel is converted to a
charge-labelled MPS only when an overlap is needed.  Required two-spin charge
blocks are formed directly, without materialising the much larger combined
d=4 walker MPS.  During the diagonal HS sweep, one conversion plus cached
left/right environments serves both field proposals at all sites.

Walker truncation happens gate by gate with the orthogonality centre moved onto
each gate first, so every split discards the smallest Schmidt values. Each
truncated charge block costs one eigh of its Gram matrix; single-row or
single-column blocks are factored in closed form.

The implementation is real-valued and assumes the spatial local basis
|0>, |alpha>, |beta>, |alpha beta>, indexed by n_alpha + 2*n_beta.

The determinant-to-MPS conversion (orbital plan, gates, charge-blocked splits,
channel interleaving) and the pyblock3 densify live in utils.py, the reference
implementation; they are imported, and so still available, here.

The trot-native version of this algorithm lives in trot's packages (trot.trial.mps,
trot.meas.mps, trot.prop.mps_cpmc, trot.gmps.dmrg, trot.gmps.driver.run_qmc_mps). This script
keeps Config and main and re-exports every name it used to define, so scripts, notebooks and
tests that use mps_cpmc_new.<name> keep working. Monkeypatching mps_cpmc_new.<name> does not
change the functions that call it: patch the defining module instead.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pyblock3.algebra.mpe import MPE
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian

from trot import walkers as wk
from trot.core.ops import MeasOps, k_energy
from trot.core.system import System
from trot.driver import make_run_blocks
from trot.gmps.dmrg import build_dmrg_hamiltonian, hubbard_dmrg_mpo, run_dmrg
from trot.gmps.driver import (
    WalkerOps,
    make_block_logger,
    make_walker_ops,
    run_qmc_fixed_chunks,
    save_result,
)
from trot.gmps.utils import (
    BondPlan,
    OrbitalPlan,
    SectorPlan,
    _assemble,
    _block_plan,
    _factor_block,
    _key,
    _move_centre,
    _rotate_mode_to_front,
    _shift_centre,
    _vector_qr,
    channel_angles,
    channel_mps,
    combine_channels,
    combined_charges,
    contract_real,
    densify_with_charges,
    flat_blocks,
    gate_pair,
    make_orbital_plan,
    plan_bonds,
    sector_plan,
    spin_occupations,
    split_pair,
)
from trot.ham.hubbard import HamHubbard, hopping_matrix
from trot.meas.mps import apply_mpo, compress_mps, hubbard_mpo
from trot.prop import blocks
from trot.prop.cpmc import init_prop_state
from trot.prop.hubbard_cpmc_ops import _build_prop_ctx, make_hubbard_cpmc_ops
from trot.prop.mps_cpmc import (
    constrain_ratio,
    init_prop_state_typed,
    make_fast_prop_ops,
    make_fast_sweep,
    right_environments,
)
from trot.prop.types import PropOps, PropState, QmcParams
from trot.stat_utils import blocking_analysis_ratio, reject_outliers
from trot.trial.auto import make_auto_trial_ops
from trot.trial.mps import (
    PHYSICAL_CHARGE,
    _charge_index,
    blocked_contract_from_blocks,
    contraction_report,
    extract_channel_blocks,
    extract_fixed_blocks,
    make_channel_block_maps,
    make_contraction_plan,
    natural_orbitals,
    one_rdm,
)
from trot.trial.uhf import UhfTrial, get_rdm1 as uhf_get_rdm1
from trot.walkers import _qr as qr_with_det

__all__ = [
    "BondPlan",
    "CFG",
    "Callable",
    "Config",
    "FCIDUMP",
    "HamHubbard",
    "Hamiltonian",
    "MPE",
    "MeasOps",
    "NamedTuple",
    "OrbitalPlan",
    "PHYSICAL_CHARGE",
    "Path",
    "PropOps",
    "PropState",
    "QmcParams",
    "SectorPlan",
    "System",
    "UhfTrial",
    "WalkerOps",
    "_assemble",
    "_block_plan",
    "_build_prop_ctx",
    "_charge_index",
    "_factor_block",
    "_key",
    "_move_centre",
    "_rotate_mode_to_front",
    "_shift_centre",
    "_vector_qr",
    "apply_mpo",
    "asdict",
    "blocked_contract_from_blocks",
    "blocking_analysis_ratio",
    "blocks",
    "build_dmrg_hamiltonian",
    "channel_angles",
    "channel_mps",
    "combine_channels",
    "combined_charges",
    "compress_mps",
    "constrain_ratio",
    "contract_real",
    "contraction_report",
    "dataclass",
    "densify_with_charges",
    "extract_channel_blocks",
    "extract_fixed_blocks",
    "flat_blocks",
    "gate_pair",
    "hopping_matrix",
    "hubbard_dmrg_mpo",
    "hubbard_mpo",
    "init_prop_state",
    "init_prop_state_typed",
    "jax",
    "jnp",
    "json",
    "k_energy",
    "main",
    "make_auto_trial_ops",
    "make_block_logger",
    "make_channel_block_maps",
    "make_contraction_plan",
    "make_fast_prop_ops",
    "make_fast_sweep",
    "make_hubbard_cpmc_ops",
    "make_orbital_plan",
    "make_run_blocks",
    "make_walker_ops",
    "math",
    "natural_orbitals",
    "np",
    "one_rdm",
    "plan_bonds",
    "qr_with_det",
    "reject_outliers",
    "right_environments",
    "run_dmrg",
    "run_qmc_fixed_chunks",
    "save_result",
    "sector_plan",
    "spin_occupations",
    "split_pair",
    "time",
    "uhf_get_rdm1",
    "wk",
]


@dataclass(frozen=True)
class Config:
    L: int = 16
    n_up: int = 8
    n_down: int = 8
    hopping: float = 1.0
    interaction: float = 4.0
    trial_chi: int = 64
    dmrg_sweeps: int = 14
    dmrg_seed: int = 0
    # rank_exact is exact for every full-rank walker before bond truncation.
    # adaptive is cheaper but is only guaranteed for the reference determinant.
    # maximal is exact but usually creates more gates than rank_exact.
    orbital_plan: str = "adaptive"  # rank_exact, adaptive, maximal
    occupation_tolerance: float = 1.0e-10
    walker_channel_chi: int | None = 4
    walker_cutoff: float = 0.0
    # Determinant that freezes the static structure: the gate circuit (make_orbital_plan)
    # and the per-sector kept counts (plan_bonds). "natural" is the determinant of the
    # trial's most occupied natural orbitals, "rhf" the free-fermion determinant.
    plan_reference: str = "natural"  # natural, rhf
    # Determinant every walker starts from. "natural" follows trot's convention
    # (natural orbitals of the trial's one-body density matrix).
    walker_start: str = "natural"  # natural, rhf
    n_walkers: int = 32
    n_blocks: int = 40
    n_equilibration: int = 15
    n_steps: int = 20
    dt: float = 0.01
    weight_floor: float = 1.0e-8
    seed: int = 1234
    n_chunks: int = 1
    result_json: str = ""
    block_log: str = ""  # if set, append every block's scalars here as it finishes
    tag: str = ""


CFG = Config()


def main(cfg=CFG):

    # The script runs unchanged on any JAX backend; log which one this is.
    device = jax.devices()[0]
    print(f"jax {jax.__version__}, backend {jax.default_backend()}, device {device.device_kind}", flush=True)

    h1 = hopping_matrix(cfg.L, cfg.hopping)
    ham = HamHubbard(h1=jnp.asarray(h1), u=cfg.interaction)
    system = System(norb=cfg.L, nelec=(cfg.n_up, cfg.n_down), walker_kind="unrestricted")
    for option in ("plan_reference", "walker_start"):
        if getattr(cfg, option) not in ("rhf", "natural"):
            raise ValueError(f"{option} must be 'rhf' or 'natural'")
    _, orbitals = np.linalg.eigh(h1)
    Ca, Cb = orbitals[:, :cfg.n_up].copy(), orbitals[:, :cfg.n_down].copy()
    print(f"L={cfg.L} ({cfg.n_up},{cfg.n_down}), U={cfg.interaction}")

    hamiltonian = build_dmrg_hamiltonian(cfg)
    dmrg_mps, dmrg_energy = run_dmrg(hamiltonian, cfg)
    trial_np, trial_charges = densify_with_charges(dmrg_mps, cfg.L)
    trial = tuple(jnp.asarray(A) for A in trial_np)
    np.testing.assert_allclose(float(contract_real(trial, trial)), 1.0, atol=1e-10)

    Htrial_np = compress_mps(apply_mpo(hubbard_mpo(cfg.L, cfg.hopping, cfg.interaction), trial_np))
    Htrial = tuple(jnp.asarray(A) for A in Htrial_np)
    trial_energy = float(contract_real(Htrial, trial) / contract_real(trial, trial))
    print(f"DMRG Davidson energy={dmrg_energy:.12f}; dense-MPS expectation={trial_energy:.12f}")
    print("trial bonds:", [A.shape[0] for A in trial] + [trial[-1].shape[-1]])
    print("H|trial> compressed bonds:", [A.shape[0] for A in Htrial] + [Htrial[-1].shape[-1]])

    determinants = {"rhf": (Ca, Cb)}
    if "natural" in (cfg.plan_reference, cfg.walker_start):
        gamma_a, gamma_b = one_rdm(trial_np)
        np.testing.assert_allclose([np.trace(gamma_a), np.trace(gamma_b)], [cfg.n_up, cfg.n_down], atol=1e-8)
        Na, occupations = natural_orbitals(gamma_a, cfg.n_up)
        Nb, _ = natural_orbitals(gamma_b, cfg.n_down)
        determinants["natural"] = (Na, Nb)
        print(f"trial natural orbitals: alpha gap n_N - n_N+1 = {occupations[cfg.n_up - 1] - occupations[cfg.n_up]:.3f}")

    # Gate circuit and kept counts are both frozen on the plan reference.
    Ra, Rb = determinants[cfg.plan_reference]
    plan_a = make_orbital_plan(Ra, cfg.orbital_plan, cfg.occupation_tolerance)
    plan_b = make_orbital_plan(Rb, cfg.orbital_plan, cfg.occupation_tolerance)
    gates = lambda plan: int(plan.block_sizes.sum() - len(plan.block_sizes))
    print(f"plan reference: {cfg.plan_reference} determinant; gates {gates(plan_a)}, {gates(plan_b)}")

    # trot starts every walker from the natural orbitals of trial_ops.get_rdm1(trial_data);
    # trial_data is only a placeholder here (the MPS trial lives in the walker ops), so
    # its orbitals choose the starting determinant.
    Sa, Sb = determinants[cfg.walker_start]
    trial_data = UhfTrial(mo_coeff_a=jnp.asarray(Sa), mo_coeff_b=jnp.asarray(Sb))
    Pa, Pb = Sa @ Sa.T, Sb @ Sb.T
    initial_energy = float(np.sum(h1 * (Pa + Pb)) + cfg.interaction * np.diag(Pa) @ np.diag(Pb))

    # |gauge|^2 is the fidelity the orbital plan keeps for a determinant (1 for the plan reference).
    def plan_fidelity(S, plan):
        _, rows = channel_angles(S, plan, xp=np)
        return float(np.linalg.det(np.stack([rows[i] for i in np.flatnonzero(plan.occupation)]))) ** 2
    print(f"walkers start from the {cfg.walker_start} determinant, E={initial_energy:.12f}; orbital-plan infidelity "
          f"{1 - plan_fidelity(Sa, plan_a):.1e}, {1 - plan_fidelity(Sb, plan_b):.1e}")

    bond_a = bond_b = None
    if cfg.walker_channel_chi is not None or cfg.walker_cutoff:
        bond_a = plan_bonds(Ra, plan_a, cfg.walker_channel_chi, cfg.walker_cutoff)
        bond_b = plan_bonds(Rb, plan_b, cfg.walker_channel_chi, cfg.walker_cutoff)
        print(f"walker truncation reference discarded weights: {bond_a.reference_discarded_weight:.3e}, "
              f"{bond_b.reference_discarded_weight:.3e}")

    ops = make_walker_ops(Ra, Rb, plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges, Htrial)
    report = contraction_report(ops.overlap_plan)
    print("walker bonds:", [len(q) for q in ops.walker_charges])
    print("overlap environment entries:", report,
          f"dense/padded={report['dense']/report['padded_blocks']:.2f}x")

    prop_ops = make_fast_prop_ops(ham, system.walker_kind, ops.overlap, ops.sweep)

    params = QmcParams(dt=cfg.dt, n_walkers=cfg.n_walkers, n_prop_steps=cfg.n_steps,
                       n_blocks=cfg.n_blocks, n_eql_blocks=cfg.n_equilibration,
                       weight_floor=cfg.weight_floor, seed=cfg.seed, n_chunks=cfg.n_chunks)
    trial_ops = make_auto_trial_ops(system, overlap_u=ops.overlap, get_rdm1=uhf_get_rdm1)
    measurement = MeasOps(overlap=ops.overlap, kernels={k_energy: ops.energy})

    block_fn = blocks.block
    if cfg.block_log:
        block_fn = make_block_logger(cfg.block_log, cfg.n_equilibration, cfg.tag)

    start = time.perf_counter()
    chunk_times = []
    mean, error, block_energies, block_weights, collapsed = run_qmc_fixed_chunks(
        sys=system, params=params, ham_data=ham, trial_data=trial_data,
        meas_ops=measurement, trial_ops=trial_ops, prop_ops=prop_ops,
        block_fn=block_fn, chunk_times=chunk_times)
    elapsed = time.perf_counter() - start

    # Steady-state throughput from the chunks after the first (which includes the compile).
    walker_steps_per_s = None
    if len(chunk_times) > 1:
        (b0, t0), (b1, t1) = chunk_times[0], chunk_times[-1]
        walker_steps_per_s = cfg.n_walkers * cfg.n_steps * (b1 - b0) / (t1 - t0)

    scalar = lambda x: None if x is None else float(x)
    print(f"CPMC energy = {scalar(mean)} +/- {scalar(error)}; elapsed={elapsed:.1f} s")
    if chunk_times:
        print(f"first chunk (compile + run) {chunk_times[0][1]:.1f} s; steady throughput "
              f"{walker_steps_per_s or float('nan'):.1f} walker-steps/s", flush=True)
    record = asdict(cfg)
    record.update(
        kind="mps", initial_energy=float(initial_energy), dmrg_energy=dmrg_energy,
        trial_energy=trial_energy, cpmc_energy=scalar(mean), cpmc_error=scalar(error),
        seconds=elapsed, walker_d4_chi=max(map(len, ops.walker_charges)), collapsed_after_block=collapsed,
        gates=gates(plan_a), backend=jax.default_backend(), device=device.device_kind,
        first_chunk_seconds=chunk_times[0][1] if chunk_times else None,
        walker_steps_per_s=walker_steps_per_s)
    save_result(cfg.result_json, record)


if __name__ == "__main__":
    main()
