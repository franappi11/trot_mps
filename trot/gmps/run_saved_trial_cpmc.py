"""CPMC conversion controls using a trial.npz from trial_bias_diagnostics.

Run separate GPU jobs, from the repository root, with the same --trial and seed::

    python -m trot.gmps.run_saved_trial_cpmc --trial diagnostics/4x4/uhf_current_chi128_seed0/trial.npz --mode baseline --out cpmc_controls/uhf_baseline
    python -m trot.gmps.run_saved_trial_cpmc --trial diagnostics/4x4/uhf_current_chi128_seed0/trial.npz --mode exact --out cpmc_controls/uhf_exact

No DMRG is run. Both modes rotate the saved trial by 90 degrees and use its saved
unrotated RDM for the natural-orbital plan and initial determinant. "exact" means
untruncated walker conversion, not an unbiased constrained-path calculation.
Defaults match the original 400-walker, dt=.005, 50-step, 50/500-block runs.
Chunking starts at one walker per chunk for memory safety; the driver may tune it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def load_saved_trial(path):
    """Read and validate the diagnostic artifact without changing it or running DMRG."""
    from trot.trial.mps import _valid_sz_labels, make_mps_trial

    with np.load(path, allow_pickle=False) as data:
        h1, u = np.asarray(data["h1"]), float(data["u"])
        nelec = tuple(int(v) for v in data["nelec"])
        if h1.ndim != 2 or h1.shape[0] != h1.shape[1] or np.iscomplexobj(h1):
            raise ValueError("saved h1 must be a real square matrix")
        n = len(h1)
        if not np.isfinite(h1).all() or not np.allclose(h1, h1.T) or not np.isfinite(u):
            raise ValueError("saved Hamiltonian must be finite and symmetric")
        if len(nelec) != 2 or not 0 < nelec[0] == nelec[1] < n:
            raise ValueError("this control runner requires 0 < N_up = N_down < nsites")
        tensors = [np.asarray(data[f"A{i}"]) for i in range(n)]
        charges = tuple(np.asarray(data[f"q{i}"]) for i in range(n + 1))
        rdm1 = np.asarray(data["rdm1"])
    if any(np.iscomplexobj(a) or not np.isfinite(a).all() for a in tensors + [rdm1]):
        raise ValueError("saved tensors and RDM must be finite and real")
    if not _valid_sz_labels(tensors, charges, nelec):
        raise ValueError("saved trial must have valid (N_up, N_down) charge labels")
    trial = make_mps_trial(tensors, charges, nelec=nelec, rdm1=rdm1)
    return h1, u, nelec, trial


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "exact"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--walkers", type=int, default=400)
    parser.add_argument("--eql", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=500)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=.005)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-chunks", type=int, default=None,
                        help="initial chunk count; default: one walker per chunk")
    parser.add_argument("--save-every", type=int, default=50, help="walker snapshots every this many blocks")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"choose a new output directory: {args.out} already exists")
    if min(args.walkers, args.blocks, args.steps, args.save_every) < 1 or args.eql < 0 or args.seed < 0:
        parser.error("walkers/blocks/steps/save-every must be positive; eql/seed must be nonnegative")
    chunks = args.walkers if args.n_chunks is None else args.n_chunks
    if chunks < 1 or args.walkers % chunks or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("n-chunks must divide walkers; dt must be finite and positive")

    from trot import config

    config.configure_once(single_precision=False, matmul_precision="highest")
    import jax
    import jax.numpy as jnp
    from trot.core.system import System
    from trot.gmps import driver
    from trot.gmps.trial_bias_diagnostics import number_blocks, prepare_htrial
    from trot.ham.hubbard import HamHubbard
    from trot.prop.types import QmcParamsMps
    from trot.trial.mps import make_mps_trial, rotate_spin, spin_rotation_y

    h1, u, nelec, original = load_saved_trial(args.trial)
    tensors, charges = [np.asarray(a) for a in original.tensors], original.charge_arrays()
    r = spin_rotation_y(90.)
    number_charges = tuple(q.sum(axis=1, keepdims=True) for q in charges)
    trial = make_mps_trial(rotate_spin(tensors, r), number_charges, nelec=nelec, rdm1=original.rdm1)
    params = QmcParamsMps(
        n_walkers=args.walkers, n_eql_blocks=args.eql, n_blocks=args.blocks,
        n_prop_steps=args.steps, dt=args.dt, seed=args.seed, n_chunks=chunks,
        auto_n_chunks=True, weight_floor=1e-8, trial_chi=max(trial.bond_dims),
        orbital_plan="adaptive" if args.mode == "baseline" else "rank_exact",
        walker_channel_chi=32 if args.mode == "baseline" else None,
        walker_cutoff=0., occupation_tolerance=1e-10, plan_reference="natural", walker_start="natural",
        energy_kernel="blocked", sector_buckets=())
    args.out.mkdir(parents=True)
    manifest = dict(trial_file=str(args.trial.resolve()), trial_sha256=hashlib.sha256(args.trial.read_bytes()).hexdigest(),
                    mode=args.mode, rotation_degrees=90., natural_rdm1="before", params=asdict(params),
                    nelec=nelec, u=u, device=str(jax.devices()[0]), save_every=args.save_every,
                    versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "jax", "jaxlib")})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Loaded {args.trial}; SHA256 {manifest['trial_sha256']}; mode {args.mode}", flush=True)
    hp, hq, energy, _ = prepare_htrial(tensors, charges, h1, u)
    hp, hq = number_blocks(hp, hq, rotation=r)
    run = driver.prepare_mps_cpmc(sys=System(len(h1), nelec, "unrestricted"), params=params,
                                  ham_data=HamHubbard(jnp.asarray(h1), u), trial=trial, htrial=(hp, hq, energy))
    np.savez(args.out / "walkers_block0000.npz", up=np.asarray(run.state.walkers[0]),
             dn=np.asarray(run.state.walkers[1]), h1=h1, u=u, nelec=nelec, block=0, tau=0.,
             reference_up=run.ops.plan.reference[0], reference_dn=run.ops.plan.reference[1])
    count = [0]

    def record(up, dn, pre_comb_weights, comb_index, block_energy, weight, e_estimate, nodes):
        count[0] += 1
        b = count[0]
        if b % args.save_every == 0 or b == args.eql + args.blocks:
            np.savez(args.out / f"walkers_block{b:04d}.npz", up=np.asarray(up), dn=np.asarray(dn),
                     h1=h1, u=u, nelec=nelec, block=b, tau=b * args.dt * args.steps,
                     energy=float(block_energy), weight=float(weight), node_encounters=int(nodes))
        return np.int32(0)

    block = driver.make_block_logger(args.out / "blocks.jsonl", args.eql, args.mode,
                                    base_block_fn=driver.make_mps_block(record=record))
    result = driver.run_prepared(run, block_fn=block)
    np.savez(args.out / "blocks.npz", energies=np.asarray(result.block_energies),
             weights=np.asarray(result.block_weights))
    summary = dict(mean_energy=float(result.mean_energy), stderr_energy=float(result.stderr_energy),
                   error_method=result.error_method, error_reliable=bool(result.error_reliable),
                   tau_int=result.tau_int, effective_sample_size=result.effective_sample_size,
                   gamma_warnings=list(result.gamma_warnings),
                   trial_sha256=manifest["trial_sha256"], mode=args.mode,
                   collapsed=bool(np.any(np.asarray(result.block_weights) == 0.)))
    (args.out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"E = {summary['mean_energy']:.9f} +/- {summary['stderr_energy']:.9f}; saved to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
