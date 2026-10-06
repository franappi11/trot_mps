"""MPS-CPMC for the Hubbard model on any lattice, CPU or GPU: one run per invocation, through trot's driver.

The lattice is given by its one-body matrix h1, built from --L (open chain), --Lx/--Ly with --bx/--by (square
lattice: open, periodic or antiperiodic sides; site x * Ly + y) or read from --h1 FILE.npy. Everything else follows
from h1 (trot.gmps.trials.describe_h1 names the run and its cache files and picks the DMRG defaults), so a chain and
a lattice run the same code: DMRG trial (cached), optional spin rotation, trot.gmps.driver.prepare_mps_cpmc and
run_prepared, on the batched engine of trot/gmps/engine.py.

Outputs in --out: <tag>.log (this run's output), blocks.jsonl (every block as it finishes), results.jsonl (one record
per finished run), and with --save-walkers <tag>_walkers.npz (every block's walkers) and dmrg_trial_<lattice>_U<U>_
chi<chi>[_rot..].npz (the trial in the notebooks' format). The run tag and the file names are those of the earlier
run_mps_sweep_gpu.py (chain) and run_sq_sweep_gpu.py (square lattice).

    python run_mps_cpmc.py --L 100 --U 8 --trial-chi 16 --chi-w 32 --walkers 400 --eql 200 --blocks 400 \\
        --dt 0.005 --dmrg-sweeps 30 --dmrg-init warm --trial-cache trial_cache_warm --out /mnt/ceph/.../L100_U8_warm
    python run_mps_cpmc.py --Lx 4 --Ly 4 --U 8 --trial-chi 256 --chi-w 32 --orbital-plan adaptive ...
    python run_mps_cpmc.py --Lx 8 --Ly 8 --U 8 --trial-chi 512 --dmrg-mpo terms --cache-htrial --prepare-only

The DMRG trial starts from the Neel product state wherever h1 and the electron counts define one (--dmrg-init auto,
the default; any bipartite lattice, with holes or doublons away from half filling), else from a random MPS; its
cache name ends in _neel.
--dmrg-init random reproduces the earlier random-start trials and their cache names, and --dmrg-init warm loads a
build_warm_trials.py trial (_warm) and never runs DMRG.

--prepare-only builds (or loads) the DMRG trial and, with --cache-htrial, the block-form H|trial>, then stops: the
CPU part of a large-lattice run, done once for every GPU run of that trial. --dmrg-reference runs a DMRG reference
energy instead (the former mps_cpmc_2d.py dmrg mode), for any lattice:

    python run_mps_cpmc.py --Lx 4 --Ly 4 --U 8 --trial-chi 2000 --dmrg-bdims 500 1000 2000 --dmrg-reference --out ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent


def add_lattice_arguments(parser):
    lattice = parser.add_argument_group("lattice (one of --L, --Lx/--Ly, --h1)")
    lattice.add_argument("--L", type=int, default=None, help="open chain of L sites")
    lattice.add_argument("--Lx", type=int, default=None)
    lattice.add_argument("--Ly", type=int, default=None)
    lattice.add_argument("--bx", default="open", choices=["open", "periodic", "antiperiodic"], help="boundary along x")
    lattice.add_argument("--by", default="open", choices=["open", "periodic", "antiperiodic"], help="boundary along y")
    lattice.add_argument("--h1", default=None, help="one-body matrix from an .npy file")
    lattice.add_argument("--hopping", type=float, default=1.0)
    lattice.add_argument("--U", type=float, default=4.0)
    lattice.add_argument("--n-up", type=int, default=None, help="default: half filling")
    lattice.add_argument("--n-down", type=int, default=None, help="default: half filling")


def check_lattice_arguments(parser, args):
    given = sum(x is not None for x in (args.L, args.Lx or args.Ly, args.h1))
    if given != 1 or (args.Lx is None) != (args.Ly is None):
        parser.error("give exactly one lattice: --L, --Lx with --Ly, or --h1")


def add_trial_arguments(group):
    """The trial's options except --trial-chi (one value here, several in the benchmark)."""
    group.add_argument("--dmrg-sweeps", type=int, default=None, help="default: 20 on a chain, 14 otherwise")
    group.add_argument("--dmrg-seed", type=int, default=0)
    group.add_argument("--dmrg-init", default="auto", choices=["auto", "neel", "random", "warm"],
                       help="DMRG initial state: auto = the Neel product state where defined, else random; warm = "
                            "load a build_warm_trials.py trial (_warm) from --trial-cache, never run DMRG")
    group.add_argument("--dmrg-mpo", default=None, choices=["terms", "qc"], help="default: terms on a chain, qc else")
    group.add_argument("--dmrg-bdims", type=int, nargs="*", default=[], help="bond-dimension ramp (plain schedule)")
    group.add_argument("--dmrg-tol", type=float, default=None, help="default: none on a chain, 1e-6 otherwise")
    group.add_argument("--trial-cache", default=str(HERE / "trial_cache"))
    group.add_argument("--cache-htrial", action="store_true",
                       help="block-form H|trial>, made once and cached next to the trial (6x6 and larger)")
    group.add_argument("--trial-rotation", type=float, default=0.0,
                       help="rotate the trial by exp(-i beta S^y), degrees (90: one-node spin projection)")
    group.add_argument("--rotated-trial", default="projected", choices=["projected", "as_is"])
    group.add_argument("--natural-rdm1", default="after", choices=["after", "before"],
                       help="natural orbitals of the trial's rdm1 after or before the rotation")


def add_walker_plan_arguments(group):
    group.add_argument("--orbital-plan", default="adaptive", choices=["adaptive", "rank_exact", "maximal"])
    group.add_argument("--plan-reference", default="natural", choices=["natural", "rhf"])
    group.add_argument("--walker-start", default="natural", choices=["natural", "rhf"])


def add_engine_arguments(group):
    group.add_argument("--linalg", default=None, help=argparse.SUPPRESS)  # accepted, unused: always batched
    group.add_argument("--walker-qr", default="auto", choices=["auto", "cholesky", "native"])
    group.add_argument("--energy", default="blocked", choices=["blocked", "dense"])
    group.add_argument("--sector-buckets", type=int, nargs="*", default=[8, 16],
                       help="sector-class size bounds (engine.compile_circuit); none given: one class per kind")
    group.add_argument("--compile-cache", default=os.path.expanduser("~/.cache/trot_jax_compile"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_lattice_arguments(parser)
    trial = parser.add_argument_group("trial")
    trial.add_argument("--trial-chi", type=int, required=True)
    add_trial_arguments(trial)

    walkers = parser.add_argument_group("walkers and CPMC")
    walkers.add_argument("--chi-w", type=int, default=32, help="walker bond per spin channel (0: exact)")
    add_walker_plan_arguments(walkers)
    walkers.add_argument("--walkers", type=int, default=400)
    walkers.add_argument("--eql", type=int, default=60)
    walkers.add_argument("--blocks", type=int, default=200)
    walkers.add_argument("--steps", type=int, default=20)
    walkers.add_argument("--dt", type=float, default=0.01)
    walkers.add_argument("--weight-floor", type=float, default=1.0e-8)
    walkers.add_argument("--seed", type=int, default=1234)

    group = parser.add_argument_group("engine")
    add_engine_arguments(group)
    group.add_argument("--n-chunks", type=int, default=0, help="walker chunks; 0: trot's automatic choice")
    group.add_argument("--no-self-check", action="store_true", help="skip the start-up conversion check")
    group.add_argument("--mem-fraction", type=float, default=None, help=argparse.SUPPRESS)  # accepted, unused

    output = parser.add_argument_group("output")
    output.add_argument("--out", default="mps_cpmc_runs")
    output.add_argument("--tag-suffix", default="", help="appended to the run tag")
    output.add_argument("--save-walkers", action="store_true",
                        help="every block's walkers to <out>/<tag>_walkers.npz and the trial to <out>/dmrg_trial_*.npz")
    output.add_argument("--prepare-only", action="store_true", help="build/cache the trial (and H|trial>), then stop")
    output.add_argument("--dmrg-reference", action="store_true",
                        help="only a DMRG reference energy (no cache, no CPMC): --trial-chi or the --dmrg-bdims ramp, "
                             "--dmrg-sweeps; a kind='dmrg' record goes to results.jsonl, the log to dmrg_<...>.log")
    args = parser.parse_args(argv)
    check_lattice_arguments(parser, args)
    return args


def make_h1(args) -> np.ndarray:
    from trot.ham.hubbard import hopping_matrix, square_hopping_matrix

    if args.L is not None:
        return hopping_matrix(args.L, args.hopping)
    if args.Lx is not None:
        return square_hopping_matrix(args.Lx, args.Ly, args.hopping, args.bx, args.by)
    h1 = np.load(args.h1)
    if h1.ndim != 2 or h1.shape[0] != h1.shape[1] or not np.allclose(h1, h1.T):
        raise SystemExit(f"{args.h1}: h1 must be a real symmetric square matrix")
    return np.asarray(h1, dtype=float)


def electrons(args, lattice) -> tuple[int, int]:
    n = lattice.n_sites
    return (args.n_up if args.n_up is not None else n // 2, args.n_down if args.n_down is not None else n // 2)


def dmrg_sweeps(args, lattice) -> int:
    return args.dmrg_sweeps or (20 if lattice.kind == "chain" else 14)


class RunTrial(NamedTuple):
    dmrg: Any  # trials.DmrgTrialData
    trial: Any  # MpsTrial
    info: dict  # trials.make_trial's
    htrial: tuple | None  # block-form (tensors, labels, trial_energy) with --cache-htrial
    htrial_file: Path | None
    seconds: dict  # setup time per stage


def build_trial(args, h1, lattice, nelec, chi, *, rotate=True, say=print) -> RunTrial:
    """The trial of a run: the DMRG trial (cached), with --cache-htrial the block-form H|trial> (cached), and the
    spin rotation; rotate=False stops before trials.make_trial (--prepare-only)."""
    from trot.gmps import trials

    clock, seconds = time.perf_counter, {}
    start = clock()
    dmrg = trials.load_or_make_dmrg_trial(h1, args.U, nelec, chi=chi, sweeps=dmrg_sweeps(args, lattice),
                                          seed=args.dmrg_seed, init=args.dmrg_init, cache_dir=args.trial_cache,
                                          mpo=args.dmrg_mpo, bdims=tuple(args.dmrg_bdims),
                                          tol="default" if args.dmrg_tol is None else args.dmrg_tol, say=say)
    seconds["trial"] = clock() - start
    htrial, gamma, htrial_file = None, None, None
    if args.cache_htrial:
        if args.trial_rotation or args.energy != "blocked":
            raise SystemExit("--cache-htrial needs an unrotated trial and --energy blocked")
        start = clock()
        (blocks, labels), hinfo = trials.load_or_make_htrial(dmrg.path, h1, args.U, dmrg.tensors, dmrg.charges,
                                                             dmrg.variational_energy, say=say)
        htrial, gamma = (blocks, labels, hinfo["trial_energy"]), hinfo["gamma"]
        htrial_file = trials.htrial_cache_file(dmrg.path) if dmrg.path is not None else None
        seconds["htrial"] = clock() - start
    if not rotate:
        return RunTrial(dmrg, None, {}, htrial, htrial_file, seconds)
    start = clock()
    trial, info = trials.make_trial(dmrg.tensors, dmrg.charges, nelec, rotation=args.trial_rotation,
                                    rotated_trial=args.rotated_trial, natural_rdm1=args.natural_rdm1, rdm1=gamma)
    if args.trial_rotation:
        say(f"trial rotated by {args.trial_rotation:g} deg about y "
            + ("and projected onto the walkers' sector" if args.rotated_trial == "projected"
               else "and used as it is (particle-number labels)")
            + f"; natural orbitals from the trial's rdm1 {args.natural_rdm1} the rotation", flush=True)
    seconds["make_trial"] = clock() - start
    return RunTrial(dmrg, trial, info, htrial, htrial_file, seconds)


def run_tag(args, lattice) -> tuple[str, str]:
    """(run tag, rotation suffix): run_mps_sweep_gpu.py's tag on a chain, run_sq_sweep_gpu.py's on a lattice."""
    chi_w = args.chi_w if args.chi_w > 0 else "exact"
    u_part = "" if lattice.kind == "chain" and args.U == 4.0 else f"_U{args.U:g}"
    tag = f"{lattice.name}{u_part}_T{args.trial_chi}_w{chi_w}"
    tag += "_NOplan" if args.plan_reference == "natural" else ""
    tag += "_NOstart" if args.walker_start == "natural" else ""
    tag += "" if args.orbital_plan == "adaptive" else f"_{args.orbital_plan}"
    rot = ""
    if args.trial_rotation:
        rot = f"_rot{args.trial_rotation:g}" + ("_asis" if args.rotated_trial == "as_is" else "")
    tag += rot + ("_NObefore" if args.trial_rotation and args.natural_rdm1 == "before" else "")
    return tag + f"_s{args.seed}" + args.tag_suffix, rot


class Tee:
    """Write to the job's stream and to this run's log file."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def main(argv=None):
    args = parse_args(argv)
    from trot import config

    config.configure_once()

    from trot.gmps import trials

    h1 = make_h1(args)
    lattice = trials.describe_h1(h1)
    tag, rot = run_tag(args, lattice)
    if args.dmrg_reference:
        if args.dmrg_init == "warm":
            raise SystemExit("--dmrg-reference runs DMRG: --dmrg-init auto, neel or random")
        ramp = "-".join(map(str, args.dmrg_bdims)) if args.dmrg_bdims else str(args.trial_chi)
        init = trials.resolve_trial_init(args.dmrg_init, h1, electrons(args, lattice))
        tag = (f"dmrg_{lattice.name}_U{args.U:g}_chi{ramp}_sw{dmrg_sweeps(args, lattice)}_seed{args.dmrg_seed}"
               f"{trials.INIT_TAGS[init]}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = (out / f"{tag}.log").open("w")
    streams = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(streams[0], log), Tee(streams[1], log)
    try:
        if args.dmrg_reference:
            return _dmrg_reference(args, h1, lattice, tag, out)
        return _run(args, h1, lattice, tag, rot, out)
    finally:
        sys.stdout.flush()
        sys.stdout, sys.stderr = streams
        log.close()


def _dmrg_reference(args, h1, lattice, tag, out):
    """A DMRG reference energy for the lattice, as mps_cpmc_2d.py's dmrg mode wrote it: e_mps is <H> of the final
    MPS (variational), e_davidson the two-site energy before the last truncation."""
    from trot.gmps import driver, trials
    from trot.gmps.dmrg import dmrg_h1, dmrg_schedule, lattice_dmrg_schedule, neel_schedule

    nelec, sweeps = electrons(args, lattice), dmrg_sweeps(args, lattice)
    defaults = trials.dmrg_defaults(lattice)
    mpo = args.dmrg_mpo or defaults["mpo"]
    schedule = "plain" if args.dmrg_bdims else defaults["schedule"]
    tol = defaults["tol"] if args.dmrg_tol is None else args.dmrg_tol
    init = trials.resolve_trial_init(args.dmrg_init, h1, nelec)
    if lattice.kind == "square":  # the line plot_cpmc_2d_runs.py reads the model from
        print(f"{lattice.Lx}x{lattice.Ly} lattice, boundaries x={lattice.boundary_x} y={lattice.boundary_y}, "
              f"({nelec[0]},{nelec[1]}), U={args.U}")
    else:
        print(f"{lattice.name} ({lattice.kind}, {lattice.n_sites} sites), ({nelec[0]},{nelec[1]}), U={args.U}")
    if schedule == "plain":
        bdims = lattice_dmrg_schedule(args.trial_chi, sweeps, tuple(args.dmrg_bdims))[0]
    else:
        bdims = (neel_schedule if init == "neel" else dmrg_schedule)(args.trial_chi, sweeps)[0]
    print(f"DMRG reference {tag}: {mpo} MPO, {schedule} schedule, bond dimensions {bdims}, tol {tol}, start "
          f"{init}", flush=True)
    start = time.perf_counter()
    tensors, _, davidson, sweep_energies, variational = dmrg_h1(
        h1, args.U, nelec, chi=args.trial_chi, n_sweeps=sweeps, seed=args.dmrg_seed, init=init, mpo=mpo,
        schedule=schedule, bdims=tuple(args.dmrg_bdims), tol=tol, iprint=0)
    record = dict(kind="dmrg", tag=tag, **lattice.record(), n_up=nelec[0], n_down=nelec[1], interaction=args.U,
                  trial_chi=args.trial_chi, dmrg_sweeps=sweeps, dmrg_seed=args.dmrg_seed, dmrg_init=init, dmrg_mpo=mpo,
                  dmrg_schedule=schedule, dmrg_tol=tol, bdims=list(bdims), e_davidson=davidson, e_mps=variational,
                  sweep_energies=sweep_energies,
                  bond_dims=[int(A.shape[0]) for A in tensors] + [int(tensors[-1].shape[-1])],
                  seconds=time.perf_counter() - start)
    print({k: record[k] for k in ("e_davidson", "e_mps", "bond_dims", "seconds")}, flush=True)
    driver.save_result(out / "results.jsonl", record)
    return record


def _run(args, h1, lattice, tag, rot, out):
    import jax
    import jax.numpy as jnp

    from trot.core.system import System
    from trot.gmps import driver, engine, trials
    from trot.ham.hubbard import HamHubbard
    from trot.prop.types import QmcParamsMps

    n, nelec, sweeps = lattice.n_sites, electrons(args, lattice), dmrg_sweeps(args, lattice)
    if args.compile_cache:
        jax.config.update("jax_compilation_cache_dir", args.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    device = jax.devices()[0]
    print(f"run {tag}: {lattice.name} ({lattice.kind}, {n} sites), nelec {nelec}, U={args.U:g}, t={lattice.hopping:g}")
    print(f"jax {jax.__version__}, backend {jax.default_backend()}, device {device.device_kind}; host CPUs "
          f"{len(os.sched_getaffinity(0))} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})", flush=True)
    eps = np.linalg.eigvalsh(h1)
    gap = [eps[k] - eps[k - 1] if 0 < k < n else float("inf") for k in nelec]
    print(f"free-fermion gap at the Fermi level: {gap[0]:.3e}, {gap[1]:.3e}", flush=True)

    built = build_trial(args, h1, lattice, nelec, args.trial_chi, rotate=not args.prepare_only)
    dmrg, trial, trial_info, htrial, htrial_file, setup = built
    print(f"DMRG trial ({dmrg.init} start): Davidson energy {dmrg.davidson_energy:.12f}"
          + ("" if dmrg.variational_energy is None else f", variational {dmrg.variational_energy:.12f}"), flush=True)
    if args.prepare_only:
        print(f"prepared {tag}: trial {'and H|trial> ' if args.cache_htrial else ''}ready "
              f"({', '.join(f'{k} {v:.1f} s' for k, v in setup.items())})", flush=True)
        return
    clock = time.perf_counter
    params = QmcParamsMps(
        dt=args.dt, n_walkers=args.walkers, n_prop_steps=args.steps, n_eql_blocks=args.eql, n_blocks=args.blocks,
        weight_floor=args.weight_floor, seed=args.seed, n_chunks=max(args.n_chunks, 1),
        auto_n_chunks=args.n_chunks <= 0, trial_chi=args.trial_chi, dmrg_sweeps=sweeps, dmrg_seed=args.dmrg_seed,
        orbital_plan=args.orbital_plan, walker_channel_chi=args.chi_w if args.chi_w > 0 else None,
        plan_reference=args.plan_reference, walker_start=args.walker_start, energy_kernel=args.energy,
        walker_qr=args.walker_qr, sector_buckets=tuple(args.sector_buckets))
    ham = HamHubbard(h1=jnp.asarray(h1), u=args.U)
    system = System(norb=n, nelec=nelec, walker_kind="unrestricted")
    start = clock()
    run = driver.prepare_mps_cpmc(sys=system, params=params, ham_data=ham, trial=trial, htrial=htrial)
    setup["prepare"] = clock() - start
    setup.update(run.info.get("setup_seconds", {}))  # its stages: walker_plan, meas_ctx, walker_start
    kernels = run.meas_ctx.kernels
    if not args.no_self_check:
        probe = kernels.jit_probe(run.state.walkers[0][0], run.state.walkers[1][0], run.meas_ctx.data())
        errors = engine.conversion_self_check(kernels, run.ops.plan.orbital_plans, run.ops.plan.bond_plans, probe)
        print(f"self-check vs NumPy channel_mps: relative error alpha {errors[0]:.1e}, beta {errors[1]:.1e}",
              flush=True)
        if max(errors) > 1e-6:
            raise AssertionError(f"device conversion disagrees with the NumPy reference: {errors}")

    rotation = dict(trial_rotation=args.trial_rotation,
                    rotated_trial=args.rotated_trial if args.trial_rotation else None,
                    natural_rdm1=args.natural_rdm1 if args.trial_rotation else None)
    start_det = tuple(np.asarray(w[0]) for w in run.state.walkers)
    trial_export = ""
    if args.save_walkers:
        trial_export = str(out / f"dmrg_trial_{lattice.name}_U{args.U:g}_chi{args.trial_chi}{rot}.npz")
        tensors, labels = [np.asarray(A) for A in trial.tensors], trial.charge_arrays()  # (N_up, N_dn) or N labels
        # gamma: one_rdm of the exported tensors, from the H|trial> cache when there is one (unrotated trial)
        trials.export_trial(trial_export, tensors, labels, h1, args.U, dmrg.davidson_energy, run.ops.plan.reference,
                            start_det, gamma=None if htrial is None else np.asarray(trial.rdm1),
                            htrial_file=htrial_file)
    block_fn = driver.make_mps_block()
    snapshots = None
    if args.save_walkers:
        snap_config = dict(L=n, N_UP=nelec[0], N_DN=nelec[1], T=lattice.hopping, U=args.U, N_WALKERS=args.walkers,
                           N_EQL=args.eql, N_BLOCKS=args.blocks, N_PROP=args.steps, DT=args.dt, SEED=args.seed,
                           DMRG_CHI_T=args.trial_chi, DMRG_SWEEPS=sweeps, DMRG_INIT=dmrg.init,
                           CHI_PROP=params.walker_channel_chi,
                           E_DMRG=dmrg.davidson_energy, E_TRIAL=run.info["trial_energy"],
                           plan_reference=args.plan_reference, walker_start=args.walker_start,
                           orbital_plan=args.orbital_plan, EPS=params.occupation_tolerance, trial_file=trial_export,
                           tag=tag, module="run_mps_cpmc", device=device.device_kind, **rotation)
        if lattice.kind == "square":
            snap_config.update(LX=lattice.Lx, LY=lattice.Ly, BOUNDARY_X=lattice.boundary_x,
                               BOUNDARY_Y=lattice.boundary_y)
        snapshots = driver.WalkerSnapshots(out / f"{tag}_walkers.npz", args.eql + args.blocks + 1, args.walkers, n,
                                           nelec[0], nelec[1], snap_config)
        snapshots.add_snapshot(run.state)
        block_fn = driver.make_mps_block(record=snapshots.add_block)
    block_fn = driver.make_block_logger(out / "blocks.jsonl", args.eql, tag, base_block_fn=block_fn)

    print("setup times: " + ", ".join(f"{k} {v:.1f} s" for k, v in setup.items()), flush=True)
    start = clock()
    result = driver.run_prepared(run, block_fn=block_fn)
    seconds = clock() - start
    weights = np.asarray(result.block_weights)
    collapsed = int(np.flatnonzero(weights == 0.0)[0]) if np.any(weights == 0.0) else None
    mean, error = float(result.mean_energy), float(result.stderr_energy)
    walker_steps = args.walkers * args.steps * (args.eql + args.blocks)
    print(f"CPMC energy = {mean} +/- {error} ({result.error_method}; blocking {result.stderr_blocking}); "
          f"elapsed={seconds:.1f} s, {walker_steps / seconds:.1f} walker-steps/s incl. compile", flush=True)
    if snapshots is not None:
        snapshots.finish(dict(E_CPMC=mean, E_CPMC_ERR=error, collapsed_after_block=collapsed),
                         dict(reference_up=np.asarray(run.ops.plan.reference[0]),
                              reference_dn=np.asarray(run.ops.plan.reference[1]),
                              start_up=start_det[0], start_dn=start_det[1], h1=h1))
    record = dict(
        kind="mps_cpmc", tag=tag, **lattice.record(), n_up=nelec[0], n_down=nelec[1], interaction=args.U,
        trial_chi=args.trial_chi, dmrg_sweeps=sweeps, dmrg_seed=args.dmrg_seed, dmrg_init=dmrg.init,
        dmrg_mpo=args.dmrg_mpo, dmrg_bdims=list(args.dmrg_bdims), walker_channel_chi=params.walker_channel_chi,
        orbital_plan=args.orbital_plan, occupation_tolerance=params.occupation_tolerance,
        plan_reference=args.plan_reference, walker_start=args.walker_start, n_walkers=args.walkers,
        n_equilibration=args.eql, n_blocks=args.blocks, n_steps=args.steps, dt=args.dt,
        weight_floor=args.weight_floor, seed=args.seed, energy=args.energy, walker_qr=run.info["walker_qr"],
        sector_buckets=list(params.sector_buckets), n_chunks=params.n_chunks, **rotation,
        trial_cache=args.trial_cache, cache_htrial=args.cache_htrial, compile_cache=args.compile_cache,
        result_json=str(out / "results.jsonl"), block_log=str(out / "blocks.jsonl"),
        walker_snapshots=str(out / f"{tag}_walkers.npz") if snapshots is not None else "", trial_export=trial_export,
        dmrg_energy=dmrg.davidson_energy, mps_energy=dmrg.variational_energy, trial_energy=run.info["trial_energy"],
        cpmc_energy=mean, cpmc_error=error, cpmc_error_blocking=result.stderr_blocking,
        error_method=result.error_method, collapsed_after_block=collapsed, seconds=seconds,
        walker_steps_per_s=walker_steps / seconds, setup_seconds={k: round(v, 1) for k, v in setup.items()},
        device=device.device_kind, backend=jax.default_backend(),
        trial_sector_weight=trial_info["sector_weight"], info=run.info)
    driver.save_result(out / "results.jsonl", record)
    print(f"saved the record to {out / 'results.jsonl'}", flush=True)
    return record


if __name__ == "__main__":
    main()
