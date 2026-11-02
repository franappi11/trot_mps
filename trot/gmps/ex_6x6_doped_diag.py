"""ex_6x6_doped.py with the walkers saved every SAVE_EVERY blocks and flags for the controls of the MPS machinery.

The doped 6x6 Hubbard model (U = 8, 16 up + 16 down, open boundaries) with a saved trial (prepare_trial_6x6.py format,
its block-form H|trial> made by prepare_htrial_saved.py --cutoff 1e-9), spin-rotated about y and used as it is, as
ex_6x6_doped.py runs it. On top of that:

* walkers: <out>/walkers_block0000.npz (the start, with the plan's reference determinant reference_up/dn) and
  walkers_block<b>.npz after every SAVE_EVERY-th block b and after the last one (equilibration blocks count): up, dn
  of shape (n_walkers, 36, 16), orthonormal, taken after the block's comb, so every walker has the same weight, with
  the comb's input (pre_comb_weights, comb_index: walker i is pre-comb walker comb_index[i]) and the block's energy,
  weight, e_estimate and node count. walker_truncation_check.py re-measures them.
* flags for the controls, production values by default: --chi-w, --orbital-plan, --plan-reference, --weight-floor,
  --rotation (0: the unrotated trial) and --trial-file (any trial in the prepare_trial_6x6.py format, e.g. the UHF
  determinant of prepare_uhf_sd_trial_6x6.py); --measure-chi converts the walkers at another bond for the energy and
  szsz only (QmcParamsMps.measure_channel_chi), e.g. --chi-w 16 --measure-chi 64; --n-chunks fixes the steps' walker
  chunks (no automatic search) and --measure-chunk the energy's (default TROT_MEASURE_CHUNK, else 20).
* --walker-start natural|rhf|uhf|walkers (tag _start<kind>; walkers: _start<--start-name>) and --seed (tag _s<seed>):
  the start-dependence test. natural is the production start; walkers restarts from --start-walkers, e.g. the
  equilibrated walkers_block<b>.npz of another run (any trial: overlaps and energies are recomputed for this one).
* --retrain-every N (and/or --retrain-at B ...): within the one run_qmc call, every N blocks (equilibration and
  sampling) a drift check trains kept counts per charge sector on the current walkers (trot.gmps.allocation,
  --retrain-rule) and, if they lower the mean discarded weight of a 128-walker sample by more than --retrain-gain
  (relative), the run continues on that walker plan: overlaps recomputed, blocks recompiled on the next chunk
  (trot.gmps.driver.make_retrain_hook through run_qmc's chunk hook). Needs --measure-chi, so the energy does not
  depend on the walker plan. Align N with run_qmc's chunks (N_EQL // 5, N_BLOCKS // 10). Every check goes to
  <out>/retrain_log.json, every plan taken to <out>/walker_plan_retrain<k>.pkl.
* --profile (tag _profile): block --profile-block (default 2, counted over equilibration and sampling) is traced with
  jax.profiler in windows of --profile-window s every --profile-every s into <out>/profile, the compiled block's
  optimized HLO is written there, and at the end the block's time is reported by loop nest, source function and line,
  and HLO instruction (trot.gmps.block_profile). Needs TROT_BLOCKS_PER_CALL=1: with a production run's flags the
  block is then that run's compiled program from the persistent cache, and N_EQL=1 N_BLOCKS=1 costs two blocks.
* <out>/manifest.json (trial, rotation, cutoff, params), <out>/walker_plan.pkl (the run's frozen orbital and bond
  plans, with the reference) and <out>/result.npz (the keys of ex_6x6_doped.py).

N_EQL / N_BLOCKS (environment) default to 100 / 200: run_qmc runs N_EQL // 5 and N_BLOCKS // 10 blocks per call, so
with N_BLOCKS = 2 N_EQL the block compiles once. <out> defaults to
/mnt/ceph/users/fnappi/trot_walkers/sq6x6_doped_diag/<trial>_w<chi_w>[_<non-default flags>]. From this folder:

    TIME=08:00:00 ./submit_ex_2d.sh h100 ex_6x6_doped_diag.py --chi 256                    # production + walkers
    N_EQL=30 N_BLOCKS=60 TIME=03:00:00 ./submit_ex_2d.sh h100 ex_6x6_doped_diag.py --chi 256 --chi-w 32
"""
from trot.config import configure_once

configure_once()
import argparse
import contextlib
import dataclasses
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trot.core.system import System
from trot.gmps import allocation, block_profile, dense_engine, driver, engine, trials
from trot.ham.hubbard import HamHubbard
from trot.prop.types import QmcParamsMps
from trot.trial.mps import make_mps_trial, rotate_spin, spin_rotation_y

TRIALS = Path("/mnt/ceph/users/fnappi/trot_trials/uhf_dmrg_6x6_U8/state_00")
WALKERS = Path("/mnt/ceph/users/fnappi/trot_walkers/sq6x6_doped_diag")
U, T_HOP, SIDE = 8.0, 1.0, 6
CUTOFF = 1e-9  # compression cutoff of the H|trial> caches (prepare_htrial_saved.py --cutoff 1e-9)


def hubbard_h1(side=SIDE, t=T_HOP):
    """Open side x side square lattice, site y * side + x (as ex_6x6_doped.py and prepare_trial_6x6.py)."""
    n = side * side
    h1 = np.zeros((n, n))
    for y in range(side):
        for x in range(side):
            site = y * side + x
            if x + 1 < side:
                h1[site, site + 1] = h1[site + 1, site] = -t
            if y + 1 < side:
                h1[site, site + side] = h1[site + side, site] = -t
    return h1


def load_trial(path, rotation, cutoff=CUTOFF, h1=None, u=U, with_htrial=True):
    """(trial, htrial, nelec) of a saved trial, built as ex_6x6_doped.py builds them: the MpsTrial spin-rotated by
    `rotation` degrees about y and used as it is (particle-number labels; for 0 the unrotated trial), with the
    unrotated trial's rdm1 (walker plan and start), and its cached block-form H|trial> rotated the same way.
    with_htrial=False (the MPO energy kernel): no H|trial> cache is needed or read, htrial is None."""
    path = Path(path)
    h1 = hubbard_h1() if h1 is None else h1
    if with_htrial and not trials.htrial_cache_file(path, cutoff).exists():
        raise SystemExit(f"no H|trial> cache for {path} at cutoff {cutoff:g}: run prepare_htrial_saved.py first")
    n = len(h1)
    with np.load(path) as z:
        nelec = tuple(int(x) for x in z["nelec"])
        if not (np.allclose(z["h1"], h1) and float(z["u"]) == u):
            raise SystemExit(f"{path} was made for another Hamiltonian")
        tensors = [z[f"A{i}"] for i in range(n)]
        charges = [z[f"q{i}"] for i in range(n + 1)]
        rdm1, energy = np.asarray(z["rdm1"]), float(z["energy"])
    blocks = labels = info = None
    if with_htrial:
        (blocks, labels), info = trials.load_or_make_htrial(path, h1, u, tensors, charges, energy,
                                                            relative_tolerance=cutoff)
    if rotation:
        trial = make_mps_trial(tensors=rotate_spin(tensors=tensors, R=spin_rotation_y(beta_deg=rotation)),
                               nelec=nelec, rdm1=rdm1)  # particle-number labels
        if with_htrial:
            blocks, labels = trials.rotate_htrial_blocks(blocks, labels, rotation)
    else:
        trial = make_mps_trial(tensors, charges, nelec=nelec, rdm1=rdm1)
    return trial, ((blocks, labels, info["trial_energy"]) if with_htrial else None), nelec


@contextlib.contextmanager
def training_pool():
    """Worker processes for the host-side training (one core each, at most 8), or None on one core. They are spawned
    with JAX on the CPU: a worker that initialised the GPU backend would try to preallocate the card. trot's
    configure_once picks the GPU from CUDA_VISIBLE_DEVICES and JAX_PLATFORM_NAME, not from JAX_PLATFORMS (with only
    JAX_PLATFORMS=cpu the workers asked for the GPU and failed: job 7279386), so all three are set."""
    workers = min(len(os.sched_getaffinity(0)), 8)
    if workers < 2:
        yield None
        return
    cpu = dict(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    saved = {name: os.environ.get(name) for name in cpu}
    os.environ.update(cpu)
    try:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            yield pool
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def walker_start(args, n_sites, nelec, n_walkers):
    """(prepare_mps_cpmc's start arguments, the manifest's record of the start) for --walker-start: natural and rhf
    are params.walker_start's determinants (no arguments), uhf the natural determinant of the UHF rdm1 [Ca Ca^T,
    Cb Cb^T] of --uhf-file (one determinant, broadcast like the others), walkers the up/dn of --start-walkers (another
    run's walkers_block<b>.npz; n_walkers of them, evenly spaced, if the file holds another number)."""
    record = dict(kind=args.walker_start)
    shapes = [(n_sites, n) for n in nelec]
    if args.walker_start == "uhf":
        with np.load(args.uhf_file) as z:
            C = [np.asarray(z["Ca"]), np.asarray(z["Cb"])]
            record.update(file=str(args.uhf_file), uhf_energy=float(z["energy"]) if "energy" in z.files else None)
        if [c.shape for c in C] != shapes:
            raise SystemExit(f"{args.uhf_file}: Ca, Cb of shapes {[c.shape for c in C]}, need {shapes}")
        C = [np.linalg.qr(c)[0] for c in C]
        return dict(start_rdm1=np.stack([c @ c.T for c in C])), record
    if args.walker_start == "walkers":
        with np.load(args.start_walkers) as z:
            up, dn = np.asarray(z["up"]), np.asarray(z["dn"])
            record.update(name=args.start_name, file=str(args.start_walkers),
                          block=int(z["block"]) if "block" in z.files else None, walkers_in_file=len(up))
        if [up.shape[1:], dn.shape[1:]] != shapes:
            raise SystemExit(f"{args.start_walkers}: walkers of shapes {up.shape[1:]}, {dn.shape[1:]}, need {shapes}")
        if len(up) != n_walkers:
            index = np.linspace(0, len(up) - 1, n_walkers).round().astype(int)
            up, dn = up[index], dn[index]
        return dict(initial_walkers=(up, dn)), record
    return {}, record


def drift(plan, walkers, *, args, chi):
    """The retrain hook's check (trot.gmps.driver.make_retrain_hook): counts trained on the current walkers
    (--retrain-walkers of them, evenly spaced; all by default) by --retrain-rule, taken if they lower the mean
    discarded weight of a 128-walker sample by more than --retrain-gain (trot.gmps.allocation.drift_check). Host
    work in CPU worker processes, or in this process if those fail (the run's GPU time is not lost over them)."""
    if args.retrain_walkers and args.retrain_walkers < len(walkers[0]):
        index = np.linspace(0, len(walkers[0]) - 1, args.retrain_walkers).round().astype(int)
        walkers = tuple(w[index] for w in walkers)
    check = lambda pool: allocation.drift_check(plan, walkers, chi, args.retrain_rule, gain=args.retrain_gain,
                                                pool=pool)
    try:
        with training_pool() as pool:
            return check(pool)
    except Exception as error:
        print(f"training in worker processes failed ({type(error).__name__}: {error}); training in this process",
              flush=True)
        return check(None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chi", type=int, default=256, help="bond dimension of the saved DMRG trial (default 256)")
    parser.add_argument("--trial-file", type=Path, default=None,
                        help="another saved trial (default: the DMRG trial dmrg_chi<chi>.npz)")
    parser.add_argument("--side", type=int, default=SIDE,
                        help="the open side x side lattice (default 6); another side needs --trial-file (its h1 must be "
                             "this lattice's; its nelec is used) and writes by default to sq<side>x<side>_doped/ (the 8x8 "
                             "runs go through ex_8x8_doped.py)")
    parser.add_argument("--chi-w", type=int, default=16, help="walker bond per spin channel (default 16)")
    parser.add_argument("--measure-chi", type=int, default=None,
                        help="walker bond of the energy and szsz measurement (default: --chi-w), e.g. 64: the walkers "
                             "propagate at --chi-w and are converted at this bond to be measured")
    parser.add_argument("--mpo", action="store_true",
                        help="energy through the MPO with dense walker legs (trot.gmps.dense_engine.mpo_numerator): no "
                             "H|trial> cache is read or built (the 18.9 GB of blocks at chi_T 512, ~76 GB at 1024 do not "
                             "exist), the walkers are measured at --measure-chi (default --chi-w); szsz is measured on "
                             "the same dense legs (dense_engine.szsz_contract) unless --no-szsz")
    parser.add_argument("--no-szsz", action="store_true",
                        help="measure the energy only: no szsz kernel is compiled or run (with the sector measurement "
                             "plan at chi_T 512 and bond 64 its environment stack needs ~24 GB per walker); the walkers "
                             "saved every --save-every blocks allow measuring it afterwards")
    parser.add_argument("--per-walker", action="store_true",
                        help="after every block (equilibration included) save every walker's own measurements before the "
                             "comb to <out>/measurements_block<b>.npz: local_energy (as measured, before the outlier "
                             "rule), e_ref, weights (as the averages use them), comb_index and, unless --no-szsz, the "
                             "(n_walkers, L, L) szsz (trot.prop.mps_cpmc.make_block's record_samples)")
    parser.add_argument("--dense", action="store_true",
                        help="dense walker legs (trot.gmps.dense_engine): bond --chi-w, each walker keeping its own "
                             "states, no frozen sector counts; needs --measure-chi (the sector plan that measures)")
    parser.add_argument("--dense-move", default="gram", choices=["gram", "cans"],
                        help="how the dense conversion moves its orthogonality centre")
    parser.add_argument("--dense-gap", type=float, default=None,
                        help="the dense gate eigh's per-charge shift in units of trace(G) (default 1e-7; dense walkers "
                             "are validated at --chi-w <= 16 only)")
    parser.add_argument("--dense-eigh", default="default", choices=["default", "jacobi"],
                        help="the dense conversion's eigh: default (XLA's choice: Jacobi up to 32 x 32, syevd above) "
                             "or jacobi (at every size)")
    parser.add_argument("--orbital-plan", default="rank_exact", choices=["adaptive", "rank_exact", "maximal"],
                        help="gate plan of the walker conversion; rank_exact (default) is exact for every walker, "
                             "adaptive was the production plan until 2026-10-20 (its runs carry no suffix in the tag, "
                             "rank_exact runs end in _rank_exact)")
    parser.add_argument("--plan-reference", default="natural", choices=["natural", "rhf"])
    parser.add_argument("--weight-floor", type=float, default=1e-8)
    parser.add_argument("--rotation", type=float, default=90.0, help="trial spin rotation about y, degrees (0: none)")
    parser.add_argument("--walker-start", default="natural", choices=["natural", "rhf", "uhf", "walkers"],
                        help="where the walkers start: natural (the production start: every walker on the natural "
                             "determinant of the unrotated trial's rdm1), rhf (the free-electron determinant), uhf (the "
                             "UHF determinant of --uhf-file) or walkers (the walkers of --start-walkers, e.g. another "
                             "run's walkers_block<b>.npz, with overlaps and energies recomputed for this trial)")
    parser.add_argument("--start-walkers", type=Path, default=None,
                        help="walkers_block<b>.npz of --walker-start walkers")
    parser.add_argument("--start-name", default="walkers",
                        help="name of a --walker-start walkers start in the tag (_start<name>, default walkers)")
    parser.add_argument("--uhf-file", type=Path, default=TRIALS / "uhf.npz",
                        help="Ca, Cb of --walker-start uhf (default: the pyscf UHF that seeded the DMRG trials)")
    parser.add_argument("--seed", type=int, default=1234, help="random seed (default 1234; another adds _s<seed>)")
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("SAVE_EVERY", 20)))
    parser.add_argument("--n-chunks", type=int, default=None,
                        help="walker chunks of the steps, fixed: no automatic search (default: the search from 1, which "
                             "compiles every candidate, ~20 min each at chi_T 2048, and leaves the energy's chunks at "
                             "--measure-chunk walkers until n_chunks exceeds 400 / --measure-chunk)")
    parser.add_argument("--measure-chunk", type=int, default=None,
                        help="walkers per chunk of the energy and szsz (default: TROT_MEASURE_CHUNK, else 20); the MPO "
                             "energy at bond 64 peaked at 41 GB in chunks of 4 at chi_T 1024 (its buffers grow with "
                             "chi_T)")
    parser.add_argument("--retrain-every", type=int, default=0,
                        help="a drift check of the walker plan's kept counts every N blocks, equilibration and "
                             "sampling (0: none); needs --measure-chi")
    parser.add_argument("--retrain-at", type=int, nargs="*", default=[],
                        help="further blocks to check at (counted over equilibration and sampling)")
    parser.add_argument("--retrain-gain", type=float, default=0.10,
                        help="relative drop of the mean discarded weight that makes a check retrain (default 0.10)")
    parser.add_argument("--retrain-rule", default="consensus", choices=list(allocation.RULES),
                        help="training rule (trot.gmps.allocation)")
    parser.add_argument("--retrain-walkers", type=int, default=0,
                        help="walkers to train on, evenly spaced (0: all of them, copies included)")
    parser.add_argument("--profile", action="store_true",
                        help="trace block --profile-block with jax.profiler in windows into <out>/profile and report "
                             "where its time goes (trot.gmps.block_profile); needs TROT_BLOCKS_PER_CALL=1")
    parser.add_argument("--profile-block", type=int, default=2,
                        help="the block --profile traces, counted over equilibration and sampling (default 2: block 1 "
                             "loads the compiled program)")
    parser.add_argument("--profile-every", type=float, default=150.0,
                        help="seconds between the starts of --profile's trace windows (default 150)")
    parser.add_argument("--profile-at", type=float, nargs="+", default=None,
                        help="instead of --profile-every: the windows' starts in seconds into the block (each window is "
                             "a profiler session; job 7358122 segfaulted stopping its 4th)")
    parser.add_argument("--profile-window", type=float, default=12.0,
                        help="seconds of every --profile trace window (default 12: ~2 half steps or ~1 measurement "
                             "chunk of the 8x8 at chi_T 2048, ~3e5 events)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.n_chunks is not None and args.n_chunks < 1 or args.measure_chunk is not None and args.measure_chunk < 1:
        raise SystemExit("--n-chunks and --measure-chunk must be positive")
    if args.measure_chunk is not None:
        engine.MEASURE_CHUNK = args.measure_chunk  # engine.measure_chunks reads it when the block is traced

    if args.side != SIDE and args.trial_file is None:
        raise SystemExit(f"--side {args.side} needs --trial-file (the saved trials of {TRIALS} are {SIDE}x{SIDE})")
    trial_path = args.trial_file or TRIALS / f"dmrg_chi{args.chi}.npz"
    tag = f"{trial_path.stem}_w{args.chi_w}"
    if args.mpo:
        if args.retrain_every or args.retrain_at:
            raise SystemExit("--mpo is not combined with retraining (it measures with its own dense plan)")
        tag += "_mpo"
    if args.dense:
        if args.measure_chi is None and not args.mpo:
            raise SystemExit("--dense needs --measure-chi (the energy and szsz are measured with a sector plan)")
        if args.retrain_every or args.retrain_at:
            raise SystemExit("--dense has no frozen counts to retrain")
        tag += f"_dense{args.dense_move if args.dense_move != 'gram' else ''}"
    if args.measure_chi is not None and (args.measure_chi != args.chi_w or args.dense):
        tag += f"_m{args.measure_chi}"
    if args.orbital_plan != "adaptive":
        tag += f"_{args.orbital_plan}"
    if args.plan_reference != "natural":
        tag += f"_ref{args.plan_reference}"
    if args.weight_floor != 1e-8:
        tag += f"_floor{args.weight_floor:g}"
    if args.rotation != 90.0:
        tag += f"_rot{args.rotation:g}"
    if (args.walker_start == "walkers") != (args.start_walkers is not None):
        raise SystemExit("--walker-start walkers and --start-walkers go together")
    start_file = {"walkers": args.start_walkers, "uhf": args.uhf_file}.get(args.walker_start)
    if start_file is not None and not start_file.exists():
        raise SystemExit(f"{start_file} not found")
    if args.walker_start != "natural":
        tag += f"_start{args.start_name if args.walker_start == 'walkers' else args.walker_start}"
    if args.seed != 1234:
        tag += f"_s{args.seed}"
    retraining = bool(args.retrain_every or args.retrain_at)
    if retraining:
        if args.measure_chi is None or args.measure_chi == args.chi_w:
            raise SystemExit("--retrain-every / --retrain-at need --measure-chi (a separate measurement plan): the "
                             "energy must not depend on the walker plan that changes during the run")
        tag += f"_rt{args.retrain_every or 'at'}{args.retrain_rule}"
    if args.profile:
        if retraining:
            raise SystemExit("--profile is not combined with retraining (both are run_qmc's chunk hook)")
        tag += "_profile"
    walkers = WALKERS if args.side == SIDE else WALKERS.with_name(f"sq{args.side}x{args.side}_doped")
    out = args.out or walkers / tag
    if any(out.glob("walkers_block*.npz")):
        raise SystemExit(f"{out} already holds the walkers of another run; pass another --out")
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {tag}: walkers every {args.save_every} blocks to {out}", flush=True)

    # persistent compile cache (as run_mps_cpmc.py --compile-cache): a rerun with the same shapes loads the block
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/trot_jax_compile"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    h1 = hubbard_h1(args.side)
    trial, htrial, nelec = load_trial(trial_path, args.rotation, h1=h1, with_htrial=not args.mpo)
    sys = System(len(h1), nelec, "unrestricted")
    ham = HamHubbard(jnp.asarray(h1), U)
    start_args, start = walker_start(args, len(h1), nelec, n_walkers=400)
    params = QmcParamsMps(
        n_walkers=400,
        n_chunks=args.n_chunks or 1,  # the steps in one batch of 400 walkers unless --n-chunks
        auto_n_chunks=args.n_chunks is None,
        n_eql_blocks=int(os.environ.get("N_EQL", 100)),
        n_blocks=int(os.environ.get("N_BLOCKS", 200)),
        dt=0.005,
        n_prop_steps=50,
        weight_floor=args.weight_floor,
        seed=args.seed,
        occupation_tolerance=1e-10,
        orbital_plan=args.orbital_plan,
        walker_channel_chi=args.chi_w,
        measure_channel_chi=args.measure_chi,
        energy_kernel="mpo" if args.mpo else "blocked",
        plan_reference=args.plan_reference,
        walker_start="rhf" if args.walker_start == "rhf" else "natural",  # uhf / walkers: prepare_mps_cpmc's start
        walker_conversion="dense" if args.dense else "sector",
        dense_move=args.dense_move,
        dense_charge_gap=args.dense_gap,
        dense_eigh=args.dense_eigh,
    )
    if args.profile:
        block_profile.check(args.profile_block, params.n_eql_blocks + params.n_blocks)
    n_measure = engine.measure_chunks(params.n_walkers, params.n_chunks)
    print(f"steps in {params.n_chunks} chunk(s){' (automatic search)' if params.auto_n_chunks else ', no search'}; "
          f"energy{'' if args.no_szsz else ' and szsz'} in {n_measure} chunks of {params.n_walkers // n_measure} walkers",
          flush=True)
    kernel_flags = dict(right_envs=dense_engine.RIGHT_ENVS, mpo_kernel=dense_engine.MPO_KERNEL,
                        measure_convert_chunk=engine.MEASURE_CONVERT_CHUNK, measure_barrier=engine.MEASURE_BARRIER,
                        szsz_group=dense_engine.SZSZ_GROUP, trial_gemm=dense_engine.TRIAL_GEMM,
                        prop_gemm=dense_engine.PROP_GEMM)
    print("kernels (TROT_RIGHT_ENVS, TROT_MPO_KERNEL, TROT_MEASURE_CONVERT_CHUNK, TROT_MEASURE_BARRIER, TROT_SZSZ_GROUP, "
          "TROT_TRIAL_GEMM, TROT_PROP_GEMM): " + ", ".join(f"{k}={v}" for k, v in kernel_flags.items()), flush=True)
    manifest = dict(trial_file=str(trial_path), htrial_cutoff=CUTOFF, rotation_degrees=args.rotation,
                    measure_chunk=engine.MEASURE_CHUNK, kernels=kernel_flags,
                    rotated_trial="as_is", natural_rdm1="before", save_every=args.save_every, tag=tag, start=start,
                    retrain=dict(every=args.retrain_every, at=args.retrain_at, gain=args.retrain_gain,
                                 rule=args.retrain_rule, walkers=args.retrain_walkers),
                    params=dataclasses.asdict(params), job=os.environ.get("SLURM_JOB_ID"),
                    device=str(jax.devices()[0]))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    run = driver.prepare_mps_cpmc(sys=sys, params=params, ham_data=ham, trial=trial, htrial=htrial, **start_args)
    meta = dict(h1=h1, u=U, nelec=np.asarray(nelec))
    Ra, Rb = run.ops.plan.reference
    # the run's own frozen plan: rebuilt elsewhere it can differ (near-degenerate gate choices: the up channel's reference
    # loss came out 0.166 on genx against 0.156 in the run), so walker_truncation_check.py reads this one
    with open(out / "walker_plan.pkl", "wb") as handle:
        pickle.dump(dict(reference=(np.asarray(Ra), np.asarray(Rb)), orbital_plans=run.ops.plan.orbital_plans,
                         bond_plans=run.ops.plan.bond_plans), handle)
    np.savez(out / "walkers_block0000.npz", up=np.asarray(run.state.walkers[0]), dn=np.asarray(run.state.walkers[1]),
             weights=np.asarray(run.state.weights), overlaps=np.asarray(run.state.overlaps), block=0, tau=0.0,
             reference_up=np.asarray(Ra), reference_dn=np.asarray(Rb), **meta)
    total, count = params.n_eql_blocks + params.n_blocks, [0]

    def record(up, dn, pre_comb_weights, comb_index, energy, weight, e_estimate, nodes):
        """Called after every block (trot.prop.mps_cpmc.make_block) with the walkers after its comb."""
        count[0] += 1
        b = count[0]
        if b % args.save_every == 0 or b == total:
            path = out / f"walkers_block{b:04d}.npz"
            np.savez(path, up=np.asarray(up), dn=np.asarray(dn), pre_comb_weights=np.asarray(pre_comb_weights),
                     comb_index=np.asarray(comb_index), block=b, tau=b * params.n_prop_steps * params.dt,
                     energy=float(energy), weight=float(weight), e_estimate=float(e_estimate),
                     node_encounters=int(nodes), **meta)
            print(f"walkers of block {b} saved to {path}", flush=True)
        return np.int32(0)

    observables = () if args.no_szsz else ("szsz",)  # szsz is the memory and time hog of the large measurements
    measured = [0]

    def record_samples(local_energy, e_ref, weights, comb_index, *observed):
        """Called after every block, before `record`, with the walkers' own measurements (before the comb)."""
        measured[0] += 1
        b = measured[0]
        values = {name: np.asarray(x) for name, x in zip(observables, observed)}
        np.savez(out / f"measurements_block{b:04d}.npz", local_energy=np.asarray(local_energy), e_ref=float(e_ref),
                 weights=np.asarray(weights), comb_index=np.asarray(comb_index), block=b,
                 sampling=bool(b > params.n_eql_blocks), **values)
        return np.int32(0)

    block_fn = driver.make_mps_block(record=record, record_samples=record_samples if args.per_walker else None)
    hook = None
    if retraining:
        hook = driver.make_retrain_hook(run, partial(drift, args=args, chi=params.walker_channel_chi),
                                        every=args.retrain_every, at=args.retrain_at)
        print(f"retrain checks at blocks {hook.points} ({args.retrain_rule}, gain {args.retrain_gain:g})", flush=True)
    profiler = None
    if args.profile:
        profiler = hook = block_profile.BlockProfiler(run, out / "profile", block=args.profile_block,
                                                      every=args.profile_every, window=args.profile_window,
                                                      at=args.profile_at)
    with profiler.capture() if profiler is not None else contextlib.nullcontext():
        result = driver.run_prepared(run, block_fn=block_fn, observable_names=observables, chunk_hook=hook)
    if retraining:
        (out / "retrain_log.json").write_text(json.dumps(hook.log, indent=2, default=float) + "\n")
        for k, plan in enumerate(hook.plans[1:], start=1):
            with open(out / f"walker_plan_retrain{k}.pkl", "wb") as handle:
                pickle.dump(dict(reference=plan.reference, orbital_plans=plan.orbital_plans,
                                 bond_plans=plan.bond_plans), handle)
        taken = sum(entry["retrained"] for entry in hook.log)
        print(f"{len(hook.log)} retrain checks, {taken} new walker plans; log in {out / 'retrain_log.json'}",
              flush=True)
    szsz = {} if args.no_szsz else dict(
        szsz=np.asarray(result.observable_means["szsz"]), szsz_err=np.asarray(result.observable_stderrs["szsz"]),
        szsz_blocks=np.asarray(result.block_observables["szsz"]))
    np.savez(out / "result.npz",
             energy=float(result.mean_energy), energy_err=float(result.stderr_energy),
             block_energies=np.asarray(result.block_energies), block_weights=np.asarray(result.block_weights),
             trial_file=str(trial_path), tag=tag, **szsz)
    print(f"{tag}: E = {float(result.mean_energy):.6f} +/- {float(result.stderr_energy):.6f}; "
          f"saved to {out / 'result.npz'}", flush=True)
    if profiler is not None:
        profiler.report()


if __name__ == "__main__":
    main()
