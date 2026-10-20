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
  szsz only (QmcParamsMps.measure_channel_chi), e.g. --chi-w 16 --measure-chi 64.
* --retrain-every N (and/or --retrain-at B ...): within the one run_qmc call, every N blocks (equilibration and
  sampling) a drift check trains kept counts per charge sector on the current walkers (trot.gmps.allocation,
  --retrain-rule) and, if they lower the mean discarded weight of a 128-walker sample by more than --retrain-gain
  (relative), the run continues on that walker plan: overlaps recomputed, blocks recompiled on the next chunk
  (trot.gmps.driver.make_retrain_hook through run_qmc's chunk hook). Needs --measure-chi, so the energy does not
  depend on the walker plan. Align N with run_qmc's chunks (N_EQL // 5, N_BLOCKS // 10). Every check goes to
  <out>/retrain_log.json, every plan taken to <out>/walker_plan_retrain<k>.pkl.
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
from trot.gmps import allocation, driver, trials
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
    parser.add_argument("--chi-w", type=int, default=16, help="walker bond per spin channel (default 16)")
    parser.add_argument("--measure-chi", type=int, default=None,
                        help="walker bond of the energy and szsz measurement (default: --chi-w), e.g. 64: the walkers "
                             "propagate at --chi-w and are converted at this bond to be measured")
    parser.add_argument("--mpo", action="store_true",
                        help="energy through the MPO with dense walker legs (trot.gmps.dense_engine.mpo_numerator): no "
                             "H|trial> cache is read or built (the 18.9 GB of blocks at chi_T 512, ~76 GB at 1024 do not "
                             "exist), the walkers are measured at --measure-chi (default --chi-w); needs --no-szsz")
    parser.add_argument("--no-szsz", action="store_true",
                        help="measure the energy only: no szsz kernel is compiled or run (at chi_T 512 and bond 64 its "
                             "environment stack needs ~24 GB per walker); the walkers saved every --save-every blocks "
                             "allow measuring it afterwards")
    parser.add_argument("--dense", action="store_true",
                        help="dense walker legs (trot.gmps.dense_engine): bond --chi-w, each walker keeping its own "
                             "states, no frozen sector counts; needs --measure-chi (the sector plan that measures)")
    parser.add_argument("--dense-move", default="gram", choices=["gram", "cans"],
                        help="how the dense conversion moves its orthogonality centre")
    parser.add_argument("--dense-gap", type=float, default=None,
                        help="the dense gate eigh's per-charge shift in units of trace(G) (default 1e-7; dense walkers "
                             "are validated at --chi-w <= 16 only)")
    parser.add_argument("--orbital-plan", default="adaptive", choices=["adaptive", "rank_exact", "maximal"])
    parser.add_argument("--plan-reference", default="natural", choices=["natural", "rhf"])
    parser.add_argument("--weight-floor", type=float, default=1e-8)
    parser.add_argument("--rotation", type=float, default=90.0, help="trial spin rotation about y, degrees (0: none)")
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("SAVE_EVERY", 20)))
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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    trial_path = args.trial_file or TRIALS / f"dmrg_chi{args.chi}.npz"
    tag = f"{trial_path.stem}_w{args.chi_w}"
    if args.mpo:
        if not args.no_szsz:
            raise SystemExit("--mpo has no szsz kernel: add --no-szsz")
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
    retraining = bool(args.retrain_every or args.retrain_at)
    if retraining:
        if args.measure_chi is None or args.measure_chi == args.chi_w:
            raise SystemExit("--retrain-every / --retrain-at need --measure-chi (a separate measurement plan): the "
                             "energy must not depend on the walker plan that changes during the run")
        tag += f"_rt{args.retrain_every or 'at'}{args.retrain_rule}"
    out = args.out or WALKERS / tag
    if any(out.glob("walkers_block*.npz")):
        raise SystemExit(f"{out} already holds the walkers of another run; pass another --out")
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {tag}: walkers every {args.save_every} blocks to {out}", flush=True)

    # persistent compile cache (as run_mps_cpmc.py --compile-cache): a rerun with the same shapes loads the block
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/trot_jax_compile"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    h1 = hubbard_h1()
    trial, htrial, nelec = load_trial(trial_path, args.rotation, with_htrial=not args.mpo)
    sys = System(len(h1), nelec, "unrestricted")
    ham = HamHubbard(jnp.asarray(h1), U)
    params = QmcParamsMps(
        n_walkers=400,
        n_chunks=1,  # the steps in one batch of 400 walkers; the energy and szsz in chunks of <= 20 (engine.MEASURE_CHUNK)
        auto_n_chunks=True,
        n_eql_blocks=int(os.environ.get("N_EQL", 100)),
        n_blocks=int(os.environ.get("N_BLOCKS", 200)),
        dt=0.005,
        n_prop_steps=50,
        weight_floor=args.weight_floor,
        seed=1234,
        occupation_tolerance=1e-10,
        orbital_plan=args.orbital_plan,
        walker_channel_chi=args.chi_w,
        measure_channel_chi=args.measure_chi,
        energy_kernel="mpo" if args.mpo else "blocked",
        plan_reference=args.plan_reference,
        walker_start="natural",
        walker_conversion="dense" if args.dense else "sector",
        dense_move=args.dense_move,
        dense_charge_gap=args.dense_gap,
    )
    manifest = dict(trial_file=str(trial_path), htrial_cutoff=CUTOFF, rotation_degrees=args.rotation,
                    rotated_trial="as_is", natural_rdm1="before", save_every=args.save_every, tag=tag,
                    retrain=dict(every=args.retrain_every, at=args.retrain_at, gain=args.retrain_gain,
                                 rule=args.retrain_rule, walkers=args.retrain_walkers),
                    params=dataclasses.asdict(params), job=os.environ.get("SLURM_JOB_ID"),
                    device=str(jax.devices()[0]))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    run = driver.prepare_mps_cpmc(sys=sys, params=params, ham_data=ham, trial=trial, htrial=htrial)
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

    block_fn = driver.make_mps_block(record=record)
    hook = None
    if retraining:
        hook = driver.make_retrain_hook(run, partial(drift, args=args, chi=params.walker_channel_chi),
                                        every=args.retrain_every, at=args.retrain_at)
        print(f"retrain checks at blocks {hook.points} ({args.retrain_rule}, gain {args.retrain_gain:g})", flush=True)
    observables = () if args.no_szsz else ("szsz",)  # szsz is the memory and time hog of the large measurements
    result = driver.run_prepared(run, block_fn=block_fn, observable_names=observables, chunk_hook=hook)
    if hook is not None:
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


if __name__ == "__main__":
    main()
