"""Make a square lattice's DMRG trial and its compressed H|trial> once, on the CPU, for
mps_cpmc_2d_gpu's GPU runs, which then load both from the cache.

Writes into --trial-cache (default /mnt/ceph/users/fnappi/trot_trials):
  <trial>.npz          the DMRG trial (mps_cpmc_2d_gpu.load_or_run_trial's file)
  <trial>_htrial.npz   H|trial> in block form, compressed per charge sector (relative cut
                       1e-13), with <trial|H|trial> and the trial's rdm1 (load_or_make_htrial)
Each is skipped when it is already there, so a job that ran out of time or memory after
the DMRG only redoes H|trial>. The log has the host BLAS rate, the DMRG sweeps, the
H|trial> bonds and sizes, and each stage's time and peak memory. <trial|H|trial> (the
nearest-neighbour MPO of hubbard_mpo_from_h1) must equal pyblock3's <H> (the DMRG MPO)
to 1e-8, or nothing is saved.

    # from trot/gmps, on genx (run_mps_sweep.sh), one job per lattice and chi_T
    sbatch -J trial_sq6x6_T128 --cpus-per-task=4 --mem=8G --time=00:30:00 \\
        --export=ALL,TARGET=prepare_sq_trial.py run_mps_sweep.sh --Lx 6 --Ly 6 --U 8 --trial-chi 128

GPU runs of the same trial then load both files:

    run_sq_sweep_gpu.py --Lx 6 --Ly 6 --U 8 --trial-chi 128 --dmrg-mpo terms --cache-htrial \\
        --trial-cache /mnt/ceph/users/fnappi/trot_trials ...
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mps_cpmc_2d_gpu as sg  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--Lx", type=int, required=True)
parser.add_argument("--Ly", type=int, required=True)
parser.add_argument("--bx", default="open", choices=list(sg.BOUNDARY_CODE), help="boundary along x")
parser.add_argument("--by", default="open", choices=list(sg.BOUNDARY_CODE), help="boundary along y")
parser.add_argument("--n-up", type=int, default=None, help="default: half filling")
parser.add_argument("--n-down", type=int, default=None, help="default: half filling")
parser.add_argument("--t", type=float, default=1.0, help="hopping")
parser.add_argument("--U", type=float, default=8.0)
parser.add_argument("--trial-chi", type=int, required=True)
parser.add_argument("--dmrg-sweeps", type=int, default=14)
parser.add_argument("--dmrg-tol", type=float, default=1.0e-6)
parser.add_argument("--dmrg-seed", type=int, default=0)
parser.add_argument("--dmrg-mpo", default="terms", choices=["terms", "qc"],
                    help="terms: from the Hubbard terms (bond 2 + 4 Ly); qc: pyblock3's quantum-chemistry MPO")
parser.add_argument("--iprint", type=int, default=1, help="pyblock3 DMRG verbosity (1: one line per sweep)")
parser.add_argument("--trial-cache", default="/mnt/ceph/users/fnappi/trot_trials")
args = parser.parse_args()


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kB on Linux


# numpy's BLAS/LAPACK go through FlexiBLAS: its NETLIB default unless a module (openblas/...) sets FLEXIBLAS
_a = np.random.default_rng(0).random((2048, 2048))
_a @ _a
_t = time.perf_counter()
_a @ _a
print(f"host: {len(os.sched_getaffinity(0))} CPUs usable, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
      f"FLEXIBLAS={os.environ.get('FLEXIBLAS', 'unset (NETLIB)')}, "
      f"numpy float64 GEMM {2 * 2048 ** 3 / (time.perf_counter() - _t) / 1e9:.0f} GFLOP/s", flush=True)
del _a

n_sites = args.Lx * args.Ly
cfg = sg.Config(Lx=args.Lx, Ly=args.Ly, boundary_x=args.bx, boundary_y=args.by,
                n_up=n_sites // 2 if args.n_up is None else args.n_up,
                n_down=n_sites // 2 if args.n_down is None else args.n_down,
                hopping=args.t, interaction=args.U, trial_chi=args.trial_chi, dmrg_sweeps=args.dmrg_sweeps,
                dmrg_tol=args.dmrg_tol, dmrg_seed=args.dmrg_seed, dmrg_mpo=args.dmrg_mpo,
                trial_cache=args.trial_cache, cache_htrial=True)
h1 = sg.lattice_hopping(cfg)
sg.describe(cfg, h1)
trial_file, htrial_file = sg._trial_cache_file(cfg), sg._htrial_cache_file(cfg)
print(f"trial file:    {trial_file}\nH|trial> file: {htrial_file}", flush=True)

start = time.perf_counter()
tensors, charges, dmrg_energy, mps_energy = sg.load_or_run_trial(cfg, h1, iprint=args.iprint)
trial_s = time.perf_counter() - start
print(f"trial: Davidson {dmrg_energy:.12f}, pyblock3 <H> {mps_energy:.12f} ({mps_energy / n_sites:.10f} per site); "
      f"bonds {[A.shape[0] for A in tensors] + [1]}; {trial_s:.0f} s, peak memory {peak_gb():.1f} GB", flush=True)

start = time.perf_counter()
(blocks, labels), info = sg.load_or_make_htrial(cfg, h1, tensors, charges, mps_energy)
htrial_s = time.perf_counter() - start
print(f"H|trial>: {htrial_s:.0f} s, peak memory {peak_gb():.1f} GB", flush=True)

sizes = ", ".join(f"{p.name} {p.stat().st_size / 1e9:.2f} GB" for p in (trial_file, htrial_file))
print(f"done: {cfg.Lx}x{cfg.Ly} U={cfg.interaction:g} chi_T={cfg.trial_chi}: "
      f"trial {trial_s:.0f} s + H|trial> {htrial_s:.0f} s, peak memory {peak_gb():.1f} GB; {sizes}", flush=True)
