"""Sweep mps_cpmc_2d_gpu over the DMRG trial bond and the walker bond on one GPU, one
fresh process per run: run_mps_sweep_gpu.py for the square lattice.

Each run appends its summary to <out>/results.jsonl and every block, as it finishes,
to <out>/blocks.jsonl. --save-walkers also writes every block's walkers to
<out>/<tag>_walkers.npz and the DMRG trial to <out>/dmrg_trial_<lattice>_U<U>_chi<chi>.npz,
the files walker_overlap_error.ipynb and plot_cpmc_runs.py read.

After each run the log splits its wall time into setup (imports, trial, H|trial>,
initial state, self-check), compile and blocks. With --project-blocks it also gives
the wall time the same run would take with that many blocks, and an --time of about
twice that. Every block has the same static shapes, so the time per block of a short
run carries over to a long one on the same kind of device (a MIG slice runs at its
share of the card: 1g.20gb 1/7, 2g.20gb 2/7).

    python run_sq_sweep_gpu.py --Lx 4 --Ly 4 --U 4 --trial-chi 128 256 --chi-w 32 64 \
        --walkers 400 --eql 60 --blocks 200 --dt 0.01 --save-walkers --out sq4x4_U4
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDARY_CODE = {"open": "o", "periodic": "p", "antiperiodic": "a"}

parser = argparse.ArgumentParser()
parser.add_argument("--Lx", type=int, default=4)
parser.add_argument("--Ly", type=int, default=4)
parser.add_argument("--bx", default="open", choices=list(BOUNDARY_CODE), help="boundary along x")
parser.add_argument("--by", default="open", choices=list(BOUNDARY_CODE), help="boundary along y")
parser.add_argument("--n-up", type=int, default=None, help="default: half filling")
parser.add_argument("--n-down", type=int, default=None, help="default: half filling")
parser.add_argument("--U", type=float, default=4.0)
parser.add_argument("--trial-chi", type=int, nargs="+", default=[128])
parser.add_argument("--chi-w", type=int, nargs="+", default=[32])
parser.add_argument("--walkers", type=int, default=400)
parser.add_argument("--eql", type=int, default=60)
parser.add_argument("--blocks", type=int, default=200)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--plan-reference", default="natural", choices=["natural", "rhf"])
parser.add_argument("--walker-start", default="natural", choices=["natural", "rhf"])
parser.add_argument("--orbital-plan", default="adaptive", choices=["adaptive", "rank_exact", "maximal"])
parser.add_argument("--dmrg-sweeps", type=int, default=14)
parser.add_argument("--dmrg-mpo", default="qc", choices=["qc", "terms"],
                    help="DMRG MPO; prepare_sq_trial.py's trials (6x6 and larger) use terms")
parser.add_argument("--cache-htrial", action="store_true",
                    help="load (or make) the block-form H|trial> next to the cached trial (prepare_sq_trial.py)")
parser.add_argument("--tag-suffix", default="", help="appended to every run tag")
parser.add_argument("--save-walkers", action="store_true",
                    help="save every block's walkers to <out>/<tag>_walkers.npz and the DMRG trial to "
                         "<out>/dmrg_trial_<lattice>_U<U>_chi<trial chi>.npz (the notebooks' formats)")
parser.add_argument("--project-blocks", type=int, default=0,
                    help="after each run, project its wall time to this many blocks (eql + sampling)")
parser.add_argument("--n-chunks", type=int, default=0, help="0 = from the memory model")
parser.add_argument("--mem-fraction", type=float, default=0.75)
parser.add_argument("--linalg", default="auto", choices=["auto", "batched", "native"])
parser.add_argument("--walker-qr", default="auto", choices=["auto", "cholesky", "native"])
parser.add_argument("--energy", default="blocked", choices=["blocked", "dense"])
parser.add_argument("--trial-cache", default=os.path.join(HERE, "trial_cache"))
parser.add_argument("--compile-cache", default=os.path.expanduser("~/.cache/trot_jax_compile"))
parser.add_argument("--out", default="sweep_sq_gpu")
args = parser.parse_args()

n_sites = args.Lx * args.Ly
n_up = n_sites // 2 if args.n_up is None else args.n_up
n_down = n_sites // 2 if args.n_down is None else args.n_down
lattice = f"sq{args.Lx}x{args.Ly}{BOUNDARY_CODE[args.bx]}{BOUNDARY_CODE[args.by]}"
if (n_up, n_down) != (n_sites // 2, n_sites // 2):
    lattice += f"_n{n_up}-{n_down}"

# The host setup is BLAS-bound: log what the step really got (srun can bind it to fewer CPUs than allocated).
_a = np.random.default_rng(0).random((2048, 2048))
_a @ _a
_t = time.perf_counter()
_a @ _a
print(f"host: {len(os.sched_getaffinity(0))} CPUs usable, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
      f"numpy float64 GEMM {2 * 2048 ** 3 / (time.perf_counter() - _t) / 1e9:.0f} GFLOP/s", flush=True)

out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
results = os.path.join(out, "results.jsonl")


def hms(seconds):
    minutes = max(10, 10 * math.ceil(seconds / 600))  # whole 10 minutes, at least 10
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def timing(tag, wall, log_path):
    """This run's wall time split into setup, compile and blocks, from its results.jsonl record."""
    record = None
    if os.path.exists(results):
        with open(results) as stream:
            for line in stream:
                r = json.loads(line)
                if r.get("tag") == tag:
                    record = r  # the last one: a rerun into the same folder appends
    if record is None:
        return f"{tag}: no result record, no timing"
    done = record["collapsed_after_block"] or record["n_equilibration"] + record["n_blocks"]
    compile_s, run_s = record["compile_seconds"], record["run_seconds"]
    per_block = run_s / max(done, 1)
    setup = wall - compile_s - run_s
    with open(log_path) as stream:
        dmrg_ran = "trial loaded from" not in stream.read()
    line = (f"{tag} on {record['device']}: wall {wall:.0f} s = setup {setup:.0f} s{' (incl. DMRG)' if dmrg_ran else ''}"
            f" + compile {compile_s:.0f} s + {done} blocks x {per_block:.2f} s"
            f"  [{record['walker_steps_per_s']:.0f} walker-steps/s, n_chunks {record['n_chunks_used']}"
            f" (energy {record['energy_chunks']}), peak {(record.get('peak_bytes') or 0) / 1e9:.1f} GB]")
    if record.get("setup_seconds"):
        line += "\n    setup stages: " + ", ".join(f"{k} {v:.0f} s" for k, v in record["setup_seconds"].items())
    if args.project_blocks:
        total = setup + compile_s + args.project_blocks * per_block
        line += (f"\n    projected for {args.project_blocks} blocks: {total / 60:.1f} min"
                 f"{' (DMRG included: cached for the next runs)' if dmrg_ran else ''}; suggested --time {hms(2 * total)}")
    return line


failed, summary = [], []
for chi, chi_w in itertools.product(args.trial_chi, args.chi_w):
    tag = f"{lattice}_U{args.U:g}_T{chi}_w{chi_w}"
    tag += ("_NOplan" if args.plan_reference == "natural" else "") + ("_NOstart" if args.walker_start == "natural" else "")
    tag += ("" if args.orbital_plan == "adaptive" else f"_{args.orbital_plan}") + f"_s{args.seed}" + args.tag_suffix
    cfg = dict(Lx=args.Lx, Ly=args.Ly, boundary_x=args.bx, boundary_y=args.by, n_up=n_up, n_down=n_down,
               interaction=args.U, trial_chi=chi, dmrg_sweeps=args.dmrg_sweeps, dmrg_mpo=args.dmrg_mpo,
               cache_htrial=args.cache_htrial, orbital_plan=args.orbital_plan,
               walker_channel_chi=chi_w, plan_reference=args.plan_reference, walker_start=args.walker_start,
               n_walkers=args.walkers, n_equilibration=args.eql, n_blocks=args.blocks, n_steps=args.steps,
               dt=args.dt, seed=args.seed, n_chunks=args.n_chunks, mem_fraction=args.mem_fraction,
               linalg=args.linalg, walker_qr=args.walker_qr, energy=args.energy, trial_cache=args.trial_cache,
               compile_cache=args.compile_cache, result_json=results,
               block_log=os.path.join(out, "blocks.jsonl"), tag=tag)
    if args.save_walkers:
        cfg.update(walker_snapshots=os.path.join(out, f"{tag}_walkers.npz"),
                   trial_export=os.path.join(out, f"dmrg_trial_{lattice}_U{args.U:g}_chi{chi}.npz"))
    code = f"import sys; sys.path.insert(0, {HERE!r}); import mps_cpmc_2d_gpu as m; m.main(m.Config(**{cfg!r}))"
    log_path = os.path.join(out, f"{tag}.log")
    start = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.run([sys.executable, "-c", code], stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - start
    print(f"[{time.strftime('%H:%M:%S')}] mps_cpmc_2d_gpu {tag}  rc={rc}  {wall:.0f} s", flush=True)
    if rc:
        failed.append(tag)
        continue
    summary.append(timing(tag, wall, log_path))
    print("    " + summary[-1], flush=True)

print("\ntiming summary:\n" + "\n".join(summary), flush=True)
if failed:
    sys.exit(f"{len(failed)} run(s) failed: {' '.join(failed)}")
