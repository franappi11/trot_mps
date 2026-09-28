"""Sweep mps_cpmc_new.py over L, U and the walker bond dimension, one fresh process per run.

Each run appends its summary to <out>/results.jsonl and every block, as it
finishes, to <out>/blocks.jsonl (see make_block_logger).

    ~/.trot_venv/bin/python run_mps_sweep.py --L 32 48 --U 4 8 --trial-chi 8 --chi-w 2 4 6 \
        --walkers 200 --eql 60 --blocks 200 --out sweep_T8
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, nargs="+", default=[32])
parser.add_argument("--trial-chi", type=int, default=8)
parser.add_argument("--chi-w", type=int, nargs="+", default=[2, 4, 6])
parser.add_argument("--walkers", type=int, default=200)
parser.add_argument("--eql", type=int, default=60)
parser.add_argument("--blocks", type=int, default=200)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--U", type=float, nargs="+", default=[4.0])
parser.add_argument("--plan-reference", default="natural", choices=["natural", "rhf"])
parser.add_argument("--walker-start", default="natural", choices=["natural", "rhf"])
parser.add_argument("--dmrg-sweeps", type=int, default=20)  # trial DMRG; 14 left L=48, U>=8 unconverged
parser.add_argument("--out", default="sweep")
args = parser.parse_args()

out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
failed = []
for L, U, chi_w in itertools.product(args.L, args.U, args.chi_w):
    tag = f"L{L}_T{args.trial_chi}_w{chi_w}" if U == 4.0 else f"L{L}_U{U:g}_T{args.trial_chi}_w{chi_w}"
    tag += ("_NOplan" if args.plan_reference == "natural" else "") + ("_NOstart" if args.walker_start == "natural" else "")
    tag += f"_s{args.seed}"
    cfg = (f"L={L}, n_up={L // 2}, n_down={L // 2}, interaction={U}, trial_chi={args.trial_chi}, "
           f"walker_channel_chi={chi_w}, plan_reference={args.plan_reference!r}, walker_start={args.walker_start!r}, n_walkers={args.walkers}, n_equilibration={args.eql}, "
           f"n_blocks={args.blocks}, seed={args.seed}, dmrg_sweeps={args.dmrg_sweeps}, tag={tag!r}, "
           f"result_json={os.path.join(out, 'results.jsonl')!r}, "
           f"block_log={os.path.join(out, 'blocks.jsonl')!r}")
    code = f"import sys; sys.path.insert(0, {HERE!r}); import mps_cpmc_new as m; m.main(m.Config({cfg}))"
    start = time.time()
    with open(os.path.join(out, f"{tag}.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-c", code], stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {tag}  rc={rc}  {time.time() - start:.0f} s", flush=True)
    if rc:
        failed.append(tag)
if failed:
    sys.exit(f"{len(failed)} run(s) failed: {' '.join(failed)}")
