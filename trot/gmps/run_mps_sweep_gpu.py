"""Sweep a CPMC module over L, U and the walker bond dimension on a GPU, one fresh
process per run (a copy of run_mps_sweep.py that can target either module).

--module mps_cpmc_gpu  (default) the GPU implementation
--module mps_cpmc_new  the original script, unchanged algorithm, for a GPU baseline

Each run appends its summary to <out>/results.jsonl and every block, as it
finishes, to <out>/blocks.jsonl.

    python run_mps_sweep_gpu.py --L 32 --U 4 --trial-chi 8 --chi-w 4 \
        --walkers 2048 --eql 60 --blocks 200 --out sweep_gpu

--trial-rotation BETA rotates the DMRG trial by exp(-i BETA S^y) and projects it onto the walkers' sector (90:
one-node spin projection); it adds _rot<BETA> to the run tag.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--module", default="mps_cpmc_gpu", choices=["mps_cpmc_gpu", "mps_cpmc_new"])
parser.add_argument("--L", type=int, nargs="+", default=[32])
parser.add_argument("--trial-chi", type=int, default=8)
parser.add_argument("--chi-w", type=int, nargs="+", default=[2, 4, 6])
parser.add_argument("--walkers", type=int, default=1024)
parser.add_argument("--eql", type=int, default=60)
parser.add_argument("--blocks", type=int, default=200)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--U", type=float, nargs="+", default=[4.0])
parser.add_argument("--plan-reference", default="natural", choices=["natural", "rhf"])
parser.add_argument("--walker-start", default="natural", choices=["natural", "rhf"])
parser.add_argument("--dmrg-sweeps", type=int, default=20)
parser.add_argument("--orbital-plan", default="adaptive", choices=["adaptive", "rank_exact", "maximal"])
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--tag-suffix", default="", help="appended to every run tag")
parser.add_argument("--trial-rotation", type=float, default=0.0,
                    help="mps_cpmc_gpu: rotate the trial by exp(-i beta S^y), degrees, and project onto the sector")
parser.add_argument("--save-walkers", action="store_true",
                    help="mps_cpmc_gpu: save every block's walkers to <out>/<tag>_walkers.npz and the DMRG trial to "
                         "<out>/dmrg_trial_L<L>_U<U>_chi<trial chi>.npz (the notebooks' formats)")
# mps_cpmc_gpu only
parser.add_argument("--n-chunks", type=int, default=0, help="0 = from the memory model")
parser.add_argument("--mem-fraction", type=float, default=0.75)
parser.add_argument("--linalg", default="auto", choices=["auto", "batched", "native"])
parser.add_argument("--walker-qr", default="auto", choices=["auto", "cholesky", "native"])
parser.add_argument("--energy", default="blocked", choices=["blocked", "dense"])
parser.add_argument("--trial-cache", default=os.path.join(HERE, "trial_cache"))
parser.add_argument("--compile-cache", default=os.path.expanduser("~/.cache/trot_jax_compile"))
parser.add_argument("--out", default="sweep_gpu")
args = parser.parse_args()

out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
failed = []
for L, U, chi_w in itertools.product(args.L, args.U, args.chi_w):
    tag = f"L{L}_T{args.trial_chi}_w{chi_w}" if U == 4.0 else f"L{L}_U{U:g}_T{args.trial_chi}_w{chi_w}"
    tag += ("_NOplan" if args.plan_reference == "natural" else "") + ("_NOstart" if args.walker_start == "natural" else "")
    tag += f"_rot{args.trial_rotation:g}" if args.trial_rotation else ""
    tag += f"_s{args.seed}" + args.tag_suffix
    rot = f"_rot{args.trial_rotation:g}" if args.trial_rotation else ""
    cfg = (f"L={L}, n_up={L // 2}, n_down={L // 2}, interaction={U}, trial_chi={args.trial_chi}, "
           f"walker_channel_chi={chi_w}, plan_reference={args.plan_reference!r}, walker_start={args.walker_start!r}, "
           f"n_walkers={args.walkers}, n_equilibration={args.eql}, n_blocks={args.blocks}, n_steps={args.steps}, "
           f"seed={args.seed}, dmrg_sweeps={args.dmrg_sweeps}, orbital_plan={args.orbital_plan!r}, dt={args.dt}, "
           f"tag={tag!r}, "
           f"result_json={os.path.join(out, 'results.jsonl')!r}, "
           f"block_log={os.path.join(out, 'blocks.jsonl')!r}")
    if args.module == "mps_cpmc_gpu":
        cfg += (f", n_chunks={args.n_chunks}, mem_fraction={args.mem_fraction}, linalg={args.linalg!r}, "
                f"walker_qr={args.walker_qr!r}, energy={args.energy!r}, trial_cache={args.trial_cache!r}, "
                f"compile_cache={args.compile_cache!r}, trial_rotation={args.trial_rotation}")
        if args.save_walkers:
            cfg += (f", walker_snapshots={os.path.join(out, tag + '_walkers.npz')!r}, "
                    f"trial_export={os.path.join(out, f'dmrg_trial_L{L}_U{U:g}_chi{args.trial_chi}{rot}.npz')!r}")
    elif args.save_walkers or args.trial_rotation:
        sys.exit("--save-walkers and --trial-rotation need --module mps_cpmc_gpu")
    code = f"import sys; sys.path.insert(0, {HERE!r}); import {args.module} as m; m.main(m.Config({cfg}))"
    start = time.time()
    with open(os.path.join(out, f"{tag}.log"), "w") as log:
        rc = subprocess.run([sys.executable, "-c", code], stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {args.module} {tag}  rc={rc}  {time.time() - start:.0f} s", flush=True)
    if rc:
        failed.append(tag)
if failed:
    sys.exit(f"{len(failed)} run(s) failed: {' '.join(failed)}")
