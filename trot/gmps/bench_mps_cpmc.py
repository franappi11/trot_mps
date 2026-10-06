"""Benchmark of MPS-CPMC through trot's driver (trot.gmps.driver), CPU or GPU, chain or any lattice.

The lattice, trial and engine flags are those of run_mps_cpmc.py (--trial-chi, --chi-w and --walkers take several
values). For every (trial chi, walker chi) the trial is built once (cached on disk) and the run prepared once at the
largest walker count (driver.prepare_mps_cpmc); every smaller count uses the first walkers of that start. For every
walker count the script compiles and times

  * step: --steps CPMC steps, the scan over 2 x steps half steps of the batched engine (engine.make_half_step),
  * energy: one local-energy measurement of every walker,
  * block: one production block as a run executes it, trot.driver.make_run_blocks with driver.make_mps_block
    (the steps, the walker QR, the energy and the comb),

and reports walker-steps/s, ms per step, compile times, XLA's memory and FLOP analysis of the step, the achieved FP64
GFLOP/s (XLA counts GEMMs and elementwise ops, not the eigh/Cholesky custom calls) and the custom calls in the step.
Timed calls run under jax.transfer_guard("disallow"): every input is on the device, so a host round trip raises.
Throughput must grow by --min-scaling from the smallest to the largest walker count, or batching is not paying off.

--profile DIR records a jax.profiler trace of one step call and summarises it (top kernels, kernel classes, busy
fraction). --circuit-only prints each conversion circuit's statistics, padding waste included, for every --bucket-sets
setting: host work only, any backend.

    python bench_mps_cpmc.py --L 100 --U 8 --trial-chi 8 16 --chi-w 32 --walkers 400 --dmrg-sweeps 30 \\
        --dmrg-init warm --trial-cache trial_cache_warm --steps 5 --out bench_mps_cpmc.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import itertools
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from trot.gmps import run_mps_cpmc as cli

# Dense FP64 peak (tensor cores where available), TFLOP/s, for the "% of peak" column.
FP64_PEAK = {"H100": 67.0, "H200": 67.0, "A100": 19.5, "V100": 7.8, "L40": 1.4, "A40": 0.6}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.add_lattice_arguments(parser)
    trial = parser.add_argument_group("trial")
    trial.add_argument("--trial-chi", type=int, nargs="+", required=True)
    cli.add_trial_arguments(trial)
    walkers = parser.add_argument_group("walkers")
    walkers.add_argument("--chi-w", type=int, nargs="+", default=[32], help="walker bond per spin channel (0: exact)")
    cli.add_walker_plan_arguments(walkers)
    walkers.add_argument("--walkers", type=int, nargs="+", default=[400])
    walkers.add_argument("--steps", type=int, default=5, help="CPMC steps per timed call and per timed block")
    walkers.add_argument("--dt", type=float, default=0.005)
    walkers.add_argument("--weight-floor", type=float, default=1.0e-8)
    walkers.add_argument("--seed", type=int, default=1234)
    group = parser.add_argument_group("engine")
    cli.add_engine_arguments(group)
    group.add_argument("--n-chunks", type=int, default=1, help="walker chunks of every timed call")
    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--no-energy", action="store_true", help="skip timing the energy measurement")
    bench.add_argument("--no-block", action="store_true", help="skip timing the production block")
    bench.add_argument("--min-scaling", type=float, default=4.0)
    bench.add_argument("--profile", default="", help="directory for a jax.profiler trace of one step call "
                       "(needs cuPTI, which the python/3.12.13 module's CUDA plugin cannot find)")
    bench.add_argument("--circuit-only", action="store_true",
                       help="only print each conversion circuit's statistics for every --bucket-sets setting")
    bench.add_argument("--bucket-sets", nargs="*", default=[],
                       help="with --circuit-only: bucket settings to compare, e.g. none 16 8,16 4,8,16")
    bench.add_argument("--out", default="bench_mps_cpmc.jsonl")
    args = parser.parse_args(argv)
    cli.check_lattice_arguments(parser, args)
    return args


# ---------------------------------------------------------------------------------------------
# Compiled-call analysis and timing
# ---------------------------------------------------------------------------------------------


def fp64_peak(kind):
    """Peak of the card, scaled to the slice's share for a MIG device ("MIG 2g.20gb": 2/7)."""
    share = 1.0
    mig = re.search(r"MIG (\d)g\.", kind)
    if mig:
        share = int(mig.group(1)) / 7
    for key, value in FP64_PEAK.items():
        if key in kind.upper():
            return value * share
    return None


def flops_of(compiled):
    try:
        cost = compiled.cost_analysis()
    except Exception:
        return None
    if isinstance(cost, (list, tuple)):
        cost = cost[0] if cost else {}
    return float(cost.get("flops", 0.0)) if cost else None


def compiled_memory(compiled):
    try:
        m = compiled.memory_analysis()
        return dict(temp_bytes=int(m.temp_size_in_bytes), argument_bytes=int(m.argument_size_in_bytes),
                    output_bytes=int(m.output_size_in_bytes))
    except Exception:  # not every backend reports it
        return {}


def hlo_census(compiled):
    try:
        text = compiled.as_text()
    except Exception:
        return {}
    calls = re.findall(r'custom_call_target="([^"]+)"', text)
    census = {name: calls.count(name) for name in sorted(set(calls))}
    return dict(custom_calls=census, while_loops=len(re.findall(r"\bwhile\(", text)),
                conditionals=len(re.findall(r"\bconditional\(", text)),
                fusions=len(re.findall(r"\bfusion\(", text)))


def compile_timed(fn, args):
    """(compiled, compile seconds) of jax.jit(fn) for args."""
    import jax

    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    return compiled, time.perf_counter() - start


def timed(compiled, args, repeats):
    """Best-of time of an AOT-compiled call, under a transfer guard: all inputs are
    already on the device, so any implicit host<->device copy raises."""
    import jax

    jax.block_until_ready(compiled(*args))  # warm-up (not guarded: first-call setup)
    best = float("inf")
    with jax.transfer_guard("disallow"):
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(compiled(*args))
            best = min(best, time.perf_counter() - start)
    return best


def is_oom(error):
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower()


def summarize_trace(trace_dir, wall_seconds, n_calls=1, top=25):
    """Top GPU kernels, kernel classes and busy fraction from the Perfetto/Chrome trace of jax.profiler."""
    files = sorted(Path(trace_dir).rglob("*perfetto_trace.json.gz")) or sorted(Path(trace_dir).rglob("*.trace.json.gz"))
    if not files:
        return dict(error=f"no trace json under {trace_dir}")
    with gzip.open(files[-1], "rt") as stream:
        data = json.load(stream)
    events = data["traceEvents"] if isinstance(data, dict) else data
    process = {e["pid"]: e.get("args", {}).get("name", "") for e in events
               if e.get("ph") == "M" and e.get("name") == "process_name"}
    thread = {(e["pid"], e.get("tid")): e.get("args", {}).get("name", "") for e in events
              if e.get("ph") == "M" and e.get("name") == "thread_name"}
    gpu_pids = {pid for pid, name in process.items() if "GPU" in name.upper() and "HOST" not in name.upper()}
    device = [e for e in events if e.get("ph") == "X" and e.get("pid") in gpu_pids and "dur" in e]
    kernels = [e for e in device if "stream" in str(thread.get((e["pid"], e.get("tid")), "")).lower()] or device
    by_name = defaultdict(lambda: [0.0, 0])
    for e in kernels:
        by_name[e.get("name", "?")][0] += e["dur"]
        by_name[e.get("name", "?")][1] += 1
    total = sum(v[0] for v in by_name.values()) or 1e-12
    patterns = [("solver", r"cusolver|syevd|syevj|geqrf|orgqr|potrf|getrf|gesvd|trsm|jacobi|cholesky|larf"),
                ("gemm", r"gemm|cutlass|cublas|xmma|matmul|dot"), ("copy", r"memcpy|memset|copy"),
                ("fusion", r"fusion|loop|reduce|select|broadcast|slice|scatter|gather|concat|transpose|iota")]
    classes = defaultdict(float)
    for name, (dur, _) in by_name.items():
        label = next((lab for lab, pat in patterns if re.search(pat, name, re.IGNORECASE)), "other")
        classes[label] += dur / 1e6
    busy, end = 0.0, -float("inf")
    for a, b in sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels):
        busy += max(0.0, b - max(a, end))
        end = max(end, b)
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:top]
    return dict(file=str(files[-1]), kernels_per_call=len(kernels) / max(n_calls, 1),
                mean_kernel_us=total / max(len(kernels), 1), kernel_seconds=total / 1e6,
                busy_fraction=busy / 1e6 / wall_seconds if wall_seconds else None, classes=dict(classes),
                top=[dict(name=n[:140], count=c, share=d / total, mean_us=d / c) for n, (d, c) in ranked])


def profile_call(compiled, args, path):
    """One traced call of compiled(*args) into path, summarised."""
    import jax

    try:
        context = jax.profiler.trace(path, create_perfetto_trace=True)
    except TypeError:
        context = jax.profiler.trace(path)
    start = time.perf_counter()
    with context:
        jax.block_until_ready(compiled(*args))
    summary = summarize_trace(path, time.perf_counter() - start)
    if "error" not in summary:
        print(f"         trace: {summary['kernels_per_call']:.0f} kernels per call, "
              f"mean {summary['mean_kernel_us']:.1f} us,"
              f" GPU busy {100 * (summary['busy_fraction'] or 0):.0f}% of the wall time; classes (s) "
              + ", ".join(f"{k} {v:.3f}" for k, v in sorted(summary["classes"].items(), key=lambda kv: -kv[1])))
        for row in summary["top"][:10]:
            print(f"           {100 * row['share']:5.1f}%  {row['count']:7d} x {row['mean_us']:8.1f} us  "
                  f"{row['name'][:90]}")
    return summary


# ---------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------


def make_params(args, chi, chi_w, n_walkers, sweeps):
    from trot.prop.types import QmcParamsMps

    return QmcParamsMps(
        dt=args.dt, n_walkers=n_walkers, n_prop_steps=args.steps, n_eql_blocks=1, n_blocks=1,
        weight_floor=args.weight_floor, seed=args.seed, n_chunks=args.n_chunks, auto_n_chunks=False, trial_chi=chi,
        dmrg_sweeps=sweeps, dmrg_seed=args.dmrg_seed, orbital_plan=args.orbital_plan,
        walker_channel_chi=chi_w if chi_w > 0 else None, plan_reference=args.plan_reference,
        walker_start=args.walker_start, energy_kernel=args.energy, walker_qr=args.walker_qr,
        sector_buckets=tuple(args.sector_buckets))


def first_walkers(state, n):
    """The state of the first n walkers (the e_estimate, RNG key and counters unchanged)."""
    return state._replace(walkers=tuple(w[:n] for w in state.walkers), weights=state.weights[:n],
                          overlaps=state.overlaps[:n])


def circuit_report(args, h1, lattice, nelec, system, ham):
    """--circuit-only: the conversion circuits of each (trial chi, walker chi) for every bucket setting, with
    their padding statistics (engine.padding_stats); host work only (plans, circuit compilation)."""
    from trot.gmps import engine
    from trot.trial.mps import make_walker_plan

    settings = [() if b in ("none", "") else tuple(int(x) for x in b.split(","))
                for b in (args.bucket_sets or ["none"])]
    keys = ("ops", "qr_calls", "eigh_calls", "eigh_le16", "max_eigh", "eigh_n_median", "eigh_n_p90", "eigh_waste",
            "qr_waste")
    for chi, chi_w in itertools.product(args.trial_chi, args.chi_w):
        trial = cli.build_trial(args, h1, lattice, nelec, chi).trial
        params = make_params(args, chi, chi_w, 1, cli.dmrg_sweeps(args, lattice))
        plan = make_walker_plan(ham, trial, system, params)
        (plan_a, plan_b), (bond_a, bond_b) = plan.orbital_plans, plan.bond_plans
        gates = int(plan_a.block_sizes.sum() - len(plan_a.block_sizes))
        print(f"\n=== {lattice.name} U={args.U:g} trial chi={chi} walker chi={chi_w}: gates {gates} ===", flush=True)
        for buckets in settings:
            converter = engine.make_converter(plan_a, plan_b, bond_a, bond_b, spin_batch=True,
                                           buckets=buckets or None)
            label = ",".join(map(str, buckets)) or "none"
            for j, circuit in enumerate(converter.circuits):
                stats = engine.circuit_stats(circuit)
                print(f"  buckets {label:>12}  circuit {j}: " + ", ".join(f"{k} {stats[k]}" for k in keys), flush=True)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    from trot import config

    config.configure_once()

    import jax
    import jax.numpy as jnp
    from jax import lax

    from trot.core.system import System
    from trot.driver import make_run_blocks
    from trot.gmps import driver, engine, trials
    from trot.ham.hubbard import HamHubbard

    if args.compile_cache:
        jax.config.update("jax_compilation_cache_dir", args.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    h1 = cli.make_h1(args)
    lattice = trials.describe_h1(h1)
    nelec, sweeps = cli.electrons(args, lattice), cli.dmrg_sweeps(args, lattice)
    system = System(norb=lattice.n_sites, nelec=nelec, walker_kind="unrestricted")
    ham = HamHubbard(h1=jnp.asarray(h1), u=args.U)
    device, backend = jax.devices()[0], jax.default_backend()
    peak, dtype = fp64_peak(device.device_kind), jnp.asarray(1.0).dtype
    print(f"jax {jax.__version__}, backend {backend}, device {device.device_kind}, default float {dtype}, "
          f"FP64 peak {peak} TFLOP/s; {lattice.name} ({lattice.kind}, {lattice.n_sites} sites), nelec {nelec}, "
          f"U={args.U:g}")
    if args.circuit_only:
        circuit_report(args, h1, lattice, nelec, system, ham)
        return
    failures = []
    if dtype != jnp.float64:
        failures.append(f"default float is {dtype}, not float64")
    if backend != "gpu":
        print(f"note: backend {backend}; the GFLOP/s and scaling checks are meant for a GPU")
    walkers = sorted(set(args.walkers))

    for chi, chi_w in itertools.product(args.trial_chi, args.chi_w):
        print(f"\n=== {lattice.name} U={args.U:g} trial chi={chi} walker chi={chi_w} ===", flush=True)
        built = cli.build_trial(args, h1, lattice, nelec, chi)
        params = make_params(args, chi, chi_w, walkers[-1], sweeps)
        start = time.perf_counter()
        run = driver.prepare_mps_cpmc(sys=system, params=params, ham_data=ham, trial=built.trial, htrial=built.htrial)
        prepare_seconds = time.perf_counter() - start
        kernels = run.meas_ctx.kernels
        info = {k: run.info.get(k) for k in ("walker_qr", "spin_batched", "circuits", "overlap_plan",
                                             "energy_plan", "trial_bonds_max", "htrial_bonds_max", "gates")}
        rates = {}
        print(f"prepared in {prepare_seconds:.1f} s (walker start at {walkers[-1]} walkers included)")
        print(f"{'walkers':>8} {'chunks':>6} {'compile s':>9} {'ms/step':>9} {'walker-steps/s':>15} {'GFLOP/s':>9} "
              f"{'%peak':>6} {'temp GB':>8} {'energy ms':>9} {'block s':>8} {'blk ms/step':>11}")
        for nw in walkers:
            p = dataclasses.replace(params, n_walkers=nw)
            n_chunks = engine.divisor_at_least(nw, p.n_chunks)
            state = jax.device_put(first_walkers(run.state, nw))
            data = run.meas_ctx.data(run.prop_ctx)
            record = dict(lattice=lattice.name, kind=lattice.kind, n_sites=lattice.n_sites, U=args.U, trial_chi=chi,
                          dmrg_init=built.dmrg.init, chi_w=chi_w, n_walkers=nw, steps=args.steps, n_chunks=n_chunks,
                          trial_rotation=args.trial_rotation, rotated_trial=args.rotated_trial,
                          natural_rdm1=args.natural_rdm1, sector_buckets=list(args.sector_buckets),
                          cache_htrial=args.cache_htrial, device=device.device_kind, jax=jax.__version__,
                          prepare_seconds=prepare_seconds, **info)
            try:
                half_step = engine.make_half_step(kernels, p, n_chunks)

                def propagate(state, data, n_half=2 * args.steps, half_step=half_step):
                    return lax.scan(lambda s, i: (half_step(s, i, data), None), state, jnp.arange(n_half))[0]

                compiled, record["compile_seconds"] = compile_timed(propagate, (state, data))
                seconds = timed(compiled, (state, data), args.repeats)
                if args.profile and nw == walkers[-1]:
                    path = os.path.join(args.profile, f"{lattice.name}_T{chi}_w{chi_w}_n{nw}")
                    record.update(profile=path, trace=profile_call(compiled, (state, data), path))
                flops = flops_of(compiled)
                record.update(seconds=seconds, ms_per_step=1e3 * seconds / args.steps,
                              walker_steps_per_s=nw * args.steps / seconds, xla_flops=flops,
                              gflops_per_s=None if flops is None else flops / seconds / 1e9,
                              **compiled_memory(compiled), **hlo_census(compiled))
                if not args.no_energy:
                    def measure(ca, cb, data, n_chunks=n_chunks):
                        return engine.chunked(lambda a, b: kernels.energies(a, b, data), n_chunks, ca, cb)

                    measured, record["energy_compile_seconds"] = compile_timed(measure, (*state.walkers, data))
                    record["energy_ms"] = 1e3 * timed(measured, (*state.walkers, data), args.repeats)
                if not args.no_block:
                    run_blocks = make_run_blocks(block_fn=driver.make_mps_block(), sys=system, params=p,
                                                 trial_ops=run.ops.trial_ops, meas_ops=run.ops.meas_ops,
                                                 prop_ops=run.ops.prop_ops)

                    def one_block(state, ham_data, trial_data, meas_ctx, prop_ctx, run_blocks=run_blocks):
                        return run_blocks(state, ham_data=ham_data, trial_data=trial_data, meas_ctx=meas_ctx,
                                          prop_ctx=prop_ctx, n_blocks=1)

                    block_args = (state, *jax.device_put((run.ham_data, run.trial, run.meas_ctx, run.prop_ctx)))
                    blocked, record["block_compile_seconds"] = compile_timed(one_block, block_args)
                    block_seconds = timed(blocked, block_args, args.repeats)
                    record.update(block_seconds=block_seconds, block_ms_per_step=1e3 * block_seconds / args.steps,
                                  block_memory=compiled_memory(blocked))
            except Exception as error:
                if not is_oom(error):
                    raise
                record["oom"] = True
                print(f"{nw:>8} {n_chunks:>6}  out of device memory; larger walker counts skipped")
                with open(args.out, "a") as stream:
                    stream.write(json.dumps(record, default=str) + "\n")
                break

            rates[nw] = record["walker_steps_per_s"]
            gf = record["gflops_per_s"]
            pct = None if (gf is None or peak is None) else 100 * gf / (1e3 * peak)
            nan = float("nan")
            print(f"{nw:>8} {n_chunks:>6} {record['compile_seconds']:9.1f} {record['ms_per_step']:9.2f} "
                  f"{record['walker_steps_per_s']:15.1f} {gf or nan:9.1f} {pct or nan:6.2f} "
                  f"{record.get('temp_bytes', 0) / 1e9:8.2f} {record.get('energy_ms', nan):9.2f} "
                  f"{record.get('block_seconds', nan):8.3f} {record.get('block_ms_per_step', nan):11.2f}", flush=True)
            if nw == walkers[0]:
                print(f"         custom calls in the step: {record.get('custom_calls')}; while loops "
                      f"{record.get('while_loops')}, conditionals {record.get('conditionals')}")
            with open(args.out, "a") as stream:
                stream.write(json.dumps(record, default=str) + "\n")

        if len(rates) > 1:
            lo, hi = min(rates), max(rates)
            scaling = rates[hi] / rates[lo]
            ok = scaling >= min(args.min_scaling, hi / lo)
            print(f"throughput scaling {lo} -> {hi} walkers: {scaling:.1f}x "
                  f"({'OK' if ok else 'POOR: the step is not batching walkers efficiently'})")
            if not ok:
                failures.append(f"T={chi} w={chi_w}: throughput only {scaling:.1f}x from {lo} to {hi} walkers")

    print("\nsummary:", "all checks passed" if not failures else "; ".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
