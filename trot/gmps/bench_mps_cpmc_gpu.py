"""End-to-end GPU utilisation benchmark for mps_cpmc_gpu.

For every (trial chi, walker chi) the setup is built once (the DMRG trial is
cached on disk). Then, for every walker count, the script compiles and times
the propagation of one block (a scan over 2*steps half steps) and one energy
measurement. It reports:

  walker-steps/s, ms per step, compile time, XLA's memory and FLOP analysis of
  the compiled propagation, achieved FP64 GFLOP/s (XLA counts GEMMs and
  elementwise ops but not the eigh/Cholesky custom calls) and the custom calls
  in the step.

and checks that the GPU is used as intended:

  * backend is a GPU and arrays are float64;
  * the timed calls run under jax.transfer_guard("disallow"), so the step makes
    no host round trips;
  * throughput scaling: walker-steps/s at the largest walker count must be at
    least --min-scaling times that at the smallest, otherwise batching is not
    paying off;
  * the memory model's prediction is printed next to XLA's actual temp bytes;
  * --baseline also times the original mps_cpmc_new step on the same GPU and
    config (walker counts up to --baseline-max-walkers), for a speedup.

    python bench_mps_cpmc_gpu.py --L 32 --trial-chi 8 32 128 --chi-w 4 8 16 \
        --walkers 128 512 2048 4096 --baseline --out bench_gpu.jsonl
"""
import argparse
import dataclasses
import itertools
import json
import os
import re
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import lax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mps_cpmc_gpu as g  # noqa: E402

# Dense FP64 peak (tensor cores where available), TFLOP/s, for the "% of peak" column.
FP64_PEAK = {"H100": 67.0, "H200": 67.0, "A100": 19.5, "V100": 7.8, "L40": 1.4, "A40": 0.6}


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


def timed(compiled, args, repeats):
    """Best-of time of an AOT-compiled call, under a transfer guard: all inputs are
    already on the device, so any implicit host<->device copy raises."""
    jax.block_until_ready(compiled(*args))  # warm-up (not guarded: first-call setup)
    best = float("inf")
    with jax.transfer_guard("disallow"):
        for _ in range(repeats):
            t = time.perf_counter()
            jax.block_until_ready(compiled(*args))
            best = min(best, time.perf_counter() - t)
    return best


def is_oom(error):
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower()


def propagate_fn(half_step, n_half):
    def propagate(state, data):
        return lax.scan(lambda s, i: (half_step(s, i, data), None), state, jnp.arange(n_half))[0]
    return propagate


def baseline_fn(setup, params, steps):
    """The original script's step (mps_cpmc_new.make_fast_prop_ops), unchanged."""
    import mps_cpmc_new as ref
    from trot.prop.hubbard_cpmc_ops import _build_prop_ctx

    (Ra, Rb), (plan_a, plan_b), (bond_a, bond_b) = setup.references, setup.plans, setup.bonds
    trial_np, trial_charges = setup.trial
    ref_ops = ref.make_walker_ops(Ra, Rb, plan_a, plan_b, bond_a, bond_b, trial_np, trial_charges)
    prop = ref.make_fast_prop_ops(setup.ham, "unrestricted", ref_ops.overlap, ref_ops.sweep)
    prop_ctx = _build_prop_ctx(setup.ham, params.dt)
    params = dataclasses.replace(params, n_chunks=1)  # the original's default: one vmap over all walkers

    def propagate(state):
        step = lambda s, _: (prop.step(s, params=params, ham_data=setup.ham, trial_data=setup.trial_data,
                                       trial_ops=None, meas_ops=None, meas_ctx=None, prop_ctx=prop_ctx), None)
        return lax.scan(step, state, None, length=steps)[0]
    return propagate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--trial-chi", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--chi-w", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--walkers", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--steps", type=int, default=5, help="CPMC steps per timed call")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dmrg-sweeps", type=int, default=10)
    parser.add_argument("--trial-cache", default=os.path.join(HERE, "trial_cache"))
    parser.add_argument("--compile-cache", default=os.path.expanduser("~/.cache/trot_jax_compile"))
    parser.add_argument("--linalg", default="auto", choices=["auto", "batched", "native"])
    parser.add_argument("--walker-qr", default="auto", choices=["auto", "cholesky", "native"])
    parser.add_argument("--n-chunks", type=int, default=0)
    parser.add_argument("--mem-fraction", type=float, default=0.75)
    parser.add_argument("--no-energy", action="store_true", help="skip timing the energy measurement")
    parser.add_argument("--baseline", action="store_true", help="also time the original mps_cpmc_new step")
    parser.add_argument("--baseline-max-walkers", type=int, default=1024)
    parser.add_argument("--min-scaling", type=float, default=4.0)
    parser.add_argument("--profile", default="",
                        help="directory for a jax.profiler trace of one call (needs cuPTI, which the "
                             "python/3.12.13 module's CUDA plugin cannot find)")
    parser.add_argument("--out", default="bench_mps_cpmc_gpu.jsonl")
    args = parser.parse_args()

    device = jax.devices()[0]
    backend = jax.default_backend()
    peak = fp64_peak(device.device_kind)
    dtype = jnp.asarray(1.0).dtype
    print(f"jax {jax.__version__}, backend {backend}, device {device.device_kind}, default float {dtype}, "
          f"FP64 peak {peak} TFLOP/s")
    failures = []
    if backend != "gpu":
        failures.append(f"backend is {backend}, not gpu")
    if dtype != jnp.float64:
        failures.append(f"default float is {dtype}, not float64")
    walkers = sorted(set(args.walkers))

    for trial_chi, chi_w in itertools.product(args.trial_chi, args.chi_w):
        cfg = g.Config(L=args.L, n_up=args.L // 2, n_down=args.L // 2, interaction=args.U, trial_chi=trial_chi,
                       walker_channel_chi=chi_w, n_walkers=walkers[-1], n_steps=args.steps,
                       dmrg_sweeps=args.dmrg_sweeps, trial_cache=args.trial_cache,
                       compile_cache=args.compile_cache, linalg=args.linalg, walker_qr=args.walker_qr,
                       mem_fraction=args.mem_fraction, self_check=False)
        print(f"\n=== L={args.L} U={args.U} trial chi={trial_chi} walker chi={chi_w} ===", flush=True)
        setup = g.build(cfg, verbose=True)
        ops = setup.ops
        limit = g.device_bytes_limit()
        budget = None if limit is None else args.mem_fraction * limit - setup.memory["data_bytes"]
        probe_fn = jax.jit(ops.probe)
        rates = {}
        print(f"{'walkers':>8} {'chunks':>6} {'compile s':>9} {'ms/step':>9} {'walker-steps/s':>15} "
              f"{'GFLOP/s':>9} {'%peak':>6} {'temp GB':>8} {'model GB':>8} {'energy ms':>9}")
        for nw in walkers:
            params = dataclasses.replace(setup.params, n_walkers=nw)
            n_chunks = args.n_chunks or g.choose_chunks(nw, setup.memory["step_bytes_per_walker"], budget)
            energy_chunks = args.n_chunks or g.choose_chunks(nw, setup.memory["energy_bytes_per_walker"], budget)
            record = dict(L=args.L, U=args.U, trial_chi=trial_chi, chi_w=chi_w, n_walkers=nw, steps=args.steps,
                          n_chunks=n_chunks, energy_chunks=energy_chunks, device=device.device_kind,
                          jax=jax.__version__, **{k: setup.info[k] for k in ("linalg", "walker_qr", "spin_batched",
                                                                            "circuit", "overlap_plan")})
            try:
                state, _ = g.init_state(ops, setup.system, setup.trial_data, params, probe_fn)
                propagate = jax.jit(propagate_fn(g.make_half_step(ops, params, n_chunks), 2 * args.steps))
                t0 = time.perf_counter()
                compiled = propagate.lower(state, ops.data).compile()
                record["compile_seconds"] = time.perf_counter() - t0
                seconds = timed(compiled, (state, ops.data), args.repeats)
                if args.profile and nw == walkers[-1]:
                    path = os.path.join(args.profile, f"T{trial_chi}_w{chi_w}_n{nw}")
                    with jax.profiler.trace(path):
                        jax.block_until_ready(compiled(state, ops.data))
                    record["profile"] = path
                flops = flops_of(compiled)
                record.update(seconds=seconds, ms_per_step=1e3 * seconds / args.steps,
                              walker_steps_per_s=nw * args.steps / seconds, xla_flops=flops,
                              gflops_per_s=None if flops is None else flops / seconds / 1e9,
                              predicted_step_bytes=nw / n_chunks * setup.memory["step_bytes_per_walker"],
                              **g.compiled_memory(compiled), **hlo_census(compiled))
                if not args.no_energy:
                    measure = jax.jit(lambda ca, cb, data, c=energy_chunks: g.chunked(
                        lambda a, b: ops.energies(a, b, data), c, ca, cb))
                    t0 = time.perf_counter()
                    measured = measure.lower(*state.walkers, ops.data).compile()
                    record["energy_compile_seconds"] = time.perf_counter() - t0
                    record["energy_ms"] = 1e3 * timed(measured, (*state.walkers, ops.data), args.repeats)
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
            print(f"{nw:>8} {n_chunks:>6} {record['compile_seconds']:9.1f} {record['ms_per_step']:9.2f} "
                  f"{record['walker_steps_per_s']:15.1f} {gf or float('nan'):9.1f} {pct or float('nan'):6.2f} "
                  f"{record.get('temp_bytes', 0) / 1e9:8.2f} {record['predicted_step_bytes'] / 1e9:8.2f} "
                  f"{record.get('energy_ms', float('nan')):9.2f}", flush=True)
            if nw == walkers[0]:
                print(f"         custom calls in the step: {record.get('custom_calls')}; while loops "
                      f"{record.get('while_loops')}, conditionals {record.get('conditionals')}")

            if args.baseline and nw <= args.baseline_max_walkers:
                try:
                    base = jax.jit(baseline_fn(setup, params, args.steps))
                    t0 = time.perf_counter()
                    base_compiled = base.lower(state).compile()
                    record["baseline_compile_seconds"] = time.perf_counter() - t0
                    base_seconds = timed(base_compiled, (state,), args.repeats)
                    record["baseline_walker_steps_per_s"] = nw * args.steps / base_seconds
                    record["speedup_vs_baseline"] = base_seconds / record["seconds"]
                    print(f"{'':>8} baseline (mps_cpmc_new on this GPU): {1e3 * base_seconds / args.steps:9.2f} "
                          f"ms/step, compile {record['baseline_compile_seconds']:.1f} s -> speedup "
                          f"{record['speedup_vs_baseline']:.1f}x", flush=True)
                except Exception as error:
                    if not is_oom(error):
                        raise
                    record["baseline_oom"] = True
                    print(f"{'':>8} baseline out of device memory")

            with open(args.out, "a") as stream:
                stream.write(json.dumps(record, default=str) + "\n")

        if len(rates) > 1:
            lo, hi = min(rates), max(rates)
            scaling = rates[hi] / rates[lo]
            ok = scaling >= min(args.min_scaling, hi / lo)
            print(f"throughput scaling {lo} -> {hi} walkers: {scaling:.1f}x "
                  f"({'OK' if ok else 'POOR: the step is not batching walkers efficiently'})")
            if not ok:
                failures.append(f"T={trial_chi} w={chi_w}: throughput only {scaling:.1f}x from {lo} to {hi} walkers")

    print("\nsummary:", "all checks passed" if not failures else "; ".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
