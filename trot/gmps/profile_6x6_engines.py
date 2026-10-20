"""Where the time of a doped 6x6 MPS-CPMC block goes, for the sector engine and the dense one, stage by stage.

The setup is ex_6x6_doped_diag.py's: the saved DMRG trial rotated by 90 degrees, its cached H|trial>, the same
parameters, walker bond --chi-w. The sector engine measures with its own plan, as the 36 s/block run 7256267 did.
The dense one (trot.gmps.dense_engine) measures with a sector plan at the same bond, as the 61 s/block runs 7284502
and 7285062 did. So the measurement stages run the same kernels in both, and only the propagation differs. The walkers
are a saved population: by default block 30 of the dense timing run, 400 walkers. Their overlaps are recomputed per
engine (prepare_mps_cpmc with recompute_overlaps). For each engine it compiles and times, on all walkers in one chunk:

  half_step    the CPMC half step as the block runs it (engine.make_half_step in a scan of --half-steps; half of them
               with the HS field sweep), per half step;
  walker_qr    the walkers' orthonormalisation;
  conversion   the walker conversion, both spins;
  right_envs   the right environments of the converted walkers (sector: with their overlap-plan blocks);
  field_sweep  the HS field sweep given them;
  energy@k     the local energies in chunks of k walkers (the block's TROT_MEASURE_CHUNK), k in --measure-chunks;
  szsz@k       the szsz observable likewise.

From these it projects a block: 2 n_prop_steps half steps, one walker QR, one energy and one szsz. It also prints the
stages' share of a half step, the half step against the sum of its stages (fusion), and with --profile a trace of the
half steps (top kernels, kernel classes, GPU busy fraction; bench_mps_cpmc.profile_call).

    ./submit_ex_2d.sh h100 profile_6x6_engines.py --engines sector --profile profiles/6x6
    ./submit_ex_2d.sh h100 profile_6x6_engines.py --engines dense --profile profiles/6x6
"""
from trot.config import configure_once

configure_once()  # float64

import argparse
import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from trot.core.system import System
from trot.gmps import dense_engine, driver, engine
from trot.gmps.bench_mps_cpmc import compile_timed, is_oom, profile_call, timed
from trot.gmps.ex_6x6_doped_diag import TRIALS, U, hubbard_h1, load_trial
from trot.ham.hubbard import HamHubbard
from trot.meas.mps import BATCHED_OBSERVABLES
from trot.prop.types import PropState, QmcParamsMps

DEFAULT_WALKERS = "/mnt/ceph/users/fnappi/trot_walkers/sq6x6_doped_diag/dense_timing/walkers_block0030.npz"

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/trot_jax_compile"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def make_params(kind, args, n_walkers):
    """ex_6x6_doped_diag.py's parameters for one engine."""
    values = dict(n_walkers=n_walkers, n_chunks=1, auto_n_chunks=False, n_eql_blocks=10, n_blocks=20, dt=0.005,
                  n_prop_steps=50, weight_floor=1e-8, seed=1234, occupation_tolerance=1e-10, orbital_plan="adaptive",
                  walker_channel_chi=args.chi_w, plan_reference="natural", walker_start="natural")
    if args.measure_chi:
        values.update(measure_channel_chi=args.measure_chi)
    if args.mpo:
        values.update(energy_kernel="mpo", dense_move=args.dense_move)
    if kind == "dense":
        values.update(walker_conversion="dense", dense_move=args.dense_move,
                      measure_channel_chi=args.measure_chi or args.chi_w)
    return QmcParamsMps(**values)


def energy_summary(values, clip=20.0):
    """Robust statistics of the walkers' local energies: the median, the mean of the walkers within `clip` of it (the
    block's outlier rule, |E - E_ref| > sqrt(2 / dt) = 20 is replaced by E_ref), the raw mean and std, and how many
    walkers are outliers or not finite. A few walkers with a vanishing overlap dominate the raw mean and std."""
    finite = np.isfinite(values)
    median = float(np.median(values[finite]))
    outlier = ~finite | (np.abs(np.where(finite, values, median) - median) > clip)
    clipped = np.where(outlier, median, values)
    return (f"median {median:.5f}, clipped mean {clipped.mean():.5f} ({int(outlier.sum())} outliers, "
            f"{int((~finite).sum())} not finite); raw mean {np.nanmean(values):.4f}, std {np.nanstd(values):.3f}")


def compare_with_stored(run, trial, htrial, ham, qa, qb, values, n_chunks, args):
    """The same walkers' local energies through the stored-H|trial> kernel (a sector measurement plan at the same bond,
    needs TROT_ENERGY_DEDUP=1 at chi_T 512 and the H|trial> cache) against the MPO kernel's `values`: each walker's
    difference is the two conversions' truncation difference (dense own states against frozen counts) and nothing
    else, so it is small where the walkers' overlaps are not, and O(1) for every walker if a kernel were wrong."""
    from trot.meas.mps import build_mps_meas_ctx
    from trot.trial.mps import measurement_plan

    chi = args.measure_chi or args.chi_w
    ctx = build_mps_meas_ctx(ham, trial, plan=measurement_plan(run.ops.plan, chi), kernel="blocked", htrial=htrial)
    kernels, data = ctx.kernels, ctx.data()
    fn = jax.jit(lambda a, b, d: engine.chunked(lambda x, y: kernels.energies(x, y, d), n_chunks, a, b))
    stored = np.asarray(jax.block_until_ready(fn(qa, qb, data)))
    print(f"  stored-H|trial> kernel at the same bond {chi}: {energy_summary(stored)}", flush=True)
    ok = np.isfinite(values) & np.isfinite(stored)
    delta = np.abs(values - stored)[ok]
    quantiles = ", ".join(f"{q}%: {np.percentile(delta, q):.2e}" for q in (50, 90, 99))
    print(f"  |MPO - stored| per walker ({int(ok.sum())} walkers): {quantiles}, max {delta.max():.2e}; "
          f"{int((delta > 0.05).sum())} walkers differ by more than 0.05; correlation "
          f"{np.corrcoef(values[ok], stored[ok])[0, 1]:.6f}", flush=True)
    if args.save_energies:
        np.savez(args.save_energies, mpo=values, stored=stored)
        print(f"  wrote {args.save_energies}", flush=True)


def device_gb(tree):
    return sum(int(x.nbytes) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "nbytes")) / 2 ** 30


def saved_state(path, n):
    """A PropState of the first n saved walkers (equal weights; overlaps recomputed by prepare_mps_cpmc)."""
    with np.load(path) as data:
        up, dn = np.asarray(data["up"])[:n], np.asarray(data["dn"])[:n]
    n = len(up)
    return PropState(walkers=(jnp.asarray(up), jnp.asarray(dn)), weights=jnp.ones(n), overlaps=jnp.ones(n),
                     rng_key=jax.random.PRNGKey(0), pop_control_ene_shift=jnp.asarray(-22.8),
                     e_estimate=jnp.asarray(-22.8), node_encounters=jnp.zeros((), jnp.int64))


def stage_functions(kind, run):
    """(conversion, right_envs, field_sweep) of an engine, each over the walker batch:
    conversion(qa, qb) -> converted walkers; right_envs(converted, data) -> (blocks, right environments, gauge);
    field_sweep(ca, cb, randoms, blocks, prefactor, right, now, data) -> the sweep's outputs."""
    kernels, plan, floor = run.meas_ctx.kernels, run.ops.plan, float(run.params.weight_floor)
    if kind == "dense":
        convert, layout = dense_engine.dense_converter(plan), kernels.overlap_plan.layout

        def conversion(qa, qb):
            return jax.vmap(convert)(qa, qb)

        def right_envs(converted, data):
            alpha, dressed, gauge = converted
            right = jax.vmap(lambda a, d: dense_engine.right_environments(a, d, data.trial, layout))(alpha, dressed)
            return (alpha, dressed), right, gauge

        def field_sweep(ca, cb, randoms, blocks, prefactor, right, now, data):
            sweep = partial(dense_engine.field_sweep, hs=data.hs, weight_floor=floor, trial=data.trial, layout=layout)
            return jax.vmap(sweep)(ca, cb, randoms, *blocks, prefactor, right, now)
        return conversion, right_envs, field_sweep

    op = kernels.overlap_plan

    def conversion(qa, qb):
        return jax.vmap(kernels.converter.convert)(qa, qb)

    def right_envs(converted, data):
        alpha, beta, (ga, gb) = converted
        wb = jax.vmap(lambda a, b: engine.walker_blocks(a, b, op))(alpha, beta)
        right = jax.vmap(lambda w: engine.right_environments(w, data.trial, op))(wb)
        return (wb,), right, ga * gb

    def field_sweep(ca, cb, randoms, blocks, prefactor, right, now, data):
        sweep = partial(engine.field_sweep, hs=data.hs, weight_floor=floor, fixed_blocks=data.trial, plan=op)
        return jax.vmap(sweep)(ca, cb, randoms, *blocks, prefactor, right, now)
    return conversion, right_envs, field_sweep


def profile_engine(kind, args, trial, htrial, nelec, h1):
    """The stage table of one engine (module docstring); returns its rows."""
    state = saved_state(args.walkers, args.n_walkers)
    nw, L = state.walkers[0].shape[:2]
    params = make_params(kind, args, nw)
    sys = System(len(h1), nelec, "unrestricted")
    ham = HamHubbard(jnp.asarray(h1), U)
    run = driver.prepare_mps_cpmc(sys=sys, params=params, ham_data=ham, trial=trial, htrial=htrial, state=state,
                                  recompute_overlaps=True)
    kernels, data = run.meas_ctx.kernels, run.meas_ctx.data(run.prop_ctx)
    measure = run.meas_ctx.measurement
    m_data = measure.data()
    if args.mpo:
        print(f"  {kind} device memory: the context's blocks {device_gb(run.meas_ctx):.2f} GB (MPO "
              f"{device_gb(m_data.htrial):.4f} GB, measurement trial {device_gb(m_data.trial):.2f} GB); no H|trial>",
              flush=True)
    else:
        energy_stats = measure.kernels.energy_plan.stats if measure.kernels.energy_plan is not None else {}
        print(f"  {kind} device memory: the context's blocks {device_gb(run.meas_ctx):.2f} GB (H|trial> "
              f"{device_gb(m_data.htrial):.2f} GB, measurement trial {device_gb(m_data.trial):.2f} GB); energy blocks "
              f"stored {'once per distinct block' if getattr(measure.kernels.energy_plan, 'dedup', False) else 'per transition'}"
              f" ({energy_stats.get('fixed_unique_entries', 0) * 8 / 2 ** 30:.2f} GB distinct, "
              f"{energy_stats.get('fixed_bucketed_entries', 0) * 8 / 2 ** 30:.2f} GB per transition)", flush=True)
    rows = []

    def stage(name, fn, inputs, per_block, of_half_step=0.0):
        try:
            compiled, compile_seconds = compile_timed(fn, inputs)
            out = jax.block_until_ready(compiled(*inputs))
            seconds = timed(compiled, inputs, args.repeats)
        except Exception as error:
            if not is_oom(error) or args.strict:
                raise
            print(f"  {kind} {name}: out of device memory", flush=True)
            return None, None
        rows.append(dict(engine=kind, stage=name, ms=1e3 * seconds, per_block=per_block, of_half_step=of_half_step,
                         compile_s=compile_seconds))
        print(f"  {kind:<6} {name:<14} {1e3 * seconds:9.2f} ms  x{per_block:<4g} = {seconds * per_block:7.2f} s/block  "
              f"(compile {compile_seconds:6.1f} s)", flush=True)
        return out, compiled

    steps = 2 * params.n_prop_steps
    wu = data.exp_h1_half @ run.state.walkers[0]
    wd = data.exp_h1_half @ run.state.walkers[1]
    if "propagation" in args.stages:
        half = engine.make_half_step(kernels, params, 1)
        n_half = args.half_steps

        def half_steps(state, data):
            return lax.scan(lambda s, i: (half(s, i, data), None), state, jnp.arange(n_half))[0]

        _, compiled = stage(f"half_step x{n_half}", half_steps, (run.state, data), steps / n_half)
        if compiled is not None and args.profile:
            print(f"  {kind} trace of {n_half} half steps:", flush=True)
            profile_call(compiled, (run.state, data), str(Path(args.profile) / kind / "half_steps"))

        # one spin's QR, counted twice (both spins) per half step and at the end of the block
        (qa, da), _ = stage("walker_qr", lambda a: kernels.batch_qr(a), (wu,), 2 * steps + 2, of_half_step=2.0)
        qb, db = jax.block_until_ready(jax.jit(kernels.batch_qr)(wd))
        conversion, right_envs, field_sweep = stage_functions(kind, run)
        converted, _ = stage("conversion", conversion, (qa, qb), steps, of_half_step=1.0)
        (blocks, right, gauge), _ = stage("right_envs", right_envs, (converted, data), steps, of_half_step=1.0)
        prefactor = da * db * gauge
        now = prefactor * right[0][:, 0, 0, 0] if kind == "dense" else prefactor * right[0][:, 0, 0, 0, 0]
        randoms = jax.random.uniform(jax.random.PRNGKey(1), (nw, L))
        stage("field_sweep", field_sweep, (wu, wd, randoms, blocks, prefactor, right, now, data), steps / 2,
              of_half_step=0.5)
        del converted, blocks, right
    else:
        qa, _ = jax.block_until_ready(jax.jit(kernels.batch_qr)(wu))
        qb, _ = jax.block_until_ready(jax.jit(kernels.batch_qr)(wd))

    for k in args.measure_chunks:
        n = engine.divisor_at_least(nw, -(-nw // k))
        energy = lambda a, b, d, n=n: engine.chunked(lambda x, y: measure.kernels.energies(x, y, d), n, a, b)
        szsz = lambda a, b, d, n=n: engine.chunked(lambda x, y: BATCHED_OBSERVABLES["szsz"](measure)(x, y, d), n, a, b)
        if "energy" in args.stages:
            values, _ = stage(f"energy@{nw // n}", energy, (qa, qb, m_data), 1)
            if values is not None:
                values = np.asarray(values)
                print(f"  {kind} local energies of these walkers: {energy_summary(values)}", flush=True)
                if args.compare_stored and k == args.measure_chunks[0]:
                    compare_with_stored(run, trial, htrial, ham, qa, qb, values, n, args)
        if "szsz" in args.stages:
            stage(f"szsz@{nw // n}", szsz, (qa, qb, m_data), 1)
    stats = jax.devices()[0].memory_stats() or {}
    if stats:
        print(f"  {kind} device peak {stats.get('peak_bytes_in_use', 0) / 2 ** 30:.1f} GB of "
              f"{stats.get('bytes_limit', 0) / 2 ** 30:.1f} GB", flush=True)
    return rows


def summarize(rows, k_list, nw):
    """The projected block per engine and measurement chunk, and the half step against its stages."""
    for kind in sorted({r["engine"] for r in rows}):
        mine = {r["stage"]: r for r in rows if r["engine"] == kind}
        half = next((r for name, r in mine.items() if name.startswith("half_step")), None)
        if half is None:
            continue
        parts = sum(r["ms"] * r["of_half_step"] for r in mine.values() if r["of_half_step"])
        print(f"{kind}: half step {half['ms'] / (100 / half['per_block']):.1f} ms measured in the scan, its stages "
              f"{parts:.1f} ms (qr x2, conversion, right_envs, half a sweep)")
        for name, r in mine.items():
            if r["of_half_step"]:
                print(f"    {name:<12} {r['ms'] * r['of_half_step']:8.1f} ms per half step "
                      f"({100 * r['ms'] * r['of_half_step'] / parts:4.1f}%)")
        for k in k_list:
            size = next((name.split("@")[1] for name in mine if name.startswith("energy@") and
                         int(name.split("@")[1]) <= k), None)
            if size is None:
                continue
            if f"energy@{size}" not in mine:
                continue
            szsz = mine.get(f"szsz@{size}", dict(ms=0.0))["ms"]  # absent with --stages without szsz (or --mpo)
            block = (half["ms"] * half["per_block"] + mine["walker_qr"]["ms"] * 2
                     + mine[f"energy@{size}"]["ms"] + szsz) / 1e3
            print(f"    projected block with the measurement in chunks of {size}: {block:.1f} s "
                  f"(propagation {half['ms'] * half['per_block'] / 1e3:.1f} s, energy "
                  f"{mine[f'energy@{size}']['ms'] / 1e3:.1f} s, szsz {szsz / 1e3:.1f} s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", nargs="+", default=["sector", "dense"], choices=["sector", "dense"])
    parser.add_argument("--chi", type=int, default=256, help="the saved DMRG trial's bond")
    parser.add_argument("--chi-w", type=int, default=16)
    parser.add_argument("--measure-chi", type=int, default=0,
                        help="walker bond of the energy and szsz (0: the walker bond); with TROT_ENERGY_DEDUP=1 the energy's "
                             "H|trial> blocks are stored once per distinct block")
    parser.add_argument("--compare-stored", action="store_true",
                        help="with --mpo: also measure the same walkers with the stored-H|trial> kernel at the same bond "
                             "(loads the H|trial> cache; TROT_ENERGY_DEDUP=1 at chi_T 512) and compare walker by walker")
    parser.add_argument("--save-energies", type=Path, default=None,
                        help="with --compare-stored: save both kernels' local energies (npz: mpo, stored)")
    parser.add_argument("--mpo", action="store_true",
                        help="energy through the MPO with dense walker legs (energy_kernel='mpo'): no H|trial> cache is "
                             "loaded; no szsz stage (use --stages propagation energy)")
    parser.add_argument("--stages", nargs="+", default=["propagation", "energy", "szsz"],
                        choices=["propagation", "energy", "szsz"],
                        help="what to compile and time (each measurement chunk size compiles again: ~20 min at chi_T 512)")
    parser.add_argument("--strict", action="store_true",
                        help="fail (nonzero exit) when a stage runs out of device memory instead of skipping it: a "
                             "feasibility test that later jobs can depend on (sbatch --dependency=afterok)")
    parser.add_argument("--dense-move", default="gram", choices=["gram", "cans"])
    parser.add_argument("--walkers", default=DEFAULT_WALKERS, help="saved walkers (up / dn)")
    parser.add_argument("--n-walkers", type=int, default=400)
    parser.add_argument("--half-steps", type=int, default=4, help="half steps in the timed scan (even: half sweep)")
    parser.add_argument("--measure-chunks", type=int, nargs="+", default=[20, 4],
                        help="walkers per measurement chunk to time (20: the default, 4: TROT_MEASURE_CHUNK=4)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", default="", help="directory for the half steps' traces (empty: no trace)")
    parser.add_argument("--out", type=Path, default=None, help="write the rows as JSON")
    args = parser.parse_args(argv)

    h1 = hubbard_h1()
    trial, htrial, nelec = load_trial(TRIALS / f"dmrg_chi{args.chi}.npz", 90.0,
                                      with_htrial=(not args.mpo) or args.compare_stored)
    print(f"device {jax.devices()[0].device_kind}; trial chi {args.chi}, walker bond {args.chi_w}, measurement bond "
          f"{args.measure_chi or args.chi_w}, energy dedup {engine.ENERGY_DEDUP}, measurement chunks "
          f"{args.measure_chunks}; walkers {args.walkers} (first {args.n_walkers})", flush=True)
    rows = []
    for kind in args.engines:
        print(f"== {kind} engine", flush=True)
        rows += profile_engine(kind, args, trial, htrial, nelec, h1)
    summarize(rows, args.measure_chunks, args.n_walkers)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
