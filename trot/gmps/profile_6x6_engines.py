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
  szsz@k       the szsz observable likewise;
  fused@k      (--stages fused; a dense plan with the MPO energy) energy and szsz as the block measures them,
               trot.meas.mps.FusedMeasurement.chunked: one conversion per walker for both, with
               TROT_MEASURE_CONVERT_CHUNK first in chunks of that many walkers.
  charges      (--stages charges; dense) no timing: the bond labels of the measured walkers under the propagation and
               the measurement plans, and the share of the trial GEMMs' multiply-adds whose environment rows can be
               nonzero (charge_report): what a layout packed by the walkers' electron numbers would skip, and what
               static packed layouts (per label, per sector, walkers stacked per sector) would still pay.
  grouped      (--stages grouped; dense, trial in sector blocks) the trial GEMM of the first walker's energy (--mpo)
               and overlap left environments at bonds --grouped-sites (grouped_bench, grouped_site): _trial_apply
               against the grouped kernel (trot.gmps.grouped_gemm; tile sizes swept at the first bond), the kernel on
               rows sorted in memory, the hybrid (cuBLAS up to a cap per sector, the kernel for the rest), cuBLAS on
               exactly each sector's own rows (the best case of static shapes), and at the first bond Triton's plain
               float64 GEMM against cuBLAS on contiguous operands; values compared (MISMATCH above 1e-9), multiply-adds
               counted. lax.ragged_dot was timed by job 7372094 (slower than _trial_apply) and is no longer. With
               --mpo also the whole site steps (site_steps) in the four TROT_TRIAL_GEMM versions, dense, grouped, fused and
               blocked: the energy's _forward_mpo_gather and a 16-member szsz _left_transfer (--grouped-steps-only: only
               these), and the blocked step's three kernels apart (blocked_parts: the fused trial GEMM, the alpha and
               the beta block contraction); --tile-sweep also sweeps the blocked step's tiles at the first bond.
               The versions are compared with dense, or where dense fails with the first one that ran; --sweep-gemm
               sets the version that carries the environment between the bonds (blocked at chi_T 4096).

The measurement stages run on the first --measure-walkers walkers (default all), the conversion, right_envs and
field_sweep stages on the first --stage-walkers (default all; at chi_T 2048 right_envs of 200 walkers runs out of
memory, so a stage that does is skipped with the stages that need its output). From these it projects a block: 2
n_prop_steps half steps, one walker QR, one energy and one szsz (or one fused measurement). It also prints the stages'
share of a half step, the half step against the sum of its stages (fusion), and with --profile a trace of the half
steps (top kernels, kernel classes, GPU busy fraction; bench_mps_cpmc.profile_call).

    ./submit_ex_2d.sh h100 profile_6x6_engines.py --engines sector --profile profiles/6x6
    ./submit_ex_2d.sh h100 profile_6x6_engines.py --engines dense --profile profiles/6x6

A/B tests of the kernel variants, one job per variant, the variant set in the environment (read at import). The
defaults since 2026-10-29 (A/B jobs 7364937-42): TROT_RIGHT_ENVS=walker (the propagation's right environments
walker-first; trial: trial-first), TROT_MEASURE_CONVERT_CHUNK=200 (the fused measurement converts its walkers first, 200
at a time; 0: in each measurement chunk), TROT_MEASURE_BARRIER=1 (a barrier between the conversion and the
measurement; 0: none), TROT_MPO_KERNEL=gather (walker: the MPO energy walker-first, 6% slower). The trial GEMMs on the
rows that can be nonzero: TROT_TRIAL_GEMM (measurement: dense, grouped, fused, blocked, cublas) and TROT_PROP_GEMM
(propagation: dense, grouped, envs, blocked; dense_engine) change the half_step, energy, szsz and fused stages (the
separate conversion, right_envs and field_sweep stages stay dense). --save-values writes the overlaps of the right_envs
stage and the measured values to an npz; --compare A.npz B.npz prints how far two runs' values are apart (their walkers
and settings must be the same) and exits.
"""
from trot.config import configure_once

configure_once()  # float64

import argparse
import json
import os
import time
from contextlib import contextmanager
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
from trot.meas.mps import BATCHED_OBSERVABLES, fused_measurement, hubbard_mpo_from_h1
from trot.prop.types import PropState, QmcParamsMps

DEFAULT_WALKERS = "/mnt/ceph/users/fnappi/trot_walkers/sq6x6_doped_diag/dense_timing/walkers_block0030.npz"

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/trot_jax_compile"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def make_params(kind, args, n_walkers):
    """ex_6x6_doped_diag.py's parameters for one engine."""
    values = dict(n_walkers=n_walkers, n_chunks=1, auto_n_chunks=False, n_eql_blocks=10, n_blocks=20, dt=0.005,
                  n_prop_steps=50, weight_floor=1e-8, seed=1234, occupation_tolerance=1e-10,
                  orbital_plan=args.orbital_plan,
                  walker_channel_chi=args.chi_w, plan_reference="natural", walker_start="natural")
    if args.measure_chi:
        values.update(measure_channel_chi=args.measure_chi)
    if args.mpo:
        values.update(energy_kernel="mpo", dense_move=args.dense_move, dense_eigh=args.dense_eigh)
    if kind == "dense":
        values.update(walker_conversion="dense", dense_move=args.dense_move, dense_eigh=args.dense_eigh,
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


def walker_labels(plan, qa, qb):
    """Per channel (alpha, beta) and bond the electron-number labels (walkers, bond) of these orthonormal walkers' bond
    states under a dense walker plan's converters; -1 for a zero-weight state (a zero column of the tensor before the
    bond), which a layout packed by label would drop."""
    dense_engine.dense_converter(plan)  # builds the plan's channel converters
    out = []
    for converter, Q in zip(plan.caches["dense_channel_converters"], (qa, qb)):
        def labels(C, converter=converter):
            tensors, charges, _ = converter.convert(C)
            norms = [jnp.sqrt(jnp.sum(A * A, axis=(0, 1))) for A in tensors]
            live = [jnp.ones(charges[0].shape, bool)] + [n > 1e-12 * jnp.max(n) for n in norms]
            return [jnp.where(keep, jnp.round(q), -1.0) for q, keep in zip(charges, live)]
        out.append([np.asarray(q).astype(int) for q in jax.jit(jax.vmap(labels))(Q)])
    return out


def mpo_channel_offsets(mpo):
    """Per MPO bond the electron-number offset of each channel, the trial's electrons left of the bond minus the
    walker's: W[w, p, q, v] != 0 (p the walker's physical state, q the trial's) gives v the offset of w plus n(q) - n(p)."""
    n = np.asarray(dense_engine.N_PHYSICAL)
    offsets = [np.zeros(1, int)]
    for W in mpo:
        W, nxt = np.asarray(W), {}
        for w, p, q, v in np.argwhere(W != 0):
            if nxt.setdefault(v, offsets[-1][w] + n[q] - n[p]) != offsets[-1][w] + n[q] - n[p]:
                raise ValueError("an MPO channel without a definite electron-number offset")
        offsets.append(np.array([nxt.get(v, 10 ** 6) for v in range(W.shape[3])]))  # 10**6: fed by nothing
    return offsets


def charge_report(name, labels, layout, offsets=None):
    """How much of the trial GEMMs' work can meet nonzero environment rows. An environment row (a, b[, channel]) of the
    left bond reaches the trial states of electron number N only if the walker labels n_a + n_b (plus the MPO
    channel's offset) equal N: every other entry of the dense (a, b[, w], t) environment is an exact zero. Per site the
    forward trial GEMM of sector N costs rows x (its width x its columns); this sums it over the sites with all rows
    (the dense walker legs of now), with the rows that can be nonzero (walker mean), with every label padded to the
    population's largest count of it (a static packed layout), with each sector's rows padded to the most any walker
    has (a static row list per sector), and with the rows of `pool` walkers stacked into one GEMM per sector (the trial
    is the same for all), padded to the chunk with the most."""
    alpha, beta = labels
    offsets = offsets or [np.zeros(1, int)] * len(alpha)
    pool = (2, 8, 25, 100)
    total = dict(dense=0.0, nonzero=0.0, padded=0.0, pair_padded=0.0, env_dense=0.0, env_nonzero=0.0)
    total.update({k: 0.0 for k in pool})
    for i, site in enumerate(layout.sites):
        a, b, offset = alpha[i], beta[i], offsets[i]
        top = int(max(a.max(), b.max())) + 1
        count = lambda x: np.stack([np.bincount(row[row >= 0], minlength=top) for row in x])  # (walkers, top)
        na, nb = count(a), count(b)
        pairs = np.stack([np.convolve(x, y) for x, y in zip(na, nb)])  # (walkers, 2 top - 1) per electron total
        capacity = np.convolve(na.max(axis=0), nb.max(axis=0))

        def rows(per_total, N):  # the rows (a, b, channel) whose labels and channel offset add up to N
            return sum((per_total[..., N - d] for d in offset if 0 <= N - d < per_total.shape[-1]),
                       np.zeros(per_total.shape[:-1]))

        dense_rows = a.shape[1] * b.shape[1] * len(offset)
        sector_of = {start: N for N, start, _ in layout.sectors[i]}
        for start, stop, cols in site.forward:
            N, mac = sector_of[start], (stop - start) * sum(width for *_, width in cols)
            per_walker = rows(pairs, N)
            total["dense"] += dense_rows * mac
            total["nonzero"] += float(np.mean(per_walker)) * mac
            total["padded"] += float(rows(capacity, N)) * mac
            total["pair_padded"] += float(np.max(per_walker)) * mac
            for k in pool:
                if len(per_walker) >= k:
                    chunks = per_walker[:len(per_walker) // k * k].reshape(-1, k)
                    total[k] += float(chunks.mean(axis=1).max()) * mac
        for N, start, stop in layout.sectors[i]:
            total["env_dense"] += dense_rows * (stop - start)
            total["env_nonzero"] += float(np.mean(rows(pairs, N))) * (stop - start)
        if i == len(layout.sites) // 2:
            mean = lambda c: {k: round(float(v), 1) for k, v in enumerate(c.mean(axis=0)) if v > 0}
            most = lambda c: {k: int(v) for k, v in enumerate(c.max(axis=0)) if v > 0}
            channels = dict(zip(*(x.tolist() for x in np.unique(offset, return_counts=True))))
            print(f"  {name}: bond {i}: alpha states per label, mean {mean(na)}, largest {most(na)}; beta, mean "
                  f"{mean(nb)}, largest {most(nb)}; trial sectors {{N: width}} "
                  f"{ {N: stop - start for N, start, stop in layout.sectors[i]} }; channels per offset {channels}",
                  flush=True)
    print(f"  {name}: trial GEMM multiply-adds on rows that can be nonzero {total['nonzero'] / total['dense']:.1%} of "
          f"the dense legs' (every label padded to the population's largest count: {total['padded'] / total['dense']:.1%});"
          f" environment entries that can be nonzero {total['env_nonzero'] / total['env_dense']:.1%}", flush=True)
    stacked = ", ".join(f"{k} {total[k] / total['dense']:.1%}" for k in pool if len(alpha[0]) >= k)
    print(f"  {name}: each sector's rows padded to the most any walker has {total['pair_padded'] / total['dense']:.1%};"
          f" the rows of k walkers stacked per sector, padded to the largest chunk: {stacked}", flush=True)


# the grouped kernel's (R, BK, BN, num_warps, num_stages) tried at the first bond of each environment kind
GROUPED_TILES = [(64, 32, 64, 4, 2), (64, 32, 64, 4, 3), (64, 16, 64, 4, 3), (32, 32, 64, 4, 2), (64, 32, 32, 4, 2),
                 (64, 32, 128, 8, 2), (128, 32, 64, 8, 2), (128, 16, 128, 8, 2)]


# the plain Triton GEMM's (BM, BN, BK, num_warps, num_stages) of the diagnostic (grouped_site, plain=True)
PLAIN_TILES = [(64, 64, 32, 4, 3), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3), (128, 128, 32, 8, 3), (128, 128, 16, 8, 2)]
# the hybrid's caps per sector: this walker's count times the factor, rounded up to a multiple of 64. +12% is what
# padding each sector to the population's largest count cost on the 8x8 (18.9% of the dense work against the mean
# walker's 16.8%, job 7366493); -10% sends about a tenth of the rows through the kernel.
HYBRID_CAPS = {"caps +12% (as the population's largest)": 1.12, "caps -10% (rows over them: the kernel)": 0.9}


def grouped_site(label, X, n_a, n_b, offsets, site, sectors, matrices, args, tiles, plain=False):
    """One trial GEMM of the grouped stage: X (rows, Dt) one walker's environment on this bond, n_a, n_b its bond
    labels, offsets the MPO channels' (None: the overlap's environment). Times:
    - with plain, the largest sector's GEMM alone on contiguous padded operands, cuBLAS against the plain Triton GEMM
      (grouped_gemm.triton_gemm, PLAIN_TILES): Triton's float64 rate without the grouped kernel's gathers and masks;
    - _trial_apply (now) and the grouped kernel for each of `tiles`;
    - with the fastest tile, the kernel on rows sorted in memory beforehand (the sort timed apart), and the hybrid
      (grouped_gemm.hybrid_trial_apply: cuBLAS on each sector's first rows up to a cap, the kernel on the rest) for
      HYBRID_CAPS;
    - cuBLAS on exactly each sector's own rows (static shapes from this walker: the best case, gathered or not).
    Every result is compared with _trial_apply's (MISMATCH above 1e-9 relative). Rates: _trial_apply's on all its
    multiply-adds, the plain GEMMs' on their padded ones, the others' on the sectors' own rows. Returns the fastest
    tile."""
    from trot.gmps import grouped_gemm as gg

    spec = gg.trial_spec(site.forward, site.right_widths, sectors)
    packed = jax.block_until_ready(gg.pack_blocks(matrices))
    g = jax.block_until_ready(jax.jit(lambda a, b: gg.row_sectors(a, b, spec, offsets))(n_a, n_b))
    g_host = np.asarray(g)
    G = len(spec.N)
    counts = np.bincount(g_host, minlength=G + 1)[:G]
    macs = gg.multiply_adds(g, spec)
    print(f"  grouped {label}: X {tuple(X.shape)}, {G} sectors; multiply-adds of _trial_apply "
          f"{macs['dense']:.3g}, on the sectors' own rows {macs['useful'] / macs['dense']:.1%}, in tiles of "
          f"{gg.TILE_ROWS} rows {macs['tiled'] / macs['dense']:.1%}", flush=True)

    def run(name, fn, inputs, work):
        try:
            compiled, compile_seconds = compile_timed(fn, inputs)
            out = jax.block_until_ready(compiled(*inputs))
            seconds = timed(compiled, inputs, args.repeats)
        except Exception as error:  # out of memory, or a lowering or Triton failure: report it and go on
            print(f"    {name:<44} failed: {type(error).__name__}: {str(error).splitlines()[0][:300]}", flush=True)
            return None, None
        rate = f"{2 * work / seconds / 1e12:6.1f} TFLOPS" if work else " " * 13
        print(f"    {name:<44} {1e3 * seconds:10.2f} ms  {rate}  (compile {compile_seconds:5.1f} s)", flush=True)
        return out, seconds

    distance = jax.jit(lambda a, b: jnp.max(jnp.abs(a - b)))
    permuted_distance = jax.jit(lambda a, b, p: jnp.max(jnp.abs(a - jnp.take(b, p, axis=0))))

    def check(got, reference, scale, perm=None):
        try:
            diff = float(distance(got, reference) if perm is None else permuted_distance(got, reference, perm))
        except Exception as error:
            print(f"      comparison failed: {type(error).__name__}: {str(error).splitlines()[0][:200]}", flush=True)
            return
        diff /= max(scale, 1e-300)
        flag = "" if diff < 1e-9 else "   MISMATCH (above 1e-9)"
        print(f"      max |difference| / max |reference| = {diff:.1e}{flag}", flush=True)

    if plain:  # Triton's float64 GEMM rate on contiguous operands: the largest sector's GEMM alone, padded
        s = int(np.argmax(counts * np.asarray(spec.width) * np.asarray(spec.ncols)))
        c, w, n = int(counts[s]), spec.width[s], spec.ncols[s]
        M, K, N = (-(-x // m) * m for x, m in ((c, 128), (w, 32), (n, 128)))
        mine = jnp.asarray(np.flatnonzero(g_host == s))
        A = jnp.zeros((M, K), X.dtype).at[:c, :w].set(X[mine, spec.start[s]:spec.start[s] + w])
        B = jnp.zeros((K, N), X.dtype).at[:w, :n].set(matrices[s])
        print(f"    the largest sector's GEMM alone, contiguous and padded: ({M}, {K}) @ ({K}, {N})", flush=True)
        reference, _ = run("cuBLAS (XLA's dot)", lambda A, B: A @ B, (A, B), M * K * N)
        for BM, BN, BK, warps, stages in PLAIN_TILES:
            def gemm(A, B, BM=BM, BN=BN, BK=BK, warps=warps, stages=stages):
                return gg.triton_gemm(A, B, BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=stages)

            got, _ = run(f"Triton plain GEMM BM{BM} BN{BN} BK{BK} warps {warps} stages {stages}", gemm, (A, B),
                         M * K * N)
            if got is not None and reference is not None:
                check(got, reference, float(jnp.max(jnp.abs(reference))))
            del got
        del A, B, reference

    blocks = tuple(matrices)
    want, t_now = run("_trial_apply (now)", lambda X, Ts: dense_engine._trial_apply(X, Ts, site.forward,
                                                                                     site.right_widths),
                      (X, blocks), macs["dense"])
    scale = float(jnp.max(jnp.abs(want))) if want is not None else 0.0
    best = None
    for R, BK, BN, warps, stages in tiles:
        def kernel(X, g, packed, R=R, BK=BK, BN=BN, warps=warps, stages=stages):
            return gg.grouped_trial_apply(X, g, spec, packed, R=R, BK=BK, BN=BN, num_warps=warps, num_stages=stages)

        got, seconds = run(f"kernel R{R} BK{BK} BN{BN} warps {warps} stages {stages}", kernel, (X, g, packed),
                           macs["useful"])
        if got is None:
            continue
        if want is not None:
            check(got, want, scale)
        if best is None or seconds < best[0]:
            best = (seconds, (R, BK, BN, warps, stages))
        del got
    if best is not None and t_now:
        print(f"    fastest kernel {best[1]}: {t_now / best[0]:.2f}x faster than _trial_apply", flush=True)

    if best is not None:
        R, BK, BN, warps, stages = best[1]
        tile = dict(R=R, BK=BK, BN=BN, num_warps=warps, num_stages=stages)
        perm = jax.block_until_ready(jax.jit(lambda g: jnp.argsort(g, stable=True))(g))
        sorted_X, t_sort = run("sort the rows in memory, X[perm]", lambda X, p: jnp.take(X, p, axis=0), (X, perm), 0)
        if sorted_X is not None:
            got, seconds = run("kernel on the sorted rows (output sorted)",
                               lambda Xs, g, packed: gg.grouped_trial_apply(Xs, g, spec, packed, presorted=True,
                                                                            **tile),
                               (sorted_X, g, packed), macs["useful"])
            if got is not None:
                if want is not None:
                    check(got, want, scale, perm)
                print(f"      with the sort {1e3 * (t_sort + seconds):.2f} ms"
                      f"{f': {t_now / (t_sort + seconds):.2f}x faster than _trial_apply' if t_now else ''}",
                      flush=True)
            del got, sorted_X
        for name, factor in HYBRID_CAPS.items():
            caps = tuple(int(-(-int(np.ceil(factor * c)) // 64) * 64) for c in counts)
            over = int(np.sum(np.maximum(counts - np.asarray(caps), 0)))

            def hybrid(X, g, Ts, packed, caps=caps):
                return gg.hybrid_trial_apply(X, g, spec, Ts, packed, caps, **tile)

            got, seconds = run(f"cuBLAS + kernel, {name}", hybrid, (X, g, blocks, packed), macs["useful"])
            if got is not None:
                if want is not None:
                    check(got, want, scale)
                print(f"      {over} rows over the caps (through the kernel); cuBLAS rows incl. padding "
                      f"{sum(caps)} for {int(counts.sum())}"
                      f"{f'; {t_now / seconds:.2f}x faster than _trial_apply' if t_now else ''}", flush=True)
            del got
    del want

    # the best case of static shapes: each sector's GEMM on exactly this walker's rows (indices built on the host)
    rows = [jnp.asarray(np.flatnonzero(g_host == s)) for s in range(G)]
    run("cuBLAS exact rows (gather + GEMM)",
        lambda X, Ts: [X[r][:, a:a + w] @ W for r, a, w, W in zip(rows, spec.start, spec.width, Ts)],
        (X, blocks), macs["useful"])
    parts = jax.block_until_ready([X[r][:, a:a + w] for r, a, w in zip(rows, spec.start, spec.width)])
    run("cuBLAS exact rows (GEMM only)", lambda P, Ts: [p @ W for p, W in zip(P, Ts)], (parts, blocks),
        macs["useful"])
    del parts
    return best[1] if best is not None else None


@contextmanager
def trial_gemm(mode):
    """dense_engine.TRIAL_GEMM = mode meanwhile: the fused kernel's GEMM (Triton, or cuBLAS for cublas) is chosen
    when a site step is traced."""
    previous = dense_engine.TRIAL_GEMM
    dense_engine.TRIAL_GEMM = mode
    try:
        yield
    finally:
        dense_engine.TRIAL_GEMM = previous


STEP_MODES = ("dense", "grouped", "fused", "blocked", "cublas")


def step_specs(layout, i):
    """Per TROT_TRIAL_GEMM version (but dense) the spec of bond i's forward site step."""
    from trot.gmps import grouped_gemm as gg

    fused = gg.layout_specs(layout, "forward", fused=True)[i]
    return dict(grouped=gg.layout_specs(layout, "forward")[i], fused=fused, blocked=fused, cublas=fused)


def step_labels(mode, n_a, n_b, specs, channel_offsets, n_sites, next_labels):
    """The `grouped` argument of a site step in version `mode` (None for dense): the walker's bond labels, the spec,
    the channels' offsets, the bound 2 L + 1 on the label sums (all but grouped) and the next bond's labels (blocked,
    cublas)."""
    if mode == "dense":
        return None
    g = (n_a, n_b, specs[mode], channel_offsets)
    return g + ((2 * n_sites + 1,) if mode != "grouped" else ()) + (
        tuple(next_labels) if mode in ("blocked", "cublas") else ())


def site_steps(label, E, stack, T, A, B, signs, gather, site, n_a, n_b, offsets, i, layout, args, next_labels,
               sweep=False):
    """The whole site step in its five versions (TROT_TRIAL_GEMM dense, grouped, fused, blocked, cublas):
    _forward_mpo_gather on the energy's environment E and _left_transfer (one weight, as szsz moves its stacks) on
    `stack`; next_labels the walker's (n_alpha, n_beta) on the site's right bond (blocked, cublas). Times them and
    compares the others with the reference, dense or, where dense fails (out of memory), the first version that ran
    (MISMATCH above 1e-9 relative; grouped and fused contract the walker tensors in XLA, so they check the label
    contractions of blocked and cublas); the blocked and cublas steps also kernel by kernel (blocked_parts). cublas
    needs the custom call's library (grouped_gemm.CUBLAS_LIB) and is reported as failed without it. sweep: also the
    blocked step's tiles and the cublas step's epilogue tile (tile_sweep)."""
    from trot.gmps import grouped_gemm as gg

    n = len(layout.sites)
    specs = step_specs(layout, i)
    fused = specs["fused"]
    distance = jax.jit(lambda a, b: jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
    plain = jnp.ones((1, 4))
    steps = [("energy step (_forward_mpo_gather)", E, offsets,
              lambda x, g: dense_engine._forward_mpo_gather(x, T, A, B, signs, gather, site, g))]
    # blocked_parts' kernels: terms, output channel offsets, zero fill of the output (mpo_contract fills only its last)
    terms = {steps[0][0]: (gg.mpo_terms(gather), gg.channel_offsets(gather, offsets), False)}
    if stack is not None:
        steps.append((f"szsz left transfer, {stack.shape[2]} members", stack, None,
                      lambda x, g: dense_engine._left_transfer(x, T, A, B, signs, plain, site, g)))
        terms[steps[1][0]] = (gg.weight_terms(stack.shape[2], plain), None, True)
    for what, x, channel_offsets, step in steps:
        want, t_ref, reference = None, None, None
        for mode in STEP_MODES:
            g = step_labels(mode, n_a, n_b, specs, channel_offsets, n, next_labels)
            try:
                with trial_gemm(mode):
                    compiled, compile_seconds = compile_timed(lambda x, g=g: step(x, g), (x,))
                out = jax.block_until_ready(compiled(x))
                seconds = timed(compiled, (x,), args.repeats)
            except Exception as error:  # out of memory, or a lowering, Triton or library failure: report it, go on
                print(f"  {label} {what} {mode}: failed: {type(error).__name__}: {str(error).splitlines()[0][:300]}",
                      flush=True)
                continue
            note = ""
            if want is None:
                want, t_ref, reference = out, seconds, mode
                note = "" if mode == "dense" else "  (the reference: dense failed)"
            else:
                diff = float(distance(out, want))
                note = (f"  {t_ref / seconds:.2f}x faster than {reference}; max |difference| / max |{reference}| = "
                        f"{diff:.1e}{'' if diff < 1e-9 else '   MISMATCH (above 1e-9)'}")
            print(f"  {label} {what} {mode:<7} {1e3 * seconds:10.2f} ms  (compile {compile_seconds:5.1f} s){note}",
                  flush=True)
            del out
        step_terms, offsets_out, fill = terms[what]
        for gemm in ("triton", "cublas"):
            blocked_parts(f"{label} {what}", x, T, A, B, signs, step_terms, fused, n_a, n_b, channel_offsets,
                          offsets_out, next_labels, 2 * n + 1, fill, args, gemm)
        if sweep and want is not None:
            g = step_labels("blocked", n_a, n_b, specs, channel_offsets, n, next_labels)
            tile_sweep(f"{label} {what} blocked", lambda x: step(x, g), x, want, distance, args, reference=reference)
            if gg.cublas_available():
                tile_sweep(f"{label} {what} cublas", lambda x: step(x, g), x, want, distance, args,
                           (("EPILOGUE_TILE", EPILOGUE_TILES),), "cublas", reference=reference)
        del want


def blocked_parts(label, x, T, A, B, signs, terms, fspec, n_a, n_b, offsets, offsets_out, next_labels, n_groups,
                  fill, args, gemm="triton"):
    """The blocked step's kernels timed apart, each on the previous one's output: the fused trial GEMM
    (fused_trial_apply without fill, as _forward_mpo_gather and _left_transfer call it; gemm "cublas": the rows sorted
    by sector and cuBLAS's products, grouped_gemm.sector_order and sector_products, then products_fused_apply), the
    alpha block contraction and the beta one, with the output's zero fill if `fill` (grouped_gemm.contract_left_alpha,
    contract_left_beta), with the module's tiles. Where the step's time goes: the GEMM or the contractions."""
    from trot.gmps import grouped_gemm as gg

    packed = jax.block_until_ready(gg.pack_blocks(T[0]))
    n_a2, n_b2 = next_labels
    spec = fspec.trial

    def products(x):
        X = x.reshape(-1, x.shape[-1])
        perm, counts, positions = gg.sector_order(n_a, n_b, spec, offsets, X.shape[0] // (len(n_a) * len(n_b)))
        return gg.sector_products(X, perm, counts, spec, packed), positions

    if gemm == "cublas":
        head = [("cuBLAS sector products", products),
                ("products epilogue", lambda zp: gg.products_fused_apply(
                    zp[0], zp[1], n_a, n_b, offsets, terms, signs, fspec, n_groups=n_groups, layout="left",
                    offsets_out=offsets_out, fill=False))]
    else:
        head = [("fused trial GEMM", lambda x: gg.fused_trial_apply(
            x.reshape(-1, x.shape[-1]), n_a, n_b, offsets, terms, signs, fspec, packed, n_groups=n_groups,
            layout="left", offsets_out=offsets_out, fill=False))]
    pieces = head + [("alpha contraction", lambda y: gg.contract_left_alpha(y, A, n_a, n_b, n_a2, offsets_out, fspec)),
                     (f"beta contraction{'' if fill else ' (no fill)'}",
                      lambda h: gg.contract_left_beta(h, B, n_b, n_a2, n_b2, offsets_out, fspec, fill))]
    total = 0.0
    for name, fn in pieces:
        try:
            compiled, _ = compile_timed(fn, (x,))
            out = jax.block_until_ready(compiled(x))
            seconds = timed(compiled, (x,), args.repeats)
        except Exception as error:
            print(f"  {label} {gemm} part {name}: failed: {type(error).__name__}: "
                  f"{str(error).splitlines()[0][:200]}", flush=True)
            return
        total += seconds
        print(f"  {label} {gemm} part {name:<24} {1e3 * seconds:8.2f} ms", flush=True)
        x = out
    print(f"  {label} {gemm} parts in all {1e3 * total:.2f} ms (FUSED_TILE {gg.FUSED_TILE}, EPILOGUE_TILE "
          f"{gg.EPILOGUE_TILE}, CONTRACT_TILE {gg.CONTRACT_TILE})", flush=True)


# the tiles --tile-sweep tries: the fused kernel's (its large-row tile, gg.FUSED_TILE: the energy's and szsz's rows)
# and label_contract's (gg.CONTRACT_TILE), each (R, BK, BN, num_warps, num_stages). Job 7375438 swept (64, 32, 128, 8,
# 2), (64, 32, 64, 4, 3), (128, 32, 64, 8, 2), (64, 16, 128, 8, 3) for the fused kernel and (32, 32, 64, 4, 2),
# (16, 32, 64, 4, 2), (32, 16, 128, 4, 2), (32, 32, 128, 8, 2) for the contraction, all slower than the defaults now;
# these are the defaults and their neighbours (smaller tiles won the contraction).
FUSED_TILES = [(32, 32, 64, 4, 2), (32, 32, 64, 4, 3), (32, 16, 64, 4, 3), (32, 32, 128, 4, 2), (16, 32, 64, 4, 2),
               (32, 64, 64, 4, 2), (32, 16, 128, 4, 3), (32, 32, 32, 2, 3)]
CONTRACT_TILES = [(16, 16, 64, 4, 3), (16, 16, 64, 2, 3), (16, 16, 32, 2, 3), (16, 16, 128, 4, 3), (16, 16, 64, 4, 4),
                  (16, 32, 64, 4, 3), (16, 16, 128, 8, 3), (16, 16, 32, 4, 3)]
# products_fused_apply's (gg.EPILOGUE_TILE; BK unused): no GEMM, a gather of cuBLAS's products, so wider column tiles
EPILOGUE_TILES = [(32, 16, 128, 4, 2), (32, 16, 64, 4, 2), (64, 16, 128, 4, 2), (64, 16, 128, 8, 2),
                  (16, 16, 128, 4, 2), (16, 16, 256, 4, 2), (32, 16, 128, 8, 2)]


def tile_sweep(label, fn, x, want, distance, args, sweeps=(("FUSED_TILE", FUSED_TILES),
                                                         ("CONTRACT_TILE", CONTRACT_TILES)), mode="blocked",
               reference="dense"):
    """fn(x) (a site step traced with TROT_TRIAL_GEMM=mode) with each sweep's tile (a grouped_gemm attribute and its
    candidates; by default the fused kernel's over FUSED_TILES, then, keeping the fastest, the contraction's over
    CONTRACT_TILES); each timed and compared with the output `want` of the `reference` version. The module's tiles
    are restored after; set the fastest with TROT_TILE_<attribute's first word>."""
    from trot.gmps import grouped_gemm as gg

    defaults = {attr: getattr(gg, attr) for attr, _ in sweeps}
    for attr, tiles in sweeps:
        results = []
        for tile in tiles:
            setattr(gg, attr, tuple(tile))
            try:
                with trial_gemm(mode):
                    compiled, _ = compile_timed(lambda x: fn(x), (x,))
                out = jax.block_until_ready(compiled(x))
                seconds = timed(compiled, (x,), args.repeats)
            except Exception as error:
                print(f"  {label} {attr} {tile}: failed: {type(error).__name__}: {str(error).splitlines()[0][:200]}",
                      flush=True)
                continue
            diff = float(distance(out, want))
            print(f"  {label} {attr} {tile}: {1e3 * seconds:9.2f} ms, max |difference| / max |{reference}| = {diff:.1e}"
                  f"{'' if diff < 1e-9 else '   MISMATCH (above 1e-9)'}", flush=True)
            results.append((seconds, tuple(tile)))
            del out
        if results:
            seconds, tile = min(results)
            setattr(gg, attr, tile)  # the next sweep runs with the fastest of this one
            print(f"  {label} fastest {attr} {tile}: {1e3 * seconds:.2f} ms (TROT_TILE_"
                  f"{attr.split('_')[0]}={','.join(map(str, tile))})", flush=True)
    for attr, tile in defaults.items():
        setattr(gg, attr, tile)


def grouped_bench(measure, m_data, qa, qb, args):
    """--stages grouped (module docstring): the first walker converted at the measurement bond, its environments swept
    left to right eagerly (op by op, no large compile), and at each bond of --grouped-sites the trial GEMM of the
    energy's environment (with --mpo) and of the overlap's timed in every variant (grouped_site), and with --mpo the
    whole site steps in their three versions (site_steps). The kernel's tiles are swept at the first bond of each
    kind; the fastest is used after."""
    from trot.gmps import grouped_gemm as gg

    layout, T = measure.kernels.overlap_plan.layout, m_data.trial
    dense_engine.dense_converter(measure.plan)  # builds the plan's channel converters
    conv_a, conv_b = measure.plan.caches["dense_channel_converters"]
    start = time.perf_counter()
    alpha, n_a, _ = jax.block_until_ready(jax.jit(conv_a.convert)(qa[0]))
    beta, n_b, _ = jax.block_until_ready(jax.jit(conv_b.convert)(qb[0]))
    print(f"  grouped: walker 0 converted at bond {measure.plan.dense.chi} in {time.perf_counter() - start:.1f} s "
          f"(with the compile)", flush=True)
    signs = [dense_engine._parity_signs(q) for q in n_b[:-1]]
    dressed = [dense_engine._dressed(B, s) for B, s in zip(beta, signs)]
    offsets = gg.mpo_channel_offsets(m_data.htrial) if args.mpo else None
    L = len(layout.sites)
    sites = sorted(set(args.grouped_sites or (L // 4, L // 2, 3 * L // 4)))
    tiles = {"energy": GROUPED_TILES, "overlap": GROUPED_TILES}
    E_mpo, E_overlap = jnp.ones((1, 1, 1, 1)), jnp.ones((1, 1, 1))
    for i in range(max(sites) + 1):
        site = layout.sites[i]
        if i in sites:
            kinds = ([("energy", E_mpo, offsets[i])] if args.mpo else []) + [("overlap", E_overlap, None)]
            for name, E, channel_offsets in kinds:
                if not args.grouped_steps_only:
                    best = grouped_site(f"bond {i} {name}", E.reshape(-1, E.shape[-1]), n_a[i], n_b[i],
                                        channel_offsets, site, layout.sectors[i], T[i][0], args, tiles[name],
                                        plain=len(tiles[name]) > 1)  # the plain GEMM once per kind, at its first bond
                    if best is not None:
                        tiles[name] = [best]
            if args.mpo:  # the whole site step: dense, grouped (stage 1), fused (stage 2)
                stack = E_overlap[:, :, None, :] * jnp.arange(1.0, 17.0)[None, None, :, None]  # 16 szsz members
                site_steps(f"bond {i}", E_mpo, stack, T[i], alpha[i], beta[i], signs[i], m_data.htrial[i], site,
                           n_a[i], n_b[i], offsets[i], i, layout, args, (n_a[i + 1], n_b[i + 1]),
                           sweep=args.tile_sweep and i == sites[0])
                del stack
        if args.mpo:  # to the next bond in the --sweep-gemm version
            g = step_labels(args.sweep_gemm, n_a[i], n_b[i], step_specs(layout, i), offsets[i], L,
                            (n_a[i + 1], n_b[i + 1]))
            with trial_gemm(args.sweep_gemm):
                E_mpo = dense_engine._forward_mpo_gather(E_mpo, T[i], alpha[i], beta[i], signs[i], m_data.htrial[i],
                                                         site, g)
        E_overlap = dense_engine._forward(E_overlap, T[i], alpha[i], dressed[i], site).sum(axis=2)


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
    rows, values = [], {}

    def stage(name, fn, inputs, per_block, of_half_step=0.0, repeats=None):
        """repeats (default --repeats) timed calls after a warm-up; 0: the time of the first call itself."""
        repeats = args.repeats if repeats is None else repeats
        try:
            compiled, compile_seconds = compile_timed(fn, inputs)
            memory = compiled.memory_analysis()  # XLA's buffers: an out-of-memory stage still says what it needed
            if memory is not None:
                print(f"  {kind} {name}: XLA memory (GB) temp {memory.temp_size_in_bytes / 2 ** 30:.2f}, argument "
                      f"{memory.argument_size_in_bytes / 2 ** 30:.2f}, output {memory.output_size_in_bytes / 2 ** 30:.2f}",
                      flush=True)
            start = time.perf_counter()
            out = jax.block_until_ready(compiled(*inputs))
            seconds = time.perf_counter() - start if repeats <= 0 else timed(compiled, inputs, repeats)
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
        ns = min(args.stage_walkers or nw, nw)  # the conversion, right_envs and field_sweep stages' walkers
        scale = nw / ns
        conversion, right_envs, field_sweep = stage_functions(kind, run)
        converted, _ = stage("conversion", conversion, (qa[:ns], qb[:ns]), steps * scale, of_half_step=scale)
        envs = None
        if converted is not None:
            envs, _ = stage("right_envs", right_envs, (converted, data), steps * scale, of_half_step=scale)
        if envs is not None:
            blocks, right, gauge = envs
            prefactor = da[:ns] * db[:ns] * gauge
            now = prefactor * right[0][:, 0, 0, 0] if kind == "dense" else prefactor * right[0][:, 0, 0, 0, 0]
            values["overlaps"] = np.asarray(now)
            print(f"  {kind} the first overlaps (TROT_RIGHT_ENVS={dense_engine.RIGHT_ENVS}): "
                  f"{np.array2string(values['overlaps'][:4], precision=13)}", flush=True)
            randoms = jax.random.uniform(jax.random.PRNGKey(1), (ns, L))
            stage("field_sweep", field_sweep, (wu[:ns], wd[:ns], randoms, blocks, prefactor, right, now, data),
                  steps / 2 * scale, of_half_step=0.5 * scale)
            del blocks, right
        del converted, envs
    else:
        qa, _ = jax.block_until_ready(jax.jit(kernels.batch_qr)(wu))
        qb, _ = jax.block_until_ready(jax.jit(kernels.batch_qr)(wd))

    nm = min(args.measure_walkers or nw, nw)  # the measurement stages' walkers
    qa, qb = qa[:nm], qb[:nm]
    if "charges" in args.stages and kind == "dense":  # both reports need the trial in sector blocks (DenseSpec.trial)
        if kernels.overlap_plan.layout is not None:
            charge_report(f"{kind} propagation (bond {args.chi_w})", walker_labels(run.ops.plan, qa, qb),
                          kernels.overlap_plan.layout)
        if measure.plan.dense is not None and measure.kernels.overlap_plan.layout is not None:
            labels, layout = walker_labels(measure.plan, qa, qb), measure.kernels.overlap_plan.layout
            charge_report(f"{kind} measurement overlap and szsz (bond {measure.plan.dense.chi})", labels, layout)
            if args.mpo:
                mpo = dense_engine.mpo_site_tensors(hubbard_mpo_from_h1(h1, U))
                charge_report(f"{kind} measurement energy (bond {measure.plan.dense.chi}, MPO)", labels, layout,
                              mpo_channel_offsets(mpo))
    if "grouped" in args.stages:
        if kind != "dense" or measure.plan.dense is None or measure.kernels.overlap_plan.layout is None:
            raise SystemExit("--stages grouped needs the dense engine with the trial in sector blocks")
        grouped_bench(measure, m_data, qa, qb, args)
    fused = fused_measurement(measure, ("szsz",), m_data) if "fused" in args.stages else None
    if "fused" in args.stages and fused is None:
        raise SystemExit("--stages fused needs a dense plan with the MPO energy (--engines dense --mpo)")
    for k in args.measure_chunks:
        n = engine.divisor_at_least(nm, -(-nm // k))
        energy = lambda a, b, d, n=n: engine.chunked(lambda x, y: measure.kernels.energies(x, y, d), n, a, b)
        szsz = lambda a, b, d, n=n: engine.chunked(lambda x, y: BATCHED_OBSERVABLES["szsz"](measure)(x, y, d), n, a, b)
        scale = nw / nm  # the block measures all nw walkers
        if "energy" in args.stages:
            energies, _ = stage(f"energy@{nm // n}", energy, (qa, qb, m_data), scale)
            if energies is not None:
                energies = values[f"energies@{nm // n}"] = np.asarray(energies)
                print(f"  {kind} local energies of these walkers: {energy_summary(energies)}", flush=True)
                print(f"  {kind} the first ones (TROT_MPO_KERNEL={dense_engine.MPO_KERNEL}): "
                      f"{np.array2string(energies[:6], precision=13)}", flush=True)
                if args.compare_stored and k == args.measure_chunks[0]:
                    compare_with_stored(run, trial, htrial, ham, qa, qb, energies, n, args)
        if "szsz" in args.stages:
            stage(f"szsz@{nm // n}", szsz, (qa, qb, m_data), scale)
        if fused is not None:
            print(f"  {kind} fused@{nm // n}: TROT_MPO_KERNEL={dense_engine.MPO_KERNEL}, TROT_MEASURE_CONVERT_CHUNK="
                  f"{engine.MEASURE_CONVERT_CHUNK} ({engine.convert_chunks(nm) if engine.MEASURE_CONVERT_CHUNK > 0 else 0}"
                  f" conversion chunks), TROT_MEASURE_BARRIER={int(engine.MEASURE_BARRIER)}", flush=True)
            out, _ = stage(f"fused@{nm // n}", lambda a, b, d, n=n: fused.chunked(a, b, d, n), (qa, qb, m_data), scale,
                           repeats=args.measure_repeats)
            if out is not None:
                energies, matrices = (np.asarray(x) for x in out)
                values[f"fused_energies@{nm // n}"], values[f"fused_szsz@{nm // n}"] = energies, matrices
                print(f"  {kind} fused local energies: {energy_summary(energies)}; the first ones "
                      f"{np.array2string(energies[:6], precision=13)}; szsz of the first walker, row 0: "
                      f"{np.array2string(matrices[0, 0, :4], precision=13)}", flush=True)
    if args.save_values:
        path = Path(f"{args.save_values}_{kind}.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)
        print(f"  {kind} values written to {path}: {', '.join(values)}", flush=True)
    stats = jax.devices()[0].memory_stats() or {}
    if stats:
        print(f"  {kind} device peak {stats.get('peak_bytes_in_use', 0) / 2 ** 30:.1f} GB of "
              f"{stats.get('bytes_limit', 0) / 2 ** 30:.1f} GB", flush=True)
    return rows


def summarize(rows):
    """Per engine the half step against its stages, and per measurement chunk size the measurement of a block (the
    stages scaled to all walkers) with the projected block when the half steps were timed."""
    for kind in sorted({r["engine"] for r in rows}):
        mine = {r["stage"]: r for r in rows if r["engine"] == kind}
        seconds = lambda name: mine[name]["ms"] * mine[name]["per_block"] / 1e3 if name in mine else 0.0
        half = next((r for name, r in mine.items() if name.startswith("half_step")), None)
        propagation = 0.0
        if half is not None:
            parts = sum(r["ms"] * r["of_half_step"] for r in mine.values() if r["of_half_step"])
            print(f"{kind}: half step {half['ms'] / (100 / half['per_block']):.1f} ms measured in the scan, its stages "
                  f"{parts:.1f} ms (qr x2, conversion, right_envs, half a sweep)")
            for name, r in mine.items():
                if r["of_half_step"]:
                    print(f"    {name:<12} {r['ms'] * r['of_half_step']:8.1f} ms per half step "
                          f"({100 * r['ms'] * r['of_half_step'] / parts:4.1f}%)")
            propagation = seconds(half["stage"]) + 2 * mine["walker_qr"]["ms"] / 1e3
        for size in sorted({name.split("@")[1] for name in mine if "@" in name}, key=int):
            if f"fused@{size}" in mine:
                measurement, what = seconds(f"fused@{size}"), f"fused {seconds(f'fused@{size}'):.1f} s"
            else:  # szsz absent with --stages without szsz
                measurement = seconds(f"energy@{size}") + seconds(f"szsz@{size}")
                what = f"energy {seconds(f'energy@{size}'):.1f} s, szsz {seconds(f'szsz@{size}'):.1f} s"
            print(f"{kind}: measurement in chunks of {size} per block {measurement:.1f} s ({what})"
                  + (f"; projected block {propagation + measurement:.1f} s (propagation {propagation:.1f} s)"
                     if half is not None else ""))


def compare_values(path_a, path_b):
    """How far two --save-values files are apart, key by key: the largest difference relative to the largest entry of
    the first, and the median relative difference per entry; then per walker its largest difference relative to its
    largest entry of the first: how many walkers lie above 1e-9, 1e-7 and 1e-5, and the worst five. A key of one
    measurement chunk size pairs with the same value at another one (fused_energies@2 with fused_energies@4)."""
    with np.load(path_a) as a, np.load(path_b) as b:
        pairs, left = [], set(b.files) - set(a.files)
        for key in sorted(a.files):
            other = key if key in b.files else \
                next((k for k in sorted(left) if k.split("@")[0] == key.split("@")[0]), None)
            left.discard(other)
            if other is None:
                print(f"  {key}: only in {path_a}")
            else:
                pairs.append((key, other))
        for key in sorted(left):
            print(f"  {key}: only in {path_b}")
        for key, other in pairs:
            x, y = np.asarray(a[key]), np.asarray(b[other])
            if other != key:
                key = f"{key} against {other}"
            if x.shape != y.shape:
                print(f"  {key}: shapes {x.shape} and {y.shape}")
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            scale = np.abs(x[finite]).max(initial=0.0)
            delta = np.abs(x - y)[finite]
            relative = delta / np.maximum(np.abs(x[finite]), 1e-300)
            print(f"  {key} {x.shape}: max |A - B| {delta.max(initial=0.0):.3e} ({delta.max(initial=0.0) / max(scale, 1e-300):.1e}"
                  f" of max |A|), median relative {np.median(relative) if relative.size else 0.0:.1e}, "
                  f"{int((~finite).sum())} not finite in either")
            if x.ndim and len(x):
                # per walker (the first axis): its largest difference over its largest |A|, a non-finite one counted
                # as the worst
                rows = lambda z: z.reshape(len(x), -1)
                walker = rows(np.abs(x - y)).max(axis=1) / np.maximum(rows(np.abs(x)).max(axis=1), 1e-300)
                walker = np.where(np.isfinite(walker), walker, np.inf)
                worst = np.argsort(walker)[::-1][:5]
                shown = (lambda i: f"{i}: {walker[i]:.1e} ({x[i]:.10f} vs {y[i]:.10f})") if x.ndim == 1 else \
                    (lambda i: f"{i}: {walker[i]:.1e}")
                print("    per walker: " + ", ".join(f"{int((walker > t).sum())} above {t:.0e}" for t in (1e-9, 1e-7, 1e-5))
                      + f" of {len(x)}; the worst " + ", ".join(shown(i) for i in worst))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", nargs="+", default=["sector", "dense"], choices=["sector", "dense"])
    parser.add_argument("--chi", type=int, default=256, help="the saved DMRG trial's bond")
    parser.add_argument("--trial-file", type=Path, default=None,
                        help="a saved trial instead of dmrg_chi<--chi>.npz (e.g. a compressed one), as "
                             "ex_6x6_doped_diag.py --trial-file")
    parser.add_argument("--side", type=int, default=6,
                        help="the open side x side lattice (default 6); another side needs --trial-file and --walkers")
    parser.add_argument("--chi-w", type=int, default=16)
    parser.add_argument("--orbital-plan", default="rank_exact", choices=["adaptive", "rank_exact", "maximal"],
                        help="gate plan of the walker conversion (the production default is rank_exact)")
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
                             "loaded; the szsz stage runs on the same dense legs (dense_engine.szsz_contract)")
    parser.add_argument("--stages", nargs="+", default=["propagation", "energy", "szsz"],
                        choices=["propagation", "energy", "szsz", "fused", "charges", "grouped"],
                        help="what to compile and time (each measurement chunk size compiles again: ~20 min at chi_T "
                             "512); fused: energy and szsz as the block measures them (a dense plan with --mpo)")
    parser.add_argument("--strict", action="store_true",
                        help="fail (nonzero exit) when a stage runs out of device memory instead of skipping it: a "
                             "feasibility test that later jobs can depend on (sbatch --dependency=afterok)")
    parser.add_argument("--dense-move", default="gram", choices=["gram", "cans"])
    parser.add_argument("--dense-eigh", default="default", choices=["default", "jacobi"],
                        help="the dense conversion's eigh (QmcParamsMps.dense_eigh)")
    parser.add_argument("--tile-sweep", action="store_true",
                        help="grouped stage: at the first bond also time the fused and the label-block kernels "
                             "(TROT_TRIAL_GEMM fused, blocked) over the tiles of FUSED_TILES and CONTRACT_TILES")
    parser.add_argument("--sweep-gemm", default="dense", choices=STEP_MODES,
                        help="grouped stage: the TROT_TRIAL_GEMM version that moves the energy's environment from bond "
                             "to bond (eagerly, so dense holds all its intermediates at once: the chi_T 2048 run "
                             "7390780 peaked at 62.4 GB; blocked for chi_T 4096)")
    parser.add_argument("--walkers", default=DEFAULT_WALKERS, help="saved walkers (up / dn)")
    parser.add_argument("--n-walkers", type=int, default=400)
    parser.add_argument("--half-steps", type=int, default=4, help="half steps in the timed scan (even: half sweep)")
    parser.add_argument("--measure-chunks", type=int, nargs="+", default=[20, 4],
                        help="walkers per measurement chunk to time (20: the default, 4: TROT_MEASURE_CHUNK=4)")
    parser.add_argument("--measure-walkers", type=int, default=0,
                        help="walkers of the measurement stages, the first ones (0: all --n-walkers); their per-block "
                             "seconds are scaled to --n-walkers")
    parser.add_argument("--grouped-sites", type=int, nargs="+", default=None,
                        help="bonds of the grouped stage (default: a quarter, half and three quarters of the lattice)")
    parser.add_argument("--grouped-steps-only", action="store_true",
                        help="grouped stage: only the whole site steps (dense, grouped, fused), not the trial GEMM's "
                             "variants (tiles, sorted rows, hybrid, cuBLAS references)")
    parser.add_argument("--stage-walkers", type=int, default=0,
                        help="walkers of the conversion, right_envs and field_sweep stages, the first ones (0: all "
                             "--n-walkers; right_envs of 200 walkers ran out of memory at chi_T 2048, job 7364937); "
                             "their per-block seconds are scaled to --n-walkers")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measure-repeats", type=int, default=0,
                        help="timed calls of the fused stage after its first (0: the first call's time; ~10 s per "
                             "chunk of 2 walkers on the 8x8 at chi_T 2048 and bond 64)")
    parser.add_argument("--profile", default="", help="directory for the half steps' traces (empty: no trace)")
    parser.add_argument("--save-values", default="",
                        help="write the overlaps and measured values to <this>_<engine>.npz (for --compare)")
    parser.add_argument("--compare", nargs=2, type=Path, default=None, metavar=("A", "B"),
                        help="print how far two --save-values files are apart and exit")
    parser.add_argument("--out", type=Path, default=None, help="write the rows as JSON")
    args = parser.parse_args(argv)
    if args.compare:
        print(f"{args.compare[1]} against {args.compare[0]}:")
        compare_values(*args.compare)
        return

    h1 = hubbard_h1(args.side)
    trial, htrial, nelec = load_trial(args.trial_file or TRIALS / f"dmrg_chi{args.chi}.npz", 90.0, h1=h1,
                                      with_htrial=(not args.mpo) or args.compare_stored)
    print(f"device {jax.devices()[0].device_kind}; trial chi {args.chi}, walker bond {args.chi_w}, measurement bond "
          f"{args.measure_chi or args.chi_w}, energy dedup {engine.ENERGY_DEDUP}, measurement chunks "
          f"{args.measure_chunks}; walkers {args.walkers} (first {args.n_walkers}, measured "
          f"{args.measure_walkers or args.n_walkers}); TROT_RIGHT_ENVS={dense_engine.RIGHT_ENVS}, TROT_MPO_KERNEL="
          f"{dense_engine.MPO_KERNEL}, TROT_MEASURE_CONVERT_CHUNK={engine.MEASURE_CONVERT_CHUNK}, TROT_MEASURE_BARRIER="
          f"{int(engine.MEASURE_BARRIER)}, TROT_SZSZ_GROUP={dense_engine.SZSZ_GROUP}, "
          f"TROT_TRIAL_GEMM={dense_engine.TRIAL_GEMM}, TROT_PROP_GEMM={dense_engine.PROP_GEMM}", flush=True)
    rows = []
    for kind in args.engines:
        print(f"== {kind} engine", flush=True)
        rows += profile_engine(kind, args, trial, htrial, nelec, h1)
    summarize(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
