"""Do the eigensolvers keep the zeros of a block-structured matrix? (the dense conversion's bond-32 labels)

The dense conversion (trot.gmps.dense_conversion) labels each kept state with an electron number. The label is exact if
the eigensolver returns sector-pure eigenvectors of a Gram matrix that is block diagonal in the electron number (the
blocks interleaved, as the rows (a, p) of a two-site matrix are). At bond 16 (32 x 32 Gram matrices) the labels were
exactly right; at bond 32 (64 x 64) they were not. This script isolates the solver. It uses batches of random
symmetric PSD matrices with the same structure: interleaved sector labels, exactly zero off-sector entries, many zero
eigenvalues, weights spread over 16 decades, and exact and near (1e-12) degeneracies across sectors. It tries sizes
around 32 with the solvers
  default   jnp.linalg.eigh on the device (XLA's choice: batched Jacobi up to 32 x 32 on GPUs),
  jacobi    jax.lax.linalg.eigh(implementation=JACOBI) on the device (GPU only),
  lapack    numpy.linalg.eigh on the host (a subset of the batch),
and the conversion's per-charge shifts (G + gap * trace * diag(labels), gap 0, 1e-7 and 2). For each it reports
  exact_pure  share of the eigenvectors whose off-sector entries are all exactly zero,
  offlabel    the largest off-sector entry of an eigenvector (outside its dominant sector),
  heavy       the largest eigenvalue (the trace is 1) of an eigenvector with more than 1e-6 of its weight off-sector,
  residual    max |Gs v - lambda v| / max |Gs|, and the compile and run time of the batch.
If default loses exact purity above 32 and jacobi does not, the bond-32 labels fail in the solver, and a sector-
preserving solver (or snapping to the sectors after it) is the fix. The compile times at 96 and 128 also test whether
the eigh is what stalled the bond-64 benchmark.

    ./submit_ex_2d.sh h100 bench_eigh_blocks.py --sizes 16 24 32 33 48 64 96 128
"""
from trot.config import configure_once

configure_once()  # float64

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

SOLVERS = ("default", "jacobi", "lapack")


def block_batch(n, batch, seed, sectors=6, zero_fraction=0.3):
    """(G (batch, n, n) of trace 1, labels (batch, n)): per matrix, random interleaved sector labels and, per sector, a
    random orthogonal eigenbasis with eigenvalues drawn from one pool shared by the sectors (exact degeneracies across
    sectors; half of them perturbed by 1e-12, near degeneracies), log-uniform over 16 decades, some exactly zero. The
    off-sector entries are exactly zero."""
    rng = np.random.default_rng(seed)
    G, labels = np.zeros((batch, n, n)), np.zeros((batch, n))
    for b in range(batch):
        lab = rng.integers(0, sectors, size=n)
        pool = 10.0 ** (-16.0 * rng.random(n))
        for s in np.unique(lab):
            idx = np.flatnonzero(lab == s)
            d = len(idx)
            lam = rng.choice(pool, size=d)
            lam = np.where(rng.random(d) < 0.5, lam * (1.0 + 1.0e-12 * rng.normal(size=d)), lam)
            lam[rng.random(d) < zero_fraction] = 0.0
            U = np.linalg.qr(rng.normal(size=(d, d)))[0]
            G[b][np.ix_(idx, idx)] = (U * lam) @ U.T
        G[b] = 0.5 * (G[b] + G[b].T)
        G[b] /= max(np.trace(G[b]), 1.0e-300)
        labels[b] = lab
    return G, labels


def metrics(G, Gs, labels, w, V, sectors):
    """The module docstring's numbers for eigenpairs (w, V) of the shifted batch Gs (G unshifted, trace 1)."""
    onehot = labels[..., None] == np.arange(sectors)  # (b, n, s)
    weights = np.einsum("bis,bik->bsk", onehot, V * V)  # each eigenvector's weight per sector
    dominant = weights.argmax(axis=1)  # (b, k)
    outside = labels[:, :, None] != dominant[:, None, :]  # (b, n, k)
    off = np.where(outside, np.abs(V), 0.0).max(axis=1)  # (b, k)
    off_weight = np.where(outside, V * V, 0.0).sum(axis=1)
    rayleigh = np.einsum("bik,bij,bjk->bk", V, G, V)
    residual = np.abs(Gs @ V - V * w[:, None, :]).max() / max(np.abs(Gs).max(), 1.0e-300)
    return dict(exact_pure=float((off == 0.0).mean()), offlabel=float(off.max()),
                heavy=float(np.where(off_weight > 1.0e-6, rayleigh, 0.0).max()), residual=float(residual))


def device_solver(name):
    if name == "jacobi":
        def solve(A):
            V, w = lax.linalg.eigh(A, implementation=lax.linalg.EighImplementation.JACOBI)
            return w, V
        return solve
    return jnp.linalg.eigh


def run(name, Gs, repeats):
    """(w, V, compile seconds, median run seconds) of one solver on the batch."""
    if name == "lapack":
        start = time.perf_counter()
        w, V = np.linalg.eigh(Gs)
        return w, V, 0.0, time.perf_counter() - start
    x = jnp.asarray(Gs)
    start = time.perf_counter()
    compiled = jax.jit(device_solver(name)).lower(x).compile()
    compile_seconds = time.perf_counter() - start
    w, V = jax.block_until_ready(compiled(x))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(x))
        times.append(time.perf_counter() - start)
    return np.asarray(w), np.asarray(V), compile_seconds, float(np.median(times))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32, 33, 48, 64, 96, 128])
    parser.add_argument("--gaps", type=float, nargs="+", default=[0.0, 1.0e-7, 2.0])
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS), choices=list(SOLVERS))
    parser.add_argument("--batch", type=int, default=400, help="matrices per device batch (the walkers)")
    parser.add_argument("--lapack-batch", type=int, default=40, help="matrices for the host solver")
    parser.add_argument("--sectors", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    device = jax.devices()[0]
    print(f"device {device.device_kind}; batch {args.batch} (lapack {args.lapack_batch}); {args.sectors} sectors; "
          f"jax {jax.__version__}", flush=True)
    print(f"{'n':>4} {'gap':>7} {'solver':>7} {'exact_pure':>10} {'offlabel':>9} {'heavy':>9} {'residual':>9} "
          f"{'compile_s':>9} {'run_ms':>9}", flush=True)
    rows = []
    for n in args.sizes:
        G, labels = block_batch(n, args.batch, seed=n, sectors=args.sectors)
        for gap in args.gaps:
            Gs = G + gap * labels[:, :, None] * np.eye(n)[None]
            for name in args.solvers:
                if name == "jacobi" and device.platform == "cpu":
                    continue
                k = args.lapack_batch if name == "lapack" else args.batch
                try:
                    w, V, compile_seconds, seconds = run(name, Gs[:k], args.repeats)
                except Exception as error:  # e.g. a solver that does not support this size
                    print(f"{n:4d} {gap:7.0e} {name:>7} failed: {type(error).__name__}: {error}", flush=True)
                    continue
                m = metrics(G[:k], Gs[:k], labels[:k], w, V, args.sectors)
                print(f"{n:4d} {gap:7.0e} {name:>7} {m['exact_pure']:10.4f} {m['offlabel']:9.1e} {m['heavy']:9.1e} "
                      f"{m['residual']:9.1e} {compile_seconds:9.1f} {1e3 * seconds:9.2f}", flush=True)
                rows.append(dict(n=n, gap=gap, solver=name, batch=k, compile_s=compile_seconds, run_ms=1e3 * seconds,
                                 **m))
    if args.out:
        args.out.write_text(json.dumps(dict(device=device.device_kind, rows=rows), indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
