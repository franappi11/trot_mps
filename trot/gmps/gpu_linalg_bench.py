"""Microbenchmark of the batched dense linear algebra mps_cpmc_gpu relies on.

For every op and matrix size n, one jitted call on a batch of b matrices is
timed and reported as microseconds per matrix, next to the time of a single
matrix (batch 1). A primitive lowered to one batched kernel costs far less per
matrix in a batch than alone ("batched"). One lowered to a loop over matrices
(a cuSOLVER routine with no batched variant) costs about the same per matrix as
a single call ("LOOPED"). A batched kernel whose per-matrix cost stops falling
with the batch is simply compute-bound, not looped. The custom-call targets of
each compiled op are printed so the lowering can be read off directly.

What the ops are in mps_cpmc_gpu (linalg="batched"):
  eigh               truncated gate sectors (one call per op and Gram side)
  qr                 exact sectors of moves and splits, and complete-QR empty modes
  cholesky_qr2       walker orthonormalisation (walker_qr="cholesky")
  det                gauge of every conversion
  gram_factor        the truncated-sector pattern on one padded sector batch

    python gpu_linalg_bench.py [--sizes 4 8 16 32 48 64 96 128] [--batches 1000 10000 100000]
"""
import argparse
import json
import os
import re
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mps_cpmc_gpu import cholesky_qr2  # noqa: E402


def gram_factor(M):
    """factor_eigh on a padded batch: normalise, row Gram, eigh, Q, R = Q^T M."""
    X = M / jnp.sqrt(jnp.sum(M * M, axis=(-2, -1), keepdims=True))
    _, U = jnp.linalg.eigh(X @ jnp.swapaxes(X, -1, -2))
    Q = U[..., ::-1][..., :M.shape[-1]]
    return Q, jnp.swapaxes(Q, -1, -2) @ M


# name -> (function, input builder: (rng, batch, n) -> array)
def _spd(rng, b, n):
    X = rng.standard_normal((b, n, n))
    return X @ np.swapaxes(X, -1, -2) / n + np.eye(n)


OPS = {
    "eigh": (lambda A: jnp.linalg.eigh(A), _spd),
    "qr": (lambda A: jnp.linalg.qr(A), lambda rng, b, n: rng.standard_normal((b, 2 * n, n))),
    "cholesky_qr2": (jax.vmap(cholesky_qr2), lambda rng, b, n: rng.standard_normal((b, 2 * n, n))),
    "cholesky": (lambda A: jnp.linalg.cholesky(A), _spd),
    "det": (lambda A: jnp.linalg.det(A), lambda rng, b, n: rng.standard_normal((b, n, n))),
    "matmul": (lambda A: A @ A, lambda rng, b, n: rng.standard_normal((b, n, n))),
    "gram_factor": (gram_factor, lambda rng, b, n: rng.standard_normal((b, n, 2 * n))),
}


def custom_calls(compiled):
    try:
        text = compiled.as_text()
    except Exception:
        return []
    return sorted(set(re.findall(r'custom_call_target="([^"]+)"', text)))


def time_call(fn, x, repeats):
    jax.block_until_ready(fn(x))  # warm-up
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn(x))
        best = min(best, time.perf_counter() - t)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+", default=list(OPS))
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 24, 32, 40, 48, 64, 96, 128])
    parser.add_argument("--batches", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mem-fraction", type=float, default=0.5, help="skip inputs larger than this share")
    parser.add_argument("--out", default="gpu_linalg_bench.jsonl")
    args = parser.parse_args()

    device = jax.devices()[0]
    limit = (device.memory_stats() or {}).get("bytes_limit") or 16e9
    print(f"jax {jax.__version__}, backend {jax.default_backend()}, device {device.device_kind}, "
          f"memory limit {limit / 1e9:.1f} GB")
    if jax.default_backend() != "gpu":
        print("WARNING: not running on a GPU; the verdicts below describe the CPU backend")

    rng = np.random.default_rng(0)
    records = []
    batches = sorted({1, *args.batches})
    header = f"{'op':>13} {'n':>4} " + " ".join(f"{'us/mat@' + str(b):>14}" for b in batches)
    print("\n" + header + "   alone/batched  verdict   custom calls")
    for name in args.ops:
        fn, build = OPS[name]
        jitted = jax.jit(fn)
        for n in args.sizes:
            per_matrix, targets = {}, []
            for b in batches:
                bytes_needed = 6 * b * 2 * (2 * n) ** 2 * 8  # input, outputs and workspace
                if bytes_needed > args.mem_fraction * limit:
                    continue
                x = jax.device_put(build(rng, b, n))
                try:
                    compiled = jitted.lower(x).compile()
                    targets = custom_calls(compiled)
                    seconds = time_call(compiled, x, args.repeats)
                except Exception as error:  # OOM or an unsupported size
                    print(f"{name:>13} {n:>4}  batch {b}: {type(error).__name__}: {str(error)[:120]}")
                    continue
                per_matrix[b] = 1e6 * seconds / b
                records.append(dict(op=name, n=n, batch=b, seconds=seconds, us_per_matrix=per_matrix[b],
                                    custom_calls=targets, device=device.device_kind, jax=jax.__version__))
                del x
            if not per_matrix:
                continue
            large = max(per_matrix)
            # A per-matrix loop pays roughly a whole single-matrix call per matrix;
            # a batched kernel pays a small fraction of it.
            gain = per_matrix[1] / per_matrix[large] if 1 in per_matrix and large > 1 else float("nan")
            verdict = "?" if gain != gain else "batched" if gain > 5 else "LOOPED" if gain < 2 else "partial"
            cells = " ".join(f"{per_matrix[b]:14.3f}" if b in per_matrix else f"{'-':>14}" for b in batches)
            print(f"{name:>13} {n:>4} {cells}   {gain:13.1f}  {verdict:<8}  {','.join(targets)[:80]}")
            for record in records:
                if record["op"] == name and record["n"] == n:
                    record["verdict"] = verdict
        print()

    with open(args.out, "a") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
