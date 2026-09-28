"""Why does a GPU conversion differ from the study's NumPy gmps? (allocation_study.py flags such cases.)

For one system and a few allocations, this takes the walkers the study checks (the test population's last
snapshot, both spins, from the study's cache), converts them with mps_cpmc_gpu and with allocation_study.gmps,
and prints per walker:

  state      1-F between the GPU and the NumPy state
  err gpu    1-F of the GPU state against the exact walker
  err numpy  1-F of the NumPy state against the exact walker
  dtheta     largest difference between the two codes' gate angles (channel_angles)
  mode gap   smallest gap between the eigenvalue channel_angles picks and its neighbour (occupations in
             [0, 1]); a tiny gap lets two eigensolvers pick different, equally valid modes: different circuits
  within     smallest gap, relative to the gate's largest, between a sector's last kept and first dropped
             squared singular value (NumPy side): a tie there is a free choice of kept subspace
  across     the same across sectors (smallest kept minus largest dropped): matters for per-walker selection

  why        for a walker whose states differ: "circuit" (the codes built different circuits: dtheta > 1e-8 or a
             mode gap < 1e-8), "cut tie" (a nearly tied truncation), or "?" (nothing explains it)

Two states that differ by a small angle can have errors differing by a large relative amount when the errors
themselves are small, so the error gap alone proves nothing. A near-degenerate choice resolved differently by the
two eigensolvers shows up as "circuit" or "cut tie", with the GPU better on some walkers and worse on others; the
study's errors (NumPy) and GPU costs (shapes only) stand either way. A bug shows up as "?" rows, or as the GPU
being worse on every differing walker.

    python allocation_gpu_check.py --L 12 --U 4 --bonds 2 3 4 --check-walkers 8
    python allocation_gpu_check.py --L 12 --U 4 --bonds --padded 2 3     # the padded scheme at nominal chi
Use the same --out, --seed and population options as the study run (defaults match allocation_study.py).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import allocation_study as st  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mps_cpmc_gpu as gpu  # noqa: E402


def mode_gaps(Q, plan):
    """For every block of channel_angles (a NumPy replica of its loop): the gap between the eigenvalue of the
    chosen mode and its neighbour (inf where the complete-QR null vector is used)."""
    rows = [np.asarray(Q)[i].copy() for i in range(Q.shape[0])]
    gaps = []
    for k, (B, reference) in enumerate(zip(plan.block_sizes, plan.references)):
        B = int(B)
        if B == 1:
            continue
        block = np.stack(rows[k:k + B])
        if plan.exact_for_all_walkers and not plan.occupation[k]:
            vectors, _ = np.linalg.qr(block, mode="complete")
            v = vectors[:, -1]
            gaps.append(np.inf)
        else:
            w, vectors = np.linalg.eigh(block @ block.T)
            if plan.occupation[k]:
                v, gap = vectors[:, -1], w[-1] - w[-2]
            else:
                v, gap = vectors[:, 0], w[1] - w[0]
            gaps.append(float(gap))
        v = list(np.where(v @ np.asarray(reference) < 0, -v, v))
        for j in range(B - 1, 0, -1):
            theta = np.arctan2(v[j], v[j - 1])
            c, s = np.cos(theta), np.sin(theta)
            v[j - 1] = c * v[j - 1] + s * v[j]
            p = k + j - 1
            rows[p], rows[p + 1] = c * rows[p] + s * rows[p + 1], -s * rows[p] + c * rows[p + 1]
    return gaps


def angle_difference(Q, plan):
    numpy_angles = np.array([t for _, t in st.channel_angles(np.asarray(Q), plan, xp=np)[0]])
    gpu_angles = np.array([float(t[0]) for t in gpu.device_channel_angles(jnp.asarray(Q)[None], [plan])[0]])
    if not len(numpy_angles):
        return 0.0
    d = np.abs(numpy_angles - gpu_angles)
    return float(np.max(np.minimum(d, 2 * np.pi - d)))


def main():
    parser = st.build_parser()
    parser.add_argument("--bonds", type=int, nargs="*", default=[2, 3, 4], help="frozen allocations to check")
    parser.add_argument("--padded", type=int, nargs="*", default=[], help="padded allocations (nominal chi)")
    parser.add_argument("--unions", type=int, nargs="*", default=[], help="union allocations (nominal chi)")
    parser.add_argument("--check-walkers", type=int, default=8)
    args = parser.parse_args()
    L, U, trial = args.L[0], args.U[0], args.trial[0]
    train_pop = st.population(L, U, trial, args.seed, args)
    test_pop = st.population(L, U, trial, args.seed + args.test_seed_offset, args)
    ref = train_pop["reference"]
    plan = st.make_orbital_plan(ref, "adaptive", st.EPS)
    up, dn = st.timing_walkers(test_pop)
    up, dn = up[:args.check_walkers], dn[:args.check_walkers]

    configs = []
    for b in args.bonds:
        counts = st.gmps(ref, plan, chi=b)[2]
        configs.append((f"frozen b={b}", counts, None,
                        lambda Q, tr, c=counts: st.gmps(Q, plan, counts=c, trace=tr)[0]))
    if args.padded or args.unions:
        train = st.sampling_walkers(train_pop, args.train_snaps, args.eql)
        for chi in args.padded:
            pad = st.padding([st.gmps(Q, plan, chi=chi)[2] for Q in train])
            configs.append((f"padded chi={chi}", pad, chi,
                            lambda Q, tr, c=pad, x=chi: st.gmps(Q, plan, chi=x, caps=c, trace=tr)[0]))
        for chi in args.unions:
            pad = st.padding([st.gmps(Q, plan, chi=chi)[2] for Q in train])
            configs.append((f"union chi={chi}", pad, None,
                            lambda Q, tr, c=pad: st.gmps(Q, plan, counts=c, trace=tr)[0]))

    device = jax.devices()[0].device_kind
    print(f"L={L} U={U:g} {trial}: plan {int((plan.block_sizes - 1).sum())} gates, {len(up)} walkers x 2 spins; "
          f"backend {jax.default_backend()} ({device})")
    for name, counts, dynamic_chi, reference in configs:
        bond = gpu.counts_bond_plan(plan, counts)
        convert = jax.jit(gpu.make_converter(plan, plan, bond, bond, linalg="batched", spin_batch=True,
                                             dynamic_chi=dynamic_chi).convert)
        print(f"\n{name}")
        print(f"  {'walker':>7s} {'state':>9s} {'err gpu':>9s} {'err numpy':>9s} {'dtheta':>9s} {'mode gap':>9s} "
              f"{'within':>9s} {'across':>9s}  why")
        worst_state = worst_gap = 0.0
        differing = unexplained = gpu_worse = 0
        for w, (qu, qd) in enumerate(zip(up, dn)):
            alpha, beta, _ = convert(jnp.asarray(qu), jnp.asarray(qd))
            for spin, Q, tensors in (("up", qu, alpha), ("dn", qd, beta)):
                trace = []
                want = reference(Q, trace)
                dev = [np.asarray(t) for t in tensors]
                exact, _ = st.exact_state(Q)
                state = st.fidelity_loss(("mps", want), dev)
                e_numpy, e_gpu = st.fidelity_loss(exact, want), st.fidelity_loss(exact, dev)
                worst_state = max(worst_state, state)
                worst_gap = max(worst_gap, abs(e_gpu - e_numpy) / max(e_numpy, 1e-12))
                within = min((t["within"] for t in trace), default=np.inf)
                across = min((t["across"] for t in trace), default=np.inf)
                dtheta, gap = angle_difference(Q, plan), min(mode_gaps(Q, plan), default=np.inf)
                why = "-"
                if state > 1e-9:
                    differing += 1
                    gpu_worse += int(e_gpu > e_numpy * (1 + 1e-3))
                    # explained: the two codes built different circuits (a nearly degenerate mode), or a cut was
                    # nearly tied (across can be negative for a frozen allocation, which ignores other sectors)
                    if dtheta > 1e-8 or gap < 1e-8:
                        why = "circuit"
                    elif within < 1e-6 or 0 <= across < 1e-6:
                        why = "cut tie"
                    else:
                        why = "?"
                        unexplained += 1
                print(f"  {w:5d}{spin:>2s} {state:9.1e} {e_gpu:9.2e} {e_numpy:9.2e} {dtheta:9.1e} "
                      f"{gap:9.1e} {within:9.1e} {across:9.1e}  {why}")
        if worst_state <= 1e-9:
            verdict = "GPU and NumPy states agree"
        elif unexplained == 0 and gpu_worse < differing:
            verdict = (f"every difference comes with a different circuit or a tied cut, and the GPU is worse on only "
                       f"{gpu_worse} of {differing}: near-degenerate choices, not a bug")
        elif unexplained == 0:
            verdict = (f"every difference is at a near-degenerate choice, but the GPU is worse on all {differing}: "
                       f"check more walkers (--check-walkers)")
        else:
            verdict = (f"{unexplained} differences with no near-degenerate choice to explain them: "
                       f"a real discrepancy between the GPU and NumPy conversions")
        print(f"  -> worst state difference {worst_state:.1e}, worst relative error difference {worst_gap:.1e}; "
              f"{verdict}")


if __name__ == "__main__":
    main()
